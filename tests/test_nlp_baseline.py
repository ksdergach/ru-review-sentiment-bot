"""N02: frozen-artifact checks, deterministic error coverage and no test access."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src import train_baseline as b
from tools import report_nlp_baseline as report
from tools import restore_nlp_splits
from test_train_baseline import prepare_project, tiny_validation, valid_config


def records():
    return [{"record_id": f"val-{a}-{c}-{i}", "true_label_id": a, "predicted_label_id": c,
             "word_count": i + 1, "text_sha256": str(i), "review_text": f"example-{a}-{c}-{i}"}
            for a, c in report.PAIRS for i in range(20)]


def test_selection_has_all_pairs_shortest_errors_and_is_order_independent():
    rows = records()
    selected = report.select_errors(rows)
    assert len(selected) == len({r["record_id"] for r in selected}) == 40
    assert all(r["word_count"] == 1 for r in selected[:4])
    assert {(r["true_label_id"], r["predicted_label_id"]) for r in selected} == set(report.PAIRS)
    assert report.select_errors(list(reversed(rows))) == selected
    assert report.select_errors(rows, seed=43) != selected


@pytest.mark.parametrize("count", [0, 1, 3, 7, 39])
def test_selection_takes_all_when_fewer_than_forty(count):
    rows = records()[:count]
    correct = {"record_id": "correct", "true_label_id": 2, "predicted_label_id": 2, "word_count": 1}
    selected = report.select_errors(rows + [correct])
    assert {r["record_id"] for r in selected} == {r["record_id"] for r in rows}


def test_selection_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        report.select_errors([records()[0]] * 2)


@pytest.fixture
def fitted_project(tmp_path, monkeypatch):
    cfg = valid_config()
    cfg.update(report_dir="reports/nlp/training", model_dir="models/nlp_baseline")
    validation = tiny_validation()
    # Deliberately produce actual errors while retaining all three true classes.
    validation["review_text"] = [f"неизвестное{i} <script>alert(1)</script>" for i in range(len(validation))]
    config_path, reads = prepare_project(tmp_path, monkeypatch, config=cfg, validation=validation)
    config_path = config_path.rename(config_path.with_name("nlp_baseline.json"))
    for relative in ("src/train_baseline.py", "tools/report_nlp_baseline.py", "tools/restore_nlp_splits.py"):
        target = tmp_path / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(report.ROOT / relative, target)
    b.run_training(config_path, project_root=tmp_path)
    result, chosen, _ = report.evaluate(tmp_path, allow_missing_annotations=True)
    b.write_json(tmp_path / "reports/nlp/baseline_error_annotations.json", {
        "author": "Codex (AI agent)", "human_review_status": "pending",
        "validation_predictions_sha256": result["validation_predictions_sha256"],
        "model_state_sha256": result["model"]["state_sha256"],
        "errors": [{**{k: r[k] for k in ("record_id", "true_label_id", "predicted_label_id", "text_sha256")},
                    "tags": ["unclear"], "explanation": "Причина не установлена."} for r in chosen],
    })
    reads.clear()
    return tmp_path, reads


def test_evaluate_is_read_only_no_fit_and_reads_only_full_validation(fitted_project, monkeypatch):
    root, reads = fitted_project
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not fit")
    monkeypatch.setattr(b.TfidfVectorizer, "fit_transform", forbidden)
    monkeypatch.setattr(b.LogisticRegression, "fit", forbidden)
    result, chosen, annotations = report.evaluate(root)
    assert reads == ["validation.parquet"]
    assert result["rows"] == 32 and result["errors"] > 0
    assert result["final_test_used"] is False and result["independent_test"] is False
    assert sum(sum(row) for row in result["confusion_matrix"]["matrix"]) == 32
    assert sum(v["support"] for v in result["per_class"].values()) == 32
    assert len(chosen) == min(result["errors"], 40)
    assert all(r["nonzero_features"] == 0 for r in chosen)
    report.write_outputs(root, result, chosen, annotations)
    public = (root / "reports/nlp/baseline_errors.csv").read_text()
    assert "<script>" not in public and "review_text" not in public
    page = (root / "artifacts/nlp/baseline_errors.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    first = (root / "reports/nlp/baseline_metrics.json").read_bytes()
    report.write_outputs(root, *report.evaluate(root))
    assert (root / "reports/nlp/baseline_metrics.json").read_bytes() == first


@pytest.mark.parametrize("relative", ["models/nlp_baseline/tfidf_logreg_baseline.joblib",
                                    "data/processed/validation.parquet", "data/processed/train.parquet",
                                    "reports/data_prep/split_ids.csv", "src/train_baseline.py"])
def test_mismatched_artifacts_fail_before_predictions(fitted_project, monkeypatch, relative):
    root, _ = fitted_project
    path = root / relative
    path.write_bytes(path.read_bytes() + b"changed")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject mismatched artifact before predicting")
    monkeypatch.setattr(b, "predict_texts", forbidden)
    with pytest.raises(ValueError, match="Checksum"):
        report.evaluate(root)


def test_annotations_cannot_silently_follow_another_model_or_invent_human_review(tmp_path):
    selected = report.select_errors(records())
    data = {"author": "Codex (AI agent)", "human_review_status": "pending", "validation_predictions_sha256": "abc",
            "model_state_sha256": "model-hash",
            "errors": [{**{k: r[k] for k in ("record_id", "true_label_id", "predicted_label_id", "text_sha256")},
                        "tags": ["unclear"], "explanation": "Причина не установлена."} for r in selected]}
    path = tmp_path / "annotations.json"
    b.write_json(path, data)
    assert report.load_annotations(path, selected, "abc", "model-hash") == data
    with pytest.raises(ValueError, match="predictions"):
        report.load_annotations(path, selected, "other", "model-hash")
    with pytest.raises(ValueError, match="model state"):
        report.load_annotations(path, selected, "abc", "other-model")
    changed = copy.deepcopy(data)
    changed["errors"][0]["true_label_id"] = 9
    b.write_json(path, changed)
    with pytest.raises(ValueError, match="mismatch"):
        report.load_annotations(path, selected, "abc", "model-hash")
    changed = copy.deepcopy(data)
    changed["human_review_status"] = "confirmed"
    b.write_json(path, changed)
    with pytest.raises(ValueError, match="evidence"):
        report.load_annotations(path, selected, "abc", "model-hash")


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for directory in (root / "reports", root / "artifacts")
            if directory.exists() for p in directory.rglob("*") if p.is_file()}


def test_changed_model_with_identical_labels_cannot_reuse_annotations(fitted_project, monkeypatch):
    root, _ = fitted_project
    before, chosen, annotations = report.evaluate(root)
    report.write_outputs(root, before, chosen, annotations)
    bundle_path = root / before["model"]["path"]
    model = b.load_bundle(bundle_path)
    texts = tiny_validation()["review_text"].tolist()
    predictions = b.predict_texts(model, texts)
    model["classifier"].coef_ *= 2
    model["classifier"].intercept_ *= 2
    assert b.predict_texts(model, texts)["label_ids"] == predictions["label_ids"]
    b.save_bundle(bundle_path, model)
    for name in ("run_metadata.json", "winner_config.json"):
        path = root / "reports/nlp/training" / name
        metadata = b.read_json(path)
        metadata["bundle_sha256"] = b.sha256_file(bundle_path)
        b.write_json(path, metadata)
    saved = snapshot(root)
    monkeypatch.setattr(report, "ROOT", root)
    assert report.main([]) == 1
    assert snapshot(root) == saved
    with pytest.raises(ValueError, match="model state"):
        report.evaluate(root)


@pytest.mark.parametrize("component", ["coef", "intercept", "idf", "vocabulary", "lowercase", "classes"])
def test_state_hash_covers_inference_components(fitted_project, component):
    root, _ = fitted_project
    model = b.load_bundle(root / "models/nlp_baseline/tfidf_logreg_baseline.joblib")
    original_hash = report.model_state_sha256(model)
    if component == "coef":
        model["classifier"].coef_[0, 0] += 0.01
    elif component == "intercept":
        model["classifier"].intercept_[0] += 0.01
    elif component == "idf":
        idf = model["vectorizer"].idf_.copy()
        idf[0] += 0.01
        model["vectorizer"].idf_ = idf
    elif component == "vocabulary":
        vocabulary = model["vectorizer"].vocabulary_
        first, second = list(vocabulary)[:2]
        vocabulary[first], vocabulary[second] = vocabulary[second], vocabulary[first]
    elif component == "lowercase":
        model["vectorizer"].lowercase = not model["vectorizer"].lowercase
    else:
        model["classifier"].classes_ = model["classifier"].classes_[::-1]
    assert report.model_state_sha256(model) != original_hash


def test_state_hash_survives_reserialization_and_memory_layout(fitted_project):
    root, _ = fitted_project
    model = b.load_bundle(root / "models/nlp_baseline/tfidf_logreg_baseline.joblib")
    original_hash = report.model_state_sha256(model)
    model["classifier"].coef_ = np.asfortranarray(model["classifier"].coef_)
    model["vectorizer"].vocabulary_ = dict(reversed(list(model["vectorizer"].vocabulary_.items())))
    path = root / "reserialized.joblib"
    b.save_bundle(path, model)
    assert report.model_state_sha256(b.load_bundle(path)) == original_hash


def test_missing_annotations_fail_without_overwriting_reports(fitted_project, monkeypatch, capsys):
    root, _ = fitted_project
    report.write_outputs(root, *report.evaluate(root))
    (root / "reports/nlp/baseline_error_annotations.json").unlink()
    saved = snapshot(root)
    monkeypatch.setattr(report, "ROOT", root)
    assert report.main([]) == 1
    assert "Missing error annotations" in capsys.readouterr().err
    assert snapshot(root) == saved
    with pytest.raises(ValueError, match="annotations are required"):
        report.write_outputs(root, {}, [], None)
    assert snapshot(root) == saved
    assert report.main(["--prepare-errors"]) == 0
    packet_path = "artifacts/nlp/baseline_error_selection.json"
    after = snapshot(root)
    assert set(after) == set(saved) | {packet_path}
    assert all(after[p] == content for p, content in saved.items())
    packet = b.read_json(root / packet_path)
    assert packet["model_state_sha256"] and packet["validation_predictions_sha256"] and packet["errors"]


def test_restore_replays_manifest_without_opening_test(tmp_path, monkeypatch):
    import pandas as pd
    from src.prepare_data import build_record_id, save_prepared_split
    root = tmp_path
    (root / "reports/data_prep").mkdir(parents=True)
    (root / "reports/baseline").mkdir(parents=True)
    (root / "NLP_dataset").mkdir()
    frames, manifest = {}, []
    audit = {"status": "completed", "final_summary": {}, "source_files": {}}
    meta = {"t03_source_parquet_sha256": {}}
    for split in ("train", "validation"):
        raw = pd.DataFrame({"movie_id": [f"{split}-{i}" for i in range(4)],
                            "review_text": [f"{split} text {i}" for i in range(4)],
                            "review_sentiment": ["NEGATIVE", "NEUTRAL", "POSITIVE", "POSITIVE"],
                            "review_language": ["ru"] * 4})
        source = root / "NLP_dataset" / f"{split}.parquet"
        raw.to_parquet(source, index=False)
        meta["t03_source_parquet_sha256"][split] = b.sha256_file(source)
        audit["source_files"][split] = {"path": f"NLP_dataset/{split}.parquet"}
        raw["source_file"] = source.name
        raw["source_row"] = range(4)
        raw["record_id"] = [build_record_id(source.name, i) for i in range(4)]
        raw["label_name"] = raw["review_sentiment"].str.casefold()
        raw["label_id"] = raw["label_name"].map(b.EXPECTED_LABEL_MAPPING).astype("int64")
        frame = raw.iloc[:3].copy()  # Fourth source row was excluded by the accepted manifest.
        frames[split] = frame
        expected = root / f"expected-{split}.parquet"
        save_prepared_split(frame, expected)
        meta[f"{split}_parquet_sha256"] = b.sha256_file(expected)
        audit["final_summary"][split] = {"rows": 3, "negative": 1, "neutral": 1, "positive": 1}
        manifest.extend(frame[["record_id", "label_id"]].assign(split=split).to_dict("records"))
    manifest.append({"record_id": "test.parquet#row=00000000", "label_id": 0, "split": "test"})
    pd.DataFrame(manifest).to_csv(root / "reports/data_prep/split_ids.csv", index=False)
    b.write_json(root / "reports/data_prep/audit.json", audit)
    for name, key in (("audit.json", "t03_audit_sha256"), ("split_ids.csv", "t03_split_ids_sha256")):
        meta[key] = b.sha256_file(root / "reports/data_prep" / name)
    b.write_json(root / "reports/baseline/run_metadata.json", meta)
    original_read = pd.read_parquet
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name in {"train.parquet", "validation.parquet"}
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    result = restore_nlp_splits.restore(root)
    assert opened == ["train.parquet", "validation.parquet"]
    assert result["train"]["rows"] == result["validation"]["rows"] == 3
    assert not (root / "data/processed/test.parquet").exists()
    assert restore_nlp_splits.restore(root) == result
    (root / "data/processed/train.parquet").write_bytes(b"foreign data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        restore_nlp_splits.restore(root)
