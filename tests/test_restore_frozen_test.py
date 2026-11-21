"""Technical restoration uses synthetic fixtures, never the actual final test."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import train_baseline as base
from src.prepare_data import build_record_id, save_prepared_split
from tools import restore_frozen_test as target


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source = tmp_path / "NLP_dataset/test-00000-of-00001.parquet"
    source.parent.mkdir()
    frame = pd.DataFrame({
        "movie_id": [10, 11, 12, 13],
        "review_text": ["Синтетический минус.", "Синтетический нейтральный.",
                        "Синтетический плюс.", "Чужой язык."],
        "review_sentiment": ["NEGATIVE", "NEUTRAL", "POSITIVE", "POSITIVE"],
        "review_language": ["ru", "ru", "ru", "kk"],
    })
    frame.to_parquet(source, index=False)
    order = [2, 0, 1]
    ids = pd.DataFrame([{
        "record_id": build_record_id(source.name, i), "split": "test",
        "label_name": ["negative", "neutral", "positive"][i], "label_id": i,
    } for i in order])
    ids_path = tmp_path / "reports/data_prep/split_ids.csv"
    ids_path.parent.mkdir(parents=True)
    ids.to_csv(ids_path, index=False)
    audit = {
        "status": "completed",
        "source_files": {"test": {"path": str(source.relative_to(tmp_path)),
                                   "sha256_after": base.sha256_file(source)}},
        "final_summary": {"test": {"rows": 3, "negative": 1, "neutral": 1, "positive": 1}},
    }
    audit_path = tmp_path / "reports/data_prep/audit.json"
    write_json(audit_path, audit)
    report_path = tmp_path / "reports/nlp/training/run_metadata.json"
    report = {
        "final_test_used": False,
        "t03_audit_sha256": base.sha256_file(audit_path),
        "t03_split_ids_sha256": base.sha256_file(ids_path),
        # Accepted training metadata intentionally has no test source entry.
        "t03_source_parquet_sha256": {"train": "0" * 64, "validation": "1" * 64},
    }
    write_json(report_path, report)

    expected = frame.iloc[order].reset_index(drop=True)
    expected["source_file"] = source.name
    expected["source_row"] = order
    expected["record_id"] = ids["record_id"].tolist()
    expected["label_name"] = ["positive", "negative", "neutral"]
    expected["label_id"] = order
    expected_path = tmp_path / "expected.parquet"
    save_prepared_split(expected, expected_path)
    write_json(tmp_path / "reports/freeze.json", {
        "schema": "t13-final-text-eval-input/draft-1", "status": "approved",
        "approved_by": ["synthetic fixture"], "approved_at": "2020-01-02",
        "final_test_used_before_freeze": False,
        "models": {"baseline": {"report_path": str(report_path.relative_to(tmp_path)),
                                  "report_sha256": base.sha256_file(report_path)}},
        "data_prep": {"audit_sha256": base.sha256_file(audit_path),
                      "split_ids_sha256": base.sha256_file(ids_path)},
        "test": {"path": "data/processed/test.parquet", "rows": 3,
                 "source_parquet_sha256": base.sha256_file(source),
                 "parquet_sha256": base.sha256_file(expected_path),
                 "ids_sha256": base.ids_sha256(expected, id_column="record_id", label_column="label_id")},
    })

    def forbidden(*args, **kwargs):
        pytest.fail("Restoration must not train or make predictions")

    monkeypatch.setattr(base, "run_training", forbidden)
    monkeypatch.setattr(base, "predict_texts", forbidden)
    return tmp_path, source, frame, order, ids_path, report_path


def test_restores_exact_recorded_order_texts_and_labels_without_inference(frozen):
    root, source, source_frame, order, _, _ = frozen
    before = base.sha256_file(source)
    result = target.restore(root)
    output = pd.read_parquet(root / result["path"])
    assert output["review_text"].tolist() == source_frame.iloc[order]["review_text"].tolist()
    assert output["label_id"].tolist() == order
    assert output["source_row"].tolist() == order
    assert result["status"] == "RESTORED" and result["rows"] == 3
    assert result["model_inference_run"] is False and result["final_metrics_computed"] is False
    assert base.sha256_file(source) == before


def test_changed_source_is_rejected_before_decoding(frozen, monkeypatch):
    root, source, *_ = frozen
    source.write_bytes(b"different source")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Source must not be decoded"))
    with pytest.raises(ValueError, match="Frozen checksum mismatch: test source"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_changed_id_manifest_is_rejected(frozen):
    root, _, _, _, ids_path, _ = frozen
    ids_path.write_text("record_id,split,label_name,label_id\n")
    with pytest.raises(ValueError, match="Frozen checksum mismatch: split_ids.csv"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_label_disagreement_is_rejected(frozen):
    root, _, _, _, ids_path, report_path = frozen
    ids = pd.read_csv(ids_path)
    ids.loc[0, "label_id"] = 0
    ids.to_csv(ids_path, index=False)
    report = json.loads(report_path.read_text())
    report["t03_split_ids_sha256"] = base.sha256_file(ids_path)
    write_json(report_path, report)
    freeze_path = root / "reports/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["data_prep"]["split_ids_sha256"] = base.sha256_file(ids_path)
    freeze["models"]["baseline"]["report_sha256"] = base.sha256_file(report_path)
    freeze["test"]["ids_sha256"] = base.ids_sha256(ids, id_column="record_id", label_column="label_id")
    write_json(freeze_path, freeze)
    with pytest.raises(ValueError, match="Test labels differ"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_different_existing_output_is_preserved(frozen):
    root, *_ = frozen
    output = root / "data/processed/test.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep this original")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        target.restore(root)
    assert output.read_bytes() == b"keep this original"


def test_repeated_restoration_does_not_rewrite_matching_output(frozen, monkeypatch):
    root, *_ = frozen
    first = target.restore(root)
    output = root / first["path"]
    before = output.stat().st_mtime_ns
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Accepted output needs no decode"))
    second = target.restore(root)
    assert second["status"] == "UNCHANGED"
    assert second["parquet_sha256"] == first["parquet_sha256"]
    assert output.stat().st_mtime_ns == before


def update_freeze(root, update):
    path = root / "reports/freeze.json"
    freeze = json.loads(path.read_text())
    update(freeze)
    write_json(path, freeze)


def no_decode(monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Reject before decoding test"))


def test_unapproved_freeze_is_rejected_before_decoding(frozen, monkeypatch):
    root, *_ = frozen
    update_freeze(root, lambda f: f.update(status="draft"))
    no_decode(monkeypatch)
    with pytest.raises(ValueError, match="approved T12 freeze"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_coherently_changed_old_metadata_cannot_replace_approved_freeze(frozen, monkeypatch):
    root, _, _, _, ids_path, report_path = frozen
    ids = pd.read_csv(ids_path).iloc[::-1]
    ids.to_csv(ids_path, index=False)
    report = json.loads(report_path.read_text())
    report["t03_split_ids_sha256"] = base.sha256_file(ids_path)
    write_json(report_path, report)  # internally consistent T03/report, frozen hashes unchanged
    no_decode(monkeypatch)
    with pytest.raises(ValueError, match="Frozen checksum mismatch: training report"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_changed_id_order_is_rejected_against_frozen_order_before_decoding(frozen, monkeypatch):
    root, _, _, _, ids_path, report_path = frozen
    pd.read_csv(ids_path).iloc[::-1].to_csv(ids_path, index=False)
    report = json.loads(report_path.read_text())
    report["t03_split_ids_sha256"] = base.sha256_file(ids_path)
    write_json(report_path, report)

    def update(freeze):
        # Exercise the independent ordered-ID gate even with consistent file hashes.
        freeze["data_prep"]["split_ids_sha256"] = base.sha256_file(ids_path)
        freeze["models"]["baseline"]["report_sha256"] = base.sha256_file(report_path)

    update_freeze(root, update)
    no_decode(monkeypatch)
    with pytest.raises(ValueError, match="Test ID order or labels differ from T12"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_frozen_row_count_is_checked_before_decoding(frozen, monkeypatch):
    root, *_ = frozen
    update_freeze(root, lambda f: f["test"].update(rows=4))
    no_decode(monkeypatch)
    with pytest.raises(ValueError, match="test ID count"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_prepared_bytes_must_match_freeze_before_publication(frozen):
    root, *_ = frozen
    update_freeze(root, lambda f: f["test"].update(parquet_sha256="0" * 64))
    before = (root / "reports/freeze.json").read_bytes()
    with pytest.raises(ValueError, match="Frozen checksum mismatch: prepared test Parquet"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()
    assert not list(root.glob("data/processed/.test.parquet.*.tmp"))
    assert (root / "reports/freeze.json").read_bytes() == before


def test_existing_reconstructed_file_is_not_accepted_with_wrong_frozen_hash(frozen, monkeypatch):
    root, *_ = frozen
    result = target.restore(root)
    output = root / result["path"]
    before = output.read_bytes(), output.stat().st_mtime_ns
    update_freeze(root, lambda f: f["test"].update(parquet_sha256="0" * 64))
    no_decode(monkeypatch)
    with pytest.raises(ValueError, match="Existing test differs from T12"):
        target.restore(root)
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_interrupted_write_leaves_no_final_file_and_retry_succeeds(frozen, monkeypatch, failure):
    root, *_ = frozen
    output = root / "data/processed/test.parquet"
    output.parent.mkdir(parents=True)
    foreign = output.with_name(".test.parquet.foreign.tmp")
    foreign.write_bytes(b"foreign temporary file")

    def interrupted(frame, path):
        assert path.parent == output.parent and path != output
        assert not output.exists()
        path.write_bytes(b"partial parquet")
        raise failure("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(target, "save_prepared_split", interrupted)
        with pytest.raises(failure):
            target.restore(root)
    assert not output.exists()
    assert list(output.parent.iterdir()) == [foreign]
    assert foreign.read_bytes() == b"foreign temporary file"
    assert target.restore(root)["status"] == "RESTORED"
    assert base.sha256_file(output) == base.sha256_file(root / "expected.parquet")


@pytest.mark.parametrize("same", [False, True])
def test_concurrent_destination_is_never_overwritten(frozen, monkeypatch, same):
    root, *_ = frozen
    output = root / "data/processed/test.parquet"
    content = (root / "expected.parquet").read_bytes() if same else b"other writer wins"
    real_link = target.os.link

    def competing_link(source, destination):
        assert source.parent == destination.parent
        assert source.read_bytes() == (root / "expected.parquet").read_bytes()
        destination.write_bytes(content)
        return real_link(source, destination)  # real FileExistsError, not a mocked success

    monkeypatch.setattr(target.os, "link", competing_link)
    if same:
        assert target.restore(root)["status"] == "UNCHANGED"
    else:
        with pytest.raises(ValueError, match="refusing to overwrite"):
            target.restore(root)
    assert output.read_bytes() == content
    assert not list(output.parent.glob(".test.parquet.*.tmp"))


@pytest.mark.parametrize("stage", ["fsync", "link"])
def test_sync_or_publication_failure_is_cleaned_and_retry_succeeds(frozen, monkeypatch, stage):
    root, *_ = frozen

    def fail(*args):
        raise OSError("simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(target.os, stage, fail)
        with pytest.raises(OSError, match="simulated publication failure"):
            target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()
    assert not list(root.glob("data/processed/.test.parquet.*.tmp"))
    assert target.restore(root)["status"] == "RESTORED"
