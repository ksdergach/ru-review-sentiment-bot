"""Evaluation adapter of the shared API on a small real baseline bundle and synthetic validation."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import classifier as c, train_baseline as tb  # noqa: E402
from test_classifier import baseline, make_real_bundle, report_of  # noqa: E402
from tools import evaluate_classifier as ev  # noqa: E402

TEXTS = ["ужасный фильм", "плохой фильм", "обычный фильм", "средний фильм", "отличный фильм", "прекрасный фильм"]
LABELS = [0, 0, 1, 1, 2, 2]


def write_split(path: Path, labels=LABELS) -> pd.DataFrame:
    names = {v: k for k, v in tb.EXPECTED_LABEL_MAPPING.items()}
    frame = pd.DataFrame({"record_id": [f"v{i}" for i in range(len(TEXTS))], "review_text": TEXTS,
                          "label_id": labels, "label_name": [names[x] for x in labels],
                          "review_language": "ru", "movie_id": list(range(len(TEXTS)))})
    frame.to_parquet(path, index=False)
    return frame


@pytest.fixture
def setup(tmp_path):
    pytest.importorskip("pyarrow")
    split = tmp_path / "validation.parquet"
    frame = write_split(split)
    version = {"validation_ids_sha256": tb.ids_sha256(frame, id_column="record_id", label_column="label_id"),
               "validation_rows": len(frame), "validation_parquet_sha256": tb.sha256_file(split)}
    model_path = make_real_bundle(tmp_path, training_update=lambda training: training.update(version))
    # Expected values from the training module directly, not through src.classifier.
    raw = tb.predict_texts(tb.load_bundle(model_path), TEXTS)
    truth = pd.Series(LABELS).to_numpy()
    predicted = pd.Series(raw["label_ids"]).to_numpy()
    report = json.loads(report_of(model_path).read_text(encoding="utf-8"))
    report["winner_validation_macro_f1"] = tb.validation_metrics(truth, predicted)["macro_f1"]
    tb.write_json(report_of(model_path), report)
    winner = {"validation_confusion_matrix": {"matrix": tb.validation_confusion_matrix(truth, predicted)}}
    tb.write_json(model_path.with_name("winner_config.json"), winner)
    return model_path, split


def test_evaluation_uses_shared_api_and_matches_report(setup, monkeypatch):
    model_path, split = setup
    model = baseline(model_path)
    calls = []
    original = model.predict_batch
    monkeypatch.setattr(model, "predict_batch", lambda texts, **kw: calls.append(len(texts)) or original(texts, **kw))
    result = ev.evaluate(model, split)
    assert result["status"] == "passed" and result["confusion_matrix_matches_report"]
    assert result["rows"] == len(TEXTS) and result["final_test_used"] is False and result["truncated_rows"] == 0
    assert calls == [len(TEXTS)]
    assert [r["label_id"] for r in model.predict_batch(TEXTS)] == [model.predict(t)["label_id"] for t in TEXTS]


def test_evaluation_reports_mismatch(setup):
    model_path, split = setup
    winner = model_path.with_name("winner_config.json")
    data = json.loads(winner.read_text(encoding="utf-8"))
    data["validation_confusion_matrix"]["matrix"][0][0] += 1
    tb.write_json(winner, data)
    assert ev.evaluate(baseline(model_path), split)["status"] == "failed"


def test_evaluation_rejects_other_validation_data(setup, tmp_path):
    model_path, _ = setup
    other = tmp_path / "other_validation.parquet"
    write_split(other, labels=[0, 1, 1, 0, 2, 2])
    with pytest.raises(ValueError, match="data version"):
        ev.evaluate(baseline(model_path), other)


def test_evaluation_refuses_test_split(setup, tmp_path):
    model_path, split = setup
    test_split = tmp_path / "test.parquet"
    test_split.write_bytes(split.read_bytes())
    with pytest.raises(ValueError, match="final test"):
        ev.evaluate(baseline(model_path), test_split)


def test_unavailable_model_exit_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(c, "_default_classifier", c.BaselineClassifier(tmp_path / "absent.joblib"))
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    write_split(tmp_path / "validation.parquet")
    (tmp_path / "configs/nlp_baseline.json").write_text('{"validation_path": "validation.parquet"}')
    assert ev.main([]) == 2
    assert "absent.joblib" in capsys.readouterr().err


def test_wrong_checksum_rejected_before_reading_texts_or_predicting(setup, tmp_path, monkeypatch):
    model_path, _ = setup
    other = tmp_path / "validation_copy.parquet"
    write_split(other, labels=[0, 1, 1, 0, 2, 2])
    model = baseline(model_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified dataset must not be decoded or passed to the model")
    monkeypatch.setattr(tb, "load_prepared_split", forbidden)
    monkeypatch.setattr(model, "predict_batch", forbidden)
    with pytest.raises(ValueError, match="data version"):
        ev.evaluate(model, other)


@pytest.mark.parametrize("field,value", [("validation_ids_sha256", "0" * 64),
                                         ("validation_rows", 999)])
def test_version_mismatch_rejected_before_prediction(setup, monkeypatch, field, value):
    model_path, split = setup
    model = baseline(model_path)
    model._get_bundle()
    model.data_version[field] = value
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible dataset must not be passed to the model")
    monkeypatch.setattr(model, "predict_batch", forbidden)
    with pytest.raises(ValueError, match="data version"):
        ev.evaluate(model, split)
