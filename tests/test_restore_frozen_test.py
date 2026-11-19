"""Technical restoration uses synthetic fixtures, never the actual final test."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import train_baseline as base
from src.prepare_data import build_record_id
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
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        target.restore(root)
    assert not (root / "data/processed/test.parquet").exists()


def test_changed_id_manifest_is_rejected(frozen):
    root, _, _, _, ids_path, _ = frozen
    ids_path.write_text("record_id,split,label_name,label_id\n")
    with pytest.raises(ValueError, match="Accepted T03 metadata changed"):
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


def test_repeated_restoration_does_not_rewrite_matching_output(frozen):
    root, *_ = frozen
    first = target.restore(root)
    output = root / first["path"]
    before = output.stat().st_mtime_ns
    second = target.restore(root)
    assert second["status"] == "UNCHANGED"
    assert second["parquet_sha256"] == first["parquet_sha256"]
    assert output.stat().st_mtime_ns == before
