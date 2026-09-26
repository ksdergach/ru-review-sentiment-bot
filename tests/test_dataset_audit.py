"""Provenance must fail closed; review packets must not damage human work."""

import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.audit_dataset import (
    LABELS, audit, inspect_file, sample_metadata, select_sample,
    sha256_file, verify_remote, write_review_packet,
)


def make_source(tmp_path):
    directory = tmp_path / "NLP_dataset"
    directory.mkdir()
    source = {
        "revision": "test-revision", "config": "pc", "files": {},
        "manual_sample": {"seed": 42, "per_class": dict.fromkeys(LABELS, 2)},
        "plan_raw_counts": {},
    }
    for split in ("train", "validation", "test"):
        rows = [
            {"movie_id": split, "review_text": f"<script>alert(1)</script> {i}",
             "review_sentiment": label.upper(), "review_language": lang}
            for label in LABELS for lang in ("ru", "kk", "cs") for i in range(5)
        ]
        table = pa.Table.from_pylist(rows)
        path = directory / f"{split}.parquet"
        pq.write_table(table, path)
        source["features"] = [{"name": f.name, "dtype": str(f.type)} for f in table.schema]
        source["files"][split] = {
            "local_file": path.name, "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size, "rows": len(rows),
        }
        source["plan_raw_counts"][split] = {"all": 45, "ru": 15, **dict.fromkeys(LABELS, 5)}
    return source


def test_audit_raw_counts_and_train_only_sample(tmp_path):
    source = make_source(tmp_path)
    report, sample = audit(tmp_path, source)
    assert report["all_match_plan"]
    assert not any(report["movie_overlap"].values())
    assert len(sample) == 6
    assert {r["label_name"] for r in sample} == set(LABELS)
    assert all(r["record_id"].startswith("train.parquet#row=") for r in sample)
    assert all(r["review_language"] == "ru" for r in sample)
    assert report["splits"]["train"]["class_counts_all"] == dict.fromkeys(LABELS, 15)
    assert report["manual_review_status"] == "pending_participant"
    assert all("review_text" not in row for row in sample_metadata(sample))
    assert select_sample(list(reversed(sample)), **source["manual_sample"]) == sample


def test_corrupt_bytes_rejected_before_parquet_read(tmp_path):
    source = make_source(tmp_path)
    (tmp_path / "NLP_dataset/train.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Source bytes"):
        audit(tmp_path, source)


@pytest.mark.parametrize("field,value", [("review_sentiment", "NOISE"), ("review_language", "en"), ("review_text", " "), ("movie_id", None)])
def test_invalid_source_values_are_not_silently_filtered(tmp_path, field, value):
    source = make_source(tmp_path)
    path = tmp_path / "NLP_dataset/train.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows), path)
    expected = source["files"]["train"]
    expected.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="field|language/label"):
        inspect_file(path, expected, source["features"])


def test_plan_disagreement_is_visible(tmp_path):
    source = make_source(tmp_path)
    source["plan_raw_counts"]["train"]["ru"] += 1
    report, _ = audit(tmp_path, source)
    assert not report["all_match_plan"]
    assert not report["splits"]["train"]["plan_counts_match"]


def test_review_packet_escapes_text_and_preserves_human_notes(tmp_path):
    source = make_source(tmp_path)
    _, sample = audit(tmp_path, source)
    output = tmp_path / "review"
    write_review_packet(sample, output)
    notes = output / "notes.md"
    notes.write_text("Human decisions, do not replace", encoding="utf-8")
    write_review_packet(sample, output)
    assert notes.read_text() == "Human decisions, do not replace"
    page = (output / "review.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    with pytest.raises(ValueError, match="different IDs"):
        write_review_packet(sample[:-1], output)
    assert notes.read_text() == "Human decisions, do not replace"


def test_insufficient_class_is_an_error(tmp_path):
    source = make_source(tmp_path)
    _, sample = audit(tmp_path, source)
    with pytest.raises(ValueError, match="Not enough"):
        select_sample(sample, 42, dict.fromkeys(LABELS, 3))


def test_remote_hash_mismatch_is_rejected(monkeypatch):
    import io
    import tools.audit_dataset as module

    source = {
        "api_url": "https://example.invalid", "repository": "owner/repo", "revision": "abc",
        "license_declared": "cc-by-4.0", "features": [],
        "files": {"train": {"remote_path": "pc/train.parquet", "sha256": "expected", "size_bytes": 10, "rows": 1}},
    }
    remote = {
        "id": "owner/repo", "sha": "abc",
        "cardData": {"license": "cc-by-4.0", "dataset_info": [{"config_name": "pc", "features": [], "splits": [{"name": "train", "num_examples": 1}]}]},
        "siblings": [{"rfilename": "pc/train.parquet", "lfs": {"sha256": "expected"}, "size": 10}],
    }
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(remote).encode()))
    verify_remote(source)
    wrong = copy.deepcopy(source)
    wrong["files"]["train"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="Remote evidence mismatch"):
        verify_remote(wrong)
