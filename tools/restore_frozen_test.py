"""Restore the T03 text test by accepted IDs for T12; never run model inference.

This is technical data preparation, not a final evaluation. It does read the
source Parquet to copy the fixed rows, but prints no texts and does not select
models, change labels, deduplicate, or overwrite an existing different output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import train_baseline as base  # noqa: E402
from src.prepare_data import build_record_id, save_prepared_split  # noqa: E402


def _check_hash(path: Path, expected: str, description: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid frozen SHA-256: {description}")
    if base.sha256_file(path) != expected:
        raise ValueError(f"Frozen checksum mismatch: {description}")


def check_inputs(root: Path = ROOT) -> tuple[dict, dict, pd.DataFrame, Path]:
    """Check approved metadata, source bytes and ordered IDs; do not decode test."""
    freeze = base.read_json(root / "reports/freeze.json")
    if (freeze.get("schema") != "t13-final-text-eval-input/draft-1"
            or freeze.get("status") != "approved"
            or not freeze.get("approved_by") or not freeze.get("approved_at")
            or freeze.get("final_test_used_before_freeze") is not False):
        raise ValueError("An approved T12 freeze is required")
    entry = freeze["test"]
    if entry["path"] != "data/processed/test.parquet":
        raise ValueError("Unexpected frozen test path")
    if type(entry["rows"]) is not int or entry["rows"] <= 0:
        raise ValueError("Invalid frozen test row count")
    for key in ("ids_sha256", "parquet_sha256", "source_parquet_sha256"):
        if not isinstance(entry[key], str) or not re.fullmatch(r"[0-9a-f]{64}", entry[key]):
            raise ValueError(f"Invalid frozen test {key}")
    model = freeze["models"]["baseline"]
    report_path = root / "reports/nlp/training/run_metadata.json"
    if model["report_path"] != "reports/nlp/training/run_metadata.json":
        raise ValueError("Unexpected frozen training report path")
    _check_hash(report_path, model["report_sha256"], "training report")
    report = base.read_json(report_path)
    if report.get("final_test_used") is not False:
        raise ValueError("The accepted training report must exclude final test use")
    audit_path = root / "reports/data_prep/audit.json"
    ids_path = root / "reports/data_prep/split_ids.csv"
    for path, key, frozen_key in ((audit_path, "t03_audit_sha256", "audit_sha256"),
                                  (ids_path, "t03_split_ids_sha256", "split_ids_sha256")):
        expected = freeze["data_prep"][frozen_key]
        _check_hash(path, expected, path.name)
        if report[key] != expected:
            raise ValueError(f"Training report differs from T12: {path.name}")
    audit = base.read_json(audit_path)
    if audit.get("status") != "completed":
        raise ValueError("T03 audit is not completed")
    source = base.resolve_path(root, audit["source_files"]["test"]["path"])
    # The training report deliberately records only train/validation sources.
    # It pins the full T03 audit above; that accepted audit identifies test.
    if audit["source_files"]["test"]["sha256_after"] != entry["source_parquet_sha256"]:
        raise ValueError("Test source in T03 differs from T12")
    _check_hash(source, entry["source_parquet_sha256"], "test source")

    ids = pd.read_csv(ids_path, dtype=str)
    selected = ids.loc[ids["split"] == "test"].reset_index(drop=True)
    if (len(selected) != entry["rows"]
            or len(selected) != audit["final_summary"]["test"]["rows"]
            or selected["record_id"].duplicated().any()):
        raise ValueError("Invalid accepted test ID count or duplicates")
    if base.ids_sha256(selected, id_column="record_id", label_column="label_id") != entry["ids_sha256"]:
        raise ValueError("Test ID order or labels differ from T12")
    return freeze, audit, selected, source


def _existing_matches(target: Path, digest: str) -> bool:
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Existing test is not a regular file; refusing to overwrite")
    if not target.exists():
        return False
    if base.sha256_file(target) != digest:
        raise ValueError("Existing test differs from T12; refusing to overwrite")
    return True


def _publish(frame: pd.DataFrame, target: Path, digest: str) -> str:
    """Write and verify beside target, then atomically link without overwriting.

    A handled failure cleans up only this invocation's temporary file. A hard
    process kill may leave that temporary file, never a partial final filename.
    Hard links must be supported by the destination filesystem; no unsafe fallback.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".test.parquet.",
                                         suffix=".tmp", delete=False) as temporary:
            candidate = Path(temporary.name)
        save_prepared_split(frame, candidate)
        with candidate.open("r+b") as stream:
            os.fsync(stream.fileno())
        _check_hash(candidate, digest, "prepared test Parquet")
        try:
            os.link(candidate, target)  # atomic appearance; fails if any target exists
        except FileExistsError:
            if not _existing_matches(target, digest):
                raise  # a concurrent target vanished; a later invocation can retry
            return "UNCHANGED"
        return "RESTORED"
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)


def restore(root: Path = ROOT) -> dict:
    freeze, audit, selected, source = check_inputs(root)
    entry = freeze["test"]
    target = root / entry["path"]
    # Reject an output path whose parent leaves this repository, also via symlinks.
    base.resolve_path(root, entry["path"])
    if _existing_matches(target, entry["parquet_sha256"]):
        return _result("UNCHANGED", entry)

    # The selected model is decided before this preparation. Copy only the
    # previously accepted T03 rows, in their recorded order; no fresh split.
    raw = pd.read_parquet(source)
    raw["source_file"] = source.name
    raw["source_row"] = range(len(raw))
    raw["record_id"] = [build_record_id(source.name, i) for i in range(len(raw))]
    frame = raw.set_index("record_id", drop=False).loc[selected["record_id"]].reset_index(drop=True)
    frame["label_name"] = frame["review_sentiment"].str.strip().str.casefold()
    frame["label_id"] = frame["label_name"].map(base.EXPECTED_LABEL_MAPPING).astype("int64")
    base.validate_prepared_frame(frame, split_name="test", text_column="review_text",
                                 label_column="label_id", id_column="record_id")
    if (frame["label_id"].tolist() != selected["label_id"].astype(int).tolist()
            or frame["label_name"].tolist() != selected["label_name"].tolist()):
        raise ValueError("Test labels differ from accepted T03 IDs")
    for name, label_id in base.EXPECTED_LABEL_MAPPING.items():
        if int((frame["label_id"] == label_id).sum()) != audit["final_summary"]["test"][name]:
            raise ValueError("Test class counts differ from accepted T03 audit")

    if base.ids_sha256(frame, id_column="record_id", label_column="label_id") != entry["ids_sha256"]:
        raise ValueError("Restored test ID order or labels differ from T12")
    return _result(_publish(frame, target, entry["parquet_sha256"]), entry)


def _result(status: str, entry: dict) -> dict:
    return {
        "status": status,
        **{key: entry[key] for key in ("path", "rows", "source_parquet_sha256",
                                     "parquet_sha256", "ids_sha256")},
        "data_preparation_only": True,
        "model_inference_run": False,
        "final_metrics_computed": False,
        "texts_printed": False,
    }


def main() -> int:
    try:
        result = restore()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Test restoration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
