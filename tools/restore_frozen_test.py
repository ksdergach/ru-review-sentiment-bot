"""Restore the T03 text test by accepted IDs for T12; never run model inference.

This is technical data preparation, not a final evaluation. It does read the
source Parquet to copy the fixed rows, but prints no texts and does not select
models, change labels, deduplicate, or overwrite an existing different output.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import train_baseline as base  # noqa: E402
from src.prepare_data import build_record_id, save_prepared_split  # noqa: E402


def restore(root: Path = ROOT) -> dict:
    report = base.read_json(root / "reports/nlp/training/run_metadata.json")
    if report.get("final_test_used") is not False:
        raise ValueError("The accepted training report must exclude final test use")
    audit_path = root / "reports/data_prep/audit.json"
    ids_path = root / "reports/data_prep/split_ids.csv"
    for path, key in ((audit_path, "t03_audit_sha256"), (ids_path, "t03_split_ids_sha256")):
        if base.sha256_file(path) != report[key]:
            raise ValueError(f"Accepted T03 metadata changed: {path.name}")
    audit = base.read_json(audit_path)
    if audit.get("status") != "completed":
        raise ValueError("T03 audit is not completed")
    source = base.resolve_path(root, audit["source_files"]["test"]["path"])
    source_sha = base.sha256_file(source)
    # The training report deliberately records only train/validation sources.
    # It pins the full T03 audit above; that accepted audit identifies test.
    if source_sha != audit["source_files"]["test"]["sha256_after"]:
        raise ValueError("Source checksum mismatch: test")

    ids = pd.read_csv(ids_path, dtype=str)
    selected = ids.loc[ids["split"] == "test"].reset_index(drop=True)
    if (len(selected) != audit["final_summary"]["test"]["rows"]
            or selected["record_id"].duplicated().any()):
        raise ValueError("Invalid accepted test ID count or duplicates")

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

    target = root / "data/processed/test.parquet"
    with tempfile.TemporaryDirectory(prefix="t12-restore-test-") as directory:
        candidate = Path(directory) / "test.parquet"
        save_prepared_split(frame, candidate)
        digest = base.sha256_file(candidate)
        if target.exists():
            if base.sha256_file(target) != digest:
                raise ValueError("Existing test differs; refusing to overwrite")
            status = "UNCHANGED"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(candidate.read_bytes())
            status = "RESTORED"
    return {
        "status": status,
        "path": "data/processed/test.parquet",
        "rows": len(frame),
        "source_parquet_sha256": source_sha,
        "parquet_sha256": digest,
        "ids_sha256": base.ids_sha256(frame, id_column="record_id", label_column="label_id"),
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
