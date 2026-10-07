"""Restore the accepted T03 train/validation by their recorded IDs; never open test."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as baseline
from src.prepare_data import build_record_id, save_prepared_split


def restore(root: Path = ROOT) -> dict:
    audit = baseline.load_t03_audit(root / "reports/data_prep/audit.json")
    original = baseline.read_json(root / "reports/baseline/run_metadata.json")
    # This is an accepted manifest, not a new deduplication or new split.
    for path, key in (("reports/data_prep/audit.json", "t03_audit_sha256"),
                      ("reports/data_prep/split_ids.csv", "t03_split_ids_sha256")):
        if baseline.sha256_file(root / path) != original[key]:
            raise ValueError(f"Accepted T03 manifest changed: {path}")
    ids = pd.read_csv(root / "reports/data_prep/split_ids.csv", dtype=str)
    ids = ids.loc[ids["split"].isin(("train", "validation"))]
    frames, result = {}, {}
    for split in ("train", "validation"):
        source = root / audit["source_files"][split]["path"]
        if baseline.sha256_file(source) != original["t03_source_parquet_sha256"][split]:
            raise ValueError(f"Source checksum mismatch: {split}")
        raw = pd.read_parquet(source)
        raw["source_file"] = source.name
        raw["source_row"] = range(len(raw))
        raw["record_id"] = [build_record_id(source.name, i) for i in range(len(raw))]
        selected_ids = ids.loc[ids["split"] == split, "record_id"].tolist()
        frame = raw.set_index("record_id", drop=False).loc[selected_ids].reset_index(drop=True)
        frame["label_name"] = frame["review_sentiment"].str.strip().str.casefold()
        frame["label_id"] = frame["label_name"].map(baseline.EXPECTED_LABEL_MAPPING).astype("int64")
        baseline.validate_prepared_frame(frame, split_name=split, text_column="review_text",
                                         label_column="label_id", id_column="record_id")
        frames[split] = frame
    baseline.validate_against_t03_audit(audit, train=frames["train"], validation=frames["validation"],
                                        label_column="label_id", train_ids_configured=False)
    baseline.validate_split_ids_against_t03(root / "reports/data_prep/split_ids.csv",
                                           train=frames["train"], validation=frames["validation"],
                                           id_column="record_id", label_column="label_id")
    baseline.validate_split_independence(frames["train"], frames["validation"],
                                         id_column="record_id", text_column="review_text")
    for split, frame in frames.items():
        target = root / f"data/processed/{split}.parquet"
        if target.exists():
            if baseline.sha256_file(target) != original[f"{split}_parquet_sha256"]:
                raise ValueError(f"Existing file differs; refusing to overwrite: {target}")
        else:
            # Verify serialization before installing the restored artifact.
            import tempfile
            with tempfile.TemporaryDirectory(prefix="nlp-restore-") as tmp:
                candidate = Path(tmp) / f"{split}.parquet"
                save_prepared_split(frame, candidate)
                if baseline.sha256_file(candidate) != original[f"{split}_parquet_sha256"]:
                    raise ValueError(f"Restored Parquet differs from accepted T03: {split}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(candidate.read_bytes())
        result[split] = {"rows": len(frame), "sha256": baseline.sha256_file(target)}
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(restore(), indent=2))
