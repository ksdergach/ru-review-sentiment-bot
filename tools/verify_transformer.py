"""Verify completed T06 evidence and load the real bundle offline; never open test."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as base
from src import train_transformer as tt


def verify(root: Path = ROOT, config_path: Path | None = None):
    config = base.read_json(config_path or root / "configs/transformer.json")
    tt.validate_config(config)
    report = base.resolve_path(root, config["report_dir"])
    bundle = base.resolve_path(root, config["model_dir"])
    checks = []
    def check(name, condition):
        if not condition:
            raise ValueError(f"Verification failed: {name}")
        checks.append(name)

    metadata = base.read_json(report / "run_metadata.json")
    _, _, version = tt.load_data(config, root)
    check("accepted_baseline_data_version", version == metadata["data_version"])
    check("no_final_test_used", metadata["final_test_used"] is False)
    budget = math.ceil(version["train_rows"] / config["batch_size"]) * config["epochs"]
    check("full_training_budget_completed", metadata["mode"] == "training" and
          metadata["steps"] == metadata["steps_budget"] == budget and
          metadata["examples_seen"] == version["train_rows"] * config["epochs"])
    check("config_checksum", base.sha256_json(config) == metadata["config_sha256"])
    check("training_script_checksum", base.sha256_file(root / "src/train_transformer.py") == metadata["script_sha256"])
    for name, digest in metadata["bundle_files_sha256"].items():
        check(f"bundle_checksum:{name}", base.sha256_file(bundle / name) == digest)
        check(f"bundle_git_ignored:{name}", base.git_ignore_status(bundle / name, root) is True)
    bundle_metadata = base.read_json(bundle / "metadata.json")
    check("bundle_metadata_matches_run", bundle_metadata == {
        k: v for k, v in metadata.items() if k != "bundle_files_sha256"})
    updates = metadata["parameter_updates"]
    check("encoder_and_head_updated", updates["encoder_updated"] and updates["head_updated"] and
          updates["encoder_max_abs_update"] > 0 and updates["head_max_abs_update"] > 0 and
          all(math.isfinite(v) and v > 0 for v in updates["sentinel_grad_norms"].values()))
    import json
    events = [json.loads(line) for line in (report / "training_log.jsonl").read_text().splitlines()]
    check("real_training_log_complete", events[0]["event"] == "start" and
          events[-1]["event"] == "complete" and events[-1]["steps"] == budget)
    logged_validation = [event for event in events if event["event"] == "validation"]
    table = pd.read_csv(report / "validation_results.csv")
    check("validation_table_matches_log", len(table) == len(logged_validation) >= 1)
    for i, row in enumerate(logged_validation):
        check(f"validation_row:{row['step']}", int(table.iloc[i]["step"]) == row["step"] and
              math.isclose(table.iloc[i]["macro_f1"], row["macro_f1"], abs_tol=1e-12))
        matrix = np.array(row["confusion_matrix"], dtype=int)
        f1 = np.divide(2 * matrix.diagonal(), matrix.sum(0) + matrix.sum(1),
                       out=np.zeros(3), where=(matrix.sum(0) + matrix.sum(1)) != 0)
        check(f"macro_f1_from_confusion:{row['step']}", matrix.shape == (3, 3) and
              int(matrix.sum()) == version["validation_rows"] and
              math.isclose(float(f1.mean()), row["macro_f1"], abs_tol=1e-12))
    winner = max(logged_validation, key=lambda row: (row["macro_f1"], -row["step"]))
    check("winner_is_max_validation_macro_f1", winner["step"] == metadata["winner"]["step"] and
          winner["macro_f1"] == metadata["winner"]["macro_f1"])
    check("honest_baseline_delta", math.isclose(metadata["delta_vs_baseline"],
          winner["macro_f1"] - version["baseline_validation_macro_f1"], abs_tol=1e-12))
    # Separate process invocation of this script verifies that no in-memory training model is needed.
    model, tokenizer, loaded_config = tt.load_bundle(bundle)
    tt.setup(loaded_config)
    fixed = tt.predict_texts(model, tokenizer, tt.FIXED_TEXTS, loaded_config)
    previous = base.read_json(report / "bundle_roundtrip.json")
    check("validation_roundtrip_recorded", previous["validation_predictions_identical_after_load"] and
          previous["validation_macro_f1_after_load"] == winner["macro_f1"])
    check("fixed_text_class_order", fixed["probability_class_ids"] == [0, 1, 2] and
          fixed["probability_labels"] == ["negative", "neutral", "positive"])
    check("fixed_text_predictions_reproduced", fixed["label_ids"] == previous["fixed_texts"]["label_ids"] and
          np.allclose(fixed["probabilities"], previous["fixed_texts"]["probabilities"], atol=1e-7, rtol=0))
    check("three_finite_probabilities", np.asarray(fixed["probabilities"]).shape == (3, 3) and
          np.isfinite(fixed["probabilities"]).all() and
          np.allclose(np.sum(fixed["probabilities"], axis=1), 1, atol=1e-6))
    result = {"passed": True, "checks": checks, "final_test_used": False,
              "fixed_predictions": fixed, "winner_step": winner["step"],
              "validation_macro_f1": winner["macro_f1"],
              "baseline_validation_macro_f1": version["baseline_validation_macro_f1"],
              "delta_vs_baseline": metadata["delta_vs_baseline"],
              "note": "Local automated verification; not an independent quality assessment."}
    base.write_json(report / "verification.json", result)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
