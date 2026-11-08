"""Validation evaluation through the shared classifier API; never reads the final test split.

    python tools/evaluate_classifier.py                          # baseline
    CLASSIFIER_BACKEND=rubert python tools/evaluate_classifier.py

Predictions come from ``src.classifier.predict_batch``: the same verified bundle, processing
and result checks as the bot and ``tools/predict_text.py``. The validation file is taken from
``configs/nlp_baseline.json`` and must match the data version recorded in the loaded model
(Parquet SHA-256, ordered IDs/labels, row count). The confusion matrix and macro-F1 are compared
with the run report of that model.

This is the evaluation adapter of the shared API, not an independent check:
``tools/verify_transformer.py`` and ``tools/report_nlp_baseline.py`` deliberately re-implement
verification without ``src.classifier``.

Exit code: 0 — matches the run report; 1 — differs; 2 — model or data unavailable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import classifier as c  # noqa: E402
from src import train_baseline as base  # noqa: E402


def _recorded(model) -> tuple[float, list[list[int]], str]:
    """Macro-F1, confusion matrix and validation Parquet SHA-256 from the model's run report."""
    report = base.read_json(model.report_path)
    if isinstance(model, c.RuBERTClassifier):
        winner = report["winner"]
        return winner["macro_f1"], winner["confusion_matrix"], report["data_version"]["checksums"]["validation_path"]
    winner = base.read_json(model.report_path.with_name("winner_config.json"))
    return (report["winner_validation_macro_f1"], winner["validation_confusion_matrix"]["matrix"],
            report["validation_parquet_sha256"])


def evaluate(model, validation_path: Path) -> dict:
    if base._looks_like_test(validation_path.stem):
        raise ValueError("The final test split is not evaluated")
    # Load and verify the model without processing any dataset text. Its report
    # pins the allowed validation file, including when a different split is renamed.
    model._get_bundle()
    recorded_f1, recorded_matrix, recorded_sha = _recorded(model)
    if base.sha256_file(validation_path) != recorded_sha:
        raise ValueError("Validation data differs from the model data version")
    frame = base.load_prepared_split(validation_path, split_name="validation", text_column="review_text",
                                     label_column="label_id", id_column="record_id")
    version = model.data_version
    if (base.ids_sha256(frame, id_column="record_id", label_column="label_id") != version["validation_ids_sha256"]
            or len(frame) != version["validation_rows"]):
        raise ValueError("Validation data differs from the model data version")
    # Preserve the training-time batch layout, but only after all data checks.
    results = model.predict_batch(frame["review_text"].tolist(), max_text_length=None)
    truth = frame["label_id"].to_numpy(dtype=int)
    predicted = np.asarray([r["label_id"] for r in results], dtype=int)
    metrics = base.validation_metrics(truth, predicted)
    matrix = base.validation_confusion_matrix(truth, predicted)
    matches = matrix == recorded_matrix and math.isclose(metrics["macro_f1"], recorded_f1, rel_tol=0, abs_tol=1e-12)
    return {"status": "passed" if matches else "failed", "backend": type(model).__name__,
            "model_path": base.display_path(model.model_path, ROOT), "split": "validation", "rows": len(frame),
            "macro_f1": metrics["macro_f1"], "recorded_macro_f1": recorded_f1, "accuracy": metrics["accuracy"],
            "confusion_matrix": matrix, "confusion_matrix_matches_report": matrix == recorded_matrix,
            "truncated_rows": sum(r["truncated"] for r in results), "final_test_used": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)  # Deliberately no dataset argument: only the recorded validation split.
    config = base.read_json(ROOT / "configs/nlp_baseline.json")
    try:
        result = evaluate(c.get_default_classifier(), base.resolve_path(ROOT, config["validation_path"]))
    except (c.ClassifierError, c.InvalidTextError, OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
