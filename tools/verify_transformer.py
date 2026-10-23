"""Independent verification of the completed T06 evidence and bundle; never opens test.

The checker deliberately does not import ``src.*``: checksums, ordered ID hashes,
metrics and the selection rule are re-implemented here, and the bundle is loaded
directly with ``transformers.AutoModelForSequenceClassification``/``AutoTokenizer``
(``local_files_only=True``). The result goes to stdout (and optionally ``--output``);
the committed ``reports/transformer/verification.json`` is never rewritten.

    HF_HUB_OFFLINE=1 python tools/verify_transformer.py
    HF_HUB_OFFLINE=1 python tools/verify_transformer.py --full-validation --pretrained-check
    python tools/verify_transformer.py --no-bundle        # evidence and data only

Exit code: 0 — all requested checks passed; 1 — a check failed; 2 — missing input/usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["negative", "neutral", "positive"]
MAPPING = {name: i for i, name in enumerate(CLASS_ORDER)}
# Fixed T06 smoke texts; their SHA-256 must equal those recorded in bundle_roundtrip.json.
FIXED_TEXTS = [
    "Фильм отличный, очень понравился.",
    "Фильм ужасный, было скучно.",
    "Обычный фильм без сильных впечатлений.",
]
COMMITTED_REPORT = "reports/transformer/verification.json"


class VerificationError(Exception):
    pass


class Checks:
    def __init__(self):
        self.passed: list[str] = []
        self.not_run: dict[str, str] = {}

    def __call__(self, name: str, condition, detail: str = "") -> None:
        if not condition:
            raise VerificationError(f"{name}{': ' + detail if detail else ''}")
        self.passed.append(name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(data) -> str:
    return sha256_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def ordered_pairs_sha256(pairs) -> str:
    digest = hashlib.sha256()
    for record_id, label in pairs:
        digest.update(f"{record_id}\t{int(label)}\n".encode("utf-8"))
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def git_ignored(path: Path, root: Path) -> bool | None:
    try:
        proc = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(path)],
                              capture_output=True, check=False)
    except OSError:
        return None
    return {0: True, 1: False}.get(proc.returncode)


def f1_from_confusion(matrix) -> list[float]:
    """Rows are true classes, columns are predictions."""
    result = []
    for k in range(3):
        denominator = sum(matrix[k]) + sum(row[k] for row in matrix)
        result.append(2 * matrix[k][k] / denominator if denominator else 0.0)
    return result


def confusion(y_true, y_pred) -> list[list[int]]:
    matrix = [[0] * 3 for _ in range(3)]
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix


def argmax(row) -> int:
    return max(range(len(row)), key=row.__getitem__)


# --------------------------------------------------------------------------- #
def verify_data(root, config, metadata, check) -> dict:
    """Recompute the accepted data version from files, not from the training module."""
    import pyarrow.parquet as pq

    p = lambda key: root / config[key]
    baseline = read_json(p("baseline_metadata_path"))
    recorded = metadata["data_version"]
    check("baseline_metadata_checksum",
          sha256_file(p("baseline_metadata_path")) == recorded["baseline_metadata_sha256"])
    check("baseline_train_ids_file_checksum",
          sha256_file(p("baseline_train_ids_path")) == recorded["baseline_train_ids_file_sha256"])
    for key, expected in (("data_prep_audit_path", "t03_audit_sha256"),
                          ("data_prep_split_ids_path", "t03_split_ids_sha256")):
        check(f"t03_checksum:{key}", sha256_file(p(key)) == baseline[expected] == recorded["checksums"][key])
    labels = {}
    for split in ("train", "validation"):
        check(f"parquet_checksum:{split}", sha256_file(p(f"{split}_path"))
              == baseline[f"{split}_parquet_sha256"] == recorded["checksums"][f"{split}_path"])
        table = pq.read_table(p(f"{split}_path"), columns=["record_id", "label_id"])
        labels[split] = list(zip(table.column("record_id").to_pylist(), table.column("label_id").to_pylist()))
    with p("baseline_train_ids_path").open(encoding="utf-8", newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: int(r["train_order"]))
    train_pairs = [(r["record_id"], int(r["label_id"])) for r in rows]
    train_parquet = dict(labels["train"])
    check("baseline_train_ids_exist_with_same_labels",
          all(train_parquet.get(rid) == label for rid, label in train_pairs))
    check("train_ordered_ids_equal_baseline",
          ordered_pairs_sha256(train_pairs) == baseline["train_ids_sha256"] == recorded["train_ids_sha256"])
    check("validation_ordered_ids_equal_baseline",
          ordered_pairs_sha256(labels["validation"]) == baseline["validation_ids_sha256"]
          == recorded["validation_ids_sha256"])
    check("row_counts", len(train_pairs) == baseline["train_rows"] == recorded["train_rows"]
          and len(labels["validation"]) == baseline["validation_rows"] == recorded["validation_rows"])
    manifest: dict[str, set] = {}
    with p("data_prep_split_ids_path").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            manifest.setdefault(r["split"], set()).add(r["record_id"])
    train_ids = {rid for rid, _ in train_pairs}
    validation_ids = {rid for rid, _ in labels["validation"]}
    check("no_train_validation_id_overlap", not train_ids & validation_ids)
    check("no_test_ids_in_train_or_validation", not (train_ids | validation_ids) & manifest.get("test", set()))
    check("ids_match_t03_manifest", train_ids <= manifest["train"] and validation_ids == manifest["validation"])
    check("recorded_no_final_test", recorded["final_test_used"] is False and metadata["final_test_used"] is False)
    check("baseline_macro_f1_recorded",
          recorded["baseline_validation_macro_f1"] == baseline["winner_validation_macro_f1"])
    return baseline


def verify_evidence(root, config, metadata, report, check, baseline_f1) -> dict:
    check("config_checksum", canonical_json_sha256(config) == metadata["config_sha256"])
    check("config_copy_matches", read_json(report / "config.json") == config)
    check("training_script_checksum", sha256_file(root / "src/train_transformer.py") == metadata["script_sha256"],
          "src/train_transformer.py differs from the script that produced the run")
    train_rows = metadata["data_version"]["train_rows"]
    steps_per_epoch = math.ceil(train_rows / config["batch_size"])
    budget = steps_per_epoch * config["epochs"]
    check("full_training_budget_completed", metadata["mode"] == "training"
          and metadata["steps"] == metadata["steps_budget"] == budget
          and metadata["examples_seen"] == train_rows * config["epochs"])
    updates = metadata["parameter_updates"]
    check("encoder_and_head_updated_at_step_1", updates["encoder_updated"] and updates["head_updated"]
          and updates["encoder_max_abs_update"] > 0 and updates["head_max_abs_update"] > 0
          and all(math.isfinite(v) and v > 0 for v in updates["sentinel_grad_norms"].values()))
    events = [json.loads(line) for line in
              (report / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    check("training_log_complete", events[0]["event"] == "start" and events[-1]["event"] == "complete"
          and events[-1]["steps"] == budget)
    logged = [e for e in events if e["event"] == "validation"]
    planned = sorted(set(range(config["eval_every_steps"], budget + 1, config["eval_every_steps"]))
                     | {steps_per_epoch * e for e in range(1, config["epochs"] + 1)})
    check("validation_only_at_preplanned_steps", [e["step"] for e in logged] == planned)
    with (report / "validation_results.csv").open(encoding="utf-8", newline="") as fh:
        table = list(csv.DictReader(fh))
    check("validation_table_matches_log", len(table) == len(logged) >= 1)
    n_validation = metadata["data_version"]["validation_rows"]
    for row, event in zip(table, logged):
        matrix = event["confusion_matrix"]
        f1 = f1_from_confusion(matrix)
        total = sum(map(sum, matrix))
        check(f"validation_row:{event['step']}", int(row["step"]) == event["step"]
              and math.isclose(float(row["macro_f1"]), event["macro_f1"], abs_tol=1e-12))
        check(f"metrics_from_confusion:{event['step']}", total == n_validation
              and math.isclose(sum(f1) / 3, event["macro_f1"], abs_tol=1e-12)
              and math.isclose(sum(matrix[k][k] for k in range(3)) / total, event["accuracy"], abs_tol=1e-12)
              and all(math.isclose(f1[k], event[f"{name}_f1"], abs_tol=1e-12) for k, name in enumerate(CLASS_ORDER)))
    winner = max(logged, key=lambda e: (e["macro_f1"], -e["step"]))
    check("winner_is_max_validation_macro_f1_earliest_tie",
          winner["step"] == metadata["winner"]["step"] and winner["macro_f1"] == metadata["winner"]["macro_f1"])
    check("honest_baseline_delta", metadata["baseline_validation_macro_f1"] == baseline_f1
          and math.isclose(metadata["delta_vs_baseline"], winner["macro_f1"] - baseline_f1, abs_tol=1e-12))
    roundtrip = read_json(report / "bundle_roundtrip.json")
    check("roundtrip_recorded", roundtrip["validation_predictions_identical_after_load"] is True
          and roundtrip["validation_macro_f1_after_load"] == winner["macro_f1"])
    check("fixed_texts_are_recorded_texts",
          [sha256_bytes(t.encode("utf-8")) for t in FIXED_TEXTS] == roundtrip["fixed_texts"]["text_sha256"])
    return {"winner": winner, "roundtrip": roundtrip}


def _predict(model, tokenizer, texts, training_config):
    """Same documented processing as training: right truncation, padding per batch."""
    import torch
    tokenizer.truncation_side = "right"
    batch = training_config["batch_size"]
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(texts), batch):
            inputs = tokenizer(list(texts[offset:offset + batch]), padding=True, truncation=True,
                               max_length=training_config["max_length"], return_tensors="pt")
            probabilities.extend(model(**inputs).logits.softmax(-1).tolist())
    return probabilities


def verify_bundle(root, config, metadata, bundle, check, evidence, args) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    expected = metadata["bundle_files_sha256"]
    # Every entry counts: a stray subdirectory or symlink is not part of the recorded bundle.
    present = sorted(p.name for p in bundle.iterdir())
    check("bundle_file_set", present == sorted(expected) and all(
        (bundle / n).is_file() and not (bundle / n).is_symlink() for n in present), f"present={present}")
    for name, digest in sorted(expected.items()):
        check(f"bundle_checksum:{name}", sha256_file(bundle / name) == digest)
    if bundle.is_relative_to(root):
        for name in sorted(expected):
            check(f"bundle_git_ignored:{name}", git_ignored(bundle / name, root) is True)
    else:
        check.not_run["bundle_git_ignored"] = "bundle is outside the repository"
    check("bundle_metadata_matches_run",
          read_json(bundle / "metadata.json") == {k: v for k, v in metadata.items() if k != "bundle_files_sha256"})
    training_config = read_json(bundle / "training_config.json")
    check("bundle_training_config_matches", training_config == config)
    check("bundle_label_mapping", read_json(bundle / "label_mapping.json") == MAPPING)

    torch.set_num_threads(config["num_threads"])
    tokenizer = AutoTokenizer.from_pretrained(bundle, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        bundle, local_files_only=True, trust_remote_code=False).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    check("hf_config_class_order", model.config.num_labels == 3
          and id2label == dict(enumerate(CLASS_ORDER)) and dict(model.config.label2id) == MAPPING)
    fixed = _predict(model, tokenizer, FIXED_TEXTS, training_config)
    check("three_finite_probabilities", len(fixed) == 3 and all(
        len(row) == 3 and all(math.isfinite(v) for v in row) and math.isclose(sum(row), 1, abs_tol=1e-6)
        for row in fixed))
    recorded = evidence["roundtrip"]["fixed_texts"]
    labels = [argmax(row) for row in fixed]
    check("fixed_text_predictions_reproduced", labels == recorded["label_ids"] and all(
        math.isclose(a, b, abs_tol=1e-6) for r1, r2 in zip(fixed, recorded["probabilities"]) for a, b in zip(r1, r2)))
    result = {"fixed_texts": {"text_sha256": recorded["text_sha256"], "label_ids": labels,
                              "labels": [CLASS_ORDER[i] for i in labels],
                              "probability_labels": CLASS_ORDER, "probabilities": fixed}}
    if args.full_validation:
        import pyarrow.parquet as pq
        table = pq.read_table(root / config["validation_path"], columns=["review_text", "label_id"])
        y_true = table.column("label_id").to_pylist()
        y_pred = [argmax(row) for row in _predict(model, tokenizer, table.column("review_text").to_pylist(),
                                                  training_config)]
        matrix = confusion(y_true, y_pred)
        macro_f1 = sum(f1_from_confusion(matrix)) / 3
        check("full_validation_confusion_matrix_reproduced", matrix == evidence["winner"]["confusion_matrix"])
        check("full_validation_macro_f1_reproduced",
              math.isclose(macro_f1, evidence["winner"]["macro_f1"], abs_tol=1e-12))
        result["full_validation"] = {"rows": len(y_true), "macro_f1": macro_f1, "confusion_matrix": matrix}
    else:
        check.not_run["full_validation"] = "use --full-validation"
    if args.pretrained_check:
        result["pretrained_comparison"] = compare_with_pretrained(model, config, args.pretrained, check)
    else:
        check.not_run["pretrained_check"] = "use --pretrained-check"
    return result


def compare_with_pretrained(model, config, source, check) -> dict:
    """The winner must differ from the pinned pretrained encoder in every encoder block."""
    from transformers import AutoModel

    kwargs = {"local_files_only": True, "trust_remote_code": False}
    if source is None:
        source, kwargs["revision"] = config["model_name"], config["revision"]
    pretrained = AutoModel.from_pretrained(source, **kwargs).state_dict()
    tuned = model.base_model.state_dict()
    check("pretrained_encoder_architecture_matches", set(pretrained) == set(tuned)
          and all(pretrained[k].shape == tuned[k].shape for k in pretrained))
    groups: dict[str, float] = {}
    for key, value in pretrained.items():
        if not value.is_floating_point():
            continue
        parts = key.split(".")
        group = ".".join(parts[:3]) if parts[:2] == ["encoder", "layer"] else parts[0]
        groups[group] = max(groups.get(group, 0.0), float((tuned[key] - value).abs().max()))
    check("winner_differs_from_pretrained_in_every_encoder_block",
          bool(groups) and all(v > 0 for v in groups.values()),
          f"unchanged={[g for g, v in groups.items() if v == 0]}")
    return {"source": str(source), "max_abs_difference_by_block": groups}


# --------------------------------------------------------------------------- #
def verify(root: Path = ROOT, config_path: Path | None = None, bundle: Path | None = None,
           args: argparse.Namespace | None = None) -> dict:
    args = args or parse_args([])
    root = root.resolve()
    config = read_json(config_path or root / "configs/transformer.json")
    check = Checks()
    check("config_class_order_and_truncation",
          config["label_mapping"] == MAPPING and config["truncation_side"] == "right")
    report = root / config["report_dir"]
    metadata = read_json(report / "run_metadata.json")
    baseline = verify_data(root, config, metadata, check)
    evidence = verify_evidence(root, config, metadata, report, check, baseline["winner_validation_macro_f1"])
    result = {"final_test_used": False, "winner_step": evidence["winner"]["step"],
              "validation_macro_f1": evidence["winner"]["macro_f1"],
              "baseline_validation_macro_f1": metadata["baseline_validation_macro_f1"],
              "delta_vs_baseline": metadata["delta_vs_baseline"]}
    if args.no_bundle:
        check.not_run["bundle"] = "--no-bundle: weights were not checked or loaded"
    else:
        bundle = (bundle or root / config["model_dir"]).resolve()
        if not (bundle / "model.safetensors").is_file():
            raise FileNotFoundError(f"Bundle not found: {bundle}. Restore it with tools/transformer_bundle.py "
                                    "or use --no-bundle for evidence-only checks.")
        result.update(verify_bundle(root, config, metadata, bundle, check, evidence, args))
    result.update(status="passed_without_bundle" if args.no_bundle else "passed",
                  checks=check.passed, checks_count=len(check.passed), not_run=check.not_run,
                  note="Re-implemented checks independent of src/*; not a quality assessment or test evaluation.")
    return result


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, help="default: <root>/configs/transformer.json")
    parser.add_argument("--bundle", type=Path, help="bundle directory (default: model_dir from the config)")
    parser.add_argument("--no-bundle", action="store_true", help="evidence and data checks only")
    parser.add_argument("--full-validation", action="store_true",
                        help="re-run inference on all validation rows and compare the confusion matrix")
    parser.add_argument("--pretrained-check", action="store_true",
                        help="compare the winner encoder with the pinned pretrained encoder (local files only)")
    parser.add_argument("--pretrained", type=Path, help="local pretrained directory instead of the HF cache")
    parser.add_argument("--output", type=Path, help="also write JSON here (never the committed report)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.output and args.output.resolve() == (args.root / COMMITTED_REPORT).resolve():
        print(f"Refusing to overwrite the committed {COMMITTED_REPORT}", file=sys.stderr)
        return 2
    try:
        result = verify(args.root, args.config, args.bundle, args)
    except VerificationError as exc:
        print(f"T06 verification FAILED: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError, OSError) as exc:
        print(f"T06 verification ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
