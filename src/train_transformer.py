"""T06: fine-tune RuBERT on baseline train IDs; select using validation only.

Run as ``python -m src.train_transformer --config configs/transformer.json``.
No test data loader or test evaluation is provided by this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set before importing torch; CUDA determinism, and no implicit HF credentials.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import train_baseline as base
from src.check_resources import (
    collect_grad_norms, evaluate_parameter_updates, select_sentinel_parameters,
    summarize_loading_info, take_parameter_snapshot,
)

MAPPING = base.EXPECTED_LABEL_MAPPING
ID2LABEL = {v: k for k, v in MAPPING.items()}
FIXED_TEXTS = base._smoke_texts()
CONFIG_KEYS = {
    "seed", "model_name", "revision", "train_path", "validation_path",
    "data_prep_audit_path", "data_prep_split_ids_path", "baseline_metadata_path",
    "baseline_train_ids_path", "report_dir", "model_dir", "label_mapping",
    "device", "num_threads", "epochs", "batch_size", "max_length", "truncation_side",
    "learning_rate", "weight_decay", "class_weight_mode", "eval_every_steps",
    "log_every_steps", "pilot_steps",
}


def validate_config(config: dict) -> None:
    base._reject_test_keys(config)
    base._check_keys("transformer", config, CONFIG_KEYS, set())
    if config["model_name"] != "cointegrated/rubert-tiny2":
        raise ValueError("T06 requires cointegrated/rubert-tiny2")
    if not isinstance(config["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("revision must be a pinned 40-character commit SHA")
    if config["label_mapping"] != MAPPING:
        raise ValueError(f"Required label mapping: {MAPPING}")
    for key in ("seed", "num_threads", "epochs", "batch_size", "max_length",
                "eval_every_steps", "log_every_steps", "pilot_steps"):
        minimum = 0 if key == "seed" else 1
        if not base._is_int(config[key]) or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["epochs"] > 3 or not 8 <= config["max_length"] <= 512 or config["pilot_steps"] > 3:
        raise ValueError("Budget: epochs <= 3, max_length in [8,512], pilot_steps <= 3")
    for key in ("learning_rate", "weight_decay"):
        value = config[key]
        if not base._is_number(value) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if config["class_weight_mode"] not in ("none", "balanced_train"):
        raise ValueError("class_weight_mode must be none or balanced_train")
    if config["device"] not in ("cpu", "cuda") or config["truncation_side"] != "right":
        raise ValueError("device must be cpu/cuda; truncation_side must be right")
    path_keys = [key for key in CONFIG_KEYS if key.endswith("_path") or key.endswith("_dir")]
    for key in path_keys:
        base._check_relative_path(key, config[key], is_input=key.endswith("_path"))
    if config["train_path"] == config["validation_path"]:
        raise ValueError("train and validation paths must differ")
    outputs = [config["model_dir"], config["report_dir"]]
    if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
           for a in outputs for b in outputs if a != b) or outputs[0] == outputs[1]:
        raise ValueError("model_dir and report_dir must be separate")


def load_data(config: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify accepted data version before any tokenization or optimization."""
    path = lambda key: base.resolve_path(root, config[key])
    metadata = base.read_json(path("baseline_metadata_path"))
    for key, expected in (("data_prep_audit_path", "t03_audit_sha256"),
                          ("data_prep_split_ids_path", "t03_split_ids_sha256")):
        if base.sha256_file(path(key)) != metadata[expected]:
            raise ValueError(f"Baseline data version mismatch: {key}")
    frames = {}
    for split in ("train", "validation"):
        if base.sha256_file(path(f"{split}_path")) != metadata[f"{split}_parquet_sha256"]:
            raise ValueError(f"Baseline Parquet checksum mismatch: {split}")
        frames[split] = base.load_prepared_split(
            path(f"{split}_path"), split_name=split,
            text_column="review_text", label_column="label_id", id_column="record_id")
    audit = base.load_t03_audit(path("data_prep_audit_path"))
    base.validate_against_t03_audit(audit, train=frames["train"], validation=frames["validation"],
                                  label_column="label_id", train_ids_configured=True)
    split_check = base.validate_split_ids_against_t03(
        path("data_prep_split_ids_path"), train=frames["train"], validation=frames["validation"],
        id_column="record_id", label_column="label_id")
    independence = base.validate_split_independence(
        frames["train"], frames["validation"], id_column="record_id", text_column="review_text")
    # Ordered ID + label equality, not just equal counts or an independent random subset.
    train = base.apply_train_id_selection(frames["train"], path("baseline_train_ids_path"),
                                          id_column="record_id", label_column="label_id",
                                          text_column="review_text")
    for split, frame in (("train", train), ("validation", frames["validation"])):
        if base.ids_sha256(frame, id_column="record_id", label_column="label_id") != metadata[f"{split}_ids_sha256"]:
            raise ValueError(f"Baseline ordered IDs/labels mismatch: {split}")
        if len(frame) != metadata[f"{split}_rows"]:
            raise ValueError(f"Baseline row count mismatch: {split}")
    version = {
        "baseline_metadata_sha256": base.sha256_file(path("baseline_metadata_path")),
        "baseline_train_ids_file_sha256": base.sha256_file(path("baseline_train_ids_path")),
        "train_ids_equal_baseline": True, "final_test_used": False,
        "split_independence": independence, "t03_split_check": split_check,
        "train_rows": len(train), "validation_rows": len(frames["validation"]),
        "baseline_validation_macro_f1": metadata["winner_validation_macro_f1"],
        "checksums": {key: base.sha256_file(path(key)) for key in (
            "train_path", "validation_path", "data_prep_audit_path", "data_prep_split_ids_path")},
        "train_ids_sha256": metadata["train_ids_sha256"],
        "validation_ids_sha256": metadata["validation_ids_sha256"],
    }
    return train, frames["validation"], version


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame["review_text"].tolist()
        self.labels = frame["label_id"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def encode(tokenizer, texts, config):
    # Identical processing in training, validation, and local inference.
    tokenizer.truncation_side = config["truncation_side"]
    return tokenizer(list(texts), padding=True, truncation=True,
                     max_length=config["max_length"], return_tensors="pt")


def make_loader(frame, tokenizer, config, *, training):
    def collate(rows):
        texts, labels = zip(*rows)
        return encode(tokenizer, texts, config), torch.tensor(labels, dtype=torch.long)
    return DataLoader(TextDataset(frame), batch_size=config["batch_size"], shuffle=training,
                      generator=torch.Generator().manual_seed(config["seed"]),
                      num_workers=0, collate_fn=collate, drop_last=False)


def setup(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose a separate CPU config")
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()


def load_pretrained(config):
    kwargs = {"revision": config["revision"], "token": False, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], **kwargs)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=3, id2label=ID2LABEL, label2id=MAPPING,
        output_loading_info=True, **kwargs)
    loading = summarize_loading_info(info)
    if not loading["only_new_head_missing"] or loading["mismatched_keys_count"]:
        raise RuntimeError(f"Pretrained encoder was not loaded completely: {loading}")
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("All encoder and head parameters must be trainable")
    return tokenizer, model.to(config["device"]), loading


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def memory_stats(device):
    import psutil
    result = {"rss_current_mib": psutil.Process().memory_info().rss / 1024**2}
    # Linux /proc high water mark; unavailable on other platforms rather than a fake peak.
    status = Path("/proc/self/status")
    result["rss_peak_mib"] = None
    if status.exists():
        match = re.search(r"^VmHWM:\s+(\d+)\s+kB", status.read_text(), re.MULTILINE)
        if match:
            result["rss_peak_mib"] = int(match[1]) / 1024
    result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None
    result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2 if device == "cuda" else None
    return result


def predict_texts(model, tokenizer, texts, config):
    if not texts:
        raise ValueError("At least one text is required")
    model.eval()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(texts), config["batch_size"]):
            inputs = encode(tokenizer, texts[offset:offset + config["batch_size"]], config)
            logits = model(**{k: v.to(config["device"]) for k, v in inputs.items()}).logits
            result.extend(logits.softmax(-1).cpu().tolist())
    probs = np.asarray(result)
    if probs.shape != (len(texts), 3) or not np.isfinite(probs).all():
        raise RuntimeError("Expected three finite probabilities per text")
    ids = probs.argmax(-1).tolist()
    return {"label_ids": ids, "labels": [ID2LABEL[i] for i in ids],
            "probability_class_ids": [0, 1, 2], "probability_labels": base.CLASS_NAMES,
            "probabilities": probs.tolist()}


def evaluate(model, tokenizer, frame, config):
    start = time.perf_counter()
    predictions = predict_texts(model, tokenizer, frame["review_text"].tolist(), config)
    sync(config["device"])
    y, pred = frame["label_id"].to_numpy(), np.asarray(predictions["label_ids"])
    return {**base.validation_metrics(y, pred),
            "confusion_matrix": base.validation_confusion_matrix(y, pred),
            "validation_seconds": time.perf_counter() - start}, pred


def select_checkpoint(rows):
    if not rows or any(not math.isfinite(r["macro_f1"]) for r in rows):
        raise ValueError("Non-empty finite validation results required")
    return max(rows, key=lambda row: (row["macro_f1"], -row["step"]))


def save_bundle(path, model, tokenizer, config, metadata):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    base.write_json(path / "training_config.json", config)
    base.write_json(path / "label_mapping.json", MAPPING)
    base.write_json(path / "metadata.json", metadata)


def load_bundle(path: Path, device="cpu"):
    config = base.read_json(path / "training_config.json")
    validate_config(config)
    if base.read_json(path / "label_mapping.json") != MAPPING:
        raise ValueError("Bundle label mapping mismatch")
    # This function cannot contact the Hub; the bundle is self-contained.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if model.config.num_labels != 3 or model.config.id2label != ID2LABEL or model.config.label2id != MAPPING:
        raise ValueError("Model class order differs from negative=0/neutral=1/positive=2")
    config["device"] = device
    return model.to(device).eval(), tokenizer, config


def write_validation(report, rows):
    flat = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in rows]
    with (report / "validation_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)


def run_training(config_path: Path, root: Path | None = None, *, pilot=False):
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    config = base.read_json(config_path)
    validate_config(config)
    train, validation, version = load_data(config, root)
    setup(config)
    report = base.resolve_path(root, config["report_dir"]) / ("pilot" if pilot else "")
    bundle = base.resolve_path(root, config["model_dir"])
    # No implicit overwrite of evidence or an accepted bundle.
    if report.exists() and any(report.iterdir()):
        if not (not pilot and all(p.name == "pilot" for p in report.iterdir())):
            raise FileExistsError(f"Report directory is not empty: {report}")
    if not pilot and bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {bundle}")
    if base.git_ignore_status(bundle / "model.safetensors", root) is False:
        raise ValueError("Weights must be Git ignored")
    report.mkdir(parents=True, exist_ok=True)
    log_path = report / "training_log.jsonl"
    def log(event, **fields):
        row = {"event": event, **fields}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    base.write_json(report / "config.json", config)
    base.write_json(report / "data_version.json", version)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    tokenizer, model, loading = load_pretrained(config)
    loader = make_loader(train, tokenizer, config, training=True)
    steps_budget = config["pilot_steps"] if pilot else len(loader) * config["epochs"]
    weights = base.compute_train_class_weight(train["label_id"].to_numpy(), config["class_weight_mode"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=None if weights is None else torch.tensor(
        [weights[i] for i in range(3)], dtype=torch.float32, device=config["device"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    sentinels = select_sentinel_parameters({n: p.ndim for n, p in model.named_parameters()})
    params = dict(model.named_parameters())
    sentinel_names = [n for names in sentinels.values() for n in names]
    before = take_parameter_snapshot([(n, params[n]) for n in sentinel_names])
    base.write_json(report / "loading_info.json", loading)
    versions = {**base.library_versions(), "torch": torch.__version__,
                "transformers": base.package_version("transformers"),
                "tokenizers": base.package_version("tokenizers")}
    log("start", mode="pilot" if pilot else "training", started_at=started_at, steps_budget=steps_budget,
        rows_per_epoch=len(train), class_weights=weights, precision="float32",
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        data_version=version, versions=versions)
    rows, step, seen, train_seconds, loss_sum, interval_steps = [], 0, 0, 0.0, 0.0, 0
    update_evidence = None
    best_predictions = None
    for epoch in range(1, config["epochs"] + 1):
        for inputs, labels in loader:
            sync(config["device"])
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{k: v.to(config["device"]) for k, v in inputs.items()}).logits
            loss = loss_fn(logits, labels.to(config["device"]))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            grads = collect_grad_norms(params, sentinel_names) if step == 0 else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            sync(config["device"])
            train_seconds += time.perf_counter() - tick
            step += 1
            seen += len(labels)
            loss_sum += loss.item()
            interval_steps += 1
            if step == 1:
                after = take_parameter_snapshot([(n, params[n]) for n in sentinel_names])
                update_evidence = evaluate_parameter_updates(before, after, grads, sentinels)
                log("parameter_updates", **update_evidence)
            if step == 1 or step % config["log_every_steps"] == 0 or step == steps_budget:
                log("train", epoch=epoch, step=step, examples_seen=seen,
                    mean_batch_loss=loss_sum / interval_steps, train_seconds=train_seconds,
                    memory=memory_stats(config["device"]))
                loss_sum, interval_steps = 0.0, 0
            if not pilot and (step % config["eval_every_steps"] == 0 or step == len(loader) * epoch):
                metrics, predictions = evaluate(model, tokenizer, validation, config)
                row = {"epoch": epoch, "step": step, "examples_seen": seen,
                       "train_seconds": train_seconds, **metrics}
                rows.append(row)
                winner = select_checkpoint(rows)
                selected = winner is row
                log("validation", selected=selected, **row)
                write_validation(report, rows)
                if selected:
                    best_predictions = predictions
                    save_bundle(bundle, model, tokenizer, config, {"data_version": version, "winner": row})
            if step == steps_budget:
                break
        if step == steps_budget:
            break
    common = {"mode": "pilot" if pilot else "training", "steps": step, "steps_budget": steps_budget,
              "started_at": started_at, "versions": versions, "loading_info": loading,
              "examples_seen": seen, "train_seconds": train_seconds,
              "elapsed_seconds": time.perf_counter() - start, "memory": memory_stats(config["device"]),
              "parameter_updates": update_evidence, "data_version": version,
              "class_weights": weights, "config_sha256": base.sha256_json(config),
              "script_sha256": base.sha256_file(Path(__file__)), "final_test_used": False}
    if pilot:
        common["projected_train_seconds"] = train_seconds / step * len(loader) * config["epochs"]
        common["weights_discarded"] = True
        base.write_json(report / "run_metadata.json", common)
        log("complete", **common)
        return common
    winner = select_checkpoint(rows)
    # Release training state before reloading the saved winner (important on small CPU hosts).
    del optimizer, model, params, inputs, logits, loss
    import gc
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    loaded, tok, loaded_config = load_bundle(bundle, config["device"])
    metrics_after, predictions_after = evaluate(loaded, tok, validation, loaded_config)
    fixed = predict_texts(loaded, tok, FIXED_TEXTS, loaded_config)
    roundtrip = {"validation_predictions_identical_after_load": bool(np.array_equal(best_predictions, predictions_after)),
                 "validation_macro_f1_after_load": metrics_after["macro_f1"],
                 "fixed_texts": {"text_sha256": [base.sha256_text(t) for t in FIXED_TEXTS], **fixed},
                 "local_files_only": True}
    if not roundtrip["validation_predictions_identical_after_load"]:
        raise RuntimeError("Saved checkpoint predictions changed after loading")
    base.write_json(report / "bundle_roundtrip.json", roundtrip)
    baseline_f1 = version["baseline_validation_macro_f1"]
    common.update(winner=winner, selection_metric="validation_macro_f1",
                  selection_tie_break="earlier_step", baseline_validation_macro_f1=baseline_f1,
                  delta_vs_baseline=winner["macro_f1"] - baseline_f1,
                  elapsed_seconds=time.perf_counter() - start,
                  memory=memory_stats(config["device"]))
    base.write_json(bundle / "metadata.json", common)
    common["bundle_files_sha256"] = {
        p.name: base.sha256_file(p) for p in bundle.iterdir() if p.is_file()}
    base.write_json(report / "run_metadata.json", common)
    base.write_json(report / "winner_config.json", {"config": config, "winner": winner})
    lines = ["# T06 — validation", "", "Выбор: максимум validation macro-F1; равенство — более ранний шаг.",
             "Финальный test не читался. Одна конфигурация; pilot не участвует в выборе.", "",
             "| epoch | step | macro-F1 | negative F1 | neutral F1 | positive F1 | выбран |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"| {row['epoch']} | {row['step']} | {row['macro_f1']:.6f} | {row['negative_f1']:.6f} | {row['neutral_f1']:.6f} | {row['positive_f1']:.6f} | {'да' if row is winner else ''} |")
    lines += ["", f"Baseline validation macro-F1: {baseline_f1:.6f}.",
              f"RuBERT validation macro-F1: {winner['macro_f1']:.6f}; разница {common['delta_vs_baseline']:+.6f}.",
              "Это сравнение по validation, не финальная оценка на test.",
              "Отсутствие преимущества допустимо; результат сохраняется без дополнительных поисков.", ""]
    (report / "validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("complete", **common)
    return common


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/transformer.json"))
    parser.add_argument("--pilot", action="store_true", help="1–3 train steps; discard weights; no validation selection")
    parser.add_argument("--check-data", action="store_true", help="verify baseline IDs and data checksums; no training")
    parser.add_argument("--verify-bundle", type=Path, help="offline load and predict fixed texts, no training")
    args = parser.parse_args(argv)
    try:
        if args.verify_bundle:
            model, tokenizer, config = load_bundle(args.verify_bundle)
            setup(config)
            print(json.dumps(predict_texts(model, tokenizer, FIXED_TEXTS, config), ensure_ascii=False, indent=2))
        elif args.check_data:
            config = base.read_json(args.config)
            validate_config(config)
            _, _, version = load_data(config, Path(__file__).resolve().parents[1])
            print(json.dumps(version, indent=2))
        else:
            run_training(args.config, pilot=args.pilot)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"T06: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
