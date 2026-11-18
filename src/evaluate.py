"""T13: финальное сравнение baseline и RuBERT на одном неизменном текстовом test.

    python -m src.evaluate --check                        # только предусловия, test не читается
    python -m src.evaluate --freeze reports/freeze.json   # финальный запуск после T12

Финальный test открывается только после подтверждённой фиксации T12: файл фиксации
должен быть утверждён, а комплекты моделей, данные T03 и сам test — совпадать с
записанными в нём суммами. Все проверки выполняются до декодирования текстов test.

Предсказания обеих моделей получаются через общий интерфейс ``src.classifier``
(тот же путь, что у бота и ``tools/evaluate_classifier.py``), метрики — функциями
``src.train_baseline`` в фиксированном порядке классов negative=0, neutral=1, positive=2.

Формат фиксации T12 ещё не утверждён. ``FREEZE_SCHEMA`` описывает минимум, который
нужен T13; T12 может принять его или добавить адаптер и проверки в ``FREEZE_CHECKS``.

Код выхода: 0 — отчёты записаны (или предусловия выполнены при --check);
2 — предусловия, модели или данные не прошли проверку; отчёты не записываются.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import classifier as c  # noqa: E402
from src import train_baseline as base  # noqa: E402

CLASS_IDS = base.CLASS_IDS
CLASS_NAMES = base.CLASS_NAMES
BACKENDS = ("baseline", "rubert")

DEFAULT_FREEZE_PATH = ROOT / "reports" / "freeze.json"
DEFAULT_OUT_DIR = ROOT / "reports" / "evaluation"

# Черновик T13, а не утверждённый протокол T12.
FREEZE_SCHEMA = "t13-final-text-eval-input/draft-1"
# Версия данных, которая должна совпадать у обеих моделей.
SHARED_VERSION_KEYS = ("train_ids_sha256", "train_rows", "validation_ids_sha256", "validation_rows",
                       "t03_split_ids_sha256", "t03_audit_sha256")
# Прогрев на фиксированных искусственных текстах: test в прогреве не участвует.
WARMUP_TEXTS = tuple(base._smoke_texts())
TEXT_COLUMN, LABEL_COLUMN, ID_COLUMN = "review_text", "label_id", "record_id"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class FreezeError(ValueError):
    """Фиксация T12 отсутствует, не утверждена или не совпадает с файлами."""


class DataMismatchError(ValueError):
    """Данные, ID, метки или версии данных моделей не совпадают."""


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Accuracy, macro-F1, метрики классов и матрица 3×3 (строки — истина, столбцы — ответ).

    Класс без предсказаний остаётся в отчёте с precision = 0 и входит в macro-F1.
    """
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise DataMismatchError("Labels and predictions must be non-empty and of equal length")
    for name, values in (("labels", truth), ("predictions", predicted)):
        invalid = sorted(set(values.tolist()) - set(CLASS_IDS))
        if invalid:
            raise DataMismatchError(f"Unexpected class id(s) in {name}: {invalid}")
    flat = base.validation_metrics(truth, predicted)
    return {
        "rows": int(len(truth)),
        "accuracy": flat["accuracy"],
        "macro_f1": flat["macro_f1"],
        "per_class": {name: {key: flat[f"{name}_{key}"] for key in ("precision", "recall", "f1", "support")}
                      for name in CLASS_NAMES},
        "predicted_counts": {name: int(np.count_nonzero(predicted == i)) for i, name in zip(CLASS_IDS, CLASS_NAMES)},
        "confusion_matrix": base.validation_confusion_matrix(truth, predicted),
    }


# --------------------------------------------------------------------------- #
# Фиксация T12
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _repo_path(root: Path, value: Any, name: str) -> Path:
    _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute(),
             f"{name} must be a relative repository path")
    try:
        return base.resolve_path(root, value)
    except ValueError as exc:
        raise FreezeError(f"{name}: {exc}") from exc


def _check_schema(freeze: Mapping[str, Any], root: Path) -> None:
    _require(freeze.get("schema") == FREEZE_SCHEMA, f"Unsupported freeze schema: {freeze.get('schema')!r}")
    _require(freeze.get("status") == "approved", "T12 freeze is not approved; the final test stays closed")
    approvers = freeze.get("approved_by")
    _require(isinstance(approvers, list) and approvers and all(isinstance(a, str) and a.strip() for a in approvers),
             "T12 freeze has no approvers")
    _require(isinstance(freeze.get("approved_at"), str) and bool(freeze["approved_at"].strip()),
             "T12 freeze has no approval date")
    _require(isinstance(freeze.get("frozen_commit"), str) and bool(_COMMIT.fullmatch(freeze["frozen_commit"])),
             "T12 freeze has no full frozen_commit")
    _require(freeze.get("final_test_used_before_freeze") is False,
             "T12 freeze must state final_test_used_before_freeze=false")
    _require(freeze.get("bot_model") in BACKENDS, "T12 freeze must name the bot model chosen on validation")
    _require(freeze.get("bot_model_selected_by") == "validation_and_resources",
             "Bot model must be selected by validation and resources, not by test")


def _check_data_prep(freeze: Mapping[str, Any], root: Path) -> None:
    prep = freeze.get("data_prep")
    _require(isinstance(prep, dict), "T12 freeze has no data_prep section")
    config = base.read_json(root / "configs/nlp_baseline.json")
    for key, config_key in (("audit_sha256", "data_prep_audit_path"), ("split_ids_sha256", "data_prep_split_ids_path")):
        _require(_is_sha256(prep.get(key)), f"T12 freeze data_prep.{key} must be SHA-256")
        path = _repo_path(root, config[config_key], config_key)
        _require(path.is_file() and base.sha256_file(path) == prep[key], f"T03 {config_key} differs from T12 freeze")


def _check_test_entry(freeze: Mapping[str, Any], root: Path) -> None:
    test = freeze.get("test")
    _require(isinstance(test, dict), "T12 freeze has no test section")
    _repo_path(root, test.get("path"), "test.path")
    for key in ("parquet_sha256", "ids_sha256"):
        _require(_is_sha256(test.get(key)), f"T12 freeze test.{key} must be SHA-256")
    rows = test.get("rows")
    _require(isinstance(rows, int) and not isinstance(rows, bool) and rows > 0, "T12 freeze test.rows must be positive")


def _check_models(freeze: Mapping[str, Any], root: Path) -> None:
    models = freeze.get("models")
    _require(isinstance(models, dict) and set(models) == set(BACKENDS),
             f"T12 freeze must list exactly the models {list(BACKENDS)}")
    for backend in BACKENDS:
        entry = models[backend]
        _require(isinstance(entry, dict), f"T12 freeze models.{backend} must be an object")
        _repo_path(root, entry.get("model_path"), f"models.{backend}.model_path")
        report = _repo_path(root, entry.get("report_path"), f"models.{backend}.report_path")
        _require(_is_sha256(entry.get("report_sha256")), f"T12 freeze models.{backend}.report_sha256 must be SHA-256")
        _require(report.is_file() and base.sha256_file(report) == entry["report_sha256"],
                 f"{backend} run report differs from T12 freeze")


def _check_inference(freeze: Mapping[str, Any], root: Path) -> None:
    inference = freeze.get("inference")
    _require(isinstance(inference, dict), "T12 freeze has no inference section")
    batch = inference.get("batch_size")
    _require(isinstance(batch, int) and not isinstance(batch, bool) and batch > 0,
             "T12 freeze inference.batch_size must be positive")
    _require(inference.get("device") == "cpu", "Only CPU inference is supported by T13")
    threads = inference.get("num_threads")
    _require(threads is None or (isinstance(threads, int) and not isinstance(threads, bool) and threads > 0),
             "T12 freeze inference.num_threads must be positive or null")


def _check_git(freeze: Mapping[str, Any], root: Path) -> None:
    state = git_state(root, freeze["frozen_commit"])
    _require(state["commit"] is not None, "Git commit of the evaluation code is unknown")
    _require(state["frozen_commit_is_ancestor"], "Evaluated code does not descend from the T12 frozen commit")
    _require(not state["tracked_changes"], "Tracked files have uncommitted changes; commit before the final run")


# T12 может дополнить список собственными проверками протокола.
FREEZE_CHECKS: list[Callable[[Mapping[str, Any], Path], None]] = [
    _check_schema, _check_data_prep, _check_test_entry, _check_models, _check_inference, _check_git,
]


def git_state(root: Path, frozen_commit: str) -> dict[str, Any]:
    ancestor = base._git(root, "merge-base", "--is-ancestor", frozen_commit, "HEAD")
    status = base._git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": base.git_commit(root),
        "frozen_commit_is_ancestor": ancestor is not None and ancestor.returncode == 0,
        "tracked_changes": status is None or status.returncode != 0 or bool(status.stdout.strip()),
    }


def load_freeze(path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Прочитать и проверить фиксацию T12. Test при этом не читается."""
    if not path.is_file():
        raise FreezeError(f"T12 freeze not found: {base.display_path(path, root)}; the final test stays closed")
    try:
        freeze = base.read_json(path)
    except (OSError, ValueError) as exc:
        raise FreezeError(f"T12 freeze is not valid JSON: {exc}") from exc
    _require(isinstance(freeze, dict), "T12 freeze must be a JSON object")
    for check in FREEZE_CHECKS:
        check(freeze, root)
    return freeze


# --------------------------------------------------------------------------- #
# Данные test
# --------------------------------------------------------------------------- #
def check_test_ids_against_t03(frame: pd.DataFrame, split_ids_path: Path) -> None:
    """ID, порядок и метки test должны совпадать с T03 split_ids.csv (только ID и метки)."""
    needed = {"split", ID_COLUMN, LABEL_COLUMN}
    ids = pd.read_csv(split_ids_path, dtype=str, keep_default_na=False, usecols=lambda col: col in needed)
    expected = ids.loc[ids["split"].str.lower() == "test"]
    if expected[ID_COLUMN].tolist() != frame[ID_COLUMN].astype(str).tolist():
        raise DataMismatchError("Test IDs or their order differ from T03 split_ids.csv")
    if expected[LABEL_COLUMN].astype(int).tolist() != frame[LABEL_COLUMN].astype(int).tolist():
        raise DataMismatchError("Test labels differ from T03 split_ids.csv")


def load_frozen_test(freeze: Mapping[str, Any], root: Path = ROOT) -> pd.DataFrame:
    """Открыть test только после совпадения SHA-256 файла с фиксацией T12."""
    entry = freeze["test"]
    path = base.resolve_path(root, entry["path"])
    if not path.is_file() or base.sha256_file(path) != entry["parquet_sha256"]:
        raise DataMismatchError("Test Parquet differs from the T12 freeze")
    frame = base.load_prepared_split(path, split_name="test", text_column=TEXT_COLUMN,
                                     label_column=LABEL_COLUMN, id_column=ID_COLUMN)
    if len(frame) != entry["rows"]:
        raise DataMismatchError(f"Test has {len(frame)} rows; T12 freeze records {entry['rows']}")
    if base.ids_sha256(frame, id_column=ID_COLUMN, label_column=LABEL_COLUMN) != entry["ids_sha256"]:
        raise DataMismatchError("Test IDs/labels differ from the T12 freeze")
    config = base.read_json(root / "configs/nlp_baseline.json")
    audit = base.load_t03_audit(base.resolve_path(root, config["data_prep_audit_path"]))
    if audit["final_summary"]["test"]["rows"] != len(frame):
        raise DataMismatchError("Test row count differs from T03 audit")
    check_test_ids_against_t03(frame, base.resolve_path(root, config["data_prep_split_ids_path"]))
    return frame


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #
def model_data_version(backend: str, data_version: Mapping[str, Any]) -> dict[str, Any]:
    """Привести версии данных baseline и RuBERT к общим ключам SHARED_VERSION_KEYS."""
    try:
        if backend == "baseline":
            version = {key: data_version[key] for key in SHARED_VERSION_KEYS}
        elif backend == "rubert":
            checksums = data_version["checksums"]
            version = {key: data_version[key] for key in SHARED_VERSION_KEYS[:4]}
            version.update(t03_split_ids_sha256=checksums["data_prep_split_ids_path"],
                           t03_audit_sha256=checksums["data_prep_audit_path"])
        else:
            raise DataMismatchError(f"Unknown backend: {backend!r}")
        if data_version["final_test_used"] is not False:
            raise DataMismatchError(f"{backend} data version reports final test use")
    except (KeyError, TypeError) as exc:
        raise DataMismatchError(f"{backend} data version is incomplete: {exc}") from exc
    return version


def check_model_versions(versions: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any]) -> None:
    """Обе модели обучены на одной версии данных T03, записанной в фиксации."""
    if set(versions) != set(BACKENDS):
        raise DataMismatchError(f"Expected models {list(BACKENDS)}; got {sorted(versions)}")
    prep = freeze["data_prep"]
    for backend, version in versions.items():
        if (version["t03_split_ids_sha256"] != prep["split_ids_sha256"]
                or version["t03_audit_sha256"] != prep["audit_sha256"]):
            raise DataMismatchError(f"{backend} was trained on a T03 version other than the T12 freeze")
    reference = versions[BACKENDS[0]]
    differing = [key for key in SHARED_VERSION_KEYS if any(v[key] != reference[key] for v in versions.values())]
    if differing:
        raise DataMismatchError(f"Model data versions differ: {', '.join(differing)}")


def timed_load(model: Any, clock: Callable[[], float] = time.perf_counter) -> float:
    """Время проверки и загрузки комплекта через src.classifier; test ещё не прочитан."""
    start = clock()
    model._get_bundle()
    return clock() - start


def run_model(model: Any, texts: Sequence[str], *, batch_size: int, load_seconds: float,
              clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Прогрев и основной проход измеряются раздельно; загрузка — заранее в timed_load.

    Прогрев — один вызов на WARMUP_TEXTS; основной проход — test целиком партиями
    по batch_size в исходном порядке.
    """
    loaded = clock()
    model.predict_batch(list(WARMUP_TEXTS), max_text_length=None)
    warmed = clock()
    results: list[dict[str, Any]] = []
    for offset in range(0, len(texts), batch_size):
        results.extend(model.predict_batch(list(texts[offset:offset + batch_size]), max_text_length=None))
    finished = clock()
    if len(results) != len(texts):
        raise DataMismatchError("Model returned a different number of predictions")
    inference = finished - warmed
    return {
        "predictions": [int(r["label_id"]) for r in results],
        "truncated_rows": int(sum(bool(r["truncated"]) for r in results)),
        "timing": {
            "load_seconds": load_seconds,
            "warmup_seconds": warmed - loaded,
            "warmup_texts": len(WARMUP_TEXTS),
            "inference_seconds": inference,
            "rows": len(texts),
            "batch_size": batch_size,
            "batches": -(-len(texts) // batch_size),
            "ms_per_text": 1000 * inference / len(texts),
            "texts_per_second": len(texts) / inference if inference > 0 else None,
        },
    }


def check_same_inputs(runs: Mapping[str, Mapping[str, Any]]) -> None:
    """Обе строки сравнения рассчитаны на одних и тех же ID, метках, порядке и версии test."""
    if set(runs) != set(BACKENDS):
        raise DataMismatchError(f"Expected runs for {list(BACKENDS)}; got {sorted(runs)}")
    reference = runs[BACKENDS[0]]
    for backend, run in runs.items():
        for key in ("test_version", "record_ids", "labels"):
            if run[key] != reference[key]:
                raise DataMismatchError(f"{backend} differs from {BACKENDS[0]} in {key}")
        if len(run["predictions"]) != len(run["record_ids"]):
            raise DataMismatchError(f"{backend} predictions do not match the number of records")


# --------------------------------------------------------------------------- #
# Повторы внутри test (чувствительность)
# --------------------------------------------------------------------------- #
def repeat_groups_in_test(frame: pd.DataFrame, candidates_path: Path, decisions_path: Path,
                          audit: Mapping[str, Any]) -> list[list[str]]:
    """Группы повторов внутри test по правилу T03, зафиксированному до обучения.

    Точные повторы — совпадение ``normalize_text_key``; близкие — пары T03 с решением
    ``duplicate``, обе записи которых в test. Группа — компонента связности; порядок
    записей внутри группы — порядок test. Число точных групп и близких пар должно
    совпасть с аудитом T03: новая дедупликация test здесь не выполняется.
    """
    order = {rid: i for i, rid in enumerate(frame[ID_COLUMN].astype(str))}
    parent = {rid: rid for rid in order}

    def find(rid: str) -> str:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=order.__getitem__)] = min(ra, rb, key=order.__getitem__)

    first_by_key: dict[str, str] = {}
    exact_groups: set[str] = set()
    for rid, text in zip(frame[ID_COLUMN].astype(str), frame[TEXT_COLUMN]):
        first = first_by_key.setdefault(base.normalize_text_key(text), rid)
        if first != rid:
            exact_groups.add(first)
            union(first, rid)

    decisions = pd.read_csv(decisions_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    confirmed = set(decisions.loc[decisions["decision"] == "duplicate", "pair_id"])
    pairs = pd.read_csv(candidates_path, dtype=str, keep_default_na=False)
    near_pairs = 0
    for row in pairs.itertuples(index=False):
        if row.pair_id in confirmed and row.left_split == "test" and row.right_split == "test":
            if row.left_record_id not in order or row.right_record_id not in order:
                raise DataMismatchError(f"T03 test duplicate pair {row.pair_id} refers to unknown test IDs")
            union(row.left_record_id, row.right_record_id)
            near_pairs += 1

    expected = (audit["exact_dedup"]["test_exact_duplicate_groups"], audit["near_duplicates"]["near_pairs_within_test"])
    if (len(exact_groups), near_pairs) != expected:
        raise DataMismatchError(f"Test repeats ({len(exact_groups)} exact groups, {near_pairs} near pairs) "
                                f"differ from T03 audit {expected}")

    groups: dict[str, list[str]] = {}
    for rid in order:
        groups.setdefault(find(rid), []).append(rid)
    return [members for members in groups.values() if len(members) > 1]


def sensitivity(runs: Mapping[str, Mapping[str, Any]], groups: Sequence[Sequence[str]],
                main: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Метрики после сохранения одной (первой) копии каждой группы; основной test не меняется."""
    rule = "keep_first_record_of_each_t03_repeat_group_in_test_order"
    if not groups:
        return {"applied": False, "rule": rule, "reason": "no repeats within test", "groups": 0}
    dropped = {rid for group in groups for rid in group[1:]}
    reference = runs[BACKENDS[0]]
    keep = [rid not in dropped for rid in reference["record_ids"]]
    labels = [label for label, k in zip(reference["labels"], keep) if k]
    models = {}
    for backend, run in runs.items():
        metrics = compute_metrics(labels, [p for p, k in zip(run["predictions"], keep) if k])
        metrics["delta_macro_f1"] = metrics["macro_f1"] - main[backend]["macro_f1"]
        metrics["delta_accuracy"] = metrics["accuracy"] - main[backend]["accuracy"]
        models[backend] = metrics
    return {"applied": True, "rule": rule, "groups": len(groups),
            "rows_in_groups": sum(len(g) for g in groups), "rows_dropped": len(dropped),
            "rows": len(labels), "models": models}


# --------------------------------------------------------------------------- #
# Окружение и отчёты
# --------------------------------------------------------------------------- #
def environment() -> dict[str, Any]:
    info: dict[str, Any] = {**base.library_versions(), "machine": platform.machine(),
                            "processor": platform.processor() or "unknown", "logical_cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_total_mib"] = psutil.virtual_memory().total / (1024 * 1024)
        info["physical_cpu_count"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        info.update(torch=torch.__version__, torch_num_threads=torch.get_num_threads())
    return info


TABLE_COLUMNS = (["model", "bot_model", "rows", "accuracy", "macro_f1"]
                 + [f"{name}_{key}" for name in CLASS_NAMES for key in ("precision", "recall", "f1", "support")]
                 + ["load_seconds", "warmup_seconds", "inference_seconds", "ms_per_text", "batch_size"])


def table_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for backend in BACKENDS:
        entry = report["models"][backend]
        metrics, timing = entry["metrics"], entry["timing"]
        row = {"model": backend, "bot_model": backend == report["bot_model"], "rows": metrics["rows"],
               "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        for name in CLASS_NAMES:
            row.update({f"{name}_{key}": value for key, value in metrics["per_class"][name].items()})
        row.update({key: timing[key] for key in ("load_seconds", "warmup_seconds", "inference_seconds",
                                                 "ms_per_text", "batch_size")})
        rows.append(row)
    return rows


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any]) -> str:
    run = report["run"]
    lines = ["# T13: финальное сравнение на текстовом test", "",
             f"- Commit: `{run['git_commit']}`; фиксация T12: `{run['freeze_path']}` (`{run['freeze_sha256']}`)",
             f"- Test: {report['test']['rows']} строк, Parquet `{report['test']['parquet_sha256']}`",
             f"- Команда: `{run['command']}`", f"- Модель бота (выбрана в T12 по validation): {report['bot_model']}", "",
             "| Модель | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | мс/текст |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table_rows(report):
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                     f"{row['negative_f1']:.4f} | {row['neutral_f1']:.4f} | {row['positive_f1']:.4f} | "
                     f"{row['ms_per_text']:.3f} |")
    for backend in BACKENDS:
        matrix = report["models"][backend]["metrics"]["confusion_matrix"]
        lines += ["", f"## Confusion matrix: {backend}", "", "| истина \\ ответ | " + " | ".join(CLASS_NAMES) + " |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {name} | " + " | ".join(str(v) for v in values) + " |" for name, values in zip(CLASS_NAMES, matrix)]
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """Агрегированные отчёты и предсказания по ID; тексты отзывов не записываются."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"metrics": out_dir / "final_metrics.json", "table": out_dir / "final_table.csv",
             "markdown": out_dir / "final_report.md", "predictions": out_dir / "predictions.csv"}
    base.write_json(paths["metrics"], report)
    _write_csv(paths["table"], TABLE_COLUMNS, [[row[col] for col in TABLE_COLUMNS] for row in table_rows(report)])
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    reference = runs[BACKENDS[0]]
    _write_csv(paths["predictions"], [ID_COLUMN, LABEL_COLUMN, *(f"{b}_label_id" for b in BACKENDS)],
               list(zip(reference["record_ids"], reference["labels"], *(runs[b]["predictions"] for b in BACKENDS))))
    for backend in BACKENDS:
        path = out_dir / f"confusion_matrix_{backend}.csv"
        matrix = report["models"][backend]["metrics"]["confusion_matrix"]
        _write_csv(path, ["true\\predicted", *CLASS_NAMES], [[name, *values] for name, values in zip(CLASS_NAMES, matrix)])
        paths[backend] = path
    return list(paths.values())


# --------------------------------------------------------------------------- #
# Финальный запуск
# --------------------------------------------------------------------------- #
def _set_torch_threads(threads: int | None) -> None:
    if threads is not None:
        import torch
        torch.set_num_threads(threads)


def evaluate(freeze_path: Path, out_dir: Path, *, root: Path = ROOT,
             factory: Callable[..., Any] = c.create_classifier,
             clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Финальная оценка обеих моделей: фиксация → модели → test → предсказания → отчёты.

    Test декодируется только после проверки фиксации, обоих комплектов и их версий данных.
    """
    freeze = load_freeze(freeze_path, root)
    inference = freeze["inference"]
    models, load_seconds, versions = {}, {}, {}
    for backend in BACKENDS:
        entry = freeze["models"][backend]
        if backend == "rubert":
            _set_torch_threads(inference["num_threads"])
        models[backend] = factory(backend, base.resolve_path(root, entry["model_path"]),
                                  report_path=base.resolve_path(root, entry["report_path"]))
        load_seconds[backend] = timed_load(models[backend], clock)
        versions[backend] = model_data_version(backend, models[backend].data_version)
    check_model_versions(versions, freeze)

    frame = load_frozen_test(freeze, root)
    record_ids = frame[ID_COLUMN].astype(str).tolist()
    labels = frame[LABEL_COLUMN].astype(int).tolist()
    texts = frame[TEXT_COLUMN].tolist()
    test_version = {"parquet_sha256": freeze["test"]["parquet_sha256"], "ids_sha256": freeze["test"]["ids_sha256"],
                    "rows": len(frame)}
    runs = {}
    for backend, model in models.items():
        run = run_model(model, texts, batch_size=inference["batch_size"], clock=clock,
                        load_seconds=load_seconds[backend])
        runs[backend] = {**run, "record_ids": record_ids, "labels": labels, "test_version": test_version}
    check_same_inputs(runs)

    main = {backend: compute_metrics(labels, run["predictions"]) for backend, run in runs.items()}
    prep_config = base.read_json(root / "configs/data_prep.json")
    candidates = base.resolve_path(root, prep_config["report_dir"]) / "near_duplicate_candidates.csv"
    decisions = base.resolve_path(root, prep_config["near_duplicates"]["decision_file"])
    audit = base.read_json(base.resolve_path(root, base.read_json(root / "configs/nlp_baseline.json")["data_prep_audit_path"]))
    groups = repeat_groups_in_test(frame, candidates, decisions, audit)

    report = {
        "task": "T13",
        "final_test_used": True,
        "bot_model": freeze["bot_model"],
        "bot_model_selected_by": freeze["bot_model_selected_by"],
        "labels": dict(zip(CLASS_NAMES, CLASS_IDS)),
        "run": {
            "command": f"python -m src.evaluate --freeze {base.display_path(freeze_path, root)}",
            "git_commit": git_state(root, freeze["frozen_commit"])["commit"],
            "frozen_commit": freeze["frozen_commit"],
            "freeze_path": base.display_path(freeze_path, root),
            "freeze_sha256": base.sha256_file(freeze_path),
            "freeze_schema": freeze["schema"],
            "approved_by": freeze["approved_by"],
            "approved_at": freeze["approved_at"],
        },
        "test": {**test_version, "path": freeze["test"]["path"], "split_ids_sha256": freeze["data_prep"]["split_ids_sha256"]},
        "data_version": versions[BACKENDS[0]],
        "inference_protocol": {
            "device": inference["device"], "batch_size": inference["batch_size"],
            "num_threads": inference["num_threads"], "max_text_length": None,
            "warmup": f"one predict_batch call on {len(WARMUP_TEXTS)} fixed non-test texts",
            "load": "bundle verification and loading via src.classifier, before the test is read",
            "inference": "whole test in original order, predict_batch per batch_size chunk; time.perf_counter",
        },
        "environment": environment(),
        "models": {
            backend: {"model_path": freeze["models"][backend]["model_path"],
                      "report_path": freeze["models"][backend]["report_path"],
                      "report_sha256": freeze["models"][backend]["report_sha256"],
                      "metrics": main[backend], "timing": runs[backend]["timing"],
                      "truncated_rows": runs[backend]["truncated_rows"]}
            for backend in BACKENDS
        },
        "repeat_sensitivity": {**sensitivity(runs, groups, main),
                               "t03_candidates_sha256": base.sha256_file(candidates),
                               "t03_decisions_sha256": base.sha256_file(decisions)},
    }
    report["outputs"] = [base.display_path(p, root) for p in write_reports(report, runs, out_dir)]
    return report


def check_only(freeze_path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Проверить предусловия финального запуска без моделей и без чтения test."""
    freeze = load_freeze(freeze_path, root)
    return {"status": "ready", "freeze_path": base.display_path(freeze_path, root),
            "freeze_sha256": base.sha256_file(freeze_path), "bot_model": freeze["bot_model"],
            "test_rows": freeze["test"]["rows"], "final_test_used": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH,
                        help="machine-readable T12 freeze (default: reports/freeze.json)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="output directory for aggregated reports (default: reports/evaluation)")
    parser.add_argument("--check", action="store_true",
                        help="only verify the T12 freeze and frozen files; do not load models or read the test")
    args = parser.parse_args(argv)
    freeze_path = args.freeze if args.freeze.is_absolute() else Path.cwd() / args.freeze
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    try:
        result = check_only(freeze_path) if args.check else evaluate(freeze_path, out_dir)
    except (c.ClassifierError, c.InvalidTextError, OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not args.check:
        result = {key: result[key] for key in ("bot_model", "test", "outputs")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
