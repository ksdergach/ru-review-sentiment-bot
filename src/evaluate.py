"""T13: финальное сравнение baseline и RuBERT на одном неизменном текстовом test.

    python -m src.evaluate --check        # готовность: фиксация, файлы, среда, модели; test не декодируется
    python -m src.evaluate --run-final    # финальный запуск по утверждённому протоколу T12

Источник истины — утверждённая фиксация T12 (``reports/freeze.json``, протокол
``reports/freeze.md``). Фиксация читается один раз: SHA-256 считается по прочитанным
байтам, и разбираются только они. До декодирования текстов test проверяются: принятая
версия файлов фиксации, суммы всех ``file_sha256``, git (оценщик закоммичен в HEAD),
среда измерений, назначение результатов, отчёты и комплекты обеих моделей, байтовая
сумма test. Любое несовпадение останавливает запуск; суммы фиксации никогда не
обновляются автоматически. Оба режима используют одну блокировку на Git worktree.
Второй процесс в этом worktree получает отказ до загрузки моделей.

Предсказания обеих моделей получаются через общий интерфейс ``src.classifier``
(тот же путь, что у бота и ``tools/evaluate_classifier.py``), метрики — функциями
``src.train_baseline`` в фиксированном порядке классов negative=0, neutral=1, positive=2.
Test не меняет модель бота, выбранную в T12 по validation.

Код выхода: 0 — отчёты записаны (или все предусловия выполнены при --check);
2 — предусловия, модели, данные или запись не прошли проверку; итоговый каталог не создаётся.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from html import escape
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

# Формат утверждённой фиксации T12 (``draft-1`` — версия формата, не статус) и версия протокола.
FREEZE_SCHEMA = "t13-final-text-eval-input/draft-1"
PROTOCOL_VERSION = "t12/1"
# Принятая версия файлов T12 (master после PR46). Изменение фиксации требует
# отдельного согласования и правки этих сумм в проверяемом коммите, а не подгонки.
ACCEPTED_T12_SHA256 = {
    "reports/freeze.json": "318712255d2b7074d1e03a75f6a2d2792cc61bcc119d9ff74313128149b452cd",
    "reports/freeze.md": "a4fc4a7525b31635a7a3fcb8bb73cd3cf50acad191c2f10c9086ec2a106a540d",
    "reports/freeze_checks.json": "46d17bc12c61580daf5e4e52bcdc2ac052ee9e2a98bd48c2d55875e471896197",
    "reports/freeze_review_fix.json": "e161f1a352a8732ae1f1f038bf95c1e7058451761fdef81b0c3cf34a827b2de7",
}
# Входы T13, которые обязаны быть закреплены в file_sha256 (плюс отчёты и конфиги моделей).
REQUIRED_PINNED = (
    "requirements.txt", "configs/nlp_baseline.json", "configs/transformer.json", "configs/data_prep.json",
    "configs/near_duplicate_decisions.csv", "reports/data_prep/audit.json", "reports/data_prep/split_ids.csv",
    "reports/data_prep/near_duplicate_candidates.csv", "src/__init__.py", "src/classifier.py",
    "src/train_baseline.py", "src/train_transformer.py", "src/prepare_data.py",
)
# Модули src, добавленные после frozen_commit самим оценщиком; остальные должны быть закреплены.
EVALUATOR_FILES = ("src/evaluate.py",)
# Объект исполняющегося модуля CPython, включая код вложенных функций.
_EVALUATOR_CODE = sys._getframe().f_code
LOCK_FILE = "t13-evaluate.lock"
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


class ConcurrentRunError(OSError):
    """Блокировка T13 для этого worktree недоступна."""


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def _class_ids(values: Sequence[Any], name: str) -> np.ndarray:
    """Целочисленные ID классов без приведения типа: 1.9, 2.0, True и "1" отклоняются."""
    items = list(values)
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in items):
        raise DataMismatchError(f"{name} must be integer class ids")
    invalid = sorted(set(items) - set(CLASS_IDS))
    if invalid:
        raise DataMismatchError(f"Unexpected class id(s) in {name}: {invalid}")
    return np.asarray(items, dtype=int)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Accuracy, macro-F1, метрики классов и матрица 3×3 (строки — истина, столбцы — ответ).

    Класс без предсказаний остаётся в отчёте с precision = 0 и входит в macro-F1.
    """
    truth = _class_ids(y_true, "labels")
    predicted = _class_ids(y_pred, "predictions")
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise DataMismatchError("Labels and predictions must be non-empty and of equal length")
    flat = base.validation_metrics(truth, predicted)
    return {
        "rows": int(len(truth)),
        "accuracy": flat["accuracy"],
        "macro_f1": flat["macro_f1"],
        "per_class": {name: {key: flat[f"{name}_{key}"] for key in ("precision", "recall", "f1", "support")}
                      for name in CLASS_NAMES},
        "predicted_counts": {name: int(np.count_nonzero(predicted == i)) for i, name in zip(CLASS_IDS, CLASS_NAMES)},
        "errors": {
            "total": int(np.count_nonzero(truth != predicted)),
            # Пропущенные записи истинного класса и ложные срабатывания предсказанного класса.
            "by_true_class": {name: int(np.count_nonzero((truth == i) & (predicted != i)))
                              for i, name in zip(CLASS_IDS, CLASS_NAMES)},
            "by_predicted_class": {name: int(np.count_nonzero((predicted == i) & (truth != i)))
                                   for i, name in zip(CLASS_IDS, CLASS_NAMES)},
        },
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


def _file_sha256(root: Path, rel: str) -> str | None:
    path = root / rel
    return base.sha256_file(path) if path.is_file() and not path.is_symlink() else None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_schema(freeze: Mapping[str, Any], root: Path) -> None:
    _require(freeze.get("schema") == FREEZE_SCHEMA, f"Unsupported freeze schema: {freeze.get('schema')!r}")
    _require(freeze.get("protocol_version") == PROTOCOL_VERSION,
             f"Unsupported T12 protocol version: {freeze.get('protocol_version')!r}")
    _require(freeze.get("protocol_path") == "reports/freeze.md", "T12 freeze must reference reports/freeze.md")
    _require(freeze.get("status") == "approved", "T12 freeze is not approved; the final test stays closed")
    _require(freeze.get("pending_approval") == [], "T12 freeze has pending approvals")
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
    decisions = freeze.get("confirmed_decisions")
    _require(isinstance(decisions, dict) and decisions.get("bot_model") == freeze["bot_model"],
             "T12 confirmed bot model differs from bot_model")
    _require(freeze.get("label_mapping") == base.EXPECTED_LABEL_MAPPING,
             "T12 label mapping must be negative=0, neutral=1, positive=2")


def _check_accepted_t12(freeze_sha256: str, root: Path) -> None:
    """Прочитанная фиксация и файлы T12 в репозитории совпадают с принятой версией.

    ``freeze_sha256`` — сумма именно тех байтов, которые разобраны (в том числе по --freeze),
    а не повторное чтение канонического файла.
    """
    _require(freeze_sha256 == ACCEPTED_T12_SHA256["reports/freeze.json"],
             "Selected T12 freeze differs from the accepted version of reports/freeze.json")
    changed = [rel for rel, digest in ACCEPTED_T12_SHA256.items() if _file_sha256(root, rel) != digest]
    _require(not changed, f"T12 freeze files differ from the accepted version: {', '.join(changed)}")


def _check_frozen_files(freeze: Mapping[str, Any], root: Path) -> None:
    """Все file_sha256 совпадают с файлами; входы T13 и все модули src, кроме оценщика, закреплены."""
    pinned = freeze.get("file_sha256")
    _require(isinstance(pinned, dict) and bool(pinned), "T12 freeze has no file_sha256")
    models = freeze.get("models") if isinstance(freeze.get("models"), dict) else {}
    required = list(REQUIRED_PINNED) + [entry.get(key) for entry in models.values() if isinstance(entry, dict)
                                        for key in ("report_path", "config_path")]
    missing = sorted({str(rel) for rel in required if rel not in pinned})
    _require(not missing, f"Required T13 inputs are not pinned in file_sha256: {', '.join(missing)}")
    changed = []
    for rel, digest in pinned.items():
        _require(_is_sha256(digest), f"file_sha256[{rel!r}] must be SHA-256")
        _repo_path(root, rel, f"file_sha256[{rel!r}]")
        if _file_sha256(root, rel) != digest:
            changed.append(rel)
    _require(not changed, f"Frozen files differ from T12 or are missing: {', '.join(sorted(changed))}")
    unpinned = [path.relative_to(root).as_posix() for path in sorted((root / "src").glob("*.py"))
                if path.relative_to(root).as_posix() not in pinned
                and path.relative_to(root).as_posix() not in EVALUATOR_FILES]
    _require(not unpinned, f"Source modules added after the freeze are not allowed: {', '.join(unpinned)}")


DATA_PREP_VERSION_KEYS = ("train_rows", "validation_rows", "train_ids_sha256", "validation_ids_sha256",
                          "train_parquet_sha256", "validation_parquet_sha256")


def _check_data_prep(freeze: Mapping[str, Any], root: Path) -> None:
    prep = freeze.get("data_prep")
    _require(isinstance(prep, dict), "T12 freeze has no data_prep section")
    config = base.read_json(root / "configs/nlp_baseline.json")
    for key, config_key in (("audit_sha256", "data_prep_audit_path"), ("split_ids_sha256", "data_prep_split_ids_path")):
        _require(_is_sha256(prep.get(key)), f"T12 freeze data_prep.{key} must be SHA-256")
        path = _repo_path(root, config[config_key], config_key)
        _require(path.is_file() and base.sha256_file(path) == prep[key], f"T03 {config_key} differs from T12 freeze")
    for key in DATA_PREP_VERSION_KEYS:
        valid = _is_positive_int(prep.get(key)) if key.endswith("_rows") else _is_sha256(prep.get(key))
        _require(valid, f"T12 freeze data_prep.{key} is missing or invalid")
    audit = base.load_t03_audit(base.resolve_path(root, config["data_prep_audit_path"]))
    for key in ("final_movie_id_overlap", "final_exact_text_overlap"):
        overlaps = audit.get(key)
        _require(isinstance(overlaps, dict) and bool(overlaps) and all(v == 0 for v in overlaps.values()),
                 f"T03 audit reports split overlap: {key}")


def _check_test_entry(freeze: Mapping[str, Any], root: Path) -> None:
    test = freeze.get("test")
    _require(isinstance(test, dict), "T12 freeze has no test section")
    _repo_path(root, test.get("path"), "test.path")
    for key in ("parquet_sha256", "ids_sha256", "source_parquet_sha256"):
        _require(_is_sha256(test.get(key)), f"T12 freeze test.{key} must be SHA-256")
    _require(_is_positive_int(test.get("rows")), "T12 freeze test.rows must be positive")
    config = base.read_json(root / "configs/nlp_baseline.json")
    audit = base.load_t03_audit(base.resolve_path(root, config["data_prep_audit_path"]))
    _require(audit["final_summary"]["test"]["rows"] == test["rows"], "T12 test size differs from T03 audit")
    # Ordered test IDs and labels from T03 split_ids.csv (no texts) must give the frozen ids_sha256.
    ids = pd.read_csv(base.resolve_path(root, config["data_prep_split_ids_path"]), dtype=str,
                      keep_default_na=False, usecols=lambda col: col in {"split", ID_COLUMN, LABEL_COLUMN})
    expected = ids.loc[ids["split"].str.lower() == "test"]
    _require(len(expected) == test["rows"] and not expected[ID_COLUMN].duplicated().any(),
             "T03 split_ids.csv test rows differ from T12 size or contain duplicates")
    _require(base.ids_sha256(expected, id_column=ID_COLUMN, label_column=LABEL_COLUMN) == test["ids_sha256"],
             "T03 split_ids.csv test IDs/labels differ from T12 ids_sha256")
    source = audit["source_files"]["test"]
    _require(source.get("sha256_after") == test["source_parquet_sha256"], "T12 test source differs from T03 audit")
    # The original Parquet is tracked in Git; only its bytes are hashed, it is not decoded.
    _require(_file_sha256(root, source["path"]) == test["source_parquet_sha256"],
             "Original test Parquet differs from T12 freeze")


def _differing_fields(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    return sorted(key for key, value in expected.items() if actual.get(key) != value)


def _check_baseline_report(entry: Mapping[str, Any], freeze: Mapping[str, Any], root: Path) -> None:
    prep, pinned = freeze["data_prep"], freeze["file_sha256"]
    _require(_is_sha256(entry.get("model_sha256")) and _is_positive_int(entry.get("bytes")),
             "T12 freeze models.baseline needs model_sha256 and bytes")
    report = base.read_json(base.resolve_path(root, entry["report_path"]))
    expected = {"bundle_path": entry["model_path"], "bundle_sha256": entry["model_sha256"],
                "config_path": entry["config_path"], "config_sha256": pinned[entry["config_path"]],
                "final_test_used": False, "selection_metric": "validation_macro_f1",
                "t03_audit_sha256": prep["audit_sha256"], "t03_split_ids_sha256": prep["split_ids_sha256"],
                **{key: prep[key] for key in DATA_PREP_VERSION_KEYS}}
    differing = _differing_fields(expected, report)
    _require(not differing, f"baseline run report differs from T12 freeze: {', '.join(differing)}")
    independence = report.get("split_independence")
    _require(isinstance(independence, dict) and bool(independence) and not any(independence.values()),
             "baseline run report shows train/validation overlap")
    config = base.read_json(base.resolve_path(root, entry["config_path"]))
    processing = {"manual_normalization": entry.get("manual_normalization"), "truncation": entry.get("truncation")}
    _require(config.get("text_processing") == processing and config.get("label_mapping") == base.EXPECTED_LABEL_MAPPING,
             "baseline configuration differs from T12 text processing or label mapping")


def _check_rubert_report(entry: Mapping[str, Any], freeze: Mapping[str, Any], root: Path) -> None:
    prep = freeze["data_prep"]
    files = entry.get("bundle_files_sha256")
    _require(isinstance(files, dict) and len(files) == 7 and all(_is_sha256(v) for v in files.values()),
             "T12 freeze models.rubert needs SHA-256 of all seven bundle files")
    report = base.read_json(base.resolve_path(root, entry["report_path"]))
    _require(report.get("bundle_files_sha256") == files, "rubert run report bundle files differ from T12 freeze")
    _require(report.get("final_test_used") is False and report.get("selection_metric") == "validation_macro_f1",
             "rubert run report must use validation selection without final test")
    version = report.get("data_version") or {}
    checksums = version.get("checksums") or {}
    expected = {"final_test_used": False, "train_ids_equal_baseline": True,
                **{key: prep[key] for key in ("train_rows", "validation_rows", "train_ids_sha256", "validation_ids_sha256")}}
    differing = _differing_fields(expected, version) + _differing_fields(
        {"train_path": prep["train_parquet_sha256"], "validation_path": prep["validation_parquet_sha256"],
         "data_prep_audit_path": prep["audit_sha256"], "data_prep_split_ids_path": prep["split_ids_sha256"]}, checksums)
    _require(not differing, f"rubert data version differs from T12 freeze: {', '.join(differing)}")
    independence = version.get("split_independence")
    _require(isinstance(independence, dict) and bool(independence) and not any(independence.values()),
             "rubert run report shows train/validation overlap")
    config = base.read_json(base.resolve_path(root, entry["config_path"]))
    expected = {"model_name": entry.get("base_model"), "revision": entry.get("base_revision"),
                "max_length": entry.get("max_length_tokens"), "truncation_side": entry.get("truncation_side"),
                "label_mapping": base.EXPECTED_LABEL_MAPPING}
    differing = _differing_fields(expected, config)
    _require(not differing, f"rubert configuration differs from T12 freeze: {', '.join(differing)}")


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
        _require(freeze["file_sha256"].get(entry["report_path"]) == entry["report_sha256"],
                 f"{backend} run report SHA-256 is inconsistent within T12 freeze")
    _check_baseline_report(models["baseline"], freeze, root)
    _check_rubert_report(models["rubert"], freeze, root)


def _check_inference(freeze: Mapping[str, Any], root: Path) -> None:
    inference = freeze.get("inference")
    _require(isinstance(inference, dict), "T12 freeze has no inference section")
    _require(_is_positive_int(inference.get("batch_size")), "T12 freeze inference.batch_size must be positive")
    _require(inference.get("device") == "cpu", "Only CPU inference is supported by T13")
    _require(_is_positive_int(inference.get("num_threads")), "T12 freeze inference.num_threads must be positive")
    protocol = freeze.get("text_protocol")
    _require(isinstance(protocol, dict), "T12 freeze has no text_protocol section")
    expected = {"models": list(BACKENDS), "runs": 1, "max_text_length": None, "labels": CLASS_IDS, "zero_division": 0}
    differing = [key for key, value in expected.items() if key not in protocol or protocol[key] != value]
    _require(not differing, f"T12 text protocol is not supported by this evaluator: {', '.join(differing)}")


def _check_git(freeze: Mapping[str, Any], root: Path, *, expected_commit: str | None = None) -> str:
    state = git_state(root, freeze["frozen_commit"], expected_commit=expected_commit)
    _require(state["commit"] is not None, "Git commit of the evaluation code is unknown")
    _require(state["frozen_commit_is_ancestor"], "Evaluated code does not descend from the T12 frozen commit")
    _require(not state["tracked_changes"], "Tracked files have uncommitted changes; commit before the final run")
    _require(state["evaluator_in_head"], f"{EVALUATOR_FILES[0]} is not committed in Git HEAD; commit it first")
    _require(state["evaluator_matches_head"],
             f"Running {EVALUATOR_FILES[0]} differs from the version committed in Git HEAD")
    return state["commit"]


def _sysctl(name: str) -> str | None:
    try:
        proc = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def machine_info() -> dict[str, Any]:
    """Характеристики машины в полях freeze.json: environment.machine."""
    system = platform.system()
    info: dict[str, Any] = {
        "os": "macOS" if system == "Darwin" else system,
        "os_version": platform.mac_ver()[0] if system == "Darwin" else platform.release(),
        "architecture": platform.machine(), "cpu": platform.processor() or "unknown",
        "physical_cores": None, "logical_cores": os.cpu_count(), "ram_bytes": None,
        "python": platform.python_version(),
    }
    if system == "Darwin":
        for key, name in (("cpu", "machdep.cpu.brand_string"), ("physical_cores", "hw.physicalcpu"),
                          ("logical_cores", "hw.logicalcpu"), ("ram_bytes", "hw.memsize")):
            value = _sysctl(name)
            if value is not None:
                info[key] = value if key == "cpu" else int(value)
    else:
        try:
            import psutil
            info.update(physical_cores=psutil.cpu_count(logical=False), ram_bytes=psutil.virtual_memory().total)
        except ImportError:
            pass
    return info


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_environment(freeze: Mapping[str, Any], root: Path) -> None:
    """Измерения выполняются только в зафиксированной среде T12; другая машина не подменяет её."""
    variables = freeze.get("runtime_environment_variables")
    environment = freeze.get("environment") or {}
    machine, packages = environment.get("machine"), environment.get("packages")
    for name, value in (("runtime_environment_variables", variables), ("environment.machine", machine),
                        ("environment.packages", packages)):
        _require(isinstance(value, dict) and bool(value), f"T12 freeze has no {name}")
    problems = []
    wrong = sorted(name for name, value in variables.items() if os.environ.get(name) != value)
    if wrong:
        problems.append(f"runtime environment variables differ from T12: {', '.join(wrong)}")
    differing = _differing_fields(machine, machine_info())
    if differing:
        problems.append(f"machine differs from the T12 measurement machine: {', '.join(differing)}")
    differing = sorted(name for name, version in packages.items() if installed_version(name) != version)
    if differing:
        problems.append(f"installed packages differ from T12: {', '.join(differing)}")
    _require(not problems, "; ".join(problems))


# Статические проверки разобранной фиксации (после _check_accepted_t12 по её байтам):
# модели и test здесь не загружаются и не декодируются.
FREEZE_CHECKS: list[Callable[[Mapping[str, Any], Path], None]] = [
    _check_schema, _check_frozen_files, _check_data_prep, _check_test_entry,
    _check_models, _check_inference, _check_environment,
]


def _git_blob(root: Path, spec: str) -> bytes | None:
    try:
        proc = subprocess.run(["git", "cat-file", "blob", spec], cwd=root, capture_output=True, check=False)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_state(root: Path, frozen_commit: str, *, expected_commit: str | None = None) -> dict[str, Any]:
    """Сверить происхождение, индекс, рабочую копию и загруженный код с одним SHA."""
    commit = base.git_commit(root) if expected_commit is None else expected_commit
    _require(isinstance(commit, str) and bool(_COMMIT.fullmatch(commit)), "Git commit of the evaluation code is unknown")
    ancestor = base._git(root, "merge-base", "--is-ancestor", frozen_commit, commit)
    index = base._git(root, "diff", "--quiet", "--cached", commit, "--")
    worktree = base._git(root, "diff", "--quiet", "--")
    rel = EVALUATOR_FILES[0]
    committed = _git_blob(root, f"{commit}:{rel}")
    path = root / rel
    matches = False
    if committed is not None and path.is_file() and not path.is_symlink() and path.read_bytes() == committed:
        try:
            matches = compile(committed, _EVALUATOR_CODE.co_filename, "exec", dont_inherit=True) == _EVALUATOR_CODE
        except (SyntaxError, ValueError) as exc:
            raise FreezeError("Cannot compile the committed evaluator") from exc
    _require(base.git_commit(root) == commit, "Git HEAD changed; evaluator commit cannot be verified")
    return {
        "commit": commit,
        "frozen_commit_is_ancestor": ancestor is not None and ancestor.returncode == 0,
        "tracked_changes": any(diff is None or diff.returncode != 0 for diff in (index, worktree)),
        "evaluator_in_head": committed is not None,
        "evaluator_matches_head": matches,
    }


def read_freeze(path: Path, root: Path = ROOT) -> tuple[dict[str, Any], str]:
    """Прочитать фиксацию один раз: вернуть разобранные байты и их SHA-256."""
    try:
        data = path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
        raise FreezeError(f"T12 freeze not found: {base.display_path(path, root)}; "
                          "the final test stays closed") from exc
    except OSError as exc:
        raise FreezeError(f"T12 freeze cannot be read: {exc}") from exc
    try:
        # utf-8-sig, как base.read_json.
        freeze = json.loads(data.decode("utf-8-sig"))
    except ValueError as exc:
        raise FreezeError(f"T12 freeze is not valid JSON: {exc}") from exc
    _require(isinstance(freeze, dict), "T12 freeze must be a JSON object")
    return freeze, hashlib.sha256(data).hexdigest()


def load_freeze(path: Path, root: Path = ROOT) -> tuple[dict[str, Any], str, str]:
    """Вернуть проверенную фиксацию T12, её SHA-256 и проверенный Git commit."""
    freeze, digest = read_freeze(path, root)
    _check_accepted_t12(digest, root)
    for check in FREEZE_CHECKS:
        try:
            check(freeze, root)
        except (KeyError, TypeError, OSError) as exc:
            raise FreezeError(f"T12 freeze check {check.__name__} failed: {type(exc).__name__}: {exc}") from exc
    return freeze, digest, _check_git(freeze, root)


@contextlib.contextmanager
def final_run_lock(root: Path = ROOT) -> Iterator[None]:
    """Общий flock для обоих режимов. Служебный файл остаётся в Git-каталоге worktree."""
    try:
        import fcntl
    except ImportError as exc:
        raise ConcurrentRunError("Exclusive final-run lock requires fcntl (Linux or macOS)") from exc
    git_dir = base._git(root, "rev-parse", "--absolute-git-dir")
    if git_dir is None or git_dir.returncode != 0:
        raise ConcurrentRunError("Cannot locate the Git worktree directory for the T13 lock")
    lock_path = Path(git_dir.stdout.removesuffix("\n")) / LOCK_FILE
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise ConcurrentRunError(f"Cannot open the final-run lock {LOCK_FILE}: {exc.strerror}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                raise ConcurrentRunError("Another final evaluation or readiness check is running in this worktree; "
                                         "refusing before loading models or reading the test") from exc
            raise ConcurrentRunError(f"Cannot lock {LOCK_FILE}: {exc}") from exc
        yield
    finally:
        os.close(fd)


def check_model_files(freeze: Mapping[str, Any], root: Path = ROOT) -> None:
    """Файлы комплектов совпадают с суммами фиксации до загрузки через src.classifier."""
    baseline = freeze["models"]["baseline"]
    path = base.resolve_path(root, baseline["model_path"])
    if not path.is_file() or path.is_symlink():
        raise c.ModelUnavailableError(f"Frozen baseline bundle not found: {baseline['model_path']}")
    if path.stat().st_size != baseline["bytes"] or base.sha256_file(path) != baseline["model_sha256"]:
        raise FreezeError("Baseline bundle differs from T12 freeze")
    rubert = freeze["models"]["rubert"]
    directory = base.resolve_path(root, rubert["model_path"])
    if not directory.is_dir():
        raise c.ModelUnavailableError(f"Frozen RuBERT bundle not found: {rubert['model_path']}")
    expected = rubert["bundle_files_sha256"]
    if {p.name for p in directory.iterdir()} != set(expected):
        raise FreezeError("RuBERT bundle files differ from T12 freeze")
    changed = sorted(name for name, digest in expected.items() if _file_sha256(directory, name) != digest)
    if changed:
        raise FreezeError(f"RuBERT bundle checksum differs from T12 freeze: {', '.join(changed)}")


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


def frozen_test_path(freeze: Mapping[str, Any], root: Path = ROOT) -> Path:
    """Путь подготовленного test: обычный файл внутри репозитория, без символьных ссылок,
    с SHA-256 из фиксации. Одна и та же проверка в --check и перед декодированием."""
    rel = freeze["test"]["path"]
    current = root
    for part in Path(rel).parts:
        current = current / part
        if current.is_symlink():
            raise DataMismatchError(f"Prepared test path must not contain symbolic links: {rel}")
    path = base.resolve_path(root, rel)
    if not path.is_file():
        raise DataMismatchError(f"Prepared test not found: {rel}; restore it with tools/restore_frozen_test.py")
    if base.sha256_file(path) != freeze["test"]["parquet_sha256"]:
        raise DataMismatchError("Test Parquet differs from the T12 freeze")
    return path


def load_frozen_test(freeze: Mapping[str, Any], root: Path = ROOT) -> pd.DataFrame:
    """Открыть test только после совпадения SHA-256 файла с фиксацией T12."""
    entry = freeze["test"]
    path = frozen_test_path(freeze, root)
    frame = base.load_prepared_split(path, split_name="test", text_column=TEXT_COLUMN,
                                     label_column=LABEL_COLUMN, id_column=ID_COLUMN)
    if len(frame) != entry["rows"]:
        raise DataMismatchError(f"Test has {len(frame)} rows; T12 freeze records {entry['rows']}")
    if base.ids_sha256(frame, id_column=ID_COLUMN, label_column=LABEL_COLUMN) != entry["ids_sha256"]:
        raise DataMismatchError("Test IDs/labels differ from the T12 freeze")
    config = base.read_json(root / "configs/nlp_baseline.json")
    audit = base.load_t03_audit(base.resolve_path(root, config["data_prep_audit_path"]))
    summary = audit["final_summary"]["test"]
    counts = {name: int((frame[LABEL_COLUMN] == i).sum()) for i, name in zip(CLASS_IDS, CLASS_NAMES)}
    if summary["rows"] != len(frame) or any(summary[name] != count for name, count in counts.items()):
        raise DataMismatchError("Test size or class counts differ from T03 audit")
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
        differing = [key for key in SHARED_VERSION_KEYS[:4] if version[key] != prep[key]]
        if differing:
            raise DataMismatchError(f"{backend} train/validation version differs from T12: {', '.join(differing)}")
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
        "predictions": _class_ids([r["label_id"] for r in results], "Model predictions").tolist(),
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
def environment(variables: Sequence[str] = ()) -> dict[str, Any]:
    """Фактическая среда запуска: машина, библиотеки, переменные протокола и потоки torch."""
    info: dict[str, Any] = {"machine": machine_info(), "libraries": base.library_versions(),
                            "runtime_environment_variables": {name: os.environ.get(name) for name in variables}}
    torch = sys.modules.get("torch")
    if torch is not None:
        info.update(torch=torch.__version__, torch_num_threads=torch.get_num_threads())
    return info


TABLE_COLUMNS = (["model", "bot_model", "rows", "accuracy", "macro_f1"]
                 + [f"{name}_{key}" for name in CLASS_NAMES for key in ("precision", "recall", "f1", "support")]
                 + ["errors_total", "load_seconds", "warmup_seconds", "inference_seconds", "ms_per_text",
                    "texts_per_second", "batch_size"])
OUTPUT_FILES = ("final_metrics.json", "final_table.csv", "final_report.md", "predictions.csv",
                *(f"confusion_matrix_{b}.{ext}" for b in BACKENDS for ext in ("csv", "svg")))


def table_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for backend in BACKENDS:
        entry = report["models"][backend]
        metrics, timing = entry["metrics"], entry["timing"]
        row = {"model": backend, "bot_model": backend == report["bot_model"], "rows": metrics["rows"],
               "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        for name in CLASS_NAMES:
            row.update({f"{name}_{key}": value for key, value in metrics["per_class"][name].items()})
        row["errors_total"] = metrics["errors"]["total"]
        row.update({key: timing[key] for key in ("load_seconds", "warmup_seconds", "inference_seconds",
                                                 "ms_per_text", "texts_per_second", "batch_size")})
        rows.append(row)
    return rows


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


# Последовательная шкала одного оттенка (светлый → тёмный); точный ноль — нейтральный серый.
_SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
               "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
_ZERO_FILL, _INK, _MUTED, _ON_DARK, _SURFACE = "#f0efec", "#1f1f1d", "#5f5e5a", "#ffffff", "#ffffff"


def confusion_matrix_svg(matrix: Sequence[Sequence[int]], title: str) -> str:
    """Статическая SVG-карта матрицы ошибок: цвет — доля строки (recall по истинному классу).

    Каждая ячейка подписана числом и долей, поэтому значение не передаётся только цветом.
    """
    cell, gap, left, top = 110, 2, 150, 92
    width, height = left + 3 * cell + 24, top + 3 * cell + 78
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
             'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
             f'<title id="title">{escape(title)}</title>',
             '<desc id="desc">Строки — истинный класс, столбцы — предсказанный. Цвет — доля строки.</desc>',
             f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>',
             f'<text x="16" y="28" font-size="16" font-weight="600" fill="{_INK}">{escape(title)}</text>',
             f'<text x="{left + 1.5 * cell}" y="{top - 34}" font-size="13" fill="{_MUTED}" '
             'text-anchor="middle">Предсказанный класс</text>',
             f'<text x="22" y="{top + 1.5 * cell}" font-size="13" fill="{_MUTED}" text-anchor="middle" '
             f'transform="rotate(-90 22 {top + 1.5 * cell})">Истинный класс</text>']
    for j, name in enumerate(CLASS_NAMES):
        parts.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 10}" font-size="13" fill="{_INK}" '
                     f'text-anchor="middle">{name}</text>')
    for i, (name, row) in enumerate(zip(CLASS_NAMES, matrix)):
        total = sum(row)
        y = top + i * cell
        parts.append(f'<text x="{left - 10}" y="{y + cell / 2 + 4}" font-size="13" fill="{_INK}" '
                     f'text-anchor="end">{name}</text>')
        for j, value in enumerate(row):
            share = value / total if total else 0.0
            fill = _ZERO_FILL if value == 0 else _SEQUENTIAL[round(share * (len(_SEQUENTIAL) - 1))]
            ink = _ON_DARK if value and share > 0.5 else _INK
            x = left + j * cell
            parts += [f'<g><title>истина {name}, ответ {CLASS_NAMES[j]}: {value} ({share:.1%} строки)</title>',
                      f'<rect x="{x + gap / 2}" y="{y + gap / 2}" width="{cell - gap}" height="{cell - gap}" '
                      f'rx="4" fill="{fill}"/>',
                      f'<text x="{x + cell / 2}" y="{y + cell / 2 - 2}" font-size="18" font-weight="600" '
                      f'fill="{ink}" text-anchor="middle">{value}</text>',
                      f'<text x="{x + cell / 2}" y="{y + cell / 2 + 18}" font-size="12" fill="{ink}" '
                      f'text-anchor="middle">{share:.1%}</text></g>']
    parts.append(f'<text x="16" y="{height - 20}" font-size="12" fill="{_MUTED}">Число записей и доля строки; '
                 'серый — 0. Таблица: соседний CSV.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _markdown(report: Mapping[str, Any]) -> str:
    run = report["run"]
    command_label = ("Шаблон команды (подставьте локальные пути)" if run["required_local_paths"] else "Команда")
    lines = ["# T13: финальное сравнение на текстовом test", "",
             f"- Commit оценщика: `{run['git_commit']}`; основа T12: `{run['frozen_commit']}`",
             f"- Фиксация: `{run['freeze_path']}` (`{run['freeze_sha256']}`), протокол {run['protocol_version']}",
             f"- Test: {report['test']['rows']} строк, Parquet `{report['test']['parquet_sha256']}`",
             f"- {command_label}: `{run['command']}`",
             f"- Модель бота выбрана в T12 по validation: {report['bot_model']}; результат test её не меняет", "",
             "| Модель | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | Ошибок | мс/текст |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table_rows(report):
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                     f"{row['negative_f1']:.4f} | {row['neutral_f1']:.4f} | {row['positive_f1']:.4f} | "
                     f"{row['errors_total']} | {row['ms_per_text']:.3f} |")
    for backend in BACKENDS:
        matrix = report["models"][backend]["metrics"]["confusion_matrix"]
        lines += ["", f"## Confusion matrix: {backend}", "", f"![{backend}](confusion_matrix_{backend}.svg)", "",
                  "| истина \\ ответ | " + " | ".join(CLASS_NAMES) + " |", "|---|---:|---:|---:|"]
        lines += [f"| {name} | " + " | ".join(str(v) for v in values) + " |" for name, values in zip(CLASS_NAMES, matrix)]
    return "\n".join(lines) + "\n"


def check_out_dir(out_dir: Path) -> None:
    """Существующие результаты не перезаписываются: каталог не должен существовать."""
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Evaluation output already exists; refusing to overwrite: {out_dir.name}")


# Флаги атомарного переименования без замещения (Linux renameat2, macOS renamex_np).
_AT_FDCWD, _RENAME_NOREPLACE, _RENAME_EXCL = -100, 1, 0x4
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}


def _rename_noreplace(source: Path, target: Path) -> None:
    """Атомарно переименовать ``source`` в ``target``; существующий target, даже пустой каталог,
    не замещается. Без поддержки операции — ошибка, без небезопасного fallback на os.rename."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(target), _RENAME_NOREPLACE)
    elif sys.platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(target), _RENAME_EXCL)
    else:
        function = None
    if function is None:
        raise OSError(errno.ENOTSUP, f"Atomic no-replace rename is unavailable on {sys.platform}; refusing to publish")
    function.argtypes, function.restype = argtypes, ctypes.c_int
    if function(*args) == 0:
        return
    code = ctypes.get_errno()
    if code in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(code, "Evaluation output already exists; refusing to overwrite", target.name)
    if code in _UNSUPPORTED:
        raise OSError(code, "File system does not support atomic no-replace rename; refusing to publish",
                      target.name)
    raise OSError(code, os.strerror(code), target.name)


def check_output_destination(out_dir: Path) -> None:
    """Назначение результатов пригодно до затратного инференса.

    Итоговый каталог не существует, родитель — существующий каталог, в нём можно создать
    временный каталог, а переименование без замещения работает на этой файловой системе
    (пробные каталоги удаляются). Итоговый каталог при проверке не создаётся.
    """
    check_out_dir(out_dir)
    parent = out_dir.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"Output parent is not an existing directory: {parent.name}")
    probe = Path(tempfile.mkdtemp(dir=parent, prefix=f".{out_dir.name}.", suffix=".probe"))
    try:
        (probe / "first").mkdir()
        (probe / "second").mkdir()
        _rename_noreplace(probe / "first", probe / "published")
        try:
            _rename_noreplace(probe / "second", probe / "published")
        except FileExistsError:
            pass
        else:
            raise OSError("No-replace rename replaced an existing directory; refusing to publish")
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def write_reports(report: dict[str, Any], runs: Mapping[str, Mapping[str, Any]], out_dir: Path,
                  *, root: Path = ROOT) -> list[Path]:
    """Записать отчёты во временный каталог рядом и опубликовать его одним переименованием.

    Тексты отзывов не записываются. При обработанной ошибке временный каталог удаляется,
    итоговый каталог не появляется; существующий каталог (в том числе пустой, созданный
    конкурентно) не заменяется: публикация — переименование без замещения.
    """
    check_out_dir(out_dir)
    staging = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}.", suffix=".tmp"))
    try:
        _write_csv(staging / "final_table.csv", TABLE_COLUMNS,
                   [[row[col] for col in TABLE_COLUMNS] for row in table_rows(report)])
        reference = runs[BACKENDS[0]]
        _write_csv(staging / "predictions.csv", [ID_COLUMN, LABEL_COLUMN, *(f"{b}_label_id" for b in BACKENDS)],
                   list(zip(reference["record_ids"], reference["labels"], *(runs[b]["predictions"] for b in BACKENDS))))
        for backend in BACKENDS:
            matrix = report["models"][backend]["metrics"]["confusion_matrix"]
            _write_csv(staging / f"confusion_matrix_{backend}.csv", ["true\\predicted", *CLASS_NAMES],
                       [[name, *values] for name, values in zip(CLASS_NAMES, matrix)])
            (staging / f"confusion_matrix_{backend}.svg").write_text(
                confusion_matrix_svg(matrix, f"Confusion matrix: {backend}, test {report['test']['rows']}"),
                encoding="utf-8")
        (staging / "final_report.md").write_text(_markdown(report), encoding="utf-8")
        report["outputs_sha256"] = {name: base.sha256_file(staging / name)
                                    for name in OUTPUT_FILES if name != "final_metrics.json"}
        base.write_json(staging / "final_metrics.json", report)
        if sorted(p.name for p in staging.iterdir()) != sorted(OUTPUT_FILES):
            raise OSError("Incomplete evaluation output")
        _check_git(report["run"], root, expected_commit=report["run"]["git_commit"])
        _rename_noreplace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return [out_dir / name for name in OUTPUT_FILES]


# --------------------------------------------------------------------------- #
# Финальный запуск
# --------------------------------------------------------------------------- #
def _set_torch_threads(threads: int) -> int:
    try:
        import torch
    except ImportError as exc:
        raise c.ModelUnavailableError(f"torch is not installed; RuBERT cannot be loaded ({exc})") from exc
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise FreezeError(f"torch uses {torch.get_num_threads()} threads; T12 requires {threads}")
    return threads


def load_models(freeze: Mapping[str, Any], *, root: Path = ROOT, factory: Callable[..., Any] = c.create_classifier,
                clock: Callable[[], float] = time.perf_counter) -> tuple[dict, dict, dict]:
    """Проверить и загрузить оба комплекта через src.classifier, baseline первым; test не читается."""
    check_model_files(freeze, root)
    models, load_seconds, versions = {}, {}, {}
    for backend in BACKENDS:
        entry = freeze["models"][backend]
        if backend == "rubert":
            _set_torch_threads(freeze["inference"]["num_threads"])
        models[backend] = factory(backend, base.resolve_path(root, entry["model_path"]),
                                  report_path=base.resolve_path(root, entry["report_path"]))
        load_seconds[backend] = timed_load(models[backend], clock)
        versions[backend] = model_data_version(backend, models[backend].data_version)
    check_model_versions(versions, freeze)
    return models, load_seconds, versions


def run_command(freeze_path: Path, out_dir: Path, root: Path) -> dict[str, Any]:
    """Аргументы для запуска из корня репозитория. Внешние пути указывает владелец."""
    argv = ["python", "-m", "src.evaluate", "--run-final"]
    required = []
    paths = {}
    for option, path in (("--freeze", freeze_path), ("--out-dir", out_dir)):
        try:
            value = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            value = "<LOCAL_FREEZE_PATH>" if option == "--freeze" else "<LOCAL_OUT_DIR>"
            required.append(option)
        if value.startswith("-"):
            value = "./" + value
        argv.extend([option, value])
        paths[option] = value
    return {"argv": argv, "command": shlex.join(argv), "required_local_paths": required,
            "freeze_path": paths["--freeze"], "out_dir": paths["--out-dir"]}


def evaluate(freeze_path: Path, out_dir: Path, *, root: Path = ROOT,
             factory: Callable[..., Any] = c.create_classifier,
             clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Финальная оценка обеих моделей под исключительной блокировкой worktree.

    Порядок: блокировка → фиксация → назначение результатов → модели → test → предсказания →
    запись временных отчётов → повторная проверка git → публикация. Test декодируется только после проверки фиксации,
    обоих комплектов и их версий данных; блокировка удерживается до публикации и очистки.
    """
    with final_run_lock(root):
        return _evaluate_locked(freeze_path, out_dir, root=root, factory=factory, clock=clock)


def _evaluate_locked(freeze_path: Path, out_dir: Path, *, root: Path, factory: Callable[..., Any],
                     clock: Callable[[], float]) -> dict[str, Any]:
    freeze, freeze_sha256, commit = load_freeze(freeze_path, root)
    check_output_destination(out_dir)
    inference = freeze["inference"]
    models, load_seconds, versions = load_models(freeze, root=root, factory=factory, clock=clock)

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

    command = run_command(freeze_path, out_dir, root)
    report = {
        "task": "T13",
        "final_test_used": True,
        "bot_model": freeze["bot_model"],
        "bot_model_selected_by": freeze["bot_model_selected_by"],
        "bot_model_changed_by_test": False,
        "labels": dict(zip(CLASS_NAMES, CLASS_IDS)),
        "run": {
            **command,
            "git_commit": commit,
            "frozen_commit": freeze["frozen_commit"],
            "freeze_sha256": freeze_sha256,
            "freeze_schema": freeze["schema"],
            "protocol_version": freeze["protocol_version"],
            "approved_by": freeze["approved_by"],
            "approved_at": freeze["approved_at"],
            "runs": freeze["text_protocol"]["runs"],
        },
        "test": {**test_version, "path": freeze["test"]["path"],
                 "source_parquet_sha256": freeze["test"]["source_parquet_sha256"],
                 "split_ids_sha256": freeze["data_prep"]["split_ids_sha256"]},
        "data_version": versions[BACKENDS[0]],
        "inference_protocol": {
            "device": inference["device"], "batch_size": inference["batch_size"],
            "num_threads": inference["num_threads"], "max_text_length": None,
            "order": list(BACKENDS),
            "warmup": f"one predict_batch call on {len(WARMUP_TEXTS)} fixed non-test texts",
            "load": "bundle verification and loading via src.classifier, before the test is read",
            "inference": "whole test in original order, predict_batch per batch_size chunk; time.perf_counter",
        },
        "environment": environment(tuple(freeze["runtime_environment_variables"])),
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
        "outputs": [str(Path(command["out_dir"]) / name) for name in OUTPUT_FILES],
    }
    write_reports(report, runs, out_dir, root=root)
    return report


def check_only(freeze_path: Path, root: Path = ROOT, *, out_dir: Path | None = None,
               factory: Callable[..., Any] = c.create_classifier) -> dict[str, Any]:
    """Готовность к финальному запуску без декодирования test и без предсказаний на нём.

    Те же проверки, что у --run-final до инференса, включая выбранный --out-dir.
    Проверки фиксации, git, среды, назначения и моделей выполняются независимо, чтобы
    показать все блокировки сразу; без утверждённой фиксации остальные шаги не выполняются.
    Test проверяется только по пути и байтовой сумме файла.
    """
    out_dir = out_dir if out_dir is not None else root / "reports" / "evaluation"
    freeze: dict[str, Any] = {}
    freeze_sha256 = None
    checks: list[dict[str, str]] = []

    def stage(name: str, action: Callable[[], Any]) -> bool:
        try:
            action()
        except (c.ClassifierError, OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            checks.append({"check": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            return False
        checks.append({"check": name, "status": "passed"})
        return True

    static = [("freeze_protocol", (lambda f, r: _check_accepted_t12(freeze_sha256, r), _check_schema)),
              ("frozen_files_and_reports", (_check_frozen_files, _check_data_prep, _check_test_entry,
                                            _check_models, _check_inference)),
              ("git", (_check_git,)), ("environment", (_check_environment,)),
              ("output_dir", (lambda f, r: check_output_destination(out_dir),))]
    with contextlib.ExitStack() as stack:
        if stage("exclusive_run", lambda: stack.enter_context(final_run_lock(root))):
            freeze, freeze_sha256 = read_freeze(freeze_path, root)
            for index, (name, functions) in enumerate(static):
                if not stage(name, lambda: [check(freeze, root) for check in functions]) and index < 2:
                    break
            else:
                stage("models", lambda: load_models(freeze, root=root, factory=factory))
                stage("test_file", lambda: frozen_test_path(freeze, root))
    ready = all(item["status"] == "passed" for item in checks)
    return {"status": "ready" if ready else "not_ready", "freeze_path": base.display_path(freeze_path, root),
            "out_dir": base.display_path(out_dir, root),
            "freeze_sha256": freeze_sha256, "bot_model": freeze.get("bot_model"),
            "test_rows": (freeze.get("test") or {}).get("rows"), "checks": checks,
            "final_test_decoded": False, "final_inference_run": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="verify freeze, frozen files, git, environment, models and test bytes; no test inference")
    mode.add_argument("--run-final", action="store_true",
                      help="run the approved final evaluation on the test (only when it is authorized)")
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH,
                        help="approved T12 freeze (default: reports/freeze.json)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="new output directory, must not exist (default: reports/evaluation)")
    args = parser.parse_args(argv)
    freeze_path = args.freeze if args.freeze.is_absolute() else Path.cwd() / args.freeze
    out_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    try:
        if args.check:
            result = check_only(freeze_path, out_dir=out_dir)
        else:
            report = evaluate(freeze_path, out_dir)
            result = {key: report[key] for key in ("bot_model", "test", "outputs")}
    except (c.ClassifierError, c.InvalidTextError, OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not args.check or result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
