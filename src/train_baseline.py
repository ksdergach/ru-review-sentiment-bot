"""T04: word-level TF-IDF + Logistic Regression baseline (три класса тональности).

Методология (см. README_T04.md):
    prepared train -> TF-IDF.fit(train) -> LogisticRegression.fit(train)
    -> несколько заранее заданных C -> оценка каждого на validation
    -> победитель по validation macro-F1 -> save -> load -> проверка ответов.

Финальный test в T04 не читается и не используется: его нет ни в конфигурации,
ни в коде загрузки данных.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

EXPECTED_LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["negative", "neutral", "positive"]
BUNDLE_FORMAT_VERSION = 3
MODEL_TYPE = "tfidf_logistic_regression"

# Схема конфигурации строгая: неизвестный ключ на любом уровне — ошибка.
# Поэтому test нельзя «спрятать» под другим именем (final_test, evaluation.test, ...).
TOP_REQUIRED = {
    "seed",
    "train_path",
    "validation_path",
    "data_prep_audit_path",
    "data_prep_split_ids_path",
    "report_dir",
    "model_dir",
    "bundle_name",
    "label_mapping",
    "text_column",
    "label_column",
    "id_column",
    "text_processing",
    "tfidf",
    "logistic_regression",
}
TOP_OPTIONAL = {"train_ids_path"}
TEXT_PROCESSING_KEYS = {"manual_normalization", "truncation"}
TFIDF_REQUIRED = {
    "analyzer",
    "lowercase",
    "ngram_range",
    "min_df",
    "max_df",
    "max_features",
    "sublinear_tf",
    "norm",
    "token_pattern",
}
TFIDF_OPTIONAL = {"use_idf", "smooth_idf"}
LR_REQUIRED = {"C_values", "solver", "max_iter", "tol", "class_weight_mode"}
# Только решатели, поддерживающие multinomial-классификацию на разреженных данных
# (liblinear в актуальном scikit-learn не поддерживает >= 3 классов).
ALLOWED_SOLVERS = {"lbfgs", "newton-cg", "sag", "saga"}
ALLOWED_CLASS_WEIGHT_MODES = {"none", "balanced_train"}

# Поля, которые T03 гарантирует в prepared train/validation (README_T03.md).
T03_EXTRA_COLUMNS = ("label_name", "review_language", "movie_id")
INPUT_SPLITS_FOR_T04 = ("train", "validation")

TIMING_KEYS = {"fit_seconds", "validation_inference_seconds"}
_TEST_TOKENS = {"test", "tests", "testing"}
# U+200B..U+200D, U+2060, U+FEFF — как в T03 (README_T03.md)
_INVISIBLE_CHARS = {cp: None for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)}


# --------------------------------------------------------------------------- #
# Общие утилиты
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    # utf-8-sig: файлы, сохранённые в Windows-редакторах с BOM, читаются без ошибки.
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def display_path(path: Path, root: Path) -> str:
    """Путь для отчётов: относительный POSIX; абсолютные пути в отчёты не попадают."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": package_version("scipy"),
        "joblib": joblib.__version__,
        "pyarrow": package_version("pyarrow"),
    }


def _git(project_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None


def git_commit(project_root: Path) -> str | None:
    proc = _git(project_root, "rev-parse", "HEAD")
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_ignore_status(path: Path, project_root: Path) -> bool | None:
    """True/False — путь игнорируется/не игнорируется Git; None — Git недоступен."""
    proc = _git(project_root, "check-ignore", "-q", str(path))
    if proc is None or proc.returncode not in (0, 1):
        return None
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
def _name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _looks_like_test(name: str) -> bool:
    """Токен `test` в имени ключа или сегмента пути (`final_test`, `testPath`,
    `test.parquet`); `latest`, `contest` не срабатывают — сравниваются токены."""
    return bool(_TEST_TOKENS & _name_tokens(name))


def _iter_key_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key)
            yield from _iter_key_paths(item, path)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _iter_key_paths(item, f"{prefix}[{i}]")


def _raise_test_forbidden(found: Sequence[str]) -> None:
    raise ValueError(
        "T04 must not receive or use the final test split. "
        f"Remove test-related config entries: {sorted(found)}"
    )


def _reject_test_keys(config: Mapping[str, Any]) -> None:
    found = [path for path, key in _iter_key_paths(config) if _looks_like_test(key)]
    if found:
        _raise_test_forbidden(found)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_keys(
    section: str, data: Mapping[str, Any], required: set[str], optional: set[str]
) -> None:
    unknown = sorted(set(data) - required - optional)
    if unknown:
        raise ValueError(f"Unknown key(s) in {section}: {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Config is missing required key(s) in {section}: {missing}")


def _check_relative_path(name: str, value: Any, *, is_input: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    win = PureWindowsPath(value)
    if win.drive or win.is_absolute() or PurePosixPath(value).is_absolute():
        raise ValueError(f"{name} must be a path relative to the repository root: {value!r}")
    parts = re.split(r"[\\/]", value)
    if ".." in parts:
        raise ValueError(f"{name} must not leave the repository root: {value!r}")
    if is_input and any(_looks_like_test(part) for part in parts):
        _raise_test_forbidden([f"{name}={value}"])


def _normalized_path_text(value: str) -> str:
    return "/".join(p for p in re.split(r"[\\/]", value) if p not in ("", "."))


def _validate_tfidf(tfidf: dict[str, Any]) -> None:
    _check_keys("tfidf", tfidf, TFIDF_REQUIRED, TFIDF_OPTIONAL)
    if tfidf["analyzer"] != "word":
        raise ValueError("T04 requires word-level TF-IDF: tfidf.analyzer must be 'word'")
    for key in ("lowercase", "sublinear_tf", "use_idf", "smooth_idf"):
        if key in tfidf and not isinstance(tfidf[key], bool):
            raise ValueError(f"tfidf.{key} must be true or false")
    if tfidf["norm"] not in ("l1", "l2", None):
        raise ValueError("tfidf.norm must be 'l1', 'l2' or null")

    ngram_range = tfidf["ngram_range"]
    if (
        not isinstance(ngram_range, list)
        or len(ngram_range) != 2
        or not all(_is_int(v) and v >= 1 for v in ngram_range)
        or ngram_range[0] > ngram_range[1]
    ):
        raise ValueError("tfidf.ngram_range must be [min_n, max_n] with positive integers")

    min_df, max_df = tfidf["min_df"], tfidf["max_df"]
    if not _is_number(min_df) or float(min_df) <= 0:
        raise ValueError("tfidf.min_df must be > 0")
    if not _is_number(max_df) or float(max_df) <= 0:
        raise ValueError("tfidf.max_df must be > 0")
    if isinstance(min_df, float) and min_df > 1:
        raise ValueError("tfidf.min_df as float must be <= 1")
    if isinstance(max_df, float) and max_df > 1:
        raise ValueError("tfidf.max_df as float must be <= 1")
    if isinstance(min_df, float) and isinstance(max_df, float) and min_df > max_df:
        raise ValueError("tfidf.min_df must not exceed tfidf.max_df")

    max_features = tfidf["max_features"]
    if max_features is not None and (not _is_int(max_features) or max_features <= 0):
        raise ValueError("tfidf.max_features must be null or a positive integer")

    pattern = tfidf["token_pattern"]
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("tfidf.token_pattern must be a non-empty string")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"tfidf.token_pattern is not a valid regex: {exc}") from exc
    if compiled.groups > 0:
        raise ValueError("tfidf.token_pattern must not contain capturing groups")


def _validate_logistic_regression(lr: dict[str, Any]) -> None:
    _check_keys("logistic_regression", lr, LR_REQUIRED, set())
    c_values = lr["C_values"]
    if not isinstance(c_values, list) or len(c_values) not in {2, 3}:
        raise ValueError("T04 requires exactly 2 or 3 regularization values in C_values")
    if any(not _is_number(v) or not math.isfinite(float(v)) or float(v) <= 0 for v in c_values):
        raise ValueError("Every C value must be a finite number > 0")
    if len({float(v) for v in c_values}) != len(c_values):
        raise ValueError("C_values must be unique")
    if lr["class_weight_mode"] not in ALLOWED_CLASS_WEIGHT_MODES:
        raise ValueError("class_weight_mode must be 'none' or 'balanced_train'")
    if not _is_int(lr["max_iter"]) or lr["max_iter"] <= 0:
        raise ValueError("logistic_regression.max_iter must be a positive integer")
    if not _is_number(lr["tol"]) or float(lr["tol"]) <= 0:
        raise ValueError("logistic_regression.tol must be > 0")
    if lr["solver"] not in ALLOWED_SOLVERS:
        raise ValueError(
            f"Unsupported logistic_regression.solver {lr['solver']!r}; "
            f"allowed for 3-class multinomial: {sorted(ALLOWED_SOLVERS)}"
        )


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")

    _reject_test_keys(config)
    _check_keys("config", config, TOP_REQUIRED, TOP_OPTIONAL)

    mapping = config["label_mapping"]
    if mapping != EXPECTED_LABEL_MAPPING or not all(_is_int(v) for v in mapping.values()):
        raise ValueError(
            f"label_mapping must be exactly {EXPECTED_LABEL_MAPPING}; got {mapping}"
        )

    seed = config["seed"]
    if not _is_int(seed) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    for key in ("train_path", "validation_path", "data_prep_audit_path", "data_prep_split_ids_path"):
        _check_relative_path(key, config[key], is_input=True)
    if config.get("train_ids_path") is not None:
        _check_relative_path("train_ids_path", config["train_ids_path"], is_input=True)
    for key in ("report_dir", "model_dir"):
        _check_relative_path(key, config[key], is_input=False)
    if _normalized_path_text(config["train_path"]) == _normalized_path_text(
        config["validation_path"]
    ):
        raise ValueError("train_path and validation_path must be different")

    bundle_name = config["bundle_name"]
    if not isinstance(bundle_name, str) or not bundle_name.strip() or re.search(r"[\\/]", bundle_name):
        raise ValueError("bundle_name must be a plain file name without directories")

    columns = [config[k] for k in ("text_column", "label_column", "id_column")]
    if any(not isinstance(c, str) or not c.strip() for c in columns):
        raise ValueError("text_column, label_column and id_column must be non-empty strings")
    if len(set(columns)) != 3:
        raise ValueError("text_column, label_column and id_column must be different")

    text_processing = config["text_processing"]
    if not isinstance(text_processing, dict):
        raise ValueError("text_processing must be an object")
    _check_keys("text_processing", text_processing, TEXT_PROCESSING_KEYS, set())
    if text_processing["manual_normalization"] != "none":
        raise ValueError(
            "T04 expects manual_normalization='none'; trainable/text handling is "
            "encapsulated in the TF-IDF vectorizer."
        )
    if text_processing["truncation"] is not None:
        raise ValueError("Baseline uses no manual text truncation; set truncation to null")

    if not isinstance(config["tfidf"], dict):
        raise ValueError("tfidf must be an object")
    _validate_tfidf(config["tfidf"])
    if not isinstance(config["logistic_regression"], dict):
        raise ValueError("logistic_regression must be an object")
    _validate_logistic_regression(config["logistic_regression"])


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def resolve_path(project_root: Path, value: str) -> Path:
    """Относительный путь -> абсолютный внутри корня репозитория."""
    path = (project_root / value).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError(f"Path leaves the repository root: {value!r}")
    return path


# --------------------------------------------------------------------------- #
# Данные T03: схема, независимость train/validation, соответствие аудиту
# --------------------------------------------------------------------------- #
def _coerce_exact_labels(series: pd.Series, split_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"{split_name}: boolean label values are not allowed")
    numeric = pd.to_numeric(series, errors="raise")
    arr = numeric.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{split_name}: label values must be finite")
    if not np.equal(arr, np.floor(arr)).all():
        raise ValueError(f"{split_name}: label values must be exact integers")
    labels = numeric.astype(int)
    invalid = sorted(set(labels.unique().tolist()) - set(CLASS_IDS))
    if invalid:
        raise ValueError(f"{split_name}: invalid label_id value(s): {invalid}")
    return labels


def validate_prepared_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    text_column: str,
    label_column: str,
    id_column: str,
) -> None:
    required = {text_column, label_column, id_column, *T03_EXTRA_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split_name}: missing required column(s): {missing}")

    if frame.empty:
        raise ValueError(f"{split_name}: prepared split is empty")

    nulls = frame[sorted(required)].isnull().sum()
    if int(nulls.sum()) > 0:
        raise ValueError(
            f"{split_name}: null values in required columns: "
            f"{ {k: int(v) for k, v in nulls.items() if v} }"
        )

    if not frame[text_column].map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"{split_name}: {text_column} must contain only strings")
    if frame[text_column].map(lambda x: not x.strip()).any():
        raise ValueError(f"{split_name}: empty/whitespace-only texts are not allowed")

    labels = _coerce_exact_labels(frame[label_column], split_name)
    found = sorted(labels.unique().tolist())
    if found != CLASS_IDS:
        raise ValueError(f"{split_name}: all three classes {CLASS_IDS} are required; found {found}")

    ids = frame[id_column].astype(str)
    if ids.map(lambda x: not x.strip()).any():
        raise ValueError(f"{split_name}: blank {id_column} values are not allowed")
    if ids.duplicated().any():
        raise ValueError(f"{split_name}: duplicate {id_column} values")

    languages = sorted(frame["review_language"].astype(str).unique().tolist())
    if languages != ["ru"]:
        raise ValueError(f"{split_name}: expected only review_language='ru'; found {languages}")

    id_to_name = {v: k for k, v in EXPECTED_LABEL_MAPPING.items()}
    expected_names = labels.map(id_to_name).to_numpy()
    actual_names = frame["label_name"].astype(str).str.lower().to_numpy()
    if not np.array_equal(actual_names, expected_names):
        raise ValueError(f"{split_name}: label_name is inconsistent with label_id")


def load_prepared_split(
    path: Path,
    *,
    split_name: str,
    text_column: str,
    label_column: str,
    id_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name}: prepared file not found: {path}")
    frame = pd.read_parquet(path)
    validate_prepared_frame(
        frame,
        split_name=split_name,
        text_column=text_column,
        label_column=label_column,
        id_column=id_column,
    )
    return frame.reset_index(drop=True)


def normalize_text_key(text: str) -> str:
    """Ключ проверки утечки; повторяет правило дедупликации T03 (README_T03.md):
    NFC -> casefold -> удаление невидимых символов -> схлопывание пробелов.
    Только для сравнения; в модель уходит исходный review_text."""
    normalized = unicodedata.normalize("NFC", text).casefold().translate(_INVISIBLE_CHARS)
    return " ".join(normalized.split())


def validate_split_independence(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    id_column: str,
    text_column: str,
) -> dict[str, int]:
    id_overlap = set(train[id_column].astype(str)) & set(validation[id_column].astype(str))
    if id_overlap:
        raise ValueError(
            f"train/validation record_id overlap detected: {len(id_overlap)}; "
            f"first: {sorted(id_overlap)[:5]}"
        )

    text_overlap = {normalize_text_key(t) for t in train[text_column]} & {
        normalize_text_key(t) for t in validation[text_column]
    }
    if text_overlap:
        raise ValueError(
            f"train/validation normalized review_text overlap detected: {len(text_overlap)}"
        )

    movie_overlap = set(train["movie_id"].dropna().astype(str)) & set(
        validation["movie_id"].dropna().astype(str)
    )
    if movie_overlap:
        raise ValueError(
            f"train/validation movie_id overlap detected: {len(movie_overlap)}; "
            f"first: {sorted(movie_overlap)[:5]}"
        )

    return {"record_id_overlap": 0, "normalized_text_overlap": 0, "movie_id_overlap": 0}


def class_distribution(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    counts = frame[label_column].astype(int).value_counts().to_dict()
    return {name: int(counts.get(idx, 0)) for name, idx in EXPECTED_LABEL_MAPPING.items()}


def load_t03_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"T03 audit not found: {path}")
    audit = read_json(path)
    if not isinstance(audit, dict):
        raise ValueError("T03 audit must be a JSON object")
    status = str(audit.get("status", "")).lower()
    if status != "completed":
        raise ValueError(
            f"T03 audit status must be 'completed'; got {audit.get('status')!r}. "
            "Re-run T03 (src/prepare_data.py) and use its fresh reports/data_prep/audit.json."
        )
    return audit


def validate_against_t03_audit(
    audit: Mapping[str, Any],
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    label_column: str,
    train_ids_configured: bool,
) -> None:
    """Prepared Parquet должны соответствовать именно этому audit.json (та же версия T03)."""
    summary = audit.get("final_summary")
    if not isinstance(summary, Mapping) or not all(s in summary for s in INPUT_SPLITS_FOR_T04):
        raise ValueError(
            "T03 audit has no final_summary for train/validation; "
            f"available top-level keys: {sorted(audit)}. Regenerate T03 outputs."
        )
    for split_name, frame in (("train", train), ("validation", validation)):
        expected = summary[split_name]
        actual = {"rows": int(len(frame)), **class_distribution(frame, label_column)}
        mismatch = {
            key: {"audit": expected.get(key), "prepared": value}
            for key, value in actual.items()
            if expected.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"Prepared {split_name} does not match T03 audit.json: {mismatch}. "
                "Parquet, split_ids.csv and audit.json come from different T03 runs; "
                "regenerate T03 outputs before T04."
            )

    subsample = audit.get("subsample") or {}
    if subsample.get("enabled") and subsample.get("path") and not train_ids_configured:
        raise ValueError(
            "T03 audit reports an enabled train subsample "
            f"({subsample.get('path')}), but train_ids_path is not set in the T04 config. "
            "Both models must use the same train IDs."
        )


def validate_split_ids_against_t03(
    split_ids_path: Path,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    id_column: str,
    label_column: str,
) -> dict[str, Any]:
    """Сверка с T03 split_ids.csv. Файл содержит только ID и метки всех сплитов;
    строки вне train/validation отбрасываются сразу после чтения и не используются."""
    if not split_ids_path.exists():
        raise FileNotFoundError(f"T03 split_ids.csv not found: {split_ids_path}")
    needed = {"split", id_column, label_column}
    ids = pd.read_csv(
        split_ids_path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in needed,
    )
    missing = sorted(needed - set(ids.columns))
    if missing:
        raise ValueError(f"T03 split_ids.csv is missing required column(s): {missing}")
    ids["split"] = ids["split"].str.lower()
    ids = ids.loc[ids["split"].isin(INPUT_SPLITS_FOR_T04)].reset_index(drop=True)

    report: dict[str, Any] = {}
    for split_name, frame in (("train", train), ("validation", validation)):
        expected = ids.loc[ids["split"] == split_name]
        expected_ids = expected[id_column].tolist()
        actual_ids = frame[id_column].astype(str).tolist()
        if len(expected_ids) != len(actual_ids):
            raise ValueError(
                f"T03 split_ids mismatch for {split_name}: "
                f"csv={len(expected_ids)} prepared={len(actual_ids)}"
            )
        if expected_ids != actual_ids:
            raise ValueError(
                f"T03 split_ids order/content mismatch for {split_name}; "
                "regenerate T03 outputs before T04"
            )
        expected_labels = _coerce_exact_labels(expected[label_column], f"split_ids-{split_name}")
        actual_labels = _coerce_exact_labels(frame[label_column], split_name)
        if not np.array_equal(expected_labels.to_numpy(), actual_labels.to_numpy()):
            raise ValueError(f"T03 split_ids label mismatch for {split_name}")
        report[f"{split_name}_rows"] = len(actual_ids)
        report[f"{split_name}_ids_sha256"] = ids_sha256(
            frame, id_column=id_column, label_column=label_column
        )
    return report


def apply_train_id_selection(
    train: pd.DataFrame,
    ids_path: Path | None,
    *,
    id_column: str,
    label_column: str,
    text_column: str,
) -> pd.DataFrame:
    """Возвращает train в точном порядке ID из файла (или весь train, если файла нет)."""
    if ids_path is None:
        return train.reset_index(drop=True)
    if not ids_path.exists():
        raise FileNotFoundError(f"train_ids_path not found: {ids_path}")

    ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
    if "split" in ids.columns:
        ids = ids.loc[ids["split"].str.lower() == "train"].copy()
    if id_column not in ids.columns:
        raise ValueError(f"train_ids_path must contain column '{id_column}'")

    ordered_ids = ids[id_column].tolist()
    if not ordered_ids:
        raise ValueError("train_ids_path contains no train IDs")
    if any(not rid.strip() for rid in ordered_ids):
        raise ValueError("train_ids_path contains blank IDs")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("train_ids_path contains duplicate IDs")

    indexed = train.assign(**{id_column: train[id_column].astype(str)}).set_index(
        id_column, drop=False
    )
    unknown = [rid for rid in ordered_ids if rid not in indexed.index]
    if unknown:
        raise ValueError(
            f"train_ids_path contains {len(unknown)} ID(s) absent from prepared train; "
            f"first: {unknown[:5]}"
        )

    selected = indexed.loc[ordered_ids].reset_index(drop=True)
    if label_column in ids.columns:
        file_labels = _coerce_exact_labels(ids[label_column], "train_ids_path")
        if not np.array_equal(file_labels.to_numpy(), selected[label_column].astype(int).to_numpy()):
            raise ValueError("train_ids_path labels disagree with prepared train labels")
    validate_prepared_frame(
        selected,
        split_name="train-selected",
        text_column=text_column,
        label_column=label_column,
        id_column=id_column,
    )
    return selected


def ids_sha256(frame: pd.DataFrame, *, id_column: str, label_column: str) -> str:
    digest = hashlib.sha256()
    for rid, label in frame[[id_column, label_column]].itertuples(index=False, name=None):
        digest.update(f"{rid}\t{int(label)}\n".encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Модель, метрики, выбор победителя
# --------------------------------------------------------------------------- #
def compute_train_class_weight(
    y_train: Sequence[int] | np.ndarray, mode: str
) -> dict[int, float] | None:
    if mode == "none":
        return None
    if mode != "balanced_train":
        raise ValueError(f"Unsupported class_weight_mode: {mode}")
    y = np.asarray(y_train, dtype=int)
    found = sorted(np.unique(y).tolist())
    if found != CLASS_IDS:
        raise ValueError(f"Class weights require all classes {CLASS_IDS}; found {found}")
    weights = compute_class_weight(
        class_weight="balanced", classes=np.asarray(CLASS_IDS, dtype=int), y=y
    )
    return {int(cls): float(w) for cls, w in zip(CLASS_IDS, weights)}


def build_vectorizer(tfidf_config: dict[str, Any]) -> TfidfVectorizer:
    params = dict(tfidf_config)
    params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)



def fit_vectorizer_on_train(vectorizer: TfidfVectorizer, texts: list[str]) -> dict[str, Any]:
    """Resolve max_features ties by Unicode term order, using train counts only.

    CountVectorizer applies min_df/max_df without a feature cap first. A fixed
    vocabulary then lets the ordinary, serializable TfidfVectorizer learn IDF
    without relying on NumPy's unspecified ordering of equal frequencies.
    """
    params = {key: value for key, value in vectorizer.get_params().items()
              if key in CountVectorizer().get_params()}
    params.update(max_features=None, vocabulary=None, dtype=np.int64)
    counter = CountVectorizer(**params)
    counts = counter.fit_transform(texts)
    terms = counter.get_feature_names_out()
    frequencies = np.asarray(counts.sum(axis=0)).ravel()
    limit = vectorizer.max_features
    order = sorted(range(len(terms)), key=lambda i: (-int(frequencies[i]), str(terms[i])))
    selected = order if limit is None else order[:limit]
    selected_terms = sorted(str(terms[i]) for i in selected)
    cutoff = int(frequencies[selected[-1]])
    above = int(np.count_nonzero(frequencies > cutoff))
    selection = {
        "policy": "train_term_frequency_desc_then_unicode_term_asc_v1",
        "eligible_features": len(terms), "selected_features": len(selected),
        "cutoff_term_frequency": cutoff,
        "features_above_cutoff": above,
        "features_tied_at_cutoff": int(np.count_nonzero(frequencies == cutoff)),
        "selected_at_cutoff": len(selected) - above,
    }
    vectorizer.set_params(vocabulary={term: i for i, term in enumerate(selected_terms)})
    return {"x": vectorizer.fit_transform(texts), "vocabulary_selection": selection}


def build_classifier(
    *,
    c_value: float,
    lr_config: dict[str, Any],
    class_weight: dict[int, float] | None,
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        solver=str(lr_config["solver"]),
        max_iter=int(lr_config["max_iter"]),
        tol=float(lr_config["tol"]),
        class_weight=class_weight,
        random_state=seed,
    )


def validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Метрики в фиксированном порядке классов [0, 1, 2]; класс без предсказаний
    получает precision = 0 (zero_division=0), но остаётся в отчёте и в macro-F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    result: dict[str, float | int] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for pos, name in enumerate(CLASS_NAMES):
        result[f"{name}_precision"] = float(precision[pos])
        result[f"{name}_recall"] = float(recall[pos])
        result[f"{name}_f1"] = float(f1[pos])
        result[f"{name}_support"] = int(support[pos])
    return result


def validation_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Строки — истинный класс, столбцы — предсказанный; порядок [0, 1, 2]."""
    return confusion_matrix(y_true, y_pred, labels=CLASS_IDS).astype(int).tolist()


def select_winner(results: Sequence[dict[str, Any]]) -> int:
    """Максимальный validation macro-F1; при точном равенстве — меньший C.
    Других критериев нет; результат не зависит от порядка запусков (C уникальны)."""
    if not results:
        raise ValueError("No validation results to select from")
    if any(not math.isfinite(float(r["macro_f1"])) for r in results):
        raise ValueError("macro_f1 must be finite for every run")
    return max(range(len(results)), key=lambda i: (float(results[i]["macro_f1"]), -float(results[i]["C"])))


def _vectorizer_state_sha256(vectorizer: TfidfVectorizer) -> str:
    vocabulary = sorted((str(term), int(index)) for term, index in vectorizer.vocabulary_.items())
    idf = [float(x) for x in np.asarray(vectorizer.idf_, dtype=float)]
    return sha256_json({"vocabulary": vocabulary, "idf": idf})


# --------------------------------------------------------------------------- #
# Комплект модели: save / load / predict
# --------------------------------------------------------------------------- #
def make_bundle(
    *,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    tfidf_config: dict[str, Any],
    lr_params: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "label_mapping": EXPECTED_LABEL_MAPPING.copy(),
        "class_ids": CLASS_IDS.copy(),
        "class_names": CLASS_NAMES.copy(),
        "text_processing": {
            "manual_normalization": "none",
            "truncation": None,
            "input_text_is_not_modified_before_vectorizer": True,
            "tfidf": tfidf_config,
        },
        "logistic_regression": lr_params,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "library_versions": library_versions(),
        "training": training,
    }


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> dict[str, Any]:
    """Загружает и проверяет самодостаточность комплекта (только доверенные файлы!)."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid bundle: expected dict")
    if bundle.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported bundle_format_version: {bundle.get('bundle_format_version')}")
    if bundle.get("model_type") != MODEL_TYPE:
        raise ValueError("Invalid bundle model_type")
    if bundle.get("label_mapping") != EXPECTED_LABEL_MAPPING:
        raise ValueError("Invalid bundle label_mapping")
    if bundle.get("class_ids") != CLASS_IDS or bundle.get("class_names") != CLASS_NAMES:
        raise ValueError("Invalid bundle class_ids/class_names; expected [0, 1, 2] / negative, neutral, positive")
    for key in ("text_processing", "training", "library_versions", "logistic_regression"):
        if not isinstance(bundle.get(key), dict):
            raise ValueError(f"Invalid bundle: '{key}' is missing or not an object")
    processing = bundle["text_processing"]
    if processing.get("manual_normalization") != "none" or processing.get("truncation") is not None:
        raise ValueError("Invalid bundle text_processing: expected no manual normalization/truncation")

    vectorizer = bundle.get("vectorizer")
    classifier = bundle.get("classifier")
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(classifier, LogisticRegression):
        raise ValueError("Invalid bundle: vectorizer/classifier missing or of wrong type")
    if vectorizer.analyzer != "word":
        raise ValueError("Invalid bundle: vectorizer must be word-level")
    try:
        check_is_fitted(vectorizer, "vocabulary_")
        check_is_fitted(classifier, "classes_")
    except Exception as exc:  # sklearn NotFittedError
        raise ValueError(f"Invalid bundle: unfitted component ({exc})") from exc
    if [int(v) for v in classifier.classes_] != CLASS_IDS:
        raise ValueError("Invalid bundle classifier classes_; expected [0, 1, 2]")
    if classifier.coef_.shape != (len(CLASS_IDS), len(vectorizer.vocabulary_)):
        raise ValueError("Invalid bundle: classifier features do not match the vectorizer vocabulary")
    return bundle


def predict_texts(bundle: dict[str, Any], texts: Sequence[str]) -> dict[str, Any]:
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("All prediction inputs must be strings")
    x = bundle["vectorizer"].transform(list(texts))
    classifier: LogisticRegression = bundle["classifier"]
    pred_ids = classifier.predict(x).astype(int)
    id_to_label = {int(v): k for k, v in bundle["label_mapping"].items()}
    class_order = [int(v) for v in classifier.classes_]
    return {
        "label_ids": pred_ids.tolist(),
        "labels": [id_to_label[int(v)] for v in pred_ids],
        "probability_class_ids": class_order,
        "probability_labels": [id_to_label[v] for v in class_order],
        "probabilities": classifier.predict_proba(x).tolist(),
    }


def _smoke_texts() -> list[str]:
    return [
        "Фильм отличный, очень понравился.",
        "Фильм ужасный, было скучно.",
        "Обычный фильм без сильных впечатлений.",
    ]


def verify_bundle_roundtrip(
    bundle: dict[str, Any], bundle_path: Path, texts: Sequence[str] | None = None
) -> dict[str, Any]:
    """Классы после load должны совпасть точно; вероятности — с допуском 1e-12."""
    texts = list(texts) if texts is not None else _smoke_texts()
    before = predict_texts(bundle, texts)
    after = predict_texts(load_bundle(bundle_path), texts)

    if before["label_ids"] != after["label_ids"] or before["labels"] != after["labels"]:
        raise AssertionError("Bundle save/load changed predicted classes")
    if before["probability_class_ids"] != after["probability_class_ids"]:
        raise AssertionError("Bundle save/load changed class order")
    max_diff = float(
        np.max(np.abs(np.asarray(before["probabilities"]) - np.asarray(after["probabilities"])))
    )
    if max_diff > 1e-12:
        raise AssertionError(f"Bundle save/load changed probabilities (max abs diff {max_diff})")

    return {
        "passed": True,
        "num_texts": len(texts),
        "text_sha256": [sha256_text(t) for t in texts],
        "predicted_label_ids": after["label_ids"],
        "predicted_labels": after["labels"],
        "probability_class_ids": after["probability_class_ids"],
        "probabilities_max_abs_diff": max_diff,
    }


def smoke_test() -> dict[str, Any]:
    """Маленький train -> save -> load -> predict; не читает Parquet и не требует GPU."""
    texts = [
        "ужасный фильм скучно плохо",
        "плохая игра актеров разочарование",
        "совсем не понравилось ужасно",
        "слабый фильм зря потратил время",
        "обычный фильм без эмоций",
        "средний фильм ничего особенного",
        "нормально можно посмотреть один раз",
        "нейтральное впечатление от фильма",
        "отличный фильм очень понравился",
        "прекрасная игра актеров рекомендую",
        "великолепно смотрел с удовольствием",
        "замечательный фильм советую всем",
    ]
    labels = np.asarray([0] * 4 + [1] * 4 + [2] * 4, dtype=int)
    tfidf_config = {
        "analyzer": "word",
        "lowercase": True,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": None,
        "sublinear_tf": True,
        "norm": "l2",
        "token_pattern": r"(?u)\b\w\w+\b",
    }
    lr_config = {"solver": "lbfgs", "max_iter": 300, "tol": 1e-4, "class_weight_mode": "balanced_train"}

    vectorizer = build_vectorizer(tfidf_config)
    x = vectorizer.fit_transform(texts)
    class_weight = compute_train_class_weight(labels, "balanced_train")
    classifier = build_classifier(c_value=1.0, lr_config=lr_config, class_weight=class_weight, seed=42)
    classifier.fit(x, labels)

    bundle = make_bundle(
        vectorizer=vectorizer,
        classifier=classifier,
        tfidf_config=tfidf_config,
        lr_params={**lr_config, "C": 1.0, "class_weights": {str(k): v for k, v in (class_weight or {}).items()}},
        training={"seed": 42, "smoke": True},
    )
    with tempfile.TemporaryDirectory(prefix="t04-smoke-") as tmp:
        path = Path(tmp) / "bundle.joblib"
        save_bundle(path, bundle)
        report = verify_bundle_roundtrip(bundle, path)
        report["bundle_loadable"] = True
        return report


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #
RESULT_COLUMNS = [
    "run_name",
    "C",
    "class_weight_mode",
    "macro_f1",
    "accuracy",
    "negative_precision",
    "negative_recall",
    "negative_f1",
    "negative_support",
    "neutral_precision",
    "neutral_recall",
    "neutral_f1",
    "neutral_support",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "positive_support",
    "fit_seconds",
    "validation_inference_seconds",
    "n_iter_max",
    "convergence_warning",
    "other_warnings",
]


def _write_train_ids(frame: pd.DataFrame, path: Path, *, id_column: str, label_column: str) -> None:
    """Только порядок, ID и метка — без текстов. Тот же файл использует RuBERT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[[id_column, label_column]].copy()
    out.columns = ["record_id", "label_id"]
    out["label_id"] = out["label_id"].astype(int)
    out.insert(0, "train_order", np.arange(len(out), dtype=int))
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_validation_results(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in results:
            clean = {k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow({key: clean.get(key, "") for key in RESULT_COLUMNS})


def _write_summary(
    path: Path,
    results: Sequence[dict[str, Any]],
    winner_index: int,
    confusion: list[list[int]],
) -> None:
    ranking = sorted(range(len(results)), key=lambda i: (-float(results[i]["macro_f1"]), float(results[i]["C"])))
    winner = results[winner_index]
    lines = [
        "# T04 — validation summary",
        "",
        "Выбор конфигурации выполнен **только по validation macro-F1**. Финальный test в T04 не читается и не используется.",
        "",
        "| Запуск | C | Веса классов | macro-F1 | F1 negative | F1 neutral | F1 positive | Сошёлся | Fit, s | Predict val, s |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for i, row in enumerate(results):
        marker = " **(winner)**" if i == winner_index else ""
        lines.append(
            "| {name}{marker} | {c:g} | {weight} | {macro:.6f} | {neg:.6f} | {neu:.6f} | {pos:.6f} | {conv} | {fit:.3f} | {infer:.3f} |".format(
                name=row["run_name"],
                marker=marker,
                c=float(row["C"]),
                weight=row["class_weight_mode"],
                macro=float(row["macro_f1"]),
                neg=float(row["negative_f1"]),
                neu=float(row["neutral_f1"]),
                pos=float(row["positive_f1"]),
                conv="нет (ConvergenceWarning)" if row["convergence_warning"] else "да",
                fit=float(row["fit_seconds"]),
                infer=float(row["validation_inference_seconds"]),
            )
        )
    lines += ["", f"Победитель: **{winner['run_name']}** (C = {float(winner['C']):g}), validation macro-F1 = {float(winner['macro_f1']):.6f}."]
    if len(ranking) > 1:
        runner = results[ranking[1] if ranking[0] == winner_index else ranking[0]]
        gap = float(winner["macro_f1"]) - float(runner["macro_f1"])
        lines.append(
            f"Ближайший конкурент: {runner['run_name']} (C = {float(runner['C']):g}), "
            f"macro-F1 = {float(runner['macro_f1']):.6f}; разница {gap:+.6f}."
        )
    lines += [
        "",
        "Правило выбора: максимальный validation macro-F1; при точном равенстве — меньший C. Других критериев нет.",
        "Accuracy — дополнительная метрика; из-за редкого neutral основной критерий — macro-F1.",
        "Время predict не включает TF-IDF transform (см. `validation_transform_seconds` в run_metadata.json).",
        "",
        "## Матрица ошибок победителя (validation)",
        "",
        "Строки — истинный класс, столбцы — предсказанный; порядок: negative(0), neutral(1), positive(2).",
        "",
        "| истина \\ прогноз | negative | neutral | positive |",
        "|---|---:|---:|---:|",
    ]
    for name, row in zip(CLASS_NAMES, confusion):
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Основной сценарий
# --------------------------------------------------------------------------- #
def _fit_capturing_warnings(fit_callable: Any) -> tuple[bool, list[str]]:
    """Выполняет fit; ConvergenceWarning фиксируется, остальные предупреждения
    не скрываются: они записываются в отчёт и повторно выводятся."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_callable()
    convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    others = sorted(
        {f"{w.category.__name__}: {w.message}" for w in caught if not issubclass(w.category, ConvergenceWarning)}
    )
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn(w.message, w.category, stacklevel=2)
    return convergence, others


def run_training(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    root = (project_root if project_root is not None else Path(__file__).resolve().parents[1]).resolve()

    train_path = resolve_path(root, config["train_path"])
    validation_path = resolve_path(root, config["validation_path"])
    audit_path = resolve_path(root, config["data_prep_audit_path"])
    split_ids_path = resolve_path(root, config["data_prep_split_ids_path"])
    report_dir = resolve_path(root, config["report_dir"])
    model_dir = resolve_path(root, config["model_dir"])
    train_ids_path = (
        resolve_path(root, config["train_ids_path"]) if config.get("train_ids_path") is not None else None
    )
    if train_path == validation_path:
        raise ValueError("Resolved train_path and validation_path must be different")
    for label, path in (
        ("train", train_path),
        ("validation", validation_path),
        ("T03 audit", audit_path),
        ("T03 split_ids", split_ids_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label}: file not found: {path}")

    text_column = config["text_column"]
    label_column = config["label_column"]
    id_column = config["id_column"]

    # Финальный test намеренно отсутствует и в загрузке данных, и в оценке.
    t03_audit = load_t03_audit(audit_path)
    full_train = load_prepared_split(
        train_path, split_name="train", text_column=text_column, label_column=label_column, id_column=id_column
    )
    validation = load_prepared_split(
        validation_path, split_name="validation", text_column=text_column, label_column=label_column, id_column=id_column
    )
    split_independence = validate_split_independence(
        full_train, validation, id_column=id_column, text_column=text_column
    )
    validate_against_t03_audit(
        t03_audit,
        train=full_train,
        validation=validation,
        label_column=label_column,
        train_ids_configured=train_ids_path is not None,
    )
    t03_split_check = validate_split_ids_against_t03(
        split_ids_path, train=full_train, validation=validation, id_column=id_column, label_column=label_column
    )
    train = apply_train_id_selection(
        full_train, train_ids_path, id_column=id_column, label_column=label_column, text_column=text_column
    )

    y_train = train[label_column].astype(int).to_numpy()
    y_validation = validation[label_column].astype(int).to_numpy()
    validation_texts = validation[text_column].tolist()

    # TF-IDF: fit только на train; validation — только transform.
    vectorizer = build_vectorizer(config["tfidf"])
    fitted_train: dict[str, Any] = {}
    start = time.perf_counter()
    _, vectorizer_warnings = _fit_capturing_warnings(
        lambda: fitted_train.update(fit_vectorizer_on_train(vectorizer, train[text_column].tolist()))
    )
    vectorizer_fit_seconds = time.perf_counter() - start
    x_train = fitted_train["x"]
    if x_train.shape[1] == 0:
        raise ValueError("TF-IDF produced zero features")
    start = time.perf_counter()
    x_validation = vectorizer.transform(validation_texts)
    validation_transform_seconds = time.perf_counter() - start
    vectorizer_state_sha256 = _vectorizer_state_sha256(vectorizer)

    lr_config = config["logistic_regression"]
    class_weight_mode = lr_config["class_weight_mode"]
    class_weight = compute_train_class_weight(y_train, class_weight_mode)  # только train

    results: list[dict[str, Any]] = []
    fitted: list[LogisticRegression] = []
    predictions: list[np.ndarray] = []
    for index, c_value in enumerate(lr_config["C_values"], start=1):
        classifier = build_classifier(
            c_value=float(c_value), lr_config=lr_config, class_weight=class_weight, seed=config["seed"]
        )
        start = time.perf_counter()
        convergence_warning, other_warnings = _fit_capturing_warnings(
            lambda: classifier.fit(x_train, y_train)
        )
        fit_seconds = time.perf_counter() - start
        if [int(v) for v in classifier.classes_] != CLASS_IDS:
            raise AssertionError(f"Classifier classes_ changed unexpectedly: {classifier.classes_.tolist()}")

        start = time.perf_counter()
        pred = classifier.predict(x_validation).astype(int)
        inference_seconds = time.perf_counter() - start
        row: dict[str, Any] = {
            "run_name": f"run_{index}_C_{float(c_value):g}",
            "C": float(c_value),
            "class_weight_mode": class_weight_mode,
            "fit_seconds": float(fit_seconds),
            "validation_inference_seconds": float(inference_seconds),
            "n_iter_max": int(np.max(classifier.n_iter_)),
            "convergence_warning": bool(convergence_warning),
            "other_warnings": " | ".join(other_warnings),
        }
        row.update(validation_metrics(y_validation, pred))
        results.append(row)
        fitted.append(classifier)
        predictions.append(pred)

    winner_index = select_winner(results)
    winner = results[winner_index]
    winner_classifier = fitted[winner_index]
    confusion = validation_confusion_matrix(y_validation, predictions[winner_index])
    winner_public = {k: v for k, v in winner.items() if k not in TIMING_KEYS}

    train_ids_digest = ids_sha256(train, id_column=id_column, label_column=label_column)
    validation_ids_digest = ids_sha256(validation, id_column=id_column, label_column=label_column)
    source_files = t03_audit.get("source_files") or {}
    weights_json = (
        {str(k): float(v) for k, v in class_weight.items()} if class_weight is not None else None
    )
    versions = library_versions()

    training_metadata = {
        "seed": int(config["seed"]),
        "train_rows": int(len(train)),
        "train_full_rows": int(len(full_train)),
        "train_is_subset": bool(len(train) != len(full_train)),
        "validation_rows": int(len(validation)),
        "train_class_distribution": class_distribution(train, label_column),
        "validation_class_distribution": class_distribution(validation, label_column),
        "train_ids_sha256": train_ids_digest,
        "validation_ids_sha256": validation_ids_digest,
        "train_selection_source": display_path(train_ids_path, root) if train_ids_path else "all_prepared_train",
        "train_ids_file_sha256": sha256_file(train_ids_path) if train_ids_path else None,
        "train_parquet_sha256": sha256_file(train_path),
        "validation_parquet_sha256": sha256_file(validation_path),
        "t03_source_parquet_sha256": {
            split: (source_files.get(split) or {}).get("sha256_after") for split in INPUT_SPLITS_FOR_T04
        },
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "vocabulary_selection": fitted_train["vocabulary_selection"],
        "tfidf_state_sha256": vectorizer_state_sha256,
        "tfidf_warnings": vectorizer_warnings,
        "class_weight_mode": class_weight_mode,
        "class_weights": weights_json,
        "winner_run_name": winner["run_name"],
        "winner_C": float(winner["C"]),
        "winner_validation_macro_f1": float(winner["macro_f1"]),
        "winner_convergence_warning": bool(winner["convergence_warning"]),
        "selection_metric": "validation_macro_f1",
        "selection_tie_break": "smaller_C",
        "final_test_used": False,
        "split_independence": split_independence,
        "t03_audit_status": t03_audit.get("status"),
        "t03_audit_sha256": sha256_file(audit_path),
        "t03_split_ids_sha256": sha256_file(split_ids_path),
        "t03_split_check": t03_split_check,
    }
    lr_params = {
        "C": float(winner["C"]),
        "solver": lr_config["solver"],
        "penalty": "l2",
        "max_iter": int(lr_config["max_iter"]),
        "tol": float(lr_config["tol"]),
        "class_weight_mode": class_weight_mode,
        "class_weights": weights_json,
    }
    bundle = make_bundle(
        vectorizer=vectorizer,
        classifier=winner_classifier,
        tfidf_config=config["tfidf"],
        lr_params=lr_params,
        training=training_metadata,
    )

    bundle_path = model_dir / config["bundle_name"]
    save_bundle(bundle_path, bundle)
    fixed_texts_report = verify_bundle_roundtrip(bundle, bundle_path)
    loaded = load_bundle(bundle_path)
    loaded_validation = np.asarray(predict_texts(loaded, validation_texts)["label_ids"], dtype=int)
    if not np.array_equal(loaded_validation, predictions[winner_index]):
        raise AssertionError("Loaded bundle predicts differently from the trained winner on validation")
    loaded_macro_f1 = validation_metrics(y_validation, loaded_validation)["macro_f1"]
    if loaded_macro_f1 != winner["macro_f1"]:
        raise AssertionError("Loaded bundle validation macro-F1 differs from the recorded winner")
    roundtrip_report = {
        "fixed_texts": fixed_texts_report,
        "validation_predictions_identical_after_load": True,
        "validation_macro_f1_after_load": float(loaded_macro_f1),
    }
    bundle_sha256 = sha256_file(bundle_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    train_ids_report_path = report_dir / "train_ids.csv"
    validation_results_path = report_dir / "validation_results.csv"
    summary_path = report_dir / "validation_summary.md"
    winner_config_path = report_dir / "winner_config.json"
    metadata_path = report_dir / "run_metadata.json"
    roundtrip_path = report_dir / "bundle_roundtrip.json"

    _write_train_ids(train, train_ids_report_path, id_column=id_column, label_column=label_column)
    _write_validation_results(validation_results_path, results)
    _write_summary(summary_path, results, winner_index, confusion)
    write_json(roundtrip_path, roundtrip_report)

    bundle_git_ignored = git_ignore_status(bundle_path, root)
    metadata = {
        **training_metadata,
        # Тайминги зависят от машины и в комплект модели не входят (он должен быть воспроизводимым).
        "tfidf_fit_seconds": float(vectorizer_fit_seconds),
        "validation_transform_seconds": float(validation_transform_seconds),
        "config_path": display_path(config_path, root),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": sha256_json(config),
        "script_sha256": sha256_file(Path(__file__)),
        "data_prep_audit_path": display_path(audit_path, root),
        "data_prep_split_ids_path": display_path(split_ids_path, root),
        "git_commit": git_commit(root),
        "library_versions": versions,
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "bundle_path": display_path(bundle_path, root),
        "bundle_sha256": bundle_sha256,
        "bundle_git_ignored": bundle_git_ignored,
        "report_paths": {
            "train_ids": display_path(train_ids_report_path, root),
            "validation_results": display_path(validation_results_path, root),
            "validation_summary": display_path(summary_path, root),
            "bundle_roundtrip": display_path(roundtrip_path, root),
        },
    }
    write_json(metadata_path, metadata)

    # winner_config.json не содержит таймингов: при тех же данных файл байт-в-байт повторяем.
    write_json(
        winner_config_path,
        {
            "selection_rule": {
                "primary": "validation_macro_f1_max",
                "tie_break": "C_min",
                "final_test_used": False,
            },
            "winner": winner_public,
            "validation_confusion_matrix": {
                "row_order": CLASS_NAMES,
                "column_order": CLASS_NAMES,
                "rows_are_true_classes": True,
                "matrix": confusion,
            },
            "label_mapping": EXPECTED_LABEL_MAPPING,
            "text_processing": config["text_processing"],
            "tfidf": config["tfidf"],
            "logistic_regression": lr_params,
            "seed": int(config["seed"]),
            "train_ids_sha256": train_ids_digest,
            "validation_ids_sha256": validation_ids_digest,
            "tfidf_state_sha256": vectorizer_state_sha256,
            "bundle_path": metadata["bundle_path"],
            "bundle_sha256": bundle_sha256,
        },
    )

    return {
        "winner": winner,
        "results": results,
        "bundle_path": bundle_path,
        "bundle_sha256": bundle_sha256,
        "bundle_git_ignored": bundle_git_ignored,
        "validation_results_path": validation_results_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "winner_config_path": winner_config_path,
        "roundtrip_path": roundtrip_path,
        "train_ids_path": train_ids_report_path,
        "final_test_used": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T04: word-level TF-IDF + Logistic Regression baseline")
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to T04 JSON config (default: configs/baseline.json)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run tiny train -> save -> load check without the corpus",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        report = smoke_test()
        print("T04 smoke test passed.")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    result = run_training(Path(args.config))
    winner = result["winner"]
    print("T04 baseline training completed.")
    for row in result["results"]:
        flag = " [ConvergenceWarning]" if row["convergence_warning"] else ""
        print(f"  {row['run_name']}: validation macro-F1={row['macro_f1']:.6f}{flag}")
    print(f"Winner: {winner['run_name']} | C={winner['C']:g} | validation macro-F1={winner['macro_f1']:.6f}")
    if winner.get("convergence_warning"):
        print("WARNING: winner emitted a ConvergenceWarning; inspect max_iter before accepting results.")
    if result["bundle_git_ignored"] is False:
        print("WARNING: the model bundle is NOT git-ignored. Add 'models/' to .gitignore; do not commit it.")
    print(f"Bundle: {result['bundle_path']}")
    print(f"Validation table: {result['validation_results_path']}")
    print(f"Summary: {result['summary_path']}")
    print("Final test split was NOT read or used in T04.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
