"""Воспроизводимая подготовка русскоязычных split'ов для sentiment classification.

T03: фильтрация языка, фиксированное сопоставление классов, аудит утечек,
точная дедупликация, поиск near-duplicate кандидатов и применение только
подтверждённых вручную решений.

Исходные Parquet-файлы никогда не изменяются.

Ключ точной дедупликации (`_exact_key`) строится как
`NFC -> casefold -> удаление невидимых символов -> схлопывание пробелов`.
Он используется ТОЛЬКО для поиска повторов и никогда не попадает в модельный
вход: `review_text` сохраняется буквально.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_VERSION = "t03.2"

# Приоритет сохранения split'ов: test > validation > train.
SPLIT_PRIORITY = {"train": 1, "validation": 2, "test": 3}
SPLITS = ("train", "validation", "test")

LABEL_ORDER = ("negative", "neutral", "positive")
REQUIRED_CLASS_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}

# Невидимые символы, которые не являются пробельными для re.\s, но делают
# визуально одинаковые тексты различными на уровне байт.
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceFile:
    split: str
    path: Path
    sha256_before: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible Russian train/validation/test splits")
    parser.add_argument("--config", default="configs/data_prep.json", help="Path to JSON config")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Default: parent directory of the config's directory.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    """Обязательный ключ: молчаливая подстановка дефолта запрещена."""
    if key not in mapping:
        raise ValueError(f"Config error: missing required key {where}.{key}")
    return mapping[key]


def _require_number(
    mapping: dict[str, Any],
    key: str,
    where: str,
    *,
    kind: type,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    value = _require(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ValueError(f"Config error: {where}.{key} must be {kind.__name__}, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Config error: {where}.{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Config error: {where}.{key} must be <= {maximum}, got {value}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Полная проверка конфигурации.

    Любое отсутствующее или некорректное значение приводит к явной ошибке:
    конфигурация определяет эксперимент, поэтому тихая подстановка дефолта
    недопустима.
    """
    for key in ("input_dir", "output_dir", "report_dir", "artifact_dir", "target_language"):
        value = _require(config, key, "config")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config error: config.{key} must be a non-empty string")

    files = _require(config, "files", "config")
    if not isinstance(files, dict):
        raise ValueError("Config error: config.files must be an object")
    for split in SPLITS:
        name = _require(files, split, "config.files")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Config error: config.files.{split} must be a non-empty string")
    if len({files[s] for s in SPLITS}) != len(SPLITS):
        raise ValueError("Config error: config.files must reference three distinct file names")

    expected_columns = _require(config, "expected_columns", "config")
    if not isinstance(expected_columns, list) or not all(isinstance(c, str) for c in expected_columns):
        raise ValueError("Config error: config.expected_columns must be a list of strings")
    for column in ("movie_id", "review_text", "review_sentiment", "review_language"):
        if column not in expected_columns:
            raise ValueError(f"Config error: config.expected_columns must contain {column!r}")

    allowed_languages = _require(config, "allowed_languages", "config")
    if not isinstance(allowed_languages, list) or not all(isinstance(x, str) for x in allowed_languages):
        raise ValueError("Config error: config.allowed_languages must be a list of strings")
    if config["target_language"] not in allowed_languages:
        raise ValueError("Config error: config.target_language must be listed in config.allowed_languages")

    class_mapping = _require(config, "class_mapping", "config")
    if not isinstance(class_mapping, dict):
        raise ValueError("Config error: config.class_mapping must be an object")
    normalized = {str(k).casefold(): v for k, v in class_mapping.items()}
    if normalized != REQUIRED_CLASS_MAPPING:
        raise ValueError(
            "Config error: T03 requires fixed class mapping negative=0, neutral=1, positive=2, "
            f"got {class_mapping}"
        )

    near = _require(config, "near_duplicates", "config")
    if not isinstance(near, dict):
        raise ValueError("Config error: config.near_duplicates must be an object")
    enabled = _require(near, "enabled", "config.near_duplicates")
    if not isinstance(enabled, bool):
        raise ValueError("Config error: config.near_duplicates.enabled must be a boolean")
    _require_number(near, "threshold", "config.near_duplicates", kind=float, minimum=0.0, maximum=1.0)
    _require_number(near, "ngram_size", "config.near_duplicates", kind=int, minimum=1, maximum=16)
    _require_number(near, "min_chars", "config.near_duplicates", kind=int, minimum=1)
    _require_number(near, "prefix_chars", "config.near_duplicates", kind=int, minimum=1)
    _require_number(near, "suffix_chars", "config.near_duplicates", kind=int, minimum=1)
    _require_number(near, "length_ratio_min", "config.near_duplicates", kind=float, minimum=0.0, maximum=1.0)
    _require_number(near, "max_block_pairs", "config.near_duplicates", kind=int, minimum=1)
    _require_number(near, "max_candidates", "config.near_duplicates", kind=int, minimum=1)
    decision_file = _require(near, "decision_file", "config.near_duplicates")
    if not isinstance(decision_file, str) or not decision_file.strip():
        raise ValueError("Config error: config.near_duplicates.decision_file must be a non-empty string")

    subsample = _require(config, "subsample", "config")
    if not isinstance(subsample, dict):
        raise ValueError("Config error: config.subsample must be an object")
    sub_enabled = _require(subsample, "enabled", "config.subsample")
    if not isinstance(sub_enabled, bool):
        raise ValueError("Config error: config.subsample.enabled must be a boolean")
    _require_number(subsample, "size", "config.subsample", kind=int, minimum=0)
    _require_number(subsample, "seed", "config.subsample", kind=int, minimum=0)
    if sub_enabled and subsample["size"] <= 0:
        raise ValueError("Config error: config.subsample.enabled is true but config.subsample.size is not positive")

    unknown = set(config) - {
        "input_dir", "files", "output_dir", "report_dir", "artifact_dir",
        "expected_columns", "allowed_languages", "target_language",
        "class_mapping", "near_duplicates", "subsample",
    }
    if unknown:
        raise ValueError(f"Config error: unknown top-level keys: {sorted(unknown)}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Config error: top-level JSON value must be an object")
    validate_config(config)
    return config


def resolve_project_root(config_path: Path, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return config_path.resolve().parent.parent


# --------------------------------------------------------------------------
# Нормализация и вспомогательные функции
# --------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_pair_id(left_id: str, right_id: str) -> str:
    a, b = sorted((left_id, right_id))
    return hashlib.sha1(f"{a}|{b}".encode("utf-8")).hexdigest()[:16]


def exact_normalize(text: str) -> str:
    """Ключ дедупликации: NFC, регистр, невидимые символы, пробелы.

    Пунктуация и эмодзи сохраняются. Результат используется только для поиска
    повторов и никогда не заменяет `review_text`.
    """
    if not isinstance(text, str):
        raise TypeError(f"review_text must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", normalized.casefold()).strip()


def near_normalize(text: str) -> str:
    """Нормализация для поиска кандидатов, не для входа модели."""
    return exact_normalize(text)


def char_ngrams(text: str, n: int) -> set[str]:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_jaccard(left: str, right: str, n: int) -> float:
    a = char_ngrams(left, n)
    b = char_ngrams(right, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_record_id(source_file: str, source_row: int) -> str:
    return f"{source_file}#row={source_row:08d}"


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV с фиксированным переводом строки: иначе Windows и Linux дают разные байты."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _rel(path: Path, project_root: Path) -> str:
    """Относительный путь с '/' на любой ОС: иначе аудит не сравнить побайтово."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


# --------------------------------------------------------------------------
# Загрузка и валидация источников
# --------------------------------------------------------------------------

def validate_source_dataframe(
    df: pd.DataFrame,
    *,
    split: str,
    expected_columns: list[str],
    allowed_languages: set[str],
) -> None:
    missing_columns = sorted(set(expected_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{split}: missing columns: {missing_columns}")

    required = ["movie_id", "review_text", "review_sentiment", "review_language"]
    nulls = df[required].isna().sum()
    bad_nulls = {col: int(n) for col, n in nulls.items() if int(n) > 0}
    if bad_nulls:
        raise ValueError(f"{split}: null values in required columns: {bad_nulls}")

    non_string = df.loc[~df["review_text"].map(lambda v: isinstance(v, str))]
    if len(non_string):
        raise ValueError(f"{split}: {len(non_string)} review_text values are not strings")

    invalid_languages = sorted(set(df["review_language"].astype(str).unique()) - allowed_languages)
    if invalid_languages:
        raise ValueError(f"{split}: unexpected review_language values: {invalid_languages}")

    # Пустой текст и текст из одних пробелов эквивалентны для дедупликации:
    # без этой проверки все такие строки схлопнулись бы в один ключ.
    blank = df["review_text"].map(lambda v: exact_normalize(v) == "")
    if bool(blank.any()):
        rows = df.loc[blank].index.tolist()[:5]
        raise ValueError(
            f"{split}: {int(blank.sum())} empty or whitespace-only review_text values "
            f"(first offending positions: {rows})"
        )


def load_source_splits(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, pd.DataFrame], list[SourceFile]]:
    input_dir = project_root / config["input_dir"]
    expected_columns = list(config["expected_columns"])
    allowed_languages = set(config["allowed_languages"])

    frames: dict[str, pd.DataFrame] = {}
    source_files: list[SourceFile] = []

    for split in SPLITS:
        source_path = input_dir / config["files"][split]
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing source file: {source_path}. "
                f"Project root resolved to {project_root}; use --project-root to override."
            )

        sha = sha256_file(source_path)
        source_files.append(SourceFile(split=split, path=source_path, sha256_before=sha))
        df = pd.read_parquet(source_path)
        validate_source_dataframe(
            df,
            split=split,
            expected_columns=expected_columns,
            allowed_languages=allowed_languages,
        )

        df = df.copy()
        df["source_file"] = source_path.name
        df["source_row"] = range(len(df))
        df["record_id"] = [build_record_id(source_path.name, i) for i in range(len(df))]
        if not df["record_id"].is_unique:
            raise ValueError(f"{split}: record_id values are not unique")
        frames[split] = df

    return frames, source_files


def assert_output_paths_safe(
    source_files: list[SourceFile], output_dir: Path, report_dir: Path, artifact_dir: Path
) -> None:
    """Исходные Parquet не должны оказаться целью записи."""
    source_paths = {sf.path.resolve() for sf in source_files}
    source_dirs = {p.parent for p in source_paths}
    for name, directory in (
        ("output_dir", output_dir),
        ("report_dir", report_dir),
        ("artifact_dir", artifact_dir),
    ):
        resolved = directory.resolve()
        if resolved in source_dirs:
            raise ValueError(
                f"Unsafe configuration: {name} ({resolved}) is the directory holding the source Parquet files"
            )
    for split in SPLITS:
        prepared = (output_dir / f"{split}.parquet").resolve()
        if prepared in source_paths:
            raise ValueError(f"Unsafe configuration: prepared output {prepared} would overwrite a source file")


def filter_and_map_labels(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    target_language = config["target_language"]
    class_mapping = {str(k).casefold(): int(v) for k, v in config["class_mapping"].items()}
    if class_mapping != REQUIRED_CLASS_MAPPING:
        raise ValueError("T03 requires fixed class mapping: negative=0, neutral=1, positive=2")

    result: dict[str, pd.DataFrame] = {}
    for split, df in frames.items():
        filtered = df.loc[df["review_language"].eq(target_language)].copy()
        labels = filtered["review_sentiment"].astype(str).str.strip().str.casefold()
        invalid = sorted(set(labels.unique()) - set(class_mapping))
        if invalid:
            raise ValueError(f"{split}: unsupported sentiment labels after normalization: {invalid}")
        filtered["label_name"] = labels
        filtered["label_id"] = labels.map(class_mapping).astype("int64")
        filtered["_exact_key"] = filtered["review_text"].map(exact_normalize)
        filtered["_near_key"] = filtered["_exact_key"]
        result[split] = filtered
    return result


# --------------------------------------------------------------------------
# Конфликты меток
# --------------------------------------------------------------------------

def detect_label_conflicts(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Одинаковый нормализованный текст с разными метками.

    Такие случаи нельзя молча схлопывать обычной дедупликацией: правило T03 —
    выявить, зафиксировать в аудите и в отдельном отчёте, а удаление оставить
    обычному приоритету split'ов (с журналом причин).

    Отчёт содержит только идентификаторы и хеш ключа, без текстов отзывов.
    """
    parts = []
    for split in SPLITS:
        df = frames[split]
        part = df[["record_id", "_exact_key", "label_name", "label_id"]].copy()
        part["split"] = split
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)

    labels_per_key = combined.groupby("_exact_key")["label_id"].nunique()
    conflict_keys = set(labels_per_key.loc[labels_per_key > 1].index)

    columns = ["exact_key_sha1", "record_id", "split", "label_name", "label_id", "scope"]
    if not conflict_keys:
        empty = pd.DataFrame(columns=columns)
        return empty, {
            "label_conflict_groups": 0,
            "label_conflict_rows": 0,
            "label_conflict_groups_within_split": 0,
            "label_conflict_groups_cross_split": 0,
        }

    conflicting = combined.loc[combined["_exact_key"].isin(conflict_keys)].copy()
    scope_by_key = (
        conflicting.groupby("_exact_key")["split"]
        .nunique()
        .map(lambda n: "cross_split" if n > 1 else "within_split")
    )
    conflicting["scope"] = conflicting["_exact_key"].map(scope_by_key)
    conflicting["exact_key_sha1"] = conflicting["_exact_key"].map(sha1_text)
    report = conflicting[columns].sort_values(
        ["exact_key_sha1", "split", "record_id"], kind="mergesort"
    ).reset_index(drop=True)

    scope_counts = scope_by_key.value_counts()
    stats = {
        "label_conflict_groups": int(len(conflict_keys)),
        "label_conflict_rows": int(len(conflicting)),
        "label_conflict_groups_within_split": int(scope_counts.get("within_split", 0)),
        "label_conflict_groups_cross_split": int(scope_counts.get("cross_split", 0)),
    }
    return report, stats


# --------------------------------------------------------------------------
# Точная дедупликация
# --------------------------------------------------------------------------

def _exclusion_row(row: pd.Series, reason: str, related_record_id: str | None = None) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "split": row.get("_split", ""),
        "reason": reason,
        "related_record_id": related_record_id or "",
    }


def exact_deduplicate(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    """Приоритет split'ов test > validation > train для нормализованных точных повторов."""
    work = {name: df.copy() for name, df in frames.items()}
    for split, df in work.items():
        df["_split"] = split

    exclusions: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    # Test неизменен: повторы внутри него только фиксируются.
    test_dup_mask = work["test"].duplicated("_exact_key", keep="first")
    stats["test_exact_duplicate_extra_rows"] = int(test_dup_mask.sum())
    stats["test_exact_duplicate_groups"] = int((work["test"].groupby("_exact_key").size() > 1).sum())

    # validation уступает test
    test_key_to_id = (
        work["test"].drop_duplicates("_exact_key").set_index("_exact_key")["record_id"].to_dict()
    )
    val_cross_mask = work["validation"]["_exact_key"].isin(test_key_to_id)
    for _, row in work["validation"].loc[val_cross_mask].iterrows():
        exclusions.append(
            _exclusion_row(row, "exact_cross_validation_test", test_key_to_id[row["_exact_key"]])
        )
    work["validation"] = work["validation"].loc[~val_cross_mask].copy()

    # затем дедупликация внутри validation, остаётся первая исходная строка
    val_dup_mask = work["validation"].duplicated("_exact_key", keep="first")
    first_val_id = (
        work["validation"].drop_duplicates("_exact_key").set_index("_exact_key")["record_id"].to_dict()
    )
    for _, row in work["validation"].loc[val_dup_mask].iterrows():
        exclusions.append(_exclusion_row(row, "exact_within_validation", first_val_id[row["_exact_key"]]))
    work["validation"] = work["validation"].loc[~val_dup_mask].copy()

    # train уступает итоговым validation и test
    eval_key_to_id: dict[str, str] = {}
    for eval_split in ("test", "validation"):
        tmp = work[eval_split].drop_duplicates("_exact_key")
        for key, rid in zip(tmp["_exact_key"], tmp["record_id"]):
            eval_key_to_id.setdefault(key, rid)

    train_cross_mask = work["train"]["_exact_key"].isin(eval_key_to_id)
    for _, row in work["train"].loc[train_cross_mask].iterrows():
        exclusions.append(_exclusion_row(row, "exact_cross_train_eval", eval_key_to_id[row["_exact_key"]]))
    work["train"] = work["train"].loc[~train_cross_mask].copy()

    train_dup_mask = work["train"].duplicated("_exact_key", keep="first")
    first_train_id = (
        work["train"].drop_duplicates("_exact_key").set_index("_exact_key")["record_id"].to_dict()
    )
    for _, row in work["train"].loc[train_dup_mask].iterrows():
        exclusions.append(_exclusion_row(row, "exact_within_train", first_train_id[row["_exact_key"]]))
    work["train"] = work["train"].loc[~train_dup_mask].copy()

    stats["removed_exact_validation_vs_test"] = int(val_cross_mask.sum())
    stats["removed_exact_within_validation"] = int(val_dup_mask.sum())
    stats["removed_exact_train_vs_eval"] = int(train_cross_mask.sum())
    stats["removed_exact_within_train"] = int(train_dup_mask.sum())

    for split in work:
        work[split] = work[split].drop(columns=["_split"])
    return work, exclusions, stats


# --------------------------------------------------------------------------
# Near-duplicates
# --------------------------------------------------------------------------

def _block_keys(text: str, prefix_chars: int, suffix_chars: int) -> tuple[str, ...]:
    if not text:
        return tuple()
    prefix = text[:prefix_chars]
    suffix = text[-suffix_chars:]
    edge_len = max(6, min(prefix_chars, suffix_chars) // 2)
    edge = f"{text[:edge_len]}|{text[-edge_len:]}"
    return (f"p:{prefix}", f"s:{suffix}", f"e:{edge}")


def find_near_duplicate_candidates(
    frames: dict[str, pd.DataFrame], settings: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "pair_id",
        "left_record_id",
        "left_split",
        "right_record_id",
        "right_split",
        "similarity",
        "left_chars",
        "right_chars",
    ]
    if not settings.get("enabled", True):
        return pd.DataFrame(columns=columns), {
            "near_blocks_skipped_large": 0,
            "near_largest_block_size": 0,
            "near_block_pairs_evaluated": 0,
            "near_candidates_truncated": False,
            "near_records_considered": 0,
        }

    threshold = float(settings["threshold"])
    ngram_size = int(settings["ngram_size"])
    min_chars = int(settings["min_chars"])
    prefix_chars = int(settings["prefix_chars"])
    suffix_chars = int(settings["suffix_chars"])
    length_ratio_min = float(settings["length_ratio_min"])
    max_block_pairs = int(settings["max_block_pairs"])
    max_candidates = int(settings["max_candidates"])

    records: dict[str, dict[str, Any]] = {}
    blocks: dict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        frame = frames[split]
        if "_near_key" not in frame.columns:
            raise ValueError(f"{split}: _near_key column is missing; call filter_and_map_labels first")
        # Явный zip вместо itertuples: itertuples переименовывает колонки,
        # начинающиеся с подчёркивания, и обращение к row._near_key молча
        # промахивается мимо предвычисленного ключа.
        for rid, text, source_row in zip(frame["record_id"], frame["_near_key"], frame["source_row"]):
            if len(text) < min_chars:
                continue
            records[rid] = {
                "split": split,
                "text": text,
                "chars": len(text),
                "source_row": int(source_row),
            }
            for key in _block_keys(text, prefix_chars, suffix_chars):
                blocks[key].append(rid)

    candidate_pairs: set[tuple[str, str]] = set()
    largest_block = 0
    block_pairs_evaluated = 0
    for key in sorted(blocks):
        ids = sorted(set(blocks[key]))
        if len(ids) < 2:
            continue
        largest_block = max(largest_block, len(ids))
        pair_count = len(ids) * (len(ids) - 1) // 2
        if pair_count > max_block_pairs:
            # Блоки никогда не отбрасываются молча. Превышение бюджета — это
            # явная ошибка конфигурации, а не повод потерять часть пар.
            raise ValueError(
                f"Blocking key of size {len(ids)} would produce {pair_count} pairs, exceeding "
                f"max_block_pairs={max_block_pairs}. Candidate search would be incomplete. "
                "Tighten blocking (prefix_chars / suffix_chars / min_chars) or raise "
                "max_block_pairs deliberately."
            )
        block_pairs_evaluated += pair_count
        for left, right in itertools.combinations(ids, 2):
            a, b = records[left], records[right]
            if a["text"] == b["text"]:
                continue
            ratio = min(a["chars"], b["chars"]) / max(a["chars"], b["chars"])
            if ratio < length_ratio_min:
                continue
            candidate_pairs.add((left, right))

    rows: list[dict[str, Any]] = []
    for left, right in sorted(candidate_pairs):
        a, b = records[left], records[right]
        similarity = ngram_jaccard(a["text"], b["text"], ngram_size)
        if similarity >= threshold:
            rows.append(
                {
                    "pair_id": stable_pair_id(left, right),
                    "left_record_id": left,
                    "left_split": a["split"],
                    "right_record_id": right,
                    "right_split": b["split"],
                    "similarity": round(similarity, 6),
                    "left_chars": a["chars"],
                    "right_chars": b["chars"],
                }
            )

    truncated = len(rows) > max_candidates
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = (
            result.sort_values(
                ["similarity", "left_record_id", "right_record_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(max_candidates)
            .reset_index(drop=True)
        )
        if not result["pair_id"].is_unique:
            raise ValueError("pair_id collision detected among near-duplicate candidates")

    stats = {
        # Блоки не отбрасываются: поле остаётся в аудите как явная фиксация
        # того, что поиск был полным в пределах схемы блокирования.
        "near_blocks_skipped_large": 0,
        "near_largest_block_size": largest_block,
        "near_block_pairs_evaluated": block_pairs_evaluated,
        "near_candidates_truncated": bool(truncated),
        "near_records_considered": len(records),
    }
    if truncated:
        print(
            f"WARNING: near-duplicate candidates truncated to max_candidates={max_candidates}; "
            f"{len(rows)} pairs were above the similarity threshold.",
            file=sys.stderr,
        )
    return result, stats


def build_near_review_artifact(candidates: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Артефакт с текстами для ручного аудита. Хранить вне Git.

    Содержит метки и movie_id обеих сторон: без них человек не может отличить
    копию от пары «похожий текст, разные метки».
    """
    columns = [
        "pair_id", "left_record_id", "left_split", "right_record_id", "right_split",
        "similarity", "left_chars", "right_chars",
        "left_label", "right_label", "labels_conflict",
        "left_movie_id", "right_movie_id", "same_movie_id",
        "left_text", "right_text", "decision", "note",
    ]
    text_lookup: dict[str, str] = {}
    label_lookup: dict[str, str] = {}
    movie_lookup: dict[str, str] = {}
    for df in frames.values():
        text_lookup.update(dict(zip(df["record_id"], df["review_text"])))
        label_lookup.update(dict(zip(df["record_id"], df["label_name"])))
        movie_lookup.update(dict(zip(df["record_id"], df["movie_id"].astype(str))))

    if candidates.empty:
        return pd.DataFrame(columns=columns)

    out = candidates.copy()
    out["left_text"] = out["left_record_id"].map(text_lookup)
    out["right_text"] = out["right_record_id"].map(text_lookup)
    out["left_label"] = out["left_record_id"].map(label_lookup)
    out["right_label"] = out["right_record_id"].map(label_lookup)
    out["labels_conflict"] = out["left_label"].ne(out["right_label"])
    out["left_movie_id"] = out["left_record_id"].map(movie_lookup)
    out["right_movie_id"] = out["right_record_id"].map(movie_lookup)
    out["same_movie_id"] = out["left_movie_id"].eq(out["right_movie_id"])
    out["decision"] = ""
    out["note"] = ""
    return out[columns]


def read_near_duplicate_decisions(path: Path, candidate_ids: set[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    allowed = {"duplicate", "not_duplicate"}
    decisions: dict[str, str] = {}
    unknown: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair_id", "decision"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Near-duplicate decision file must contain {sorted(required)}")
        for row in reader:
            pair_id = (row.get("pair_id") or "").strip()
            decision = (row.get("decision") or "").strip().casefold()
            if not pair_id or not decision:
                continue
            if decision not in allowed:
                raise ValueError(f"Invalid near-duplicate decision {decision!r} for {pair_id}")
            if pair_id in decisions and decisions[pair_id] != decision:
                raise ValueError(
                    f"Conflicting decisions for pair {pair_id}: {decisions[pair_id]!r} and {decision!r}"
                )
            if pair_id not in candidate_ids:
                unknown.append(pair_id)
                continue
            decisions[pair_id] = decision
    if unknown:
        # Собираем все проблемные pair_id сразу: один запуск должен показать
        # весь список, а не первый попавшийся.
        raise ValueError(
            f"{len(unknown)} decision(s) reference pairs absent from the current candidate set: "
            f"{sorted(unknown)}. This happens when the candidate-search config changed, or when one "
            "side of the pair was already removed by exact deduplication. Re-review the affected "
            "pairs and update the decision file; do not reuse stale decisions."
        )
    return decisions


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # итеративное сжатие пути
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def apply_confirmed_near_duplicates(
    frames: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    decisions: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    empty_stats = {
        "near_pairs_confirmed_duplicate": 0,
        "near_rows_removed": 0,
        "near_test_rows_preserved": 0,
        "near_pairs_within_test": 0,
        "near_components": 0,
        "near_components_larger_than_pair": 0,
        "near_indirect_pairs_in_components": 0,
        "near_removed_rows_with_label_conflict": 0,
    }
    duplicate_pairs = candidates.loc[candidates["pair_id"].map(decisions).eq("duplicate")].copy()
    if duplicate_pairs.empty:
        return {k: v.copy() for k, v in frames.items()}, [], empty_stats

    lookup: dict[str, dict[str, Any]] = {}
    for split, df in frames.items():
        for row in df[["record_id", "source_row", "label_name"]].itertuples(index=False):
            lookup[row.record_id] = {
                "split": split,
                "source_row": int(row.source_row),
                "label_name": row.label_name,
            }

    uf = _UnionFind()
    for row in duplicate_pairs.itertuples(index=False):
        uf.union(row.left_record_id, row.right_record_id)

    components: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(uf.parent):  # явный детерминированный порядок
        components[uf.find(rid)].append(rid)

    # Транзитивность: сколько пар внутри компонент человек НЕ подтверждал напрямую.
    direct_pairs = {
        tuple(sorted((row.left_record_id, row.right_record_id)))
        for row in duplicate_pairs.itertuples(index=False)
    }
    indirect = 0
    larger_than_pair = 0
    for member_ids in components.values():
        if len(member_ids) <= 2:
            continue
        larger_than_pair += 1
        for pair in itertools.combinations(sorted(member_ids), 2):
            if pair not in direct_pairs:
                indirect += 1

    to_remove: dict[str, tuple[str, str]] = {}
    preserved_test = 0
    for component_ids in components.values():
        ids = sorted(component_ids)
        tests = [rid for rid in ids if lookup[rid]["split"] == "test"]
        validations = [rid for rid in ids if lookup[rid]["split"] == "validation"]
        trains = [rid for rid in ids if lookup[rid]["split"] == "train"]

        if tests:
            keep_ids = sorted(tests)
            preserved_test += len(keep_ids)
            representative = keep_ids[0]
            for rid in validations + trains:
                to_remove[rid] = ("near_duplicate_confirmed_lower_priority", representative)
        elif validations:
            representative = min(validations, key=lambda rid: (lookup[rid]["source_row"], rid))
            for rid in validations:
                if rid != representative:
                    to_remove[rid] = ("near_duplicate_confirmed_within_validation", representative)
            for rid in trains:
                to_remove[rid] = ("near_duplicate_confirmed_lower_priority", representative)
        else:
            representative = min(trains, key=lambda rid: (lookup[rid]["source_row"], rid))
            for rid in trains:
                if rid != representative:
                    to_remove[rid] = ("near_duplicate_confirmed_within_train", representative)

    # Инвариант приоритета: удаляемая строка никогда не выше представителя.
    for rid, (_, representative) in to_remove.items():
        if SPLIT_PRIORITY[lookup[rid]["split"]] > SPLIT_PRIORITY[lookup[representative]["split"]]:
            raise AssertionError(
                f"T03 invariant violated: {rid} ({lookup[rid]['split']}) would be removed in favour of "
                f"{representative} ({lookup[representative]['split']})"
            )

    label_conflicts = sum(
        1 for rid, (_, rep) in to_remove.items() if lookup[rid]["label_name"] != lookup[rep]["label_name"]
    )

    within_test_pairs = int(
        (duplicate_pairs["left_split"].eq("test") & duplicate_pairs["right_split"].eq("test")).sum()
    )

    result: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    for split, df in frames.items():
        remove_ids = set(df["record_id"]) & set(to_remove)
        if split == "test" and remove_ids:
            raise AssertionError("T03 invariant violated: test rows must never be removed")
        for _, row in df.loc[df["record_id"].isin(remove_ids)].iterrows():
            reason, related = to_remove[row["record_id"]]
            row = row.copy()
            row["_split"] = split
            exclusions.append(_exclusion_row(row, reason, related))
        result[split] = df.loc[~df["record_id"].isin(remove_ids)].copy()

    stats = {
        "near_pairs_confirmed_duplicate": int(len(duplicate_pairs)),
        "near_rows_removed": int(len(to_remove)),
        "near_test_rows_preserved": int(preserved_test),
        "near_pairs_within_test": within_test_pairs,
        "near_components": int(len(components)),
        "near_components_larger_than_pair": int(larger_than_pair),
        "near_indirect_pairs_in_components": int(indirect),
        "near_removed_rows_with_label_conflict": int(label_conflicts),
    }
    return result, exclusions, stats


# --------------------------------------------------------------------------
# Инварианты и сводки
# --------------------------------------------------------------------------

def overlap_counts(frames: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a = set(frames[left][column].astype(str))
        b = set(frames[right][column].astype(str))
        result[f"{left}__{right}"] = len(a & b)
    return result


def assert_final_invariants(frames: dict[str, pd.DataFrame], target_language: str = "ru") -> None:
    movie_overlap = overlap_counts(frames, "movie_id")
    if any(movie_overlap.values()):
        raise ValueError(f"movie_id leakage remains after preparation: {movie_overlap}")

    exact_overlap = overlap_counts(frames, "_exact_key")
    if any(exact_overlap.values()):
        raise ValueError(f"Normalized exact-text leakage remains after preparation: {exact_overlap}")

    for split, df in frames.items():
        if not df["review_language"].eq(target_language).all():
            raise ValueError(f"{split}: non-{target_language} rows remain")
        if not set(df["label_id"].unique()).issubset({0, 1, 2}):
            raise ValueError(f"{split}: invalid label_id values")
        if not set(df["label_name"].unique()).issubset(set(LABEL_ORDER)):
            raise ValueError(f"{split}: invalid label_name values")
        if not df["record_id"].is_unique:
            raise ValueError(f"{split}: record_id values are not unique")


def stratified_subsample_ids(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or size >= len(df):
        out = df[["record_id", "label_name", "label_id"]].copy()
        return out.reset_index(drop=True)

    counts = df["label_name"].value_counts().sort_index()
    raw = counts / counts.sum() * size
    quotas = raw.apply(math.floor).astype(int)
    remainder = size - int(quotas.sum())
    if remainder > 0:
        fractional = (raw - quotas).sort_values(ascending=False, kind="mergesort")
        for label in fractional.index[:remainder]:
            quotas.loc[label] += 1

    sampled_parts = []
    for i, label in enumerate(sorted(quotas.index)):
        group = df.loc[df["label_name"].eq(label)]
        n = min(int(quotas.loc[label]), len(group))
        sampled_parts.append(group.sample(n=n, random_state=seed + i))

    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) < size:
        remaining = df.loc[~df.index.isin(sampled.index)]
        sampled = pd.concat([sampled, remaining.sample(n=size - len(sampled), random_state=seed + 1000)])
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled[["record_id", "label_name", "label_id"]]


def split_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLITS:
        df = frames[split]
        counts = df["label_name"].value_counts().reindex(list(LABEL_ORDER), fill_value=0)
        summary[split] = {
            "rows": int(len(df)),
            "negative": int(counts["negative"]),
            "neutral": int(counts["neutral"]),
            "positive": int(counts["positive"]),
            "unique_movie_id": int(df["movie_id"].nunique()),
        }
    return summary


def render_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# T03 — аудит подготовки данных",
        "",
        "Отчёт генерируется детерминированно командой "
        "`python src/prepare_data.py --config configs/data_prep.json`.",
        "",
        f"- Версия конвейера: **{audit['pipeline_version']}**",
        f"- Статус запуска: **{audit['status']}**",
        "",
        "## Контрольные суммы исходных файлов",
        "",
        "| Split | SHA-256 до | SHA-256 после | Не изменён |",
        "|---|---|---|---|",
    ]
    for split in SPLITS:
        s = audit["source_files"][split]
        lines.append(f"| {split} | `{s['sha256_before']}` | `{s['sha256_after']}` | {s['unchanged']} |")

    lines += [
        "",
        "## Итоговые split'ы",
        "",
        "| Split | Строк | Negative | Neutral | Positive | Фильмов |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        s = audit["final_summary"][split]
        lines.append(
            f"| {split} | {s['rows']} | {s['negative']} | {s['neutral']} | {s['positive']} | {s['unique_movie_id']} |"
        )

    lines += [
        "",
        "## Утечки после очистки",
        "",
        f"- Пересечения `movie_id`: `{audit['final_movie_id_overlap']}`",
        f"- Пересечения нормализованных точных текстов: `{audit['final_exact_text_overlap']}`",
        "",
        "## Точная дедупликация",
        "",
    ]
    for key, value in sorted(audit["exact_dedup"].items()):
        lines.append(f"- {key}: **{value}**")

    lines += ["", "## Конфликты меток", ""]
    lc = audit["label_conflicts"]
    lines.append(f"- Групп «одинаковый текст, разные метки»: **{lc['label_conflict_groups']}**")
    lines.append(f"- Строк в таких группах: **{lc['label_conflict_rows']}**")
    lines.append(f"- Из них групп внутри одного split: **{lc['label_conflict_groups_within_split']}**")
    lines.append(f"- Из них групп между split'ами: **{lc['label_conflict_groups_cross_split']}**")
    lines.append(
        "- Конфликты не скрываются дедупликацией: полный список идентификаторов — "
        "`reports/data_prep/label_conflicts.csv` (без текстов отзывов)."
    )

    lines += ["", "## Похожие тексты", ""]
    nd = audit["near_duplicates"]
    lines.append(f"- Зафиксированный порог Jaccard char-n-gram: **{nd['threshold']}**")
    lines.append(f"- Размер n-граммы: **{nd['ngram_size']}**")
    lines.append(f"- Найдено кандидатов для ручного аудита: **{nd['candidate_pairs']}**")
    lines.append(f"- Кандидатов БЕЗ ручного решения (не обработаны): **{nd['candidates_without_decision']}**")
    lines.append(f"- Подтверждено вручную как копии: **{nd['confirmed_duplicate_pairs']}**")
    lines.append(f"- Удалено строк по подтверждённым near-duplicates: **{nd['rows_removed']}**")
    lines.append(f"- Подтверждённых пар целиком внутри test: **{nd['near_pairs_within_test']}**")
    lines.append(f"- Строк test, сохранённых в near-duplicate компонентах: **{nd['near_test_rows_preserved']}**")
    lines.append(f"- Компонент связности: **{nd['near_components']}**")
    lines.append(f"- Компонент крупнее пары: **{nd['near_components_larger_than_pair']}**")
    lines.append(
        f"- Пар внутри компонент БЕЗ прямого подтверждения человеком: **{nd['near_indirect_pairs_in_components']}**"
    )
    lines.append(
        f"- Удалённых строк, метка которых отличается от представителя: "
        f"**{nd['near_removed_rows_with_label_conflict']}**"
    )
    lines.append(f"- Блоков, пропущенных при поиске: **{nd['near_blocks_skipped_large']}** (блоки не отбрасываются)")
    lines.append(f"- Самый большой блок: **{nd['near_largest_block_size']}** записей")
    lines.append(f"- Пар, сопоставленных внутри блоков: **{nd['near_block_pairs_evaluated']}**")
    lines.append(f"- Список кандидатов обрезан по max_candidates: **{nd['near_candidates_truncated']}**")
    lines.append(
        "- Высокое сходство само по себе не удаляет запись: удаляются только пары, "
        "отмеченные `duplicate` в файле решений."
    )

    lines += ["", "## Артефакты", ""]
    for item in audit["artifacts"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def save_prepared_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_cols = [c for c in df.columns if str(c).startswith("_")]
    out = df.drop(columns=drop_cols).copy()
    leftover = [c for c in out.columns if str(c).startswith("_")]
    if leftover:
        raise AssertionError(f"Internal columns leaked into prepared output: {leftover}")
    out.to_parquet(path, index=False)


# --------------------------------------------------------------------------
# Конвейер
# --------------------------------------------------------------------------

def run(config_path: Path, project_root_override: str | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    project_root = resolve_project_root(config_path, project_root_override)
    config = load_config(config_path)

    output_dir = project_root / config["output_dir"]
    report_dir = project_root / config["report_dir"]
    artifact_dir = project_root / config["artifact_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    audit_json = report_dir / "audit.json"
    audit_md = report_dir / "audit_report.md"

    # Аудит помечается незавершённым СРАЗУ: упавший запуск не должен оставлять
    # рядом свежие кандидаты и правдоподобный, но устаревший отчёт.
    _write_json(audit_json, {"status": "in_progress", "pipeline_version": PIPELINE_VERSION})
    _write_text(
        audit_md,
        "# T03 — аудит подготовки данных\n\n"
        "**Запуск не завершён.** Этот файл будет перезаписан по успешном завершении\n"
        "`python src/prepare_data.py --config configs/data_prep.json`.\n",
    )

    source_frames, source_files = load_source_splits(config, project_root)
    assert_output_paths_safe(source_files, output_dir, report_dir, artifact_dir)

    source_summary = {split: {"rows": int(len(df))} for split, df in source_frames.items()}
    ru_frames = filter_and_map_labels(source_frames, config)
    ru_summary_before_cleaning = split_summary(ru_frames)

    label_conflict_report, label_conflict_stats = detect_label_conflicts(ru_frames)

    exact_clean, exact_exclusions, exact_stats = exact_deduplicate(ru_frames)

    near_settings = config["near_duplicates"]
    candidates, near_search_stats = find_near_duplicate_candidates(exact_clean, near_settings)
    review_artifact = build_near_review_artifact(candidates, exact_clean)

    # Кандидаты и артефакт ревью пишутся до применения решений: без них человек
    # не может выполнить ручной аудит при первом запуске.
    candidates_report_path = report_dir / "near_duplicate_candidates.csv"
    review_artifact_path = artifact_dir / "near_duplicate_review.csv"
    _write_csv(candidates, candidates_report_path)
    _write_csv(review_artifact, review_artifact_path)

    decision_path = project_root / near_settings["decision_file"]
    decisions = read_near_duplicate_decisions(decision_path, set(candidates["pair_id"]))

    # Кандидат без решения не удаляется — и это обязано быть видно, иначе
    # новая пара молча остаётся непросмотренной.
    undecided = candidates.loc[~candidates["pair_id"].isin(decisions)].copy()

    final_frames, near_exclusions, near_stats = apply_confirmed_near_duplicates(
        exact_clean, candidates, decisions
    )

    assert_final_invariants(final_frames, config["target_language"])

    excluded = pd.DataFrame(
        exact_exclusions + near_exclusions,
        columns=["record_id", "split", "reason", "related_record_id"],
    )
    if not excluded.empty:
        excluded = excluded.sort_values(["split", "record_id", "reason"], kind="mergesort")

    split_ids_rows = []
    for split in SPLITS:
        tmp = final_frames[split][["record_id", "label_name", "label_id"]].copy()
        tmp.insert(1, "split", split)
        split_ids_rows.append(tmp)
    split_ids = pd.concat(split_ids_rows, ignore_index=True)

    prepared_paths = {}
    for split in SPLITS:
        path = output_dir / f"{split}.parquet"
        save_prepared_split(final_frames[split], path)
        prepared_paths[split] = _rel(path, project_root)

    subsample_cfg = config["subsample"]
    subsample_path = report_dir / "train_subsample_ids.csv"
    subsample_enabled = bool(subsample_cfg["enabled"])
    subsample_size = int(subsample_cfg["size"])
    if subsample_enabled and subsample_size > 0:
        sample = stratified_subsample_ids(final_frames["train"], subsample_size, int(subsample_cfg["seed"]))
        _write_csv(sample, subsample_path)
    elif subsample_path.exists():
        subsample_path.unlink()

    source_hashes: dict[str, dict[str, Any]] = {}
    for src in source_files:
        after = sha256_file(src.path)
        source_hashes[src.split] = {
            "path": _rel(src.path, project_root),
            "sha256_before": src.sha256_before,
            "sha256_after": after,
            "unchanged": src.sha256_before == after,
        }
        if src.sha256_before != after:
            raise AssertionError(f"Source file was modified: {src.path}")

    excluded_path = report_dir / "excluded_ids.csv"
    split_ids_path = report_dir / "split_ids.csv"
    label_conflicts_path = report_dir / "label_conflicts.csv"
    undecided_path = report_dir / "undecided_candidates.csv"
    _write_csv(excluded, excluded_path)
    _write_csv(split_ids, split_ids_path)
    _write_csv(label_conflict_report, label_conflicts_path)
    _write_csv(
        undecided[["pair_id", "left_record_id", "left_split", "right_record_id", "right_split", "similarity"]],
        undecided_path,
    )

    audit = {
        "status": "completed",
        "pipeline_version": PIPELINE_VERSION,
        "config": config,
        "source_files": source_hashes,
        "source_summary": source_summary,
        "ru_summary_before_cleaning": ru_summary_before_cleaning,
        "label_conflicts": label_conflict_stats,
        "exact_dedup": exact_stats,
        "near_duplicates": {
            "threshold": near_settings["threshold"],
            "ngram_size": near_settings["ngram_size"],
            "candidate_pairs": int(len(candidates)),
            "manual_decisions_total": int(len(decisions)),
            "candidates_without_decision": int(len(undecided)),
            "confirmed_duplicate_pairs": int(near_stats["near_pairs_confirmed_duplicate"]),
            "rows_removed": int(near_stats["near_rows_removed"]),
            **{k: v for k, v in near_stats.items() if k not in {"near_pairs_confirmed_duplicate", "near_rows_removed"}},
            **near_search_stats,
        },
        "excluded_rows": int(len(excluded)),
        "final_summary": split_summary(final_frames),
        "final_movie_id_overlap": overlap_counts(final_frames, "movie_id"),
        "final_exact_text_overlap": overlap_counts(final_frames, "_exact_key"),
        "prepared_files": prepared_paths,
        "subsample": {
            "enabled": subsample_enabled,
            "size": subsample_size if subsample_enabled else 0,
            "seed": int(subsample_cfg["seed"]),
            "path": _rel(subsample_path, project_root) if subsample_path.exists() else None,
        },
        "artifacts": [
            _rel(excluded_path, project_root),
            _rel(split_ids_path, project_root),
            _rel(label_conflicts_path, project_root),
            _rel(undecided_path, project_root),
            _rel(candidates_report_path, project_root),
            _rel(review_artifact_path, project_root),
            _rel(audit_json, project_root),
            _rel(audit_md, project_root),
        ],
    }

    _write_json(audit_json, audit)
    _write_text(audit_md, render_audit_markdown(audit))

    print("T03 data preparation completed.")
    print(f"Prepared data: {output_dir}")
    print(f"Audit report: {audit_md}")
    print(f"Near-duplicate human-review artifact: {review_artifact_path}")
    if label_conflict_stats["label_conflict_groups"]:
        print(
            f"NOTE: {label_conflict_stats['label_conflict_groups']} identical-text group(s) carry "
            f"conflicting sentiment labels. See {label_conflicts_path}."
        )
    if len(undecided):
        print(
            f"WARNING: {len(undecided)} near-duplicate candidate(s) have no manual decision and were "
            f"therefore NOT acted upon. Review them in "
            f"{review_artifact_path.name} and record decisions in {decision_path.name}. "
            f"List: {undecided_path}",
            file=sys.stderr,
        )
    return audit


def main() -> None:
    args = parse_args()
    run(Path(args.config), args.project_root)


if __name__ == "__main__":
    main()
