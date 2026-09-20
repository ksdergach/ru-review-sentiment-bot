"""Тесты T03.

Большинство тестов работает на синтетических DataFrame и не требует pyarrow.
Интеграционный тест создаёт настоящие Parquet-файлы и помечен skipif:
в облачной среде ревьюера pyarrow недоступен, поэтому соответствующая
runtime-проверка там отмечена как NOT VERIFIED IN CLOUD ENVIRONMENT.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from src.prepare_data import (
    apply_confirmed_near_duplicates,
    assert_final_invariants,
    build_near_review_artifact,
    detect_label_conflicts,
    exact_deduplicate,
    exact_normalize,
    filter_and_map_labels,
    find_near_duplicate_candidates,
    overlap_counts,
    read_near_duplicate_decisions,
    stable_pair_id,
    stratified_subsample_ids,
    validate_config,
    validate_source_dataframe,
)

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None

NEAR_SETTINGS = {
    "enabled": True,
    "threshold": 0.85,
    "ngram_size": 3,
    "min_chars": 40,
    "prefix_chars": 20,
    "suffix_chars": 20,
    "length_ratio_min": 0.8,
    "max_block_pairs": 2000000,
    "max_candidates": 100,
}


def _frame(rows, filename="synthetic.parquet"):
    df = pd.DataFrame(rows)
    df["source_file"] = filename
    df["source_row"] = range(len(df))
    df["record_id"] = [f"{filename}#row={i:08d}" for i in range(len(df))]
    return df


def _config():
    return {
        "target_language": "ru",
        "class_mapping": {"negative": 0, "neutral": 1, "positive": 2},
    }


def _row(movie_id, text, sentiment="POSITIVE", language="ru"):
    return {
        "movie_id": movie_id,
        "review_text": text,
        "review_sentiment": sentiment,
        "review_language": language,
    }


def _mapped(train=(), validation=(), test=()):
    frames = {
        "train": _frame(list(train) or [_row("t0", "Базовый обучающий отзыв про фильм номер один.")], "train.parquet"),
        "validation": _frame(
            list(validation) or [_row("v0", "Базовый проверочный отзыв про фильм номер два.")], "validation.parquet"
        ),
        "test": _frame(list(test) or [_row("x0", "Базовый тестовый отзыв про фильм номер три.")], "test.parquet"),
    }
    return filter_and_map_labels(frames, _config())


def _full_config():
    return {
        "input_dir": "NLP_dataset",
        "files": {
            "train": "train-00000-of-00001.parquet",
            "validation": "validation-00000-of-00001.parquet",
            "test": "test-00000-of-00001.parquet",
        },
        "output_dir": "data/processed",
        "report_dir": "reports/data_prep",
        "artifact_dir": "artifacts/data_prep",
        "expected_columns": ["movie_id", "review_text", "review_sentiment", "review_language"],
        "allowed_languages": ["ru", "kk", "cs"],
        "target_language": "ru",
        "class_mapping": {"negative": 0, "neutral": 1, "positive": 2},
        "near_duplicates": {
            "enabled": True,
            "threshold": 0.90,
            "ngram_size": 3,
            "min_chars": 40,
            "prefix_chars": 24,
            "suffix_chars": 24,
            "length_ratio_min": 0.85,
            "max_block_pairs": 2000000,
            "max_candidates": 5000,
            "decision_file": "configs/near_duplicate_decisions.csv",
        },
        "subsample": {"enabled": False, "size": 0, "seed": 42},
    }


# --------------------------------------------------------------------------
# Язык, метки, сохранность текста
# --------------------------------------------------------------------------

def test_language_mapping_and_model_text_preserved():
    text = "Фильм  ОЧЕНЬ хороший! 😊"
    frames = {
        "train": _frame([
            _row("m1", text, "POSITIVE", "ru"),
            _row("m2", "Жаман", "NEGATIVE", "kk"),
        ]),
        "validation": _frame([_row("m3", "Смешанные чувства", " Neutral ", "ru")]),
        "test": _frame([_row("m4", "Плохо.", "negative", "ru")]),
    }
    out = filter_and_map_labels(frames, _config())
    assert len(out["train"]) == 1
    assert out["train"].iloc[0]["review_text"] == text
    assert out["train"].iloc[0]["label_name"] == "positive"
    assert out["train"].iloc[0]["label_id"] == 2
    assert out["validation"].iloc[0]["label_id"] == 1
    assert out["test"].iloc[0]["label_id"] == 0


def test_class_mapping_is_fixed():
    frames = {s: _frame([_row("m", "Текст отзыва")]) for s in ("train", "validation", "test")}
    bad = {"target_language": "ru", "class_mapping": {"negative": 2, "neutral": 1, "positive": 0}}
    with pytest.raises(ValueError, match="fixed class mapping"):
        filter_and_map_labels(frames, bad)


def test_unsupported_sentiment_label_raises():
    frames = {
        "train": _frame([_row("m1", "Текст", "MIXED", "ru")]),
        "validation": _frame([_row("m2", "Текст два")]),
        "test": _frame([_row("m3", "Текст три")]),
    }
    with pytest.raises(ValueError, match="unsupported sentiment labels"):
        filter_and_map_labels(frames, _config())


def test_non_target_language_is_excluded_entirely():
    frames = {
        "train": _frame([_row("m1", "Русский отзыв", "POSITIVE", "ru"), _row("m2", "Cesky", "POSITIVE", "cs")]),
        "validation": _frame([_row("m3", "Ещё отзыв")]),
        "test": _frame([_row("m4", "И ещё")]),
    }
    out = filter_and_map_labels(frames, _config())
    assert out["train"]["review_language"].tolist() == ["ru"]


# --------------------------------------------------------------------------
# Валидация источника
# --------------------------------------------------------------------------

def _validate(df):
    validate_source_dataframe(
        df,
        split="train",
        expected_columns=["movie_id", "review_text", "review_sentiment", "review_language"],
        allowed_languages={"ru", "kk", "cs"},
    )


def test_missing_required_column_raises():
    df = _frame([_row("m1", "Текст")]).drop(columns=["movie_id"])
    with pytest.raises(ValueError, match="missing columns"):
        _validate(df)


def test_null_in_required_column_raises():
    df = _frame([_row("m1", "Текст")])
    df.loc[0, "review_sentiment"] = None
    with pytest.raises(ValueError, match="null values"):
        _validate(df)


def test_empty_review_text_raises():
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _validate(_frame([_row("m1", "")]))


def test_whitespace_only_review_text_raises():
    # Без этой проверки все такие строки схлопнулись бы в один _exact_key.
    for blank in ("   ", "\t\n", "  ", "​"):
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            _validate(_frame([_row("m1", blank)]))


def test_unexpected_language_raises():
    with pytest.raises(ValueError, match="unexpected review_language"):
        _validate(_frame([_row("m1", "Текст", "POSITIVE", "en")]))


def test_non_string_review_text_raises():
    df = _frame([_row("m1", "Текст")])
    df["review_text"] = pd.Series([42], dtype=object)
    with pytest.raises(ValueError, match="not strings"):
        _validate(df)


# --------------------------------------------------------------------------
# Нормализация и Unicode
# --------------------------------------------------------------------------

def test_exact_normalize_only_case_and_whitespace_for_ascii_punctuation():
    assert exact_normalize("Фильм   ХОРОШИЙ!!! 😊") == "фильм хороший!!! 😊"


def test_exact_normalize_applies_nfc():
    composed = "Фильм добрый"          # й как один код-пойнт
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert exact_normalize(composed) == exact_normalize(decomposed)


def test_exact_normalize_strips_zero_width_characters():
    assert exact_normalize("фильм​хороший") == exact_normalize("фильмхороший")


def test_unicode_equivalent_texts_are_caught_by_exact_dedup():
    """Регрессия: канонически одинаковый текст в train и validation — это утечка.

    До исправления такие пары не ловились exact-дедупликацией и попадали в
    near-duplicate кандидаты, то есть зависели от порога и min_chars.
    """
    composed = "Фильм добрый, есть смешные моменты, правда затянули чуток"
    decomposed = unicodedata.normalize("NFD", composed)
    frames = _mapped(
        train=[_row("m1", decomposed)],
        validation=[_row("m2", composed)],
    )
    cleaned, exclusions, stats = exact_deduplicate(frames)
    assert len(cleaned["train"]) == 0
    assert len(cleaned["validation"]) == 1
    assert stats["removed_exact_train_vs_eval"] == 1
    assert exclusions[0]["reason"] == "exact_cross_train_eval"


def test_review_text_is_never_rewritten_by_normalization():
    raw = "  Фильм​   ХОРОШИЙ!!! 😊  "
    frames = _mapped(train=[_row("m1", raw)])
    assert frames["train"].iloc[0]["review_text"] == raw
    assert frames["train"].iloc[0]["_exact_key"] != raw


# --------------------------------------------------------------------------
# Exact dedup и приоритет split'ов
# --------------------------------------------------------------------------

def test_exact_duplicate_priority_and_test_is_immutable():
    train = _frame([
        _row("t1", "ФИЛЬМ   классный!"),
        _row("t2", "Повтор внутри train"),
        _row("t3", "повтор   внутри TRAIN"),
    ])
    validation = _frame([
        _row("v1", "Фильм классный!"),
        _row("v2", "Валидация уникальна", "NEUTRAL"),
    ])
    test = _frame([
        _row("x1", "фильм классный!"),
        _row("x2", "Тестовый дубль", "NEUTRAL"),
        _row("x3", "ТЕСТОВЫЙ   ДУБЛЬ", "NEUTRAL"),
    ])
    frames = filter_and_map_labels({"train": train, "validation": validation, "test": test}, _config())
    cleaned, exclusions, stats = exact_deduplicate(frames)
    assert len(cleaned["test"]) == 3
    assert "Фильм классный!" not in cleaned["validation"]["review_text"].tolist()
    assert "ФИЛЬМ   классный!" not in cleaned["train"]["review_text"].tolist()
    assert len(cleaned["train"]) == 1
    assert stats["test_exact_duplicate_extra_rows"] == 1
    assert stats["test_exact_duplicate_groups"] == 1
    reasons = {r["reason"] for r in exclusions}
    assert "exact_cross_validation_test" in reasons
    assert "exact_cross_train_eval" in reasons
    assert "exact_within_train" in reasons


def test_exact_duplicate_validation_loses_to_test():
    frames = _mapped(
        validation=[_row("v1", "Совпадающий отзыв про фильм")],
        test=[_row("x1", "совпадающий   ОТЗЫВ про фильм")],
    )
    cleaned, exclusions, stats = exact_deduplicate(frames)
    assert len(cleaned["validation"]) == 0
    assert len(cleaned["test"]) == 1
    assert stats["removed_exact_validation_vs_test"] == 1
    assert exclusions[0]["reason"] == "exact_cross_validation_test"


def test_exact_duplicate_within_validation_keeps_first_source_row():
    frames = _mapped(validation=[
        _row("v1", "Повтор внутри валидации"),
        _row("v2", "ПОВТОР   внутри валидации"),
    ])
    cleaned, exclusions, stats = exact_deduplicate(frames)
    assert len(cleaned["validation"]) == 1
    assert cleaned["validation"].iloc[0]["source_row"] == 0
    assert stats["removed_exact_within_validation"] == 1
    assert exclusions[0]["reason"] == "exact_within_validation"
    assert exclusions[0]["related_record_id"] == "validation.parquet#row=00000000"


def test_exact_dedup_never_removes_test_rows():
    frames = _mapped(test=[
        _row("x1", "Одинаковый тестовый отзыв"),
        _row("x2", "ОДИНАКОВЫЙ  тестовый отзыв"),
    ])
    cleaned, exclusions, stats = exact_deduplicate(frames)
    assert len(cleaned["test"]) == 2
    assert stats["test_exact_duplicate_extra_rows"] == 1
    assert all(not r["reason"].startswith("exact_within_test") for r in exclusions)


# --------------------------------------------------------------------------
# Конфликты меток
# --------------------------------------------------------------------------

def test_label_conflict_same_text_different_labels_is_detected():
    frames = _mapped(
        train=[_row("m1", "Один и тот же длинный текст отзыва", "POSITIVE")],
        test=[_row("m2", "один и тот же ДЛИННЫЙ текст отзыва", "NEGATIVE")],
    )
    report, stats = detect_label_conflicts(frames)
    assert stats["label_conflict_groups"] == 1
    assert stats["label_conflict_rows"] == 2
    assert stats["label_conflict_groups_cross_split"] == 1
    assert set(report["scope"]) == {"cross_split"}
    assert "review_text" not in report.columns  # отчёт идёт в Git: текстов быть не должно


def test_label_conflict_within_split_is_detected():
    frames = _mapped(train=[
        _row("m1", "Повторяющийся отзыв", "POSITIVE"),
        _row("m2", "повторяющийся ОТЗЫВ", "NEGATIVE"),
    ])
    report, stats = detect_label_conflicts(frames)
    assert stats["label_conflict_groups"] == 1
    assert stats["label_conflict_groups_within_split"] == 1
    assert len(report) == 2


def test_no_label_conflict_when_labels_agree():
    frames = _mapped(train=[
        _row("m1", "Повторяющийся отзыв", "POSITIVE"),
        _row("m2", "повторяющийся ОТЗЫВ", "POSITIVE"),
    ])
    report, stats = detect_label_conflicts(frames)
    assert stats["label_conflict_groups"] == 0
    assert report.empty


# --------------------------------------------------------------------------
# Near-duplicates
# --------------------------------------------------------------------------

LONG_A = "Очень хороший фильм с прекрасной актёрской игрой и отличным финалом. Рекомендую всем."
LONG_B = "Очень хороший фильм с прекрасной актёрской игрой и отличным финалом! Рекомендую всем."


def test_near_duplicate_requires_manual_confirmation():
    frames = _mapped(
        train=[_row("t1", LONG_A)],
        validation=[_row("v1", LONG_B)],
        test=[_row("x1", "Совсем другой длинный отзыв о фильме, который не похож на предыдущий текст.", "NEGATIVE")],
    )
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    assert len(candidates) == 1

    unchanged, exclusions, stats = apply_confirmed_near_duplicates(frames, candidates, {})
    assert len(unchanged["train"]) == 1
    assert len(unchanged["validation"]) == 1
    assert stats["near_rows_removed"] == 0
    assert exclusions == []

    pair_id = candidates.iloc[0]["pair_id"]
    cleaned, exclusions, stats = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "duplicate"})
    assert len(cleaned["train"]) == 0
    assert len(cleaned["validation"]) == 1
    assert stats["near_rows_removed"] == 1
    assert exclusions[0]["reason"] == "near_duplicate_confirmed_lower_priority"


def test_not_duplicate_decision_preserves_both_rows():
    frames = _mapped(train=[_row("t1", LONG_A)], validation=[_row("v1", LONG_B)])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    pair_id = candidates.iloc[0]["pair_id"]
    cleaned, exclusions, stats = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "not_duplicate"})
    assert len(cleaned["train"]) == 1
    assert len(cleaned["validation"]) == 1
    assert stats["near_rows_removed"] == 0
    assert exclusions == []


def test_confirmed_near_duplicate_never_removes_test_rows():
    frames = _mapped(test=[_row("x1", LONG_A, "NEGATIVE"), _row("x2", LONG_B, "NEGATIVE")])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    assert len(candidates) == 1
    pair_id = candidates.iloc[0]["pair_id"]
    cleaned, exclusions, stats = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "duplicate"})
    assert len(cleaned["test"]) == 2
    assert stats["near_rows_removed"] == 0
    assert stats["near_test_rows_preserved"] == 2
    assert stats["near_pairs_within_test"] == 1


def test_confirmed_near_duplicate_train_loses_to_test():
    frames = _mapped(train=[_row("t1", LONG_A, "NEGATIVE")], test=[_row("x1", LONG_B, "NEGATIVE")])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    pair_id = candidates.iloc[0]["pair_id"]
    cleaned, _, stats = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "duplicate"})
    assert len(cleaned["train"]) == 0
    assert len(cleaned["test"]) == 1
    assert stats["near_test_rows_preserved"] == 1


def test_near_duplicate_within_train_keeps_lowest_source_row():
    frames = _mapped(train=[_row("t1", LONG_A), _row("t2", LONG_B)])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    pair_id = candidates.iloc[0]["pair_id"]
    cleaned, exclusions, _ = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "duplicate"})
    assert cleaned["train"]["source_row"].tolist() == [0]
    assert exclusions[0]["reason"] == "near_duplicate_confirmed_within_train"
    assert exclusions[0]["related_record_id"] == "train.parquet#row=00000000"


def test_connected_components_report_indirect_pairs():
    """Транзитивность: A≈B и B≈C подтверждены, пара A–C — нет.

    Такая компонента удаляет запись, которую человек напрямую дубликатом не
    подтверждал. Поведение сохранено, но факт обязан попасть в аудит.
    """
    a = "Совершенно замечательный фильм про дружбу и приключения в горах aaaa"
    b = "Совершенно замечательный фильм про дружбу и приключения в горах bbbb"
    c = "Совершенно замечательный фильм про дружбу и приключения в горах cccc"
    frames = _mapped(train=[_row("m1", a), _row("m2", b), _row("m3", c)])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    ids = {
        "ab": stable_pair_id("train.parquet#row=00000000", "train.parquet#row=00000001"),
        "bc": stable_pair_id("train.parquet#row=00000001", "train.parquet#row=00000002"),
    }
    present = set(candidates["pair_id"])
    assert ids["ab"] in present and ids["bc"] in present
    decisions = {ids["ab"]: "duplicate", ids["bc"]: "duplicate"}
    cleaned, _, stats = apply_confirmed_near_duplicates(frames, candidates, decisions)
    assert stats["near_components"] == 1
    assert stats["near_components_larger_than_pair"] == 1
    assert stats["near_indirect_pairs_in_components"] == 1
    assert len(cleaned["train"]) == 1


def test_removed_near_duplicate_with_conflicting_label_is_counted():
    frames = _mapped(train=[_row("t1", LONG_A, "POSITIVE"), _row("t2", LONG_B, "NEGATIVE")])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    pair_id = candidates.iloc[0]["pair_id"]
    _, _, stats = apply_confirmed_near_duplicates(frames, candidates, {pair_id: "duplicate"})
    assert stats["near_removed_rows_with_label_conflict"] == 1


def test_near_duplicate_search_requires_precomputed_key():
    """Регрессия: itertuples переименовывает колонки с подчёркиванием.

    Старый код читал row._near_key через itertuples, промахивался и молча
    уходил в запасную ветку. Теперь отсутствие колонки — явная ошибка.
    """
    frames = _mapped(train=[_row("t1", LONG_A)], validation=[_row("v1", LONG_B)])
    frames["train"] = frames["train"].drop(columns=["_near_key"])
    with pytest.raises(ValueError, match="_near_key column is missing"):
        find_near_duplicate_candidates(frames, NEAR_SETTINGS)


def test_near_duplicate_search_uses_the_normalised_key():
    """Ключ поиска — нормализованный текст, а не сырой review_text."""
    frames = _mapped(train=[_row("t1", "  " + LONG_A.upper() + "  ")], validation=[_row("v1", LONG_B)])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    assert len(candidates) == 1


def test_near_duplicate_search_is_deterministic():
    frames = _mapped(train=[_row("t1", LONG_A), _row("t2", LONG_B)], validation=[_row("v1", LONG_A + " Ещё раз.")])
    first, stats_a = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    second, stats_b = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    pd.testing.assert_frame_equal(first, second)
    assert stats_a == stats_b


def test_short_texts_below_min_chars_are_not_candidates():
    # Оба train-текста короче min_chars=40 и вообще не попадают в поиск,
    # хотя их similarity была бы выше порога.
    frames = _mapped(train=[_row("t1", "Отличный фильм!"), _row("t2", "Отличный фильм.")])
    candidates, stats = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    assert candidates.empty
    considered_train = [r for r in frames["train"]["_near_key"] if len(r) >= NEAR_SETTINGS["min_chars"]]
    assert considered_train == []
    # учтены только два длинных текста по умолчанию (validation и test)
    assert stats["near_records_considered"] == 2


def test_large_blocks_are_never_skipped_silently():
    """Регрессия: раньше крупный блок отбрасывался целиком и пары терялись.

    На реальных данных это стоило двух настоящих near-duplicate пар.
    """
    common = "Фильм очень понравился, всем советую посмотреть обязательно "
    rows = [_row(f"m{i}", common + f"вариант номер {i:03d} с отдельным хвостом") for i in range(12)]
    frames = _mapped(train=rows)
    candidates, stats = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    assert stats["near_blocks_skipped_large"] == 0
    assert stats["near_largest_block_size"] >= 12
    assert stats["near_block_pairs_evaluated"] >= 12 * 11 // 2
    # все 12 текстов делят префикс, значит пары внутри блока реально посчитаны
    assert len(candidates) > 0


def test_oversized_block_raises_instead_of_skipping():
    common = "Фильм очень понравился, всем советую посмотреть обязательно "
    rows = [_row(f"m{i}", common + f"вариант номер {i:03d} с отдельным хвостом") for i in range(12)]
    frames = _mapped(train=rows)
    settings = dict(NEAR_SETTINGS, max_block_pairs=5)
    with pytest.raises(ValueError, match="max_block_pairs"):
        find_near_duplicate_candidates(frames, settings)


def test_near_review_artifact_exposes_labels_and_movie_ids():
    frames = _mapped(train=[_row("m1", LONG_A, "POSITIVE")], validation=[_row("m2", LONG_B, "NEGATIVE")])
    candidates, _ = find_near_duplicate_candidates(frames, NEAR_SETTINGS)
    artifact = build_near_review_artifact(candidates, frames)
    row = artifact.iloc[0]
    assert {"left_label", "right_label", "labels_conflict", "left_movie_id", "right_movie_id"} <= set(artifact.columns)
    assert bool(row["labels_conflict"]) is True
    assert bool(row["same_movie_id"]) is False


def test_pair_id_is_stable_and_order_independent():
    assert stable_pair_id("a", "b") == stable_pair_id("b", "a")
    assert stable_pair_id(
        "train-00000-of-00001.parquet#row=00008603",
        "train-00000-of-00001.parquet#row=00008746",
    ) == "1b617879404ca482"


# --------------------------------------------------------------------------
# Файл решений
# --------------------------------------------------------------------------

def _decisions_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "near_duplicate_decisions.csv"
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def test_decisions_file_reads_utf8_bom_and_valid_values(tmp_path):
    path = _decisions_file(tmp_path, "﻿pair_id,decision,note\nabc,duplicate,ok\ndef,not_duplicate,\n")
    assert read_near_duplicate_decisions(path, {"abc", "def"}) == {"abc": "duplicate", "def": "not_duplicate"}


def test_decisions_file_rejects_invalid_decision(tmp_path):
    path = _decisions_file(tmp_path, "pair_id,decision\nabc,maybe\n")
    with pytest.raises(ValueError, match="Invalid near-duplicate decision"):
        read_near_duplicate_decisions(path, {"abc"})


def test_decisions_file_rejects_unknown_pair_and_lists_all(tmp_path):
    path = _decisions_file(tmp_path, "pair_id,decision\nabc,duplicate\nzzz,duplicate\nyyy,duplicate\n")
    with pytest.raises(ValueError, match="yyy.*zzz|zzz.*yyy") as excinfo:
        read_near_duplicate_decisions(path, {"abc"})
    assert "2 decision(s)" in str(excinfo.value)


def test_decisions_file_rejects_conflicting_duplicate_rows(tmp_path):
    path = _decisions_file(tmp_path, "pair_id,decision\nabc,duplicate\nabc,not_duplicate\n")
    with pytest.raises(ValueError, match="Conflicting decisions"):
        read_near_duplicate_decisions(path, {"abc"})


def test_missing_decisions_file_is_empty(tmp_path):
    assert read_near_duplicate_decisions(tmp_path / "nope.csv", {"abc"}) == {}


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

def test_valid_config_passes():
    validate_config(_full_config())


def test_config_missing_required_key_raises():
    cfg = _full_config()
    del cfg["target_language"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_config(cfg)


def test_config_missing_near_duplicate_key_is_not_silently_defaulted():
    cfg = _full_config()
    del cfg["near_duplicates"]["ngram_size"]
    with pytest.raises(ValueError, match="near_duplicates.ngram_size"):
        validate_config(cfg)


def test_config_threshold_out_of_range_raises():
    cfg = _full_config()
    cfg["near_duplicates"]["threshold"] = 1.5
    with pytest.raises(ValueError, match="threshold must be <= 1"):
        validate_config(cfg)


def test_config_wrong_type_raises():
    cfg = _full_config()
    cfg["near_duplicates"]["ngram_size"] = "3"
    with pytest.raises(ValueError, match="must be int"):
        validate_config(cfg)


def test_config_rejects_altered_class_mapping():
    cfg = _full_config()
    cfg["class_mapping"] = {"negative": 0, "neutral": 1, "positive": 2, "mixed": 3}
    with pytest.raises(ValueError, match="fixed class mapping"):
        validate_config(cfg)


def test_config_rejects_unknown_top_level_key():
    cfg = _full_config()
    cfg["balance_validation"] = True
    with pytest.raises(ValueError, match="unknown top-level keys"):
        validate_config(cfg)


def test_config_target_language_must_be_allowed():
    cfg = _full_config()
    cfg["target_language"] = "en"
    with pytest.raises(ValueError, match="must be listed in"):
        validate_config(cfg)


def test_config_subsample_enabled_requires_positive_size():
    cfg = _full_config()
    cfg["subsample"] = {"enabled": True, "size": 0, "seed": 42}
    with pytest.raises(ValueError, match="size is not positive"):
        validate_config(cfg)


def test_config_default_disables_subsample():
    assert _full_config()["subsample"]["enabled"] is False


# --------------------------------------------------------------------------
# Подвыборка
# --------------------------------------------------------------------------

def _subsample_frame():
    rows = []
    for label_name, label_id, count in [("negative", 0, 20), ("neutral", 1, 10), ("positive", 2, 70)]:
        for i in range(count):
            rows.append({"record_id": f"{label_name}-{i}", "label_name": label_name, "label_id": label_id})
    return pd.DataFrame(rows)


def test_stratified_subsample_is_reproducible():
    df = _subsample_frame()
    a = stratified_subsample_ids(df, size=30, seed=42)
    b = stratified_subsample_ids(df, size=30, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30
    assert a["label_name"].value_counts().to_dict() == {"positive": 21, "negative": 6, "neutral": 3}


def test_stratified_subsample_uses_existing_record_ids():
    df = _subsample_frame()
    sample = stratified_subsample_ids(df, size=30, seed=42)
    assert set(sample["record_id"]) <= set(df["record_id"])
    assert sample["record_id"].is_unique


def test_stratified_subsample_full_when_disabled_size():
    df = _subsample_frame()
    assert len(stratified_subsample_ids(df, size=0, seed=42)) == len(df)


# --------------------------------------------------------------------------
# Инварианты и утечки
# --------------------------------------------------------------------------

def test_overlap_counts():
    frames = {
        "train": pd.DataFrame({"movie_id": ["a", "b"]}),
        "validation": pd.DataFrame({"movie_id": ["c"]}),
        "test": pd.DataFrame({"movie_id": ["d"]}),
    }
    assert overlap_counts(frames, "movie_id") == {
        "train__validation": 0,
        "train__test": 0,
        "validation__test": 0,
    }


def test_assert_final_invariants_detects_movie_overlap():
    frames = _mapped(train=[_row("shared", "Отзыв про фильм один")], validation=[_row("shared", "Другой отзыв")])
    with pytest.raises(ValueError, match="movie_id leakage"):
        assert_final_invariants(frames)


def test_assert_final_invariants_detects_exact_text_leakage():
    frames = _mapped(train=[_row("m1", "Одинаковый текст отзыва")], validation=[_row("m2", "ОДИНАКОВЫЙ текст отзыва")])
    with pytest.raises(ValueError, match="exact-text leakage"):
        assert_final_invariants(frames)


def test_assert_final_invariants_accepts_clean_frames():
    assert_final_invariants(_mapped())


# --------------------------------------------------------------------------
# Интеграционный тест на настоящем Parquet
# --------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow is not installed in this environment")
def test_end_to_end_pipeline_on_real_parquet(tmp_path):
    """NOT VERIFIED IN CLOUD ENVIRONMENT: требует pyarrow."""
    from src.prepare_data import run

    root = tmp_path
    (root / "NLP_dataset").mkdir()
    (root / "configs").mkdir()

    def write(split, rows):
        pd.DataFrame(rows).to_parquet(root / "NLP_dataset" / f"{split}-00000-of-00001.parquet", index=False)

    long_a = "Очень хороший фильм с прекрасной актёрской игрой и отличным финалом. Рекомендую."
    long_b = "Очень хороший фильм с прекрасной актёрской игрой и отличным финалом! Рекомендую."
    write("train", [
        _row("m1", "Уникальный обучающий отзыв про первый фильм", "POSITIVE"),
        _row("m1", "УНИКАЛЬНЫЙ   обучающий отзыв про первый фильм", "POSITIVE"),   # повтор внутри train
        _row("m2", long_a, "POSITIVE"),
        _row("m9", "Казахский отзыв", "POSITIVE", "kk"),                            # не ru
    ])
    write("validation", [_row("m3", "Проверочный отзыв про третий фильм", "NEUTRAL")])
    write("test", [
        _row("m4", long_b, "POSITIVE"),
        _row("m5", "Тестовый отзыв про пятый фильм", "NEGATIVE"),
    ])

    cfg = _full_config()
    config_path = root / "configs" / "data_prep.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8", newline="\n")

    sources = sorted((root / "NLP_dataset").glob("*.parquet"))
    before = {p.name: p.read_bytes() for p in sources}

    audit_one = run(config_path, str(root))
    assert audit_one["status"] == "completed"
    # near-duplicate кандидат найден, но решения нет -> ничего не удалено
    assert audit_one["near_duplicates"]["candidate_pairs"] == 1
    assert audit_one["near_duplicates"]["rows_removed"] == 0
    assert audit_one["exact_dedup"]["removed_exact_within_train"] == 1
    assert audit_one["final_summary"]["test"]["rows"] == 2

    for path in sources:
        assert path.read_bytes() == before[path.name], "исходный Parquet изменён"

    prepared = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    assert not [c for c in prepared.columns if str(c).startswith("_")]
    assert "review_text" in prepared.columns and "record_id" in prepared.columns
    assert prepared["review_language"].eq("ru").all()

    split_ids = pd.read_csv(root / "reports" / "data_prep" / "split_ids.csv")
    assert set(split_ids.loc[split_ids["split"].eq("train"), "record_id"]) == set(prepared["record_id"])

    # Подтверждаем near-duplicate: train уступает test
    pair_id = pd.read_csv(root / "reports" / "data_prep" / "near_duplicate_candidates.csv").iloc[0]["pair_id"]
    (root / "configs" / "near_duplicate_decisions.csv").write_text(
        f"pair_id,decision,note\n{pair_id},duplicate,integration test\n", encoding="utf-8", newline="\n"
    )

    audit_two = run(config_path, str(root))
    assert audit_two["near_duplicates"]["rows_removed"] == 1
    assert audit_two["final_summary"]["test"]["rows"] == 2

    # Повторный запуск при неизменных входах даёт побайтово те же отчёты
    reports = root / "reports" / "data_prep"
    snapshot = {p.name: p.read_bytes() for p in sorted(reports.glob("*"))}
    run(config_path, str(root))
    for name, content in snapshot.items():
        assert (reports / name).read_bytes() == content, f"{name} не воспроизводится побайтово"


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow is not installed in this environment")
def test_pipeline_refuses_to_write_over_source_directory(tmp_path):
    """NOT VERIFIED IN CLOUD ENVIRONMENT: требует pyarrow."""
    from src.prepare_data import run

    root = tmp_path
    (root / "NLP_dataset").mkdir()
    (root / "configs").mkdir()
    for split in ("train", "validation", "test"):
        pd.DataFrame([_row(f"m-{split}", f"Отзыв про фильм {split} достаточно длинный")]).to_parquet(
            root / "NLP_dataset" / f"{split}-00000-of-00001.parquet", index=False
        )
    cfg = copy.deepcopy(_full_config())
    cfg["output_dir"] = "NLP_dataset"
    config_path = root / "configs" / "data_prep.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="Unsafe configuration"):
        run(config_path, str(root))
