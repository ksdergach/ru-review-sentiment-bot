from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # работает и с `pytest`, и с `python -m pytest` из корня репозитория
    sys.path.insert(0, str(ROOT))

import src.train_baseline as tb  # noqa: E402
from src.train_baseline import (  # noqa: E402
    EXPECTED_LABEL_MAPPING,
    apply_train_id_selection,
    build_classifier,
    build_vectorizer,
    compute_train_class_weight,
    ids_sha256,
    load_bundle,
    load_config,
    load_t03_audit,
    make_bundle,
    normalize_text_key,
    predict_texts,
    resolve_path,
    run_training,
    save_bundle,
    select_winner,
    smoke_test,
    validate_against_t03_audit,
    validate_config,
    validate_prepared_frame,
    validate_split_ids_against_t03,
    validate_split_independence,
    validation_confusion_matrix,
    validation_metrics,
)

NAMES = ["negative", "neutral", "positive"]
POOLS = {
    0: ["ужасный", "скучно", "плохо", "разочарование", "слабый", "зря"],
    1: ["обычный", "средний", "нормально", "ничего", "особенного", "посредственный"],
    2: ["отличный", "прекрасно", "рекомендую", "великолепно", "замечательный", "удовольствие"],
}
ALPHABET = "абвгдежзиклмнопрстуфхцчшэюя"
KW = {"text_column": "review_text", "label_column": "label_id", "id_column": "record_id"}


# --------------------------------------------------------------------------- #
# Синтетические данные и проект во временной папке
# --------------------------------------------------------------------------- #
def _letters(i: int) -> str:
    out = ""
    for _ in range(3):
        i, r = divmod(i, len(ALPHABET))
        out += ALPHABET[r]
    return out


def make_split(prefix: str, counts: tuple[int, int, int], *, offset: int, movie_base: int) -> pd.DataFrame:
    rows, n = [], 0
    for label, count in enumerate(counts):
        pool = POOLS[label]
        for k in range(count):
            text = f"Фильм {pool[k % 6]} {pool[(k // 6 + 1) % 6]} {_letters(offset + n)}"
            rows.append((f"{prefix}#row={n:08d}", text, label, NAMES[label], f"m{movie_base + n}"))
            n += 1
    frame = pd.DataFrame(rows, columns=["record_id", "review_text", "label_id", "label_name", "movie_id"])
    return frame.assign(review_language="ru")


def tiny_train() -> pd.DataFrame:
    return make_split("train.parquet", (12, 4, 30), offset=0, movie_base=0)


def tiny_validation() -> pd.DataFrame:
    return make_split("validation.parquet", (8, 4, 20), offset=5000, movie_base=1000)


def valid_config() -> dict:
    return {
        "seed": 42,
        "train_path": "data/processed/train.parquet",
        "validation_path": "data/processed/validation.parquet",
        "train_ids_path": None,
        "data_prep_audit_path": "reports/data_prep/audit.json",
        "data_prep_split_ids_path": "reports/data_prep/split_ids.csv",
        "report_dir": "reports/baseline",
        "model_dir": "models/baseline",
        "bundle_name": "tfidf_logreg_baseline.joblib",
        "label_mapping": EXPECTED_LABEL_MAPPING.copy(),
        "text_column": "review_text",
        "label_column": "label_id",
        "id_column": "record_id",
        "text_processing": {"manual_normalization": "none", "truncation": None},
        "tfidf": {
            "analyzer": "word",
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": None,
            "sublinear_tf": True,
            "norm": "l2",
            "token_pattern": r"(?u)\b\w\w+\b",
        },
        "logistic_regression": {
            "C_values": [0.5, 1.0],
            "solver": "lbfgs",
            "max_iter": 300,
            "tol": 0.0001,
            "class_weight_mode": "balanced_train",
        },
    }


def _summary(frame: pd.DataFrame) -> dict:
    counts = frame["label_id"].value_counts()
    return {"rows": len(frame), **{n: int(counts.get(i, 0)) for i, n in enumerate(NAMES)}}


def write_t03_support(
    tmp_path: Path, train: pd.DataFrame, validation: pd.DataFrame, *, audit: dict | None = None
) -> None:
    report_dir = tmp_path / "reports" / "data_prep"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "final_summary": {"train": _summary(train), "validation": _summary(validation)},
        "subsample": {"enabled": False, "size": 0, "seed": 42, "path": None},
        "source_files": {
            "train": {"sha256_after": "a" * 64},
            "validation": {"sha256_after": "b" * 64},
        },
    }
    payload.update(audit or {})
    (report_dir / "audit.json").write_text(json.dumps(payload), encoding="utf-8")
    test_rows = pd.DataFrame(
        {"record_id": ["test.parquet#row=00000000"], "split": ["test"], "label_name": ["negative"], "label_id": [0]}
    )
    split_ids = pd.concat(
        [
            train[["record_id", "label_name", "label_id"]].assign(split="train"),
            validation[["record_id", "label_name", "label_id"]].assign(split="validation"),
            test_rows,
        ],
        ignore_index=True,
    )[["record_id", "split", "label_name", "label_id"]]
    split_ids.to_csv(report_dir / "split_ids.csv", index=False)


def prepare_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
    config: dict | None = None,
    audit: dict | None = None,
) -> tuple[Path, list[str]]:
    train = tiny_train() if train is None else train
    validation = tiny_validation() if validation is None else validation
    config = valid_config() if config is None else config
    config_path = tmp_path / "configs" / "baseline.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    write_t03_support(tmp_path, train, validation, audit=audit)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.parquet").write_bytes(b"fake-train")
    (data_dir / "validation.parquet").write_bytes(b"fake-validation")
    (data_dir / "test.parquet").write_bytes(b"THIS-MUST-NOT-BE-READ")

    reads: list[str] = []

    def fake_read_parquet(path, *args, **kwargs):
        name = Path(path).name
        reads.append(name)
        if name == "train.parquet":
            return train.copy()
        if name == "validation.parquet":
            return validation.copy()
        raise AssertionError(f"Unexpected Parquet read: {name}")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    return config_path, reads


def read_reports(report_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in report_dir.iterdir() if p.is_file()}


def fitted_bundle(config: dict | None = None) -> dict:
    config = valid_config() if config is None else config
    train = tiny_train()
    vectorizer = build_vectorizer(config["tfidf"])
    x = vectorizer.fit_transform(train["review_text"].tolist())
    weights = compute_train_class_weight(train["label_id"].to_numpy(), "balanced_train")
    classifier = build_classifier(
        c_value=1.0, lr_config=config["logistic_regression"], class_weight=weights, seed=42
    )
    classifier.fit(x, train["label_id"].to_numpy())
    return make_bundle(
        vectorizer=vectorizer,
        classifier=classifier,
        tfidf_config=config["tfidf"],
        lr_params={"C": 1.0},
        training={"seed": 42},
    )


# --------------------------------------------------------------------------- #
# Конфигурация: test-защита, строгая схема, значения
# --------------------------------------------------------------------------- #
def test_valid_config_and_repository_config_are_accepted() -> None:
    validate_config(valid_config())
    repo_config = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
    if repo_config.exists():
        load_config(repo_config)


def _add_top(key: str, value: object = "x"):
    def mutate(cfg: dict) -> None:
        cfg[key] = value

    return mutate


def _add_nested(*path: str):
    def mutate(cfg: dict) -> None:
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = "x"

    return mutate


def _set_value(key: str, value: object):
    def mutate(cfg: dict) -> None:
        cfg[key] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _add_top("test_path"),
        _add_top("test_file"),
        _add_top("final_test"),
        _add_top("testPath"),
        _add_top("test-dataset"),
        _add_nested("evaluation", "test"),
        _add_nested("data", "test"),
        _add_nested("paths", "test"),
        _add_nested("tfidf", "test_file"),
        _set_value("validation_path", "data/processed/test.parquet"),
        _set_value("train_path", "data/processed/test-00000-of-00001.parquet"),
        _set_value("train_ids_path", "reports/test_ids.csv"),
        _set_value("data_prep_split_ids_path", "reports\\data_prep\\Test\\split_ids.csv"),
    ],
)
def test_config_rejects_every_way_to_smuggle_final_test(mutate) -> None:
    config = valid_config()
    mutate(config)
    with pytest.raises(ValueError, match="final test"):
        validate_config(config)


def test_config_does_not_reject_words_that_merely_contain_test() -> None:
    config = valid_config()
    config["validation_path"] = "data/processed/latest_validation.parquet"
    config["report_dir"] = "reports/contest"
    validate_config(config)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_add_top("unexpected_key"), "Unknown key"),
        (_add_nested("tfidf", "vocabulary"), "Unknown key"),
        (_add_nested("tfidf", "tokenizer"), "Unknown key"),
        (_add_nested("logistic_regression", "penalty"), "Unknown key"),
        (_add_nested("text_processing", "lemmatize"), "Unknown key"),
        (_set_value("train_path", "C:\\Users\\someone\\train.parquet"), "relative"),
        (_set_value("train_path", "/abs/train.parquet"), "relative"),
        (_set_value("model_dir", "../outside"), "leave the repository"),
        (_set_value("bundle_name", "sub/model.joblib"), "plain file name"),
        (_set_value("validation_path", "data\\processed\\train.parquet"), "must be different"),
        (_set_value("seed", "42"), "seed"),
        (_set_value("seed", True), "seed"),
        (_set_value("seed", -1), "seed"),
        (_set_value("label_mapping", {"negative": 1, "neutral": 0, "positive": 2}), "label_mapping must be exactly"),
        (_set_value("label_mapping", {"negative": False, "neutral": True, "positive": 2}), "label_mapping must be exactly"),
        (_set_value("id_column", "review_text"), "must be different"),
    ],
)
def test_config_strict_schema_and_paths(mutate, message) -> None:
    config = valid_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    "c_values,message",
    [
        ([1.0], "exactly 2 or 3"),
        ([], "exactly 2 or 3"),
        ([0.1, 0.2, 0.3, 0.4], "exactly 2 or 3"),
        ([1.0, 1.0], "unique"),
        ([1.0, 1], "unique"),
        ([0, 1.0], "> 0"),
        ([-1.0, 1.0], "> 0"),
        ([float("nan"), 1.0], "finite"),
        ([float("inf"), 1.0], "finite"),
        (["1", 2.0], "finite number"),
        ([True, 2.0], "finite number"),
    ],
)
def test_config_c_values_are_validated(c_values, message) -> None:
    config = valid_config()
    config["logistic_regression"]["C_values"] = c_values
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("logistic_regression", "solver", "liblinear", "Unsupported"),
        ("logistic_regression", "class_weight_mode", "auto", "class_weight_mode"),
        ("logistic_regression", "max_iter", 0, "max_iter"),
        ("logistic_regression", "max_iter", 10.5, "max_iter"),
        ("logistic_regression", "tol", 0, "tol"),
        ("tfidf", "analyzer", "char", "word-level"),
        ("tfidf", "ngram_range", [2, 1], "ngram_range"),
        ("tfidf", "ngram_range", [0, 1], "ngram_range"),
        ("tfidf", "min_df", 0, "min_df"),
        ("tfidf", "max_df", 1.5, "max_df"),
        ("tfidf", "norm", "max", "norm"),
        ("tfidf", "lowercase", "yes", "lowercase"),
        ("tfidf", "max_features", 0, "max_features"),
        ("tfidf", "token_pattern", "(", "token_pattern"),
        ("tfidf", "token_pattern", "(\\w+)", "capturing"),
        ("text_processing", "manual_normalization", "lower", "manual_normalization"),
        ("text_processing", "truncation", 512, "truncation"),
    ],
)
def test_config_model_and_tfidf_parameters_are_validated(section, key, value, message) -> None:
    config = valid_config()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_config_with_utf8_bom_can_be_loaded(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(valid_config()), encoding="utf-8-sig")
    assert load_config(path)["seed"] == 42


def test_resolve_path_stays_inside_repository(tmp_path: Path) -> None:
    assert resolve_path(tmp_path.resolve(), "reports/x.csv") == (tmp_path / "reports" / "x.csv").resolve()
    with pytest.raises(ValueError, match="leaves the repository"):
        resolve_path(tmp_path.resolve(), "../x.csv")


# --------------------------------------------------------------------------- #
# Контракт данных T03
# --------------------------------------------------------------------------- #
def test_prepared_frame_accepts_t03_shape() -> None:
    validate_prepared_frame(tiny_train(), split_name="train", **KW)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda f: f.drop(columns=["label_name"]), "missing required column"),
        (lambda f: f.drop(columns=["review_language"]), "missing required column"),
        (lambda f: f.drop(columns=["movie_id"]), "missing required column"),
        (lambda f: f.query("label_id != 1"), "all three classes"),
        (lambda f: f.assign(label_id=f["label_id"].where(f.index != 0, 3)), "invalid label_id"),
        (lambda f: f.assign(label_id=f["label_id"].astype(float).where(f.index != 0, 0.5)), "exact integers"),
        (lambda f: f.assign(label_name=f["label_name"].where(f.index != 0, "positive")), "inconsistent"),
        (lambda f: f.assign(record_id=f["record_id"].where(f.index != 0, "  ")), "blank record_id"),
        (lambda f: f.assign(record_id=f["record_id"].where(f.index != 1, f["record_id"].iloc[0])), "duplicate record_id"),
        (lambda f: f.assign(review_text=f["review_text"].where(f.index != 0, "   ")), "empty/whitespace"),
        (lambda f: f.assign(review_text=f["review_text"].astype(object).where(f.index != 0, 5)), "only strings"),
        (lambda f: f.assign(review_language=f["review_language"].where(f.index != 0, "kk")), "review_language"),
        (lambda f: f.assign(movie_id=f["movie_id"].where(f.index != 0, None)), "null values"),
    ],
)
def test_prepared_frame_rejects_violations_before_training(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prepared_frame(mutate(tiny_train()), split_name="train", **KW)


def test_split_independence_detects_record_id_and_movie_overlap() -> None:
    train, validation = tiny_train(), tiny_validation()
    bad = validation.copy()
    bad.loc[0, "record_id"] = train.loc[0, "record_id"]
    with pytest.raises(ValueError, match="record_id overlap"):
        validate_split_independence(train, bad, id_column="record_id", text_column="review_text")
    bad = validation.copy()
    bad.loc[0, "movie_id"] = train.loc[0, "movie_id"]
    with pytest.raises(ValueError, match="movie_id overlap"):
        validate_split_independence(train, bad, id_column="record_id", text_column="review_text")


def test_split_independence_detects_normalized_text_overlap() -> None:
    train, validation = tiny_train(), tiny_validation()
    original = train.loc[0, "review_text"]
    variants = [
        original.upper(),  # регистр
        "  " + original.replace(" ", "  ") + " ",  # пробелы
        original.replace("й", "и\u0306"),  # разложенная «й» (NFD)
        original + "\u200b",  # невидимый символ
    ]
    for variant in variants:
        bad = validation.copy()
        bad.loc[0, "review_text"] = variant
        assert normalize_text_key(variant) == normalize_text_key(original)
        with pytest.raises(ValueError, match="normalized review_text overlap"):
            validate_split_independence(train, bad, id_column="record_id", text_column="review_text")


def test_t03_audit_must_be_completed(tmp_path: Path) -> None:
    for payload in ({"status": "in_progress"}, {"final_summary": {}}):  # второй — старый audit без status
        path = tmp_path / "audit.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="must be 'completed'"):
            load_t03_audit(path)


def test_prepared_data_must_match_audit_summary() -> None:
    train, validation = tiny_train(), tiny_validation()
    audit = {"final_summary": {"train": _summary(train), "validation": _summary(validation)}}
    kw = {"train": train, "validation": validation, "label_column": "label_id", "train_ids_configured": False}
    validate_against_t03_audit(audit, **kw)

    stale = copy.deepcopy(audit)
    stale["final_summary"]["train"]["rows"] += 1
    with pytest.raises(ValueError, match="does not match T03 audit"):
        validate_against_t03_audit(stale, **kw)

    stale = copy.deepcopy(audit)
    stale["final_summary"]["validation"]["neutral"] -= 1
    with pytest.raises(ValueError, match="does not match T03 audit"):
        validate_against_t03_audit(stale, **kw)

    with pytest.raises(ValueError, match="no final_summary"):
        validate_against_t03_audit({"status": "completed"}, **kw)


def test_enabled_t03_subsample_requires_train_ids_path() -> None:
    train, validation = tiny_train(), tiny_validation()
    audit = {
        "final_summary": {"train": _summary(train), "validation": _summary(validation)},
        "subsample": {"enabled": True, "path": "reports/data_prep/train_subsample_ids.csv"},
    }
    kw = {"train": train, "validation": validation, "label_column": "label_id"}
    with pytest.raises(ValueError, match="train_ids_path is not set"):
        validate_against_t03_audit(audit, train_ids_configured=False, **kw)
    validate_against_t03_audit(audit, train_ids_configured=True, **kw)


def test_t03_split_ids_must_match_prepared_order_and_labels(tmp_path: Path) -> None:
    train, validation = tiny_train(), tiny_validation()
    write_t03_support(tmp_path, train, validation)
    path = tmp_path / "reports" / "data_prep" / "split_ids.csv"
    kw = {"train": train, "validation": validation, "id_column": "record_id", "label_column": "label_id"}
    report = validate_split_ids_against_t03(path, **kw)
    assert report["train_rows"] == len(train) and report["validation_rows"] == len(validation)

    rows = pd.read_csv(path)
    swapped = rows.copy()
    a, b = swapped.index[swapped["split"] == "train"][:2]
    swapped.loc[[a, b], "record_id"] = swapped.loc[[b, a], "record_id"].to_numpy()
    swapped.to_csv(path, index=False)
    with pytest.raises(ValueError, match="order/content mismatch"):
        validate_split_ids_against_t03(path, **kw)

    relabeled = rows.copy()
    first_train = relabeled.index[relabeled["split"] == "train"][0]
    relabeled.loc[first_train, "label_id"] = 1
    relabeled.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label mismatch"):
        validate_split_ids_against_t03(path, **kw)

    rows.iloc[:-1].query("split != 'validation'").to_csv(path, index=False)
    with pytest.raises(ValueError, match="mismatch"):
        validate_split_ids_against_t03(path, **kw)


def test_test_rows_of_split_ids_never_affect_validation_of_train_and_validation(tmp_path: Path) -> None:
    train, validation = tiny_train(), tiny_validation()
    write_t03_support(tmp_path, train, validation)
    path = tmp_path / "reports" / "data_prep" / "split_ids.csv"
    rows = pd.read_csv(path, dtype=str)
    rows.loc[rows["split"] == "test", ["record_id", "label_id"]] = ["garbage", "999"]
    rows.to_csv(path, index=False)
    validate_split_ids_against_t03(
        path, train=train, validation=validation, id_column="record_id", label_column="label_id"
    )


# --------------------------------------------------------------------------- #
# Общий train / subset
# --------------------------------------------------------------------------- #
def test_train_id_selection_preserves_order_and_checks_content(tmp_path: Path) -> None:
    train = tiny_train()
    ids = [train.loc[40, "record_id"], train.loc[3, "record_id"], train.loc[14, "record_id"], train.loc[20, "record_id"]]
    path = tmp_path / "ids.csv"
    pd.DataFrame({"record_id": ids}).to_csv(path, index=False)
    selected = apply_train_id_selection(train, path, **KW)
    assert selected["record_id"].tolist() == ids

    def select(frame: pd.DataFrame):
        frame.to_csv(path, index=False)
        return apply_train_id_selection(train, path, **KW)

    with pytest.raises(ValueError, match="all three classes"):
        select(pd.DataFrame({"record_id": ids[:2]}))
    with pytest.raises(ValueError, match="absent from prepared train"):
        select(pd.DataFrame({"record_id": ids + ["validation.parquet#row=00000000"]}))
    with pytest.raises(ValueError, match="duplicate IDs"):
        select(pd.DataFrame({"record_id": ids + ids[:1]}))
    with pytest.raises(ValueError, match="blank IDs"):
        select(pd.DataFrame({"record_id": ids + [" "], "note": ["a"] * 5}))
    with pytest.raises(ValueError, match="labels disagree"):
        select(pd.DataFrame({"record_id": ids, "label_id": [2, 2, 2, 2]}))
    only_train = pd.DataFrame({"record_id": ids + ["x"], "split": ["train"] * 4 + ["validation"]})
    assert select(only_train)["record_id"].tolist() == ids


def test_ids_sha_depends_on_order_and_labels() -> None:
    train = tiny_train()
    digest = ids_sha256(train, id_column="record_id", label_column="label_id")
    assert digest != ids_sha256(train.iloc[::-1].reset_index(drop=True), id_column="record_id", label_column="label_id")
    changed = train.copy()
    changed.loc[0, "label_id"] = 1
    assert digest != ids_sha256(changed, id_column="record_id", label_column="label_id")


# --------------------------------------------------------------------------- #
# Метрики и выбор победителя
# --------------------------------------------------------------------------- #
def test_metrics_keep_rare_class_and_fixed_label_order() -> None:
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 0, 2, 2])  # neutral ни разу не предсказан
    m = validation_metrics(y_true, y_pred)
    assert m["neutral_precision"] == 0.0 and m["neutral_recall"] == 0.0 and m["neutral_f1"] == 0.0
    assert m["neutral_support"] == 1
    assert m["negative_precision"] == 0.5 and m["negative_recall"] == 1.0
    assert m["positive_f1"] == 1.0 and m["positive_support"] == 2
    assert abs(m["macro_f1"] - (2 / 3 + 0.0 + 1.0) / 3) < 1e-12
    assert m["accuracy"] == 0.75


def test_macro_f1_counts_all_three_labels_even_if_one_is_absent() -> None:
    m = validation_metrics(np.array([0, 2]), np.array([0, 2]))
    assert abs(m["macro_f1"] - 2 / 3) < 1e-12 and m["neutral_support"] == 0


def test_confusion_matrix_rows_are_true_classes() -> None:
    matrix = validation_confusion_matrix(np.array([0, 0, 1, 2, 2, 2]), np.array([1, 0, 1, 0, 2, 2]))
    assert matrix == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_winner_uses_macro_f1_then_smaller_c_and_ignores_order() -> None:
    results = [
        {"C": 2.0, "macro_f1": 0.80, "neutral_f1": 0.99, "accuracy": 0.99},
        {"C": 1.0, "macro_f1": 0.81, "neutral_f1": 0.10, "accuracy": 0.10},
        {"C": 0.5, "macro_f1": 0.81, "neutral_f1": 0.01, "accuracy": 0.01},
    ]
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]):
        shuffled = [results[i] for i in order]
        assert shuffled[select_winner(shuffled)]["C"] == 0.5
    with pytest.raises(ValueError, match="finite"):
        select_winner([{"C": 1.0, "macro_f1": float("nan")}, {"C": 2.0, "macro_f1": 0.5}])


def test_class_weights_follow_the_balanced_formula_of_the_given_labels() -> None:
    y = np.asarray([0] * 6 + [1] * 2 + [2] * 12)
    weights = compute_train_class_weight(y, "balanced_train")
    assert weights is not None
    for cls, count in ((0, 6), (1, 2), (2, 12)):
        assert abs(weights[cls] - len(y) / (3 * count)) < 1e-12
    assert compute_train_class_weight(y, "none") is None
    with pytest.raises(ValueError, match="all classes"):
        compute_train_class_weight(np.asarray([0, 2, 2]), "balanced_train")


# --------------------------------------------------------------------------- #
# Полный сценарий на синтетических данных (Parquet подменён)
# --------------------------------------------------------------------------- #
def test_run_touches_only_train_and_validation_and_never_test(tmp_path: Path, monkeypatch) -> None:
    config_path, reads = prepare_project(tmp_path, monkeypatch)
    hashed: list[str] = []
    original_sha256_file = tb.sha256_file
    monkeypatch.setattr(tb, "sha256_file", lambda p: (hashed.append(Path(p).name), original_sha256_file(p))[1])
    csv_reads: list[str] = []
    original_read_csv = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: (csv_reads.append(Path(p).name), original_read_csv(p, *a, **k))[1])

    result = run_training(config_path, project_root=tmp_path)

    assert reads == ["train.parquet", "validation.parquet"]
    assert "test.parquet" not in hashed
    assert csv_reads == ["split_ids.csv"]
    assert result["final_test_used"] is False
    assert json.loads((tmp_path / "reports" / "baseline" / "run_metadata.json").read_text(encoding="utf-8"))["final_test_used"] is False


def test_tfidf_and_class_weights_use_train_only_and_no_refit_on_validation(tmp_path: Path, monkeypatch) -> None:
    train, validation = tiny_train(), tiny_validation()
    leak_token = "валидационноеслово"
    validation.loc[0, "review_text"] += f" {leak_token}"
    config = valid_config()
    config_path, _ = prepare_project(tmp_path, monkeypatch, train=train, validation=validation, config=config)
    result = run_training(config_path, project_root=tmp_path)
    bundle = load_bundle(result["bundle_path"])
    metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))

    # 1) словарь и IDF — ровно как при fit на одном train; validation-слово не попало в словарь.
    reference = build_vectorizer(config["tfidf"])
    x_train = reference.fit_transform(train["review_text"].tolist())
    assert leak_token not in bundle["vectorizer"].vocabulary_
    assert tb._vectorizer_state_sha256(reference) == tb._vectorizer_state_sha256(bundle["vectorizer"])
    assert metadata["tfidf_state_sha256"] == tb._vectorizer_state_sha256(reference)

    # 2) веса классов — из распределения train (12/4/30), а не validation (8/4/20).
    counts = np.bincount(train["label_id"].to_numpy())
    expected = {str(i): len(train) / (3 * counts[i]) for i in range(3)}
    assert metadata["class_weights"].keys() == expected.keys()
    for key in expected:
        assert abs(metadata["class_weights"][key] - expected[key]) < 1e-12

    # 3) коэффициенты победителя совпадают с независимой подгонкой только на train
    #    (значит, нет дообучения на train + validation).
    winner_c = metadata["winner_C"]
    independent = build_classifier(
        c_value=winner_c,
        lr_config=config["logistic_regression"],
        class_weight={i: expected[str(i)] for i in range(3)},
        seed=42,
    ).fit(x_train, train["label_id"].to_numpy())
    assert np.allclose(independent.coef_, bundle["classifier"].coef_, atol=1e-8)


def test_vectorizer_state_hash_covers_idf_not_only_vocabulary() -> None:
    train_texts = ["фильм хороший ужасно", "фильм плохой ужасно"]
    first = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b").fit(train_texts)
    # тот же словарь, но IDF посчитан по другим документам (например, с участием validation)
    second = TfidfVectorizer(vocabulary=first.vocabulary_, token_pattern=r"(?u)\b\w\w+\b").fit(
        train_texts + ["фильм фильм фильм", "фильм ужасно"]
    )
    assert first.vocabulary_ == second.vocabulary_
    assert tb._vectorizer_state_sha256(first) != tb._vectorizer_state_sha256(second)


def test_winner_is_the_argmax_of_validation_macro_f1_in_reports(tmp_path: Path, monkeypatch) -> None:
    config = valid_config()
    config["logistic_regression"]["C_values"] = [0.01, 1.0, 100.0]
    config_path, _ = prepare_project(tmp_path, monkeypatch, config=config)
    result = run_training(config_path, project_root=tmp_path)
    table = pd.read_csv(result["validation_results_path"])
    assert len(table) == 3
    best = table.sort_values(["macro_f1", "C"], ascending=[False, True]).iloc[0]
    winner = json.loads(result["winner_config_path"].read_text(encoding="utf-8"))
    assert winner["winner"]["C"] == best["C"]
    assert {"negative_precision", "neutral_recall", "positive_f1", "accuracy"} <= set(table.columns)
    assert winner["validation_confusion_matrix"]["row_order"] == ["negative", "neutral", "positive"]
    assert sum(map(sum, winner["validation_confusion_matrix"]["matrix"])) == len(tiny_validation())
    assert "(winner)" in result["summary_path"].read_text(encoding="utf-8")


def test_validation_is_neither_reduced_nor_balanced_when_train_is_a_subset(tmp_path: Path, monkeypatch) -> None:
    train, validation = tiny_train(), tiny_validation()
    ids = pd.concat([train.iloc[[0, 1, 2, 3]], train.iloc[[12, 13, 14]], train.iloc[[16, 17, 18, 19, 20]]]).sample(frac=1.0, random_state=7)
    config = valid_config()
    config["train_ids_path"] = "reports/data_prep/train_subsample_ids.csv"
    audit = {"subsample": {"enabled": True, "size": len(ids), "seed": 42, "path": config["train_ids_path"]}}
    config_path, _ = prepare_project(tmp_path, monkeypatch, train=train, validation=validation, config=config, audit=audit)
    ids_file = tmp_path / config["train_ids_path"]
    ids[["record_id", "label_name", "label_id"]].to_csv(ids_file, index=False)

    result = run_training(config_path, project_root=tmp_path)
    metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["train_rows"] == len(ids) and metadata["train_is_subset"] is True
    assert metadata["train_full_rows"] == len(train)
    assert metadata["validation_rows"] == len(validation)
    assert metadata["validation_class_distribution"] == {"negative": 8, "neutral": 4, "positive": 20}
    assert metadata["train_selection_source"] == "reports/data_prep/train_subsample_ids.csv"
    assert metadata["train_ids_file_sha256"] == tb.sha256_file(ids_file)

    written = pd.read_csv(result["train_ids_path"])
    assert list(written.columns) == ["train_order", "record_id", "label_id"]
    assert written["record_id"].tolist() == ids["record_id"].tolist()  # порядок сохранён
    assert written["label_id"].tolist() == ids["label_id"].tolist()


def test_full_train_run_is_recorded_as_such_and_train_ids_have_no_text(tmp_path: Path, monkeypatch) -> None:
    train, validation = tiny_train(), tiny_validation()
    config_path, _ = prepare_project(tmp_path, monkeypatch, train=train, validation=validation)
    result = run_training(config_path, project_root=tmp_path)
    metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["train_selection_source"] == "all_prepared_train"
    assert metadata["train_is_subset"] is False and metadata["train_rows"] == len(train)

    reports = read_reports(tmp_path / "reports" / "baseline")
    assert reports["train_ids.csv"].splitlines()[0] == "train_order,record_id,label_id"
    assert len(reports["train_ids.csv"].splitlines()) == len(train) + 1
    for name, content in reports.items():  # ни в одном отчёте нет текстов отзывов
        for text in list(train["review_text"]) + list(validation["review_text"]):
            assert text not in content, f"review text leaked into {name}"


def test_reports_contain_no_absolute_paths(tmp_path: Path, monkeypatch) -> None:
    ids_path = "reports/data_prep/train_subsample_ids.csv"
    config = valid_config()
    config["train_ids_path"] = ids_path
    train = tiny_train()
    audit = {"subsample": {"enabled": True, "path": ids_path}}
    config_path, _ = prepare_project(tmp_path, monkeypatch, config=config, audit=audit)
    train[["record_id", "label_name", "label_id"]].to_csv(tmp_path / ids_path, index=False)
    run_training(config_path, project_root=tmp_path)
    for name, content in read_reports(tmp_path / "reports" / "baseline").items():
        assert tmp_path.as_posix() not in content and str(tmp_path) not in content, name
        assert not re.search(r"[A-Za-z]:\\\\", content), name


def test_run_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    config_path, _ = prepare_project(tmp_path, monkeypatch)
    texts = ["хороший фильм", "плохой фильм", "обычный фильм"]
    first = run_training(config_path, project_root=tmp_path)
    first_files = {
        name: (tmp_path / "reports" / "baseline" / name).read_bytes()
        for name in ("winner_config.json", "train_ids.csv")
    }
    first_pred = predict_texts(load_bundle(first["bundle_path"]), texts)
    second = run_training(config_path, project_root=tmp_path)
    for name, content in first_files.items():
        assert (tmp_path / "reports" / "baseline" / name).read_bytes() == content, name
    assert predict_texts(load_bundle(second["bundle_path"]), texts) == first_pred
    assert first["winner"]["C"] == second["winner"]["C"]


def test_stale_t03_audit_stops_before_any_model_is_written(tmp_path: Path, monkeypatch) -> None:
    train, validation = tiny_train(), tiny_validation()
    audit = {"final_summary": {"train": {**_summary(train), "rows": len(train) + 1}, "validation": _summary(validation)}}
    config_path, _ = prepare_project(tmp_path, monkeypatch, train=train, validation=validation, audit=audit)
    with pytest.raises(ValueError, match="does not match T03 audit"):
        run_training(config_path, project_root=tmp_path)
    assert not (tmp_path / "models").exists() and not (tmp_path / "reports" / "baseline").exists()


def test_train_validation_overlap_stops_before_training(tmp_path: Path, monkeypatch) -> None:
    train, validation = tiny_train(), tiny_validation()
    validation.loc[0, "review_text"] = train.loc[0, "review_text"].upper()
    config_path, _ = prepare_project(tmp_path, monkeypatch, train=train, validation=validation)
    with pytest.raises(ValueError, match="normalized review_text overlap"):
        run_training(config_path, project_root=tmp_path)
    assert not (tmp_path / "models").exists()


def test_convergence_failure_is_recorded_and_not_silent(tmp_path: Path, monkeypatch) -> None:
    config = valid_config()
    config["logistic_regression"]["max_iter"] = 1
    config_path, _ = prepare_project(tmp_path, monkeypatch, config=config)
    result = run_training(config_path, project_root=tmp_path)
    table = pd.read_csv(result["validation_results_path"])
    assert table["convergence_warning"].astype(str).str.lower().eq("true").all()
    assert json.loads(result["metadata_path"].read_text(encoding="utf-8"))["winner_convergence_warning"] is True
    assert "ConvergenceWarning" in result["summary_path"].read_text(encoding="utf-8")


def test_non_convergence_warnings_are_recorded_and_reemitted(tmp_path: Path, monkeypatch) -> None:
    real_fit = LogisticRegression.fit

    def noisy_fit(self, x, y, *args, **kwargs):
        warnings.warn("custom-training-problem", UserWarning)
        return real_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", noisy_fit)
    config_path, _ = prepare_project(tmp_path, monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_training(config_path, project_root=tmp_path)
    table = pd.read_csv(result["validation_results_path"])
    assert table["other_warnings"].str.contains("custom-training-problem").all()
    assert any("custom-training-problem" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# Комплект модели: save / load
# --------------------------------------------------------------------------- #
def test_bundle_roundtrip_keeps_classes_order_and_probabilities(tmp_path: Path) -> None:
    bundle = fitted_bundle()
    fixed = ["очень хороший фильм", "совсем плохой фильм", "обычный фильм", "", "🙂 !!!"]
    before = predict_texts(bundle, fixed)
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)
    loaded = load_bundle(path)
    after = predict_texts(loaded, fixed)
    assert before["label_ids"] == after["label_ids"] and before["labels"] == after["labels"]
    assert np.allclose(before["probabilities"], after["probabilities"], rtol=0, atol=1e-12)
    assert after["probability_class_ids"] == [0, 1, 2]
    assert loaded["label_mapping"] == EXPECTED_LABEL_MAPPING
    assert loaded["library_versions"]["scikit-learn"]


def test_saved_bundle_is_self_contained_in_a_fresh_process(tmp_path: Path) -> None:
    bundle = fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)
    texts = ["Фильм отличный рекомендую", "Фильм ужасный скучно плохо", "Обычный средний фильм"]
    code = (
        "import json, sys, joblib\n"
        "b = joblib.load(sys.argv[1])\n"
        "x = b['vectorizer'].transform(json.loads(sys.argv[2]))\n"
        "print(json.dumps(b['classifier'].predict(x).tolist()))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path), json.dumps(texts)],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    assert json.loads(proc.stdout) == predict_texts(bundle, texts)["label_ids"]


def _break(key: str, value: object):
    def mutate(bundle: dict) -> dict:
        bundle[key] = value
        return bundle

    return mutate


def _drop(key: str):
    def mutate(bundle: dict) -> dict:
        del bundle[key]
        return bundle

    return mutate


def _unfitted_vectorizer(bundle: dict) -> dict:
    bundle["vectorizer"] = TfidfVectorizer()
    return bundle


def _mismatched_vocabulary(bundle: dict) -> dict:
    bundle["vectorizer"] = TfidfVectorizer().fit(["совсем другой словарь слов", "ещё один текст"])
    return bundle


def _classifier_without_neutral(bundle: dict) -> dict:
    train = tiny_train().query("label_id != 1")
    vectorizer = build_vectorizer(valid_config()["tfidf"])
    classifier = build_classifier(c_value=1.0, lr_config=valid_config()["logistic_regression"], class_weight=None, seed=42)
    classifier.fit(vectorizer.fit_transform(train["review_text"].tolist()), train["label_id"].to_numpy())
    bundle.update(vectorizer=vectorizer, classifier=classifier)
    return bundle


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_break("bundle_format_version", 1), "bundle_format_version"),
        (_break("model_type", "other"), "model_type"),
        (_break("label_mapping", {"negative": 1, "neutral": 0, "positive": 2}), "label_mapping"),
        (_break("class_names", ["positive", "neutral", "negative"]), "class_ids/class_names"),
        (_break("text_processing", {"manual_normalization": "none", "truncation": 512}), "text_processing"),
        (_break("vectorizer", "not-a-vectorizer"), "wrong type"),
        (_drop("classifier"), "wrong type"),
        (_drop("training"), "'training'"),
        (_unfitted_vectorizer, "unfitted"),
        (_mismatched_vocabulary, "do not match"),
        (_classifier_without_neutral, "classes_"),
    ],
)
def test_corrupted_bundle_is_rejected_on_load(tmp_path: Path, mutate, message) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump(mutate(fitted_bundle()), path)
    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_bundle_that_is_not_a_dict_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump(["not", "a", "bundle"], path)
    with pytest.raises(ValueError, match="expected dict"):
        load_bundle(path)


ORIGINAL_LOAD_BUNDLE = tb.load_bundle


def _tampered_loader(*, flip_classes: bool = False, intercept_shift: float = 0.0):
    """Загрузчик, «портящий» комплект после чтения: имитирует некорректный save/load."""

    def load(path):
        bundle = ORIGINAL_LOAD_BUNDLE(path)
        if flip_classes:
            bundle["classifier"].coef_ = bundle["classifier"].coef_[::-1].copy()
            bundle["classifier"].intercept_ = bundle["classifier"].intercept_[::-1].copy()
        if intercept_shift:
            bundle["classifier"].intercept_ = bundle["classifier"].intercept_ + np.array([intercept_shift, 0.0, 0.0])
        return bundle

    return load


def test_roundtrip_check_detects_changed_classes_and_probabilities(tmp_path: Path, monkeypatch) -> None:
    bundle = fitted_bundle()
    path = tmp_path / "bundle.joblib"
    save_bundle(path, bundle)
    texts = ["Фильм отличный рекомендую", "Фильм ужасный скучно плохо", "Обычный средний фильм"]
    tb.verify_bundle_roundtrip(bundle, path, texts)  # без искажений — проходит

    monkeypatch.setattr(tb, "load_bundle", _tampered_loader(flip_classes=True))
    with pytest.raises(AssertionError, match="classes"):
        tb.verify_bundle_roundtrip(bundle, path, texts)

    monkeypatch.setattr(tb, "load_bundle", _tampered_loader(intercept_shift=1e-6))
    with pytest.raises(AssertionError, match="probabilities"):
        tb.verify_bundle_roundtrip(bundle, path, texts)


def test_run_fails_if_saved_bundle_does_not_reproduce_predictions(tmp_path: Path, monkeypatch) -> None:
    config_path, _ = prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(tb, "load_bundle", _tampered_loader(flip_classes=True))
    with pytest.raises(AssertionError):
        run_training(config_path, project_root=tmp_path)


def test_display_path_never_exposes_paths_outside_the_repository(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    assert tb.display_path(root / "reports" / "a.csv", root) == "reports/a.csv"
    assert tb.display_path(tmp_path / "elsewhere" / "cfg.json", root) == "cfg.json"


def test_config_outside_repository_does_not_leak_its_absolute_path(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    config_path, _ = prepare_project(project, monkeypatch)
    external = tmp_path / "somewhere_else" / "cfg.json"
    external.parent.mkdir()
    external.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    result = run_training(external, project_root=project)
    metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["config_path"] == "cfg.json"
    for name, content in read_reports(project / "reports" / "baseline").items():
        assert tmp_path.as_posix() not in content, name


def test_smoke_test_needs_no_corpus_and_reads_no_files(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("smoke test must not read data files")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    report = smoke_test()
    assert report["passed"] is True and report["bundle_loadable"] is True
    assert report["probability_class_ids"] == [0, 1, 2]
    assert report["probabilities_max_abs_diff"] <= 1e-12


def test_real_parquet_roundtrip_when_pyarrow_is_available(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    train, validation = tiny_train(), tiny_validation()
    config = valid_config()
    config_path = tmp_path / "configs" / "baseline.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    write_t03_support(tmp_path, train, validation)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    validation.to_parquet(data_dir / "validation.parquet", index=False)
    result = run_training(config_path, project_root=tmp_path)
    assert result["bundle_path"].exists() and result["final_test_used"] is False
