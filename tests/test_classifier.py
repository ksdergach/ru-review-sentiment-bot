from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.classifier as classifier
from src import train_baseline as tb


def fake_prediction(
    label_id: int = 2,
    label: str = "positive",
) -> dict[str, Any]:
    return {
        "label_ids": [label_id],
        "labels": [label],
        "probability_class_ids": [0, 1, 2],
        "probability_labels": ["negative", "neutral", "positive"],
        "probabilities": [[0.10, 0.20, 0.70]],
    }


def make_real_bundle(tmp_path: Path, training_update=None) -> Path:
    """Создать маленький настоящий TF-IDF + LR bundle для интеграционных тестов."""
    texts = [
        "ужасный фильм",
        "плохой скучный фильм",
        "обычный фильм",
        "средний фильм",
        "отличный фильм",
        "прекрасный фильм",
    ]
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=int)

    tfidf_config = {
        "analyzer": "word",
        "lowercase": True,
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": None,
        "sublinear_tf": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "token_pattern": r"(?u)\b\w\w+\b",
    }

    lr_config = {
        "solver": "lbfgs",
        "max_iter": 300,
        "tol": 1e-4,
        "class_weight_mode": "none",
    }

    vectorizer = tb.build_vectorizer(tfidf_config)
    x_train = vectorizer.fit_transform(texts)

    model = tb.build_classifier(
        c_value=1.0,
        lr_config=lr_config,
        class_weight=None,
        seed=42,
    )
    model.fit(x_train, labels)

    bundle = tb.make_bundle(
        vectorizer=vectorizer,
        classifier=model,
        tfidf_config=tfidf_config,
        lr_params={
            **lr_config,
            "C": 1.0,
        },
        training=valid_training(vectorizer),
    )
    if training_update is not None:
        training_update(bundle["training"])

    path = tmp_path / "real_baseline.joblib"
    tb.save_bundle(path, bundle)
    write_report(path, bundle["training"])
    return path


def valid_training(vectorizer) -> dict[str, Any]:
    """Версия данных той же схемы, что пишет train_baseline.run_training."""
    return {
        "seed": 42,
        "train_rows": 6,
        "validation_rows": 2,
        "train_ids_sha256": "1" * 64,
        "validation_ids_sha256": "2" * 64,
        "train_parquet_sha256": "3" * 64,
        "validation_parquet_sha256": "4" * 64,
        "tfidf_state_sha256": tb._vectorizer_state_sha256(vectorizer),
        "class_weight_mode": "none",
        "winner_C": 1.0,
        "selection_metric": "validation_macro_f1",
        "final_test_used": False,
        "split_independence": {"movie_id_overlap": 0, "normalized_text_overlap": 0, "record_id_overlap": 0},
    }


def report_of(model_path: Path) -> Path:
    return model_path.with_name("run_metadata.json")


def write_report(model_path: Path, training: dict[str, Any], **changes: Any) -> Path:
    """Отчёт запуска для искусственного комплекта: тот же формат, что у T04."""
    report = {**training, "bundle_sha256": tb.sha256_file(model_path),
              "config_path": "configs/nlp_baseline.json", **changes}
    tb.write_json(report_of(model_path), report)
    return report_of(model_path)


def baseline(model_path: Path) -> classifier.BaselineClassifier:
    return classifier.BaselineClassifier(model_path, report_path=report_of(model_path))


def predict_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: dict[str, Any]) -> classifier.BaselineClassifier:
    """Настоящий проверенный комплект с подменённым результатом predict_texts."""
    monkeypatch.setattr(classifier, "predict_texts", lambda bundle, texts: raw)
    return baseline(make_real_bundle(tmp_path))


def test_predict_returns_label_confidence_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = predict_with(monkeypatch, tmp_path, fake_prediction())
    result = model.predict("Отличный фильм")

    assert result["label"] == "positive"
    assert result["label_id"] == 2
    assert result["confidence"] == pytest.approx(0.70)
    assert result["probabilities"] == {
        "negative": pytest.approx(0.10),
        "neutral": pytest.approx(0.20),
        "positive": pytest.approx(0.70),
    }


def test_model_is_loaded_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = classifier.load_bundle
    calls = 0

    def counting_load(path: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(classifier, "load_bundle", counting_load)
    model = predict_with(monkeypatch, tmp_path, fake_prediction())

    model.predict("Первый отзыв")
    model.predict("Второй отзыв")

    assert calls == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_empty_text_is_rejected(text: str) -> None:
    model = classifier.BaselineClassifier("missing.joblib")

    with pytest.raises(classifier.EmptyTextError):
        model.predict(text)


def test_text_over_2000_characters_is_rejected() -> None:
    model = classifier.BaselineClassifier("missing.joblib")

    with pytest.raises(classifier.TextTooLongError):
        model.predict("а" * 2001)


def test_whitespace_over_2000_characters_is_still_empty() -> None:
    model = classifier.BaselineClassifier("missing.joblib")

    with pytest.raises(classifier.EmptyTextError):
        model.predict(" " * 2001)


def test_exactly_2000_characters_are_allowed(tmp_path: Path) -> None:
    model = baseline(make_real_bundle(tmp_path))

    result = model.predict("а" * 2000)

    assert result["label"] in classifier.EXPECTED_CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0


def test_missing_model_has_clear_error(tmp_path: Path) -> None:
    model = classifier.BaselineClassifier(
        tmp_path / "missing.joblib"
    )

    with pytest.raises(classifier.ModelUnavailableError, match="FileNotFoundError") as error:
        model.predict("Обычный отзыв")
    assert str(tmp_path / "missing.joblib") in str(error.value)


def test_corrupted_model_has_clear_error(tmp_path: Path) -> None:
    model_path = tmp_path / "broken.joblib"
    model_path.write_bytes(b"this is not a joblib bundle")
    write_report(model_path, {})

    with pytest.raises(classifier.ModelUnavailableError):
        baseline(model_path).predict("Обычный отзыв")


def test_wrong_label_mapping_is_rejected(tmp_path: Path) -> None:
    model_path = make_real_bundle(tmp_path)
    bundle = tb.load_bundle(model_path)
    bundle["label_mapping"] = {"negative": 2, "neutral": 1, "positive": 0}
    tb.save_bundle(model_path, bundle)
    write_report(model_path, bundle["training"])
    with pytest.raises(classifier.ModelUnavailableError, match="label_mapping"):
        baseline(model_path).predict("Обычный отзыв")


def _set(component: str, name: str, value: Any):
    def change(bundle: dict[str, Any]) -> None:
        target = {"vectorizer": bundle["vectorizer"], "weighting": bundle["vectorizer"]._tfidf,
                  "classifier": bundle["classifier"]}.get(component)
        if target is None:
            bundle[component][name] = value
        else:
            setattr(target, name, value)
    return change


@pytest.mark.parametrize("change, reason", [
    (_set("vectorizer", "lowercase", False), "TF-IDF parameters differ.*lowercase"),
    (_set("vectorizer", "lowercase", 1), "TF-IDF parameters differ.*lowercase"),
    (_set("vectorizer", "ngram_range", (1, 2)), "TF-IDF parameters differ.*ngram_range"),
    (_set("vectorizer", "ngram_range", [1, 1]), "TF-IDF parameters differ.*ngram_range"),
    (_set("vectorizer", "token_pattern", r"(?u)\b\w+\b"), "TF-IDF parameters differ.*token_pattern"),
    (_set("vectorizer", "min_df", 2), "TF-IDF parameters differ.*min_df"),
    (_set("vectorizer", "max_df", 0.5), "TF-IDF parameters differ.*max_df"),
    (_set("vectorizer", "max_features", 10), "TF-IDF parameters differ.*max_features"),
    (_set("vectorizer", "sublinear_tf", False), "TF-IDF parameters differ.*sublinear_tf"),
    (_set("vectorizer", "norm", "l1"), "TF-IDF parameters differ.*norm"),
    (_set("vectorizer", "use_idf", False), "TF-IDF parameters differ.*use_idf"),
    (_set("vectorizer", "smooth_idf", False), "TF-IDF parameters differ.*smooth_idf"),
    (_set("vectorizer", "binary", True), "TF-IDF parameters differ.*binary"),
    (_set("vectorizer", "strip_accents", "unicode"), "TF-IDF parameters differ.*strip_accents"),
    (_set("weighting", "sublinear_tf", False), "TF-IDF weighting differs.*sublinear_tf"),
    (_set("weighting", "norm", "l1"), "TF-IDF weighting differs.*norm"),
    (_set("weighting", "use_idf", False), "TF-IDF weighting differs.*use_idf"),
    (_set("text_processing", "tfidf", None), "no declared text_processing.tfidf"),
    (_set("classifier", "multi_class", "ovr"), "LogisticRegression parameters differ.*multi_class"),
    (_set("classifier", "solver", "liblinear"), "LogisticRegression parameters differ.*solver"),
    (_set("classifier", "C", 2.0), "LogisticRegression parameters differ.*C"),
    (_set("classifier", "class_weight", {0: 1.0, 1: 1.0, 2: 1.0}),
     "LogisticRegression parameters differ.*class_weight"),
])
def test_inference_parameters_must_match_declared_configuration(tmp_path: Path, change, reason) -> None:
    model_path = make_real_bundle(tmp_path)
    bundle = tb.load_bundle(model_path)
    change(bundle)
    tb.save_bundle(model_path, bundle)
    report = write_report(model_path, bundle["training"])
    # The changed bundle has its own matching checksum: rejection comes from the parameter check.
    assert tb.read_json(report)["bundle_sha256"] == tb.sha256_file(model_path)
    with pytest.raises(classifier.ModelUnavailableError, match=reason):
        baseline(model_path).predict("Отличный фильм")


def test_declared_tfidf_change_is_rejected_even_if_vectorizer_matches(tmp_path: Path) -> None:
    model_path = make_real_bundle(tmp_path)
    bundle = tb.load_bundle(model_path)
    bundle["text_processing"]["tfidf"]["lowercase"] = False
    tb.save_bundle(model_path, bundle)
    write_report(model_path, bundle["training"])
    with pytest.raises(classifier.ModelUnavailableError, match="lowercase"):
        baseline(model_path).predict("Отличный фильм")


def test_wrong_probability_order_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_result = fake_prediction()
    bad_result["probability_class_ids"] = [2, 1, 0]
    model = predict_with(monkeypatch, tmp_path, bad_result)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_wrong_probability_label_order_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_result = fake_prediction()
    bad_result["probability_labels"] = [
        "positive",
        "neutral",
        "negative",
    ]
    model = predict_with(monkeypatch, tmp_path, bad_result)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_inconsistent_predicted_label_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_result = fake_prediction(
        label_id=2,
        label="negative",
    )
    model = predict_with(monkeypatch, tmp_path, bad_result)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_real_bundle_contract(tmp_path: Path) -> None:
    """Проверить реальный контракт train_baseline -> BaselineClassifier."""
    model = baseline(make_real_bundle(tmp_path))

    result = model.predict("отличный фильм")
    probabilities = result["probabilities"]

    assert list(probabilities) == [
        "negative",
        "neutral",
        "positive",
    ]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert result["label"] == max(
        probabilities,
        key=probabilities.get,
    )
    assert result["confidence"] == pytest.approx(
        probabilities[result["label"]]
    )
    assert result["label_id"] == (
        classifier.EXPECTED_LABEL_MAPPING[result["label"]]
    )
    assert model.data_version["train_ids_sha256"] == "1" * 64


def test_batch_matches_single_predictions(tmp_path: Path) -> None:
    model = baseline(make_real_bundle(tmp_path))
    texts = ["отличный фильм", "ужасный фильм", "средний фильм"]
    assert model.predict_batch(texts) == [model.predict(text) for text in texts]
    with pytest.raises(classifier.TextTooLongError):
        model.predict_batch(["фильм", "а" * 2001])
    assert len(model.predict_batch(["фильм " * 500], max_text_length=None)) == 1


def test_default_model_path_follows_configuration(tmp_path, monkeypatch):
    import json
    model_path = make_real_bundle(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_dir": str(tmp_path), "bundle_name": model_path.name,
                                  "report_dir": str(tmp_path)}))
    monkeypatch.setattr(classifier, "DEFAULT_CONFIG_PATH", config)
    model = classifier.BaselineClassifier()
    assert model.predict("отличный фильм")["label"] == "positive"
    assert model.model_path == model_path
    assert model.report_path == report_of(model_path)


def test_invalid_configuration_is_model_error(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"model_dir": 123}')
    monkeypatch.setattr(classifier, "DEFAULT_CONFIG_PATH", config)
    with pytest.raises(classifier.ModelUnavailableError):
        classifier.BaselineClassifier().predict("Фильм")


def test_concurrent_first_predictions_load_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    model_path = make_real_bundle(tmp_path)
    original = classifier.load_bundle
    calls = []
    def load(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(classifier, "load_bundle", load)
    model = baseline(model_path)
    gate = Barrier(4)
    def predict(_):
        gate.wait(timeout=5)
        return model.predict("отличный фильм")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict, range(4)))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)


def test_checksum_is_verified_before_deserialization(tmp_path, monkeypatch):
    model_path = make_real_bundle(tmp_path)
    write_report(model_path, tb.load_bundle(model_path)["training"], bundle_sha256="0" * 64)
    def forbidden(path):
        pytest.fail("A file that differs from the run report must not be deserialized")
    monkeypatch.setattr(classifier, "load_bundle", forbidden)
    with pytest.raises(classifier.ModelUnavailableError, match="checksum differs from run report"):
        baseline(model_path).predict("Фильм")


def test_default_report_rejects_other_bundle(tmp_path):
    """Без собственного отчёта используется отчёт принятого запуска T04."""
    with pytest.raises(classifier.ModelUnavailableError, match="checksum differs from run report"):
        classifier.BaselineClassifier(make_real_bundle(tmp_path)).predict("Фильм")


def test_report_without_bundle_checksum_is_rejected(tmp_path):
    model_path = make_real_bundle(tmp_path)
    write_report(model_path, tb.load_bundle(model_path)["training"], bundle_sha256=None)
    with pytest.raises(classifier.ModelUnavailableError, match="bundle_sha256"):
        baseline(model_path).predict("Фильм")


def _drop(key):
    return lambda training: training.pop(key)


def _set(key, value):
    return lambda training: training.update({key: value})


@pytest.mark.parametrize("update, reason", [
    (lambda training: training.clear(), "train_ids_sha256"),
    (_set("train_ids_sha256", "abc"), "train_ids_sha256"),
    (_set("validation_ids_sha256", "G" * 64), "validation_ids_sha256"),
    (_set("train_parquet_sha256", None), "train_parquet_sha256"),
    (_set("train_rows", 0), "train_rows"),
    (_set("validation_rows", True), "validation_rows"),
    (_drop("validation_parquet_sha256"), "validation_parquet_sha256"),
    (_drop("tfidf_state_sha256"), "tfidf_state_sha256"),
    (_set("final_test_used", True), "final test"),
    (_drop("final_test_used"), "final test"),
    (_set("split_independence", {"record_id_overlap": 3}), "overlap"),
    (_drop("split_independence"), "overlap"),
    (_set("selection_metric", "test_macro_f1"), "selection"),
    (_set("winner_C", 2.0), "differs from model configuration"),
    (_set("class_weight_mode", "balanced_train"), "differs from model configuration"),
    (_set("tfidf_state_sha256", "5" * 64), "TF-IDF state"),
])
def test_invalid_or_incompatible_data_version_is_rejected(tmp_path, update, reason):
    """Отчёт согласован с изменённым комплектом: отказ даёт именно проверка версии данных."""
    model_path = make_real_bundle(tmp_path, training_update=update)
    with pytest.raises(classifier.ModelUnavailableError, match=reason):
        baseline(model_path).predict("Фильм")


@pytest.mark.parametrize("field, value", [("train_ids_sha256", "9" * 64), ("final_test_used", True),
                                          ("validation_rows", 99)])
def test_data_version_must_match_run_report(tmp_path, field, value):
    model_path = make_real_bundle(tmp_path)
    write_report(model_path, tb.load_bundle(model_path)["training"], **{field: value})
    with pytest.raises(classifier.ModelUnavailableError, match="final test" if field == "final_test_used" else field):
        baseline(model_path).predict("Фильм")
