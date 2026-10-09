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


def make_fake_model(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.joblib"
    path.write_bytes(b"fake model")
    return path


def make_real_bundle(tmp_path: Path) -> Path:
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
        training={
            "seed": 42,
            "smoke": True,
        },
    )

    path = tmp_path / "real_baseline.joblib"
    tb.save_bundle(path, bundle)
    return path


def test_predict_returns_label_confidence_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = make_fake_model(tmp_path)

    monkeypatch.setattr(
        classifier,
        "load_bundle",
        lambda path: {
            "label_mapping": classifier.EXPECTED_LABEL_MAPPING.copy()
        },
    )
    monkeypatch.setattr(
        classifier,
        "predict_texts",
        lambda bundle, texts: fake_prediction(),
    )

    model = classifier.BaselineClassifier(model_path)
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
    model_path = make_fake_model(tmp_path)
    calls = 0

    def fake_load(path: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "label_mapping": classifier.EXPECTED_LABEL_MAPPING.copy()
        }

    monkeypatch.setattr(classifier, "load_bundle", fake_load)
    monkeypatch.setattr(
        classifier,
        "predict_texts",
        lambda bundle, texts: fake_prediction(),
    )

    model = classifier.BaselineClassifier(model_path)

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
    model_path = make_real_bundle(tmp_path)
    model = classifier.BaselineClassifier(model_path)

    result = model.predict("а" * 2000)

    assert result["label"] in classifier.EXPECTED_CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0


def test_missing_model_has_clear_error(tmp_path: Path) -> None:
    model = classifier.BaselineClassifier(
        tmp_path / "missing.joblib"
    )

    with pytest.raises(classifier.ModelUnavailableError):
        model.predict("Обычный отзыв")


def test_corrupted_model_has_clear_error(tmp_path: Path) -> None:
    model_path = tmp_path / "broken.joblib"
    model_path.write_bytes(b"this is not a joblib bundle")

    model = classifier.BaselineClassifier(model_path)

    with pytest.raises(classifier.ModelUnavailableError):
        model.predict("Обычный отзыв")


def test_wrong_label_mapping_is_rejected(tmp_path: Path) -> None:
    model_path = make_real_bundle(tmp_path)
    bundle = tb.load_bundle(model_path)
    bundle["label_mapping"] = {"negative": 2, "neutral": 1, "positive": 0}
    tb.save_bundle(model_path, bundle)
    with pytest.raises(classifier.ModelUnavailableError):
        classifier.BaselineClassifier(model_path).predict("Обычный отзыв")


def test_wrong_probability_order_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = make_fake_model(tmp_path)

    monkeypatch.setattr(
        classifier,
        "load_bundle",
        lambda path: {
            "label_mapping": classifier.EXPECTED_LABEL_MAPPING.copy()
        },
    )

    bad_result = fake_prediction()
    bad_result["probability_class_ids"] = [2, 1, 0]

    monkeypatch.setattr(
        classifier,
        "predict_texts",
        lambda bundle, texts: bad_result,
    )

    model = classifier.BaselineClassifier(model_path)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_wrong_probability_label_order_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = make_fake_model(tmp_path)

    monkeypatch.setattr(
        classifier,
        "load_bundle",
        lambda path: {
            "label_mapping": classifier.EXPECTED_LABEL_MAPPING.copy()
        },
    )

    bad_result = fake_prediction()
    bad_result["probability_labels"] = [
        "positive",
        "neutral",
        "negative",
    ]

    monkeypatch.setattr(
        classifier,
        "predict_texts",
        lambda bundle, texts: bad_result,
    )

    model = classifier.BaselineClassifier(model_path)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_inconsistent_predicted_label_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = make_fake_model(tmp_path)

    monkeypatch.setattr(
        classifier,
        "load_bundle",
        lambda path: {
            "label_mapping": classifier.EXPECTED_LABEL_MAPPING.copy()
        },
    )

    bad_result = fake_prediction(
        label_id=2,
        label="negative",
    )

    monkeypatch.setattr(
        classifier,
        "predict_texts",
        lambda bundle, texts: bad_result,
    )

    model = classifier.BaselineClassifier(model_path)

    with pytest.raises(classifier.ClassifierError):
        model.predict("Обычный отзыв")


def test_real_bundle_contract(tmp_path: Path) -> None:
    """Проверить реальный контракт train_baseline -> BaselineClassifier."""
    model_path = make_real_bundle(tmp_path)
    model = classifier.BaselineClassifier(model_path)

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

def test_default_model_path_follows_configuration(tmp_path, monkeypatch):
    import json
    model_path = make_real_bundle(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_dir": str(tmp_path), "bundle_name": model_path.name}))
    monkeypatch.setattr(classifier, "DEFAULT_CONFIG_PATH", config)
    model = classifier.BaselineClassifier()
    assert model.predict("отличный фильм")["label"] == "positive"
    assert model.model_path == model_path


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
    model = classifier.BaselineClassifier(model_path)
    gate = Barrier(4)
    def predict(_):
        gate.wait(timeout=5)
        return model.predict("отличный фильм")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict, range(4)))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
