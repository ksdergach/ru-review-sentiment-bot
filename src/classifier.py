"""Общий интерфейс baseline-классификатора для приложения.

Модель загружается лениво при первом предсказании и затем переиспользуется.
"""

from __future__ import annotations

import json
from threading import Lock
from pathlib import Path
from typing import Any

from src.train_baseline import (
    CLASS_IDS as EXPECTED_CLASS_IDS,
    CLASS_NAMES as EXPECTED_CLASS_NAMES,
    EXPECTED_LABEL_MAPPING,
    load_bundle,
    predict_texts,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = ROOT / "configs" / "nlp_baseline.json"

MAX_TEXT_LENGTH = 2000


class ClassifierError(RuntimeError):
    """Общая ошибка классификатора."""


class ModelUnavailableError(ClassifierError):
    """Модель отсутствует или не может быть корректно загружена."""


class InvalidTextError(ValueError):
    """Текст не подходит для классификации."""


class EmptyTextError(InvalidTextError):
    """Текст отзыва пуст."""


class TextTooLongError(InvalidTextError):
    """Текст превышает допустимую длину."""


class BaselineClassifier:
    """Ленивая обёртка над сохранённым TF-IDF + Logistic Regression."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._load_lock = Lock()
        self._bundle: dict[str, Any] | None = None

    def _get_bundle(self) -> dict[str, Any]:
        """Загрузить модель один раз и проверить отображение классов."""
        if self._bundle is not None:
            return self._bundle

        # to_thread and direct callers may race on the first request.
        with self._load_lock:
            if self._bundle is not None:
                return self._bundle
            try:
                if self.model_path is None:
                    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
                    model_dir, bundle_name = config["model_dir"], config["bundle_name"]
                    if not all(isinstance(v, str) and v.strip() for v in (model_dir, bundle_name)):
                        raise ValueError("Invalid model location in baseline configuration")
                    self.model_path = ROOT / model_dir / bundle_name
                # load_bundle validates the shared class mapping and fitted components.
                self._bundle = load_bundle(self.model_path)
            except Exception as exc:
                raise ModelUnavailableError("Could not load configured baseline model") from exc
            return self._bundle

    @staticmethod
    def _validate_text(text: str) -> None:
        """Проверить текст до обращения к модели."""
        if not isinstance(text, str):
            raise InvalidTextError("Review text must be a string")

        if not text.strip():
            raise EmptyTextError("Review text must not be empty")

        if len(text) > MAX_TEXT_LENGTH:
            raise TextTooLongError(
                f"Review text must not exceed {MAX_TEXT_LENGTH} characters"
            )

    def predict(self, text: str) -> dict[str, Any]:
        """Классифицировать один киноотзыв."""
        self._validate_text(text)
        bundle = self._get_bundle()

        try:
            raw = predict_texts(bundle, [text])
        except Exception as exc:
            raise ClassifierError("Baseline prediction failed") from exc

        class_ids = raw.get("probability_class_ids")
        class_names = raw.get("probability_labels")

        if class_ids != EXPECTED_CLASS_IDS:
            raise ClassifierError(
                f"Unexpected probability class order: {class_ids}"
            )

        if class_names != EXPECTED_CLASS_NAMES:
            raise ClassifierError(
                f"Unexpected probability label order: {class_names}"
            )

        try:
            label_id = int(raw["label_ids"][0])
            label = str(raw["labels"][0])
            probabilities = [
                float(value) for value in raw["probabilities"][0]
            ]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ClassifierError(
                "Invalid prediction result returned by baseline"
            ) from exc

        if len(probabilities) != len(EXPECTED_CLASS_NAMES):
            raise ClassifierError(
                "Unexpected number of class probabilities"
            )

        expected_label = next(
            (
                name
                for name, value in EXPECTED_LABEL_MAPPING.items()
                if value == label_id
            ),
            None,
        )

        if expected_label != label:
            raise ClassifierError(
                "Predicted label is inconsistent with label mapping"
            )

        probability_map = dict(
            zip(EXPECTED_CLASS_NAMES, probabilities, strict=True)
        )

        confidence = probability_map[label]

        return {
            "label": label,
            "label_id": label_id,
            "confidence": confidence,
            "probabilities": probability_map,
        }


_default_classifier = BaselineClassifier()


def predict(text: str) -> dict[str, Any]:
    """Общий predict, используемый ботом и прямыми вызовами."""
    return _default_classifier.predict(text)
