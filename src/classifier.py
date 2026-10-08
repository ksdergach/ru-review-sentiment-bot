"""Общий интерфейс baseline-классификатора для приложения.

Модель загружается лениво при первом предсказании и затем переиспользуется.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.train_baseline import load_bundle, predict_texts


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    ROOT / "models" / "nlp_baseline" / "tfidf_logreg_baseline.joblib"
)

EXPECTED_LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

EXPECTED_CLASS_IDS = [0, 1, 2]
EXPECTED_CLASS_NAMES = ["negative", "neutral", "positive"]

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

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._bundle: dict[str, Any] | None = None

    def _get_bundle(self) -> dict[str, Any]:
        """Загрузить модель один раз и проверить отображение классов."""
        if self._bundle is not None:
            return self._bundle

        if not self.model_path.is_file():
            raise ModelUnavailableError(
                f"Baseline model not found: {self.model_path}"
            )

        try:
            bundle = load_bundle(self.model_path)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load baseline model: {self.model_path}"
            ) from exc

        if bundle.get("label_mapping") != EXPECTED_LABEL_MAPPING:
            raise ModelUnavailableError(
                "Unexpected label mapping in baseline model"
            )

        self._bundle = bundle
        return bundle

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