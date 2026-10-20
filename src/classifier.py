"""Общий интерфейс baseline и RuBERT для приложения и оценки.

Модель загружается лениво при первом предсказании и затем переиспользуется.
"""

from __future__ import annotations

import json
import math
import os
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
        self.data_version = None

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
                self.data_version = dict(self._bundle.get("training", {}))
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

        return _format_prediction(raw, truncated=False)


def _format_prediction(raw: dict[str, Any], *, truncated: bool) -> dict[str, Any]:
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
            "Invalid prediction result returned by classifier"
        ) from exc

    if len(probabilities) != len(EXPECTED_CLASS_NAMES):
        raise ClassifierError(
            "Unexpected number of class probabilities"
        )

    if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ClassifierError("Expected finite normalized class probabilities")
    if label_id not in EXPECTED_CLASS_IDS or probabilities[label_id] != max(probabilities):
        raise ClassifierError("Predicted label does not maximize probability")

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
        "truncated": truncated,
    }


class RuBERTClassifier:
    """Offline inference with the checkpoint and preprocessing accepted in T06.

    report_path pins the bundle's checksums and data version. Alternate bundles
    require their own run report; files are checked before loading model weights.
    """

    def __init__(self, model_path: str | Path | None = None, *,
                 report_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else ROOT / "models/transformer"
        self.report_path = Path(report_path) if report_path is not None else ROOT / "reports/transformer/run_metadata.json"
        self._load_lock = Lock()
        self._predict_lock = Lock()
        self._bundle = None
        self.data_version = None

    def _get_bundle(self):
        if self._bundle is not None:
            return self._bundle
        with self._load_lock:
            if self._bundle is not None:
                return self._bundle
            try:
                # Lazy import: using baseline must not initialize torch/transformers.
                from src import train_transformer as transformer
                from src.train_baseline import sha256_file, sha256_json
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
                expected = report["bundle_files_sha256"]
                required = {"model.safetensors", "config.json", "tokenizer.json",
                            "tokenizer_config.json", "training_config.json",
                            "label_mapping.json", "metadata.json"}
                if set(expected) != required:
                    raise ValueError("Unexpected transformer bundle manifest")
                if {p.name for p in self.model_path.iterdir()} != required:
                    raise ValueError("Unexpected transformer bundle files")
                for name, digest in expected.items():
                    path = self.model_path / name
                    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                        raise ValueError(f"Transformer bundle checksum mismatch: {name}")
                metadata = json.loads((self.model_path / "metadata.json").read_text(encoding="utf-8"))
                if metadata != {k: v for k, v in report.items() if k != "bundle_files_sha256"}:
                    raise ValueError("Transformer metadata differs from run report")
                config = json.loads((self.model_path / "training_config.json").read_text(encoding="utf-8"))
                if sha256_json(config) != metadata["config_sha256"]:
                    raise ValueError("Transformer preprocessing differs from training")
                version = metadata["data_version"]
                if not version["train_ids_equal_baseline"] or version["final_test_used"]:
                    raise ValueError("Incompatible transformer data version")
                for key in ("train_ids_sha256", "validation_ids_sha256"):
                    if not isinstance(version[key], str) or len(version[key]) != 64:
                        raise ValueError("Missing transformer data provenance")
                # load_bundle validates mapping and config and uses local_files_only.
                loaded = transformer.load_bundle(self.model_path, device="cpu")
                if loaded[2]["max_length"] > loaded[0].config.max_position_embeddings:
                    raise ValueError("max_length exceeds model positions")
                self.data_version = dict(version)
                self._bundle = loaded
            except Exception as exc:
                raise ModelUnavailableError("Could not load verified RuBERT bundle") from exc
            return self._bundle

    def predict(self, text: str) -> dict[str, Any]:
        BaselineClassifier._validate_text(text)
        model, tokenizer, config = self._get_bundle()
        try:
            from src.train_transformer import predict_texts as transformer_predict
            # Fast tokenizers carry mutable truncation state; serialize direct callers.
            with self._predict_lock:
                ids = tokenizer(text, truncation=False, padding=False,
                                add_special_tokens=True)["input_ids"]
                truncated = len(ids) > config["max_length"]
                raw = transformer_predict(model, tokenizer, [text], config)
            return _format_prediction(raw, truncated=truncated)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierError("RuBERT prediction failed") from exc


def create_classifier(backend: str = "baseline", model_path: str | Path | None = None,
                      *, report_path: str | Path | None = None):
    if backend == "baseline":
        if report_path is not None:
            raise ValueError("report_path is only used for RuBERT")
        return BaselineClassifier(model_path)
    if backend == "rubert":
        return RuBERTClassifier(model_path, report_path=report_path)
    raise ValueError("CLASSIFIER_BACKEND must be baseline or rubert")


_default_classifier = None
_default_lock = Lock()


def predict(text: str) -> dict[str, Any]:
    """Shared bot/evaluation entry point; backend is fixed at first use."""
    global _default_classifier
    BaselineClassifier._validate_text(text)
    if _default_classifier is None:
        with _default_lock:
            if _default_classifier is None:
                try:
                    _default_classifier = create_classifier(
                        os.getenv("CLASSIFIER_BACKEND", "baseline"),
                        os.getenv("CLASSIFIER_MODEL_PATH") or None,
                        report_path=os.getenv("CLASSIFIER_REPORT_PATH") or None)
                except ValueError as exc:
                    raise ModelUnavailableError("Invalid classifier configuration") from exc
    return _default_classifier.predict(text)
