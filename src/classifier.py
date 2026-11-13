"""Общий интерфейс baseline и RuBERT для приложения и оценки.

Модель загружается лениво при первом предсказании и затем переиспользуется.
``predict`` (бот, CLI) и ``predict_batch`` (оценка на размеченных текстах)
используют одну и ту же загрузку, обработку и проверку результата.
"""

from __future__ import annotations

import json
import math
import os
from threading import Lock
from pathlib import Path
from typing import Any, Sequence

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

# Data provenance that every baseline bundle must carry (src/train_baseline.py::run_training).
BASELINE_SHA256_FIELDS = ("train_ids_sha256", "validation_ids_sha256", "train_parquet_sha256",
                          "validation_parquet_sha256", "tfidf_state_sha256")
BASELINE_ROW_FIELDS = ("train_rows", "validation_rows")


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


def _validate_text(text: str, max_length: int | None = MAX_TEXT_LENGTH) -> None:
    """Проверить текст до обращения к модели; max_length=None — без ограничения длины."""
    if not isinstance(text, str):
        raise InvalidTextError("Review text must be a string")

    if not text.strip():
        raise EmptyTextError("Review text must not be empty")

    if max_length is not None and len(text) > max_length:
        raise TextTooLongError(
            f"Review text must not exceed {max_length} characters"
        )


def _validate_texts(texts: Sequence[str], max_length: int | None) -> list[str]:
    if isinstance(texts, str) or not isinstance(texts, Sequence) or not texts:
        raise InvalidTextError("Expected a non-empty sequence of review texts")
    for text in texts:
        _validate_text(text, max_length)
    return list(texts)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _same(expected: Any, actual: Any) -> bool:
    """Равенство с учётом типа: True не равно 1, 2.0 не равно 2 (для sklearn это разные значения)."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(_same(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(_same(a, b) for a, b in zip(expected, actual))
    return expected == actual


def _differing(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    return sorted(k for k in expected.keys() | actual.keys()
                  if k not in expected or k not in actual or not _same(expected[k], actual[k]))


def _check_baseline_components(bundle: dict[str, Any]) -> None:
    """Параметры, которые читает инференс, должны совпадать с объявленными в комплекте.

    Ожидаемые объекты строятся теми же функциями, что и при обучении, поэтому
    сравниваются все параметры TfidfVectorizer и LogisticRegression, включая
    значения по умолчанию (``ngram_range``: список в JSON, tuple в sklearn).
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    from src.train_baseline import _validate_tfidf, build_classifier, build_vectorizer

    declared = bundle["text_processing"].get("tfidf")
    if not isinstance(declared, dict):
        raise ValueError("Baseline bundle has no declared text_processing.tfidf")
    _validate_tfidf(declared)
    vectorizer = bundle["vectorizer"]
    expected = build_vectorizer(declared).get_params()
    actual = vectorizer.get_params()
    # vocabulary_ and idf_ are checked by tfidf_state_sha256; transform reads vocabulary_.
    expected.pop("vocabulary")
    actual.pop("vocabulary")
    differing = _differing(expected, actual)
    if differing:
        raise ValueError(f"Baseline TF-IDF parameters differ from text_processing.tfidf: {', '.join(differing)}")
    # transform applies sublinear_tf, IDF and norm from the transformer created by fit.
    weighting = getattr(vectorizer, "_tfidf", None)
    if not isinstance(weighting, TfidfTransformer):
        raise ValueError("Baseline TF-IDF vectorizer has no fitted weighting transformer")
    differing = _differing({k: expected[k] for k in ("norm", "use_idf", "smooth_idf", "sublinear_tf")},
                           weighting.get_params())
    if differing or hasattr(weighting, "idf_") is not expected["use_idf"]:
        raise ValueError(f"Baseline TF-IDF weighting differs from text_processing.tfidf: "
                         f"{', '.join(differing) or 'idf_'}")

    lr = bundle["logistic_regression"]
    weights = lr.get("class_weights")
    try:
        class_weight = None if weights is None else {int(k): float(v) for k, v in weights.items()}
        expected = build_classifier(c_value=lr["C"], lr_config=lr, class_weight=class_weight,
                                    seed=0).get_params()
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid baseline logistic_regression parameters") from exc
    if "penalty" in lr:
        expected["penalty"] = lr["penalty"]
    actual = bundle["classifier"].get_params()
    # random_state affects only fitting with sag/saga; inference does not read it.
    expected.pop("random_state")
    actual.pop("random_state")
    differing = _differing(expected, actual)
    if differing:
        raise ValueError(f"Baseline LogisticRegression parameters differ from logistic_regression: "
                         f"{', '.join(differing)}")


def _check_baseline_provenance(bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Проверить версию данных комплекта и её совпадение с отчётом запуска."""
    from src.train_baseline import _vectorizer_state_sha256

    training = bundle["training"]
    for key in BASELINE_SHA256_FIELDS:
        if not _is_sha256(training.get(key)):
            raise ValueError(f"Missing or invalid baseline data provenance: {key}")
    for key in BASELINE_ROW_FIELDS:
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Missing or invalid baseline data provenance: {key}")
    if training.get("final_test_used") is not False or report.get("final_test_used") is not False:
        raise ValueError("Incompatible baseline data version: final test must not be used")
    independence = training.get("split_independence")
    if (not isinstance(independence, dict) or not independence
            or any(value != 0 for value in independence.values())):
        raise ValueError("Incompatible baseline data version: train/validation overlap")
    if training.get("selection_metric") != "validation_macro_f1":
        raise ValueError("Incompatible baseline training: selection is not validation macro-F1")
    lr = bundle["logistic_regression"]
    if training.get("winner_C") != lr.get("C") or training.get("class_weight_mode") != lr.get("class_weight_mode"):
        raise ValueError("Baseline training metadata differs from model configuration")
    if _vectorizer_state_sha256(bundle["vectorizer"]) != training["tfidf_state_sha256"]:
        raise ValueError("Baseline TF-IDF state differs from recorded training")
    # The run report stores every training field next to run-specific extras.
    recorded = json.loads(json.dumps(training))
    differing = sorted(key for key, value in recorded.items() if key not in report or report[key] != value)
    if differing:
        raise ValueError(f"Baseline training metadata differs from run report: {', '.join(differing)}")
    return recorded


class BaselineClassifier:
    """Ленивая обёртка над сохранённым TF-IDF + Logistic Regression.

    report_path — run_metadata.json запуска, создавшего комплект (по умолчанию —
    ``<report_dir>/run_metadata.json`` из configs/nlp_baseline.json). SHA-256 файла
    сверяется с отчётом до joblib.load: десериализуется только файл, записанный в
    отчёте из Git. Это не делает произвольный pickle безопасным. Другой комплект
    требует собственного отчёта.
    """

    def __init__(self, model_path: str | Path | None = None, *,
                 report_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.report_path = Path(report_path) if report_path is not None else None
        self._load_lock = Lock()
        self._bundle: dict[str, Any] | None = None
        self.data_version = None

    def _resolve_paths(self) -> None:
        if self.model_path is not None and self.report_path is not None:
            return
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        if self.model_path is None:
            model_dir, bundle_name = config["model_dir"], config["bundle_name"]
            if not all(isinstance(v, str) and v.strip() for v in (model_dir, bundle_name)):
                raise ValueError("Invalid model location in baseline configuration")
            self.model_path = ROOT / model_dir / bundle_name
        if self.report_path is None:
            report_dir = config["report_dir"]
            if not isinstance(report_dir, str) or not report_dir.strip():
                raise ValueError("Invalid report location in baseline configuration")
            self.report_path = ROOT / report_dir / "run_metadata.json"

    def _get_bundle(self) -> dict[str, Any]:
        """Загрузить проверенный комплект один раз."""
        if self._bundle is not None:
            return self._bundle

        # to_thread and direct callers may race on the first request.
        with self._load_lock:
            if self._bundle is not None:
                return self._bundle
            try:
                from src.train_baseline import sha256_file
                self._resolve_paths()
                if not self.model_path.is_file() or self.model_path.is_symlink():
                    raise FileNotFoundError("baseline bundle file not found")
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict) or not _is_sha256(report.get("bundle_sha256")):
                    raise ValueError("Run report has no valid bundle_sha256")
                if sha256_file(self.model_path) != report["bundle_sha256"]:
                    raise ValueError("Baseline bundle checksum differs from run report")
                # load_bundle validates the shared class mapping and fitted components.
                bundle = load_bundle(self.model_path)
                _check_baseline_components(bundle)
                self.data_version = _check_baseline_provenance(bundle, report)
                self._bundle = bundle
            except Exception as exc:
                raise ModelUnavailableError(
                    f"Could not load verified baseline bundle {self.model_path}: {_reason(exc)}") from exc
            return self._bundle

    # Kept as a static method for existing callers.
    _validate_text = staticmethod(_validate_text)

    def predict(self, text: str) -> dict[str, Any]:
        """Классифицировать один киноотзыв."""
        _validate_text(text)
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str], *,
                      max_text_length: int | None = MAX_TEXT_LENGTH) -> list[dict[str, Any]]:
        """Классифицировать несколько текстов тем же путём, что и predict."""
        texts = _validate_texts(texts, max_text_length)
        bundle = self._get_bundle()

        try:
            raw = predict_texts(bundle, texts)
        except Exception as exc:
            raise ClassifierError("Baseline prediction failed") from exc

        return [_format_prediction(raw, truncated=False, index=i) for i in range(len(texts))]


def _reason(exc: Exception) -> str:
    """Cause of a load failure for the error message shown to the caller."""
    return f"{type(exc).__name__}: {exc}"


def _format_prediction(raw: dict[str, Any], *, truncated: bool, index: int = 0) -> dict[str, Any]:
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
        label_id = int(raw["label_ids"][index])
        label = str(raw["labels"][index])
        probabilities = [
            float(value) for value in raw["probabilities"][index]
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
                if not self.model_path.is_dir():
                    raise FileNotFoundError(
                        "bundle directory not found; install the accepted model with "
                        "python tools/fetch_transformer.py")
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
                raise ModelUnavailableError(
                    f"Could not load verified RuBERT bundle {self.model_path}: {_reason(exc)}") from exc
            return self._bundle

    def predict(self, text: str) -> dict[str, Any]:
        _validate_text(text)
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str], *,
                      max_text_length: int | None = MAX_TEXT_LENGTH) -> list[dict[str, Any]]:
        """Классифицировать несколько текстов тем же путём, что и predict."""
        texts = _validate_texts(texts, max_text_length)
        model, tokenizer, config = self._get_bundle()
        try:
            from src.train_transformer import predict_texts as transformer_predict
            # Fast tokenizers carry mutable truncation state; serialize direct callers.
            with self._predict_lock:
                lengths = [len(ids) for ids in tokenizer(texts, truncation=False, padding=False,
                                                         add_special_tokens=True)["input_ids"]]
                raw = transformer_predict(model, tokenizer, texts, config)
            return [_format_prediction(raw, truncated=length > config["max_length"], index=i)
                    for i, length in enumerate(lengths)]
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierError("RuBERT prediction failed") from exc


def create_classifier(backend: str = "baseline", model_path: str | Path | None = None,
                      *, report_path: str | Path | None = None):
    if backend == "baseline":
        return BaselineClassifier(model_path, report_path=report_path)
    if backend == "rubert":
        return RuBERTClassifier(model_path, report_path=report_path)
    raise ValueError("CLASSIFIER_BACKEND must be baseline or rubert")


_default_classifier = None
_default_lock = Lock()


def get_default_classifier():
    """Classifier selected by CLASSIFIER_* variables; fixed at first use."""
    global _default_classifier
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
    return _default_classifier


def predict(text: str) -> dict[str, Any]:
    """Shared bot/CLI entry point; backend is fixed at first use."""
    _validate_text(text)
    return get_default_classifier().predict(text)


def predict_batch(texts: Sequence[str], *,
                  max_text_length: int | None = MAX_TEXT_LENGTH) -> list[dict[str, Any]]:
    """Shared evaluation entry point: same model and processing as predict.

    max_text_length=None lifts only the interactive length limit for dataset texts.
    """
    _validate_texts(texts, max_text_length)
    return get_default_classifier().predict_batch(texts, max_text_length=max_text_length)
