"""T14: влияние ошибок Whisper на классификацию тональности 36 голосовых отзывов T11.

Для каждой записи один и тот же классификатор получает два входа:

1. эталонную транскрипцию T11 (режим reference);
2. результат Whisper для аудио той же записи (режим ASR).

Обе классификации идут через ``predict_batch`` общего интерфейса ``src.classifier``
с одинаковым ``max_text_length``; распознавание — через ``src.speech.SpeechRecognizer``
с параметрами из фиксации. Метрики классов считает ``validation_metrics`` из
``src.train_baseline``; служебное предсказание ``error`` кодируется вне трёх классов,
поэтому оно снижает recall истинного класса, но не входит в среднее F1.

Финальный запуск возможен только по утверждённой фиксации T12. Её формат ещё не
утверждён, поэтому реестры ``PROTOCOL_ADAPTERS`` и ``APPROVED_NORMALIZATIONS`` пусты
и любой запуск отказывает до чтения приватных данных T11 и загрузки моделей::

    python -m src.evaluate_voice --freeze <фиксация T12> --check
    python -m src.evaluate_voice --freeze <фиксация T12>
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, fields
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from src.classifier import ClassifierError, InvalidTextError, create_classifier
from src.speech import SpeechConfig, SpeechRecognizer
from src.train_baseline import (
    CLASS_IDS,
    CLASS_NAMES,
    EXPECTED_LABEL_MAPPING,
    _smoke_texts,
    display_path,
    git_ignore_status,
    library_versions,
    package_version,
    read_json,
    sha256_file,
    sha256_json,
    validation_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_voice_test.py"
DEFAULT_PRIVATE_DIR = ROOT / "artifacts" / "t14"
DEFAULT_PUBLIC_DIR = ROOT / "reports" / "voice_eval"
PRIVATE_RECORDS_NAME = "records.json"
PUBLIC_METRICS_NAME = "metrics.json"
PUBLIC_REPORT_NAME = "report.md"

LABELS = tuple(CLASS_NAMES)  # negative=0, neutral=1, positive=2
ERROR = "error"  # служебное «нет предсказания», не четвёртый класс
ERROR_ID = -1  # вне CLASS_IDS: validation_metrics(labels=[0, 1, 2]) не усредняет его
BACKENDS = ("baseline", "rubert")

REFERENCE_STATUSES = ("ok", "classifier_error")
ASR_STATUSES = ("ok", "empty", "error")  # SpeechResult.status
ASR_CLASSIFICATION_STATUSES = ("ok", "classifier_error", "not_run")
# Варианты численного учёта технического сбоя Whisper в WER. Какой из них
# применяется, решает фиксация T12; значения по умолчанию нет.
ASR_FAILURE_RULES = ("all_deletions", "excluded")

RECORD_FIELDS = (
    "record_id", "true_label",
    "reference_prediction", "reference_status", "reference_error_kind",
    "asr_status", "asr_error_code",
    "asr_prediction", "asr_classification_status", "asr_classifier_error_kind",
    "wer_substitutions", "wer_deletions", "wer_insertions", "reference_word_count",
    "asr_seconds", "asr_model_load_seconds", "reference_classifier_seconds",
    "asr_classifier_seconds", "total_seconds", "audio_duration_seconds",
)
# Только в приватном файле результатов; в публичные отчёты не попадает.
PRIVATE_RECORD_FIELDS = ("asr_text",)
TIMING_FIELDS = ("asr_seconds", "reference_classifier_seconds", "asr_classifier_seconds", "total_seconds")
# Ключи, которые не должны встречаться в публичных отчётах ни в каком виде.
FORBIDDEN_PUBLIC_KEYS = ("asr_text", "reference_text", "reference_transcript", "true_label",
                         "reference_prediction", "asr_prediction")
MIN_GUARDED_WORDS = 3  # короче — совпадение со служебным текстом отчёта не является утечкой

LIMITATIONS = (
    "Малый размер голосового теста: 36 записей, по 12 на класс.",
    "Предметная область — только отзывы о фильмах.",
    "Ограниченное число дикторов (4).",
    "Часть записей сделана с умеренным бытовым шумом.",
    "Интонация и эмоциональные акустические признаки не анализируются: классифицируется только текст.",
    "Результаты нельзя автоматически распространять на любые сообщения Telegram.",
    "Время измерено локально, без сетевой задержки Telegram API.",
)

# Заполняются при интеграции с утверждённой фиксацией T12 (этап Б), не раньше.
# PROTOCOL_ADAPTERS: schema фиксации -> функция, возвращающая поля EvaluationProtocol
# (кроме freeze_path и freeze_sha256). APPROVED_NORMALIZATIONS: ID утверждённых правил
# нормализации WER -> функция «текст -> список слов».
PROTOCOL_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
APPROVED_NORMALIZATIONS: dict[str, Callable[[str], list[str]]] = {}


class EvaluationError(Exception):
    """Оценка не может быть выполнена или завершена."""


class PreconditionError(EvaluationError):
    """Предусловие финального запуска не выполнено; данные T11 и модели не тронуты."""


class DataMismatchError(EvaluationError):
    """Входные данные или результаты неполны или противоречивы."""


# --------------------------------------------------------------------------- #
# Нормализация и WER
# --------------------------------------------------------------------------- #
def _validator():
    """Штатный валидатор T11 (tools/ не пакет); загружается один раз и лениво."""
    module = sys.modules.get("validate_voice_test")
    if module is None:
        spec = importlib.util.spec_from_file_location("validate_voice_test", VALIDATOR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["validate_voice_test"] = module
        spec.loader.exec_module(module)
    return module


def provisional_words(text: str) -> list[str]:
    """Черновая токенизация для разработки: та же, что ``words`` валидатора T11
    (нижний регистр, ё -> е, последовательности ``\\w+``).

    Это НЕ утверждённая нормализация WER: правила числительных, дефисов и
    пунктуации утверждает T12, и финальный запуск её не принимает.
    """
    return _validator().words(text)


def wer_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    """(S, D, I) минимального выравнивания слов по Левенштейну.

    Сумма S + D + I всегда равна расстоянию редактирования. Среди равноценных
    выравниваний выбирается совпадение/замена, затем удаление, затем вставка.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            cost[i][j] = min(diagonal, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    substitutions = deletions = insertions = 0
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            substitutions += reference[i - 1] != hypothesis[j - 1]
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


def corpus_wer(records: Sequence[Mapping[str, Any]], failure_rule: str) -> dict[str, Any]:
    """WER = (ΣS + ΣD + ΣI) / ΣN по всем учитываемым записям, не среднее по записям.

    Пустая транскрипция уже записана как D = N. Для технического сбоя Whisper
    (S/D/I не измерены) правило ``failure_rule`` из фиксации T12 решает:
    ``all_deletions`` — D = N, ``excluded`` — запись не входит ни в числитель,
    ни в знаменатель. В обоих случаях сбои считаются отдельно.
    """
    if failure_rule not in ASR_FAILURE_RULES:
        raise ValueError(f"failure_rule must be one of {', '.join(ASR_FAILURE_RULES)}")
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    counted = failures = 0
    for record in records:
        words = record["reference_word_count"]
        if record["asr_status"] == "error":
            failures += 1
            if failure_rule == "excluded":
                continue
            s, d, i = 0, words, 0
        else:
            s, d, i = record["wer_substitutions"], record["wer_deletions"], record["wer_insertions"]
        totals["substitutions"] += s
        totals["deletions"] += d
        totals["insertions"] += i
        totals["reference_words"] += words
        counted += 1
    if totals["reference_words"] == 0:
        raise DataMismatchError("Corpus WER is undefined: no reference words in counted records")
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    return {
        "wer": errors / totals["reference_words"],
        **totals,
        "word_errors": errors,
        "records_counted": counted,
        "asr_failures": failures,
        "asr_failure_rule": failure_rule,
        "empty_transcripts": sum(r["asr_status"] == "empty" for r in records),
    }


# --------------------------------------------------------------------------- #
# Метрики классификации
# --------------------------------------------------------------------------- #
def _label_ids(values: Sequence[str], *, allow_error: bool) -> np.ndarray:
    allowed = set(LABELS) | ({ERROR} if allow_error else set())
    if any(value not in allowed for value in values):
        raise DataMismatchError(f"Labels must be one of {', '.join(sorted(allowed))}")
    return np.array([ERROR_ID if v == ERROR else EXPECTED_LABEL_MAPPING[v] for v in values], dtype=int)


def mode_metrics(true_labels: Sequence[str], predictions: Sequence[str]) -> dict[str, Any]:
    """Метрики одного режима по всем записям, включая записи без предсказания.

    ``error`` засчитывается как пропуск истинного класса; macro-F1 усредняется
    строго по negative, neutral и positive, даже если класс ни разу не предсказан.
    Матрица ошибок: строки — истинные классы, столбцы — три класса и ``error``.
    """
    if len(true_labels) != len(predictions) or not true_labels:
        raise DataMismatchError("True labels and predictions must be non-empty and of equal length")
    y_true = _label_ids(true_labels, allow_error=False)
    y_pred = _label_ids(predictions, allow_error=True)
    metrics = validation_metrics(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[*CLASS_IDS, ERROR_ID])[:len(CLASS_IDS)]
    correct = int((y_true == y_pred).sum())
    no_prediction = int((y_pred == ERROR_ID).sum())
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class": {name: {key: metrics[f"{name}_{key}"] for key in ("precision", "recall", "f1", "support")}
                      for name in LABELS},
        "confusion_matrix": {"rows": list(LABELS), "columns": [*LABELS, ERROR],
                             "values": matrix.astype(int).tolist()},
        "records": len(y_true),
        "correct": correct,
        "incorrect": len(y_true) - correct,
        "wrong_label": len(y_true) - correct - no_prediction,
        "no_prediction": no_prediction,
    }


# --------------------------------------------------------------------------- #
# Проверка записей результатов
# --------------------------------------------------------------------------- #
def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_seconds(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _record_problem(record: Mapping[str, Any]) -> str | None:
    """Имя первого противоречивого поля записи или None; значения не возвращаются."""
    if record["true_label"] not in LABELS:
        return "true_label"
    if record["reference_status"] not in REFERENCE_STATUSES:
        return "reference_status"
    if (record["reference_prediction"] in LABELS) != (record["reference_status"] == "ok") \
            or record["reference_prediction"] not in (*LABELS, ERROR):
        return "reference_prediction"
    if not _is_seconds(record["reference_classifier_seconds"]):
        return "reference_classifier_seconds"
    status = record["asr_status"]
    if status not in ASR_STATUSES:
        return "asr_status"
    classification = record["asr_classification_status"]
    if classification not in ASR_CLASSIFICATION_STATUSES or (classification == "not_run") != (status != "ok"):
        return "asr_classification_status"
    if (record["asr_prediction"] in LABELS) != (classification == "ok") \
            or record["asr_prediction"] not in (*LABELS, ERROR):
        return "asr_prediction"
    if (record["asr_classifier_seconds"] is None) != (classification == "not_run") or (
            record["asr_classifier_seconds"] is not None and not _is_seconds(record["asr_classifier_seconds"])):
        return "asr_classifier_seconds"
    for key in ("asr_seconds", "asr_model_load_seconds", "total_seconds"):
        if not _is_seconds(record[key]):
            return key
    words = record["reference_word_count"]
    if not _is_count(words):
        return "reference_word_count"
    counts = [record[f"wer_{key}"] for key in ("substitutions", "deletions", "insertions")]
    if status == "error":
        if any(value is not None for value in counts):
            return "wer_substitutions"
    elif not all(_is_count(value) for value in counts) or counts[0] + counts[1] > words:
        return "wer_substitutions"
    elif status == "empty" and counts != [0, words, 0]:
        return "wer_deletions"
    return None


def validate_records(records: Sequence[Mapping[str, Any]],
                     expected_ids: Sequence[str] | None = None) -> None:
    """Все записи полны и непротиворечивы; при ``expected_ids`` — тот же состав и порядок."""
    if not records:
        raise DataMismatchError("No evaluation records")
    ids = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DataMismatchError(f"record #{index + 1}: expected an object")
        where = record.get("record_id")
        where = where if isinstance(where, str) and where else f"record #{index + 1}"
        if missing := [key for key in RECORD_FIELDS if key not in record]:
            raise DataMismatchError(f"{where}: missing fields {', '.join(missing)}")
        if (problem := _record_problem(record)) is not None:
            raise DataMismatchError(f"{where}: inconsistent {problem}")
        ids.append(record["record_id"])
    if len(set(ids)) != len(ids):
        raise DataMismatchError("Duplicate record_id in evaluation records")
    if expected_ids is not None and ids != list(expected_ids):
        missing, extra = set(expected_ids) - set(ids), set(ids) - set(expected_ids)
        raise DataMismatchError(f"Evaluation records differ from expected IDs: missing {len(missing)}, "
                                f"extra {len(extra)}" + ("" if missing or extra else ", different order"))


# --------------------------------------------------------------------------- #
# Классификация, распознавание и запись результата
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoiceInput:
    record_id: str
    true_label: str
    reference_text: str = field(repr=False)
    audio_path: Path
    audio_sha256: str = ""


@dataclass(frozen=True)
class ClassifierOutcome:
    prediction: str
    status: str
    error_kind: str | None
    seconds: float | None


def _error_kind(exc: BaseException) -> str:
    """Тип ошибки и её причины без сообщения: в сообщениях бывают пути и текст."""
    cause = exc.__cause__
    return type(exc).__name__ if cause is None else f"{type(exc).__name__} ({type(cause).__name__})"


def classify_text(classifier: Any, text: str, *, max_text_length: int | None,
                  clock: Callable[[], float] = time.perf_counter) -> ClassifierOutcome:
    """Один текст через общий ``predict_batch``; ошибка контракта -> служебный ``error``.

    Отказ классификатора не превращается в какой-либо класс тональности.
    Прочие исключения означают дефект и прерывают оценку.
    """
    start = clock()
    try:
        result = classifier.predict_batch([text], max_text_length=max_text_length)
    except (ClassifierError, InvalidTextError) as exc:
        return ClassifierOutcome(ERROR, "classifier_error", _error_kind(exc), clock() - start)
    seconds = clock() - start
    if not isinstance(result, list) or len(result) != 1 or result[0].get("label") not in LABELS:
        raise EvaluationError("Classifier returned a result outside the shared contract")
    return ClassifierOutcome(result[0]["label"], "ok", None, seconds)


def evaluate_record(item: VoiceInput, *, recognizer: Any, classifier: Any,
                    tokenize: Callable[[str], list[str]], max_text_length: int | None,
                    clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Оба режима для одной записи одним классификатором и одной предобработкой."""
    reference = classify_text(classifier, item.reference_text, max_text_length=max_text_length, clock=clock)
    reference_words = tokenize(item.reference_text)

    start = clock()
    speech = recognizer.transcribe(item.audio_path)
    if speech.status not in ASR_STATUSES:
        raise EvaluationError(f"{item.record_id}: unexpected speech status")
    if speech.status == "ok":
        asr = classify_text(classifier, speech.text, max_text_length=max_text_length, clock=clock)
    else:
        asr = ClassifierOutcome(ERROR, "not_run", None, None)
    total = clock() - start

    if speech.status == "ok":
        s, d, i = wer_counts(reference_words, tokenize(speech.text))
    elif speech.status == "empty":
        s, d, i = 0, len(reference_words), 0  # все слова эталона удалены
    else:
        s = d = i = None  # не измерено; учёт задаёт правило фиксации T12
    return {
        "record_id": item.record_id,
        "true_label": item.true_label,
        "reference_prediction": reference.prediction,
        "reference_status": reference.status,
        "reference_error_kind": reference.error_kind,
        "asr_status": speech.status,
        "asr_error_code": speech.error_code,
        "asr_prediction": asr.prediction,
        "asr_classification_status": asr.status,
        "asr_classifier_error_kind": asr.error_kind,
        "wer_substitutions": s,
        "wer_deletions": d,
        "wer_insertions": i,
        "reference_word_count": len(reference_words),
        "asr_seconds": speech.elapsed_seconds,
        "asr_model_load_seconds": speech.model_load_seconds,
        "reference_classifier_seconds": reference.seconds,
        "asr_classifier_seconds": asr.seconds,
        "total_seconds": total,
        "audio_duration_seconds": speech.duration_seconds,
        "asr_text": speech.text,
    }


def evaluate_records(inputs: Sequence[VoiceInput], **kwargs: Any) -> list[dict[str, Any]]:
    """Все записи по порядку; ни одна запись не пропускается из-за сбоя."""
    ids = [item.record_id for item in inputs]
    if not ids or len(set(ids)) != len(ids):
        raise DataMismatchError("Voice inputs must be non-empty with unique record_id")
    records = [evaluate_record(item, **kwargs) for item in inputs]
    validate_records(records, ids)
    return records


# --------------------------------------------------------------------------- #
# Агрегаты
# --------------------------------------------------------------------------- #
def timing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in TIMING_FIELDS:
        values = [float(r[key]) for r in records if r[key] is not None]
        result[key] = {
            "count": len(values),
            "total": sum(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return result


def summarize(records: Sequence[Mapping[str, Any]], *, failure_rule: str,
              expected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Метрики обоих режимов, ΔF1, корпусный WER, сбои и время — по всем записям."""
    validate_records(records, expected_ids)
    true_labels = [r["true_label"] for r in records]
    reference = mode_metrics(true_labels, [r["reference_prediction"] for r in records])
    asr = mode_metrics(true_labels, [r["asr_prediction"] for r in records])

    def ids(condition: Callable[[Mapping[str, Any]], bool]) -> list[str]:
        return [r["record_id"] for r in records if condition(r)]

    return {
        "records": len(records),
        "record_ids": [r["record_id"] for r in records],
        "reference": reference,
        "asr": asr,
        "delta_macro_f1": asr["macro_f1"] - reference["macro_f1"],
        "wer": corpus_wer(records, failure_rule),
        "failures": {
            "asr_status": {status: sum(r["asr_status"] == status for r in records) for status in ASR_STATUSES},
            "asr_error_codes": dict(sorted(Counter(r["asr_error_code"] for r in records
                                                   if r["asr_status"] == "error").items())),
            "asr_error_record_ids": ids(lambda r: r["asr_status"] == "error"),
            "asr_empty_record_ids": ids(lambda r: r["asr_status"] == "empty"),
            "reference_classifier_errors": sum(r["reference_status"] == "classifier_error" for r in records),
            "asr_classifier_errors": sum(r["asr_classification_status"] == "classifier_error" for r in records),
            "classifier_error_kinds": dict(sorted(Counter(
                kind for r in records for kind in (r["reference_error_kind"], r["asr_classifier_error_kind"])
                if kind is not None).items())),
        },
        "timing": timing_summary(records),
    }


def assert_no_private_text(payloads: Iterable[str], private_texts: Iterable[str]) -> None:
    """Публичные отчёты не содержат приватных текстов и полей отдельных записей.

    Проверяются тексты от MIN_GUARDED_WORDS слов: одно-два слова могут совпасть
    со служебным текстом отчёта без утечки.
    """
    payloads = list(payloads)
    for payload in payloads:
        if any(f'"{key}"' in payload for key in FORBIDDEN_PUBLIC_KEYS):
            raise EvaluationError("Public report contains a per-record private field; nothing written")
    for text in private_texts:
        text = " ".join(text.split())
        if len(text.split()) >= MIN_GUARDED_WORDS and any(text in " ".join(p.split()) for p in payloads):
            raise EvaluationError("Public report contains a private transcript; nothing written")


def build_public_summary(records: Sequence[Mapping[str, Any]], *, protocol: EvaluationProtocol,
                         run: Mapping[str, Any], expected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Только агрегаты, ID, версии и контрольные суммы; без текстов и меток отдельных записей."""
    return {
        "schema": "t14-voice-eval-summary/draft-1",
        "task": "T14",
        "protocol": {
            "freeze_path": protocol.freeze_path,
            "freeze_sha256": protocol.freeze_sha256,
            "classifier_backend": protocol.classifier_backend,
            "classifier_report_path": protocol.classifier_report_path,
            "classifier_report_sha256": protocol.classifier_report_sha256,
            "max_text_length": protocol.max_text_length,
            "speech_config": dict(protocol.speech_config),
            "t11_report_sha256": protocol.t11_report_sha256,
            "t11_content_sha256": protocol.t11_content_sha256,
            "normalization": protocol.normalization,
            "asr_failure_wer_rule": protocol.asr_failure_wer_rule,
        },
        "run": dict(run),
        "results": summarize(records, failure_rule=protocol.asr_failure_wer_rule, expected_ids=expected_ids),
        "limitations": list(LIMITATIONS),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(summary: Mapping[str, Any]) -> str:
    """Markdown-отчёт по публичной сводке; других источников данных у него нет."""
    results, protocol, run = summary["results"], summary["protocol"], summary["run"]
    ref, asr, wer = results["reference"], results["asr"], results["wer"]
    lines = [
        "# T14: влияние Whisper на классификацию тональности",
        "",
        f"Записей: {results['records']}. Классификатор: `{protocol['classifier_backend']}`; "
        f"Whisper: `{protocol['speech_config']['model']}` @ `{protocol['speech_config']['revision']}`.",
        f"Фиксация T12: `{protocol['freeze_path']}` (`{protocol['freeze_sha256']}`); "
        f"commit `{run.get('git_commit')}`; T11 `content_sha256` `{protocol['t11_content_sha256']}`.",
        "",
        "## Классификация",
        "",
        "| Режим | macro-F1 | accuracy | верно | неверный класс | нет предсказания |",
        "|---|---|---|---|---|---|",
    ]
    for name, mode in (("Эталонный текст", ref), ("Whisper", asr)):
        lines.append(f"| {name} | {_fmt(mode['macro_f1'])} | {_fmt(mode['accuracy'])} | {mode['correct']} | "
                     f"{mode['wrong_label']} | {mode['no_prediction']} |")
    lines += ["", f"ΔF1 = F1_ASR − F1_reference = {_fmt(results['delta_macro_f1'])}", "",
              "| Класс | P ref | R ref | F1 ref | P ASR | R ASR | F1 ASR | support |", "|---|---|---|---|---|---|---|---|"]
    for label in LABELS:
        r, a = ref["per_class"][label], asr["per_class"][label]
        lines.append(f"| {label} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | "
                     f"{_fmt(a['precision'])} | {_fmt(a['recall'])} | {_fmt(a['f1'])} | {r['support']} |")
    for name, mode in (("эталонный текст", ref), ("Whisper", asr)):
        matrix = mode["confusion_matrix"]
        lines += ["", f"Матрица ошибок, {name} (строки — истинный класс):", "",
                  "| | " + " | ".join(matrix["columns"]) + " |", "|---" * (len(matrix["columns"]) + 1) + "|"]
        lines += [f"| {row} | " + " | ".join(str(v) for v in values) + " |"
                  for row, values in zip(matrix["rows"], matrix["values"])]
    failures = results["failures"]
    lines += [
        "", "## Распознавание", "",
        f"Корпусный WER = (ΣS + ΣD + ΣI) / ΣN = ({wer['substitutions']} + {wer['deletions']} + "
        f"{wer['insertions']}) / {wer['reference_words']} = {_fmt(wer['wer'])}; "
        f"нормализация `{protocol['normalization']}`, учёт сбоев `{wer['asr_failure_rule']}`.",
        "",
        f"Whisper: ok {failures['asr_status']['ok']}, пустых {failures['asr_status']['empty']}, "
        f"технических сбоев {failures['asr_status']['error']}. Ошибок классификатора: "
        f"эталон {failures['reference_classifier_errors']}, Whisper {failures['asr_classifier_errors']}.",
        "", "## Время", "",
        "| Этап | n | всего, s | среднее, s | медиана, s |", "|---|---|---|---|---|",
    ]
    for key, stats in results["timing"].items():
        lines.append(f"| {key} | {stats['count']} | {_fmt(stats['total'], 3)} | {_fmt(stats['mean'], 3)} | "
                     f"{_fmt(stats['median'], 3)} |")
    lines += ["", "Загрузка и прогрев моделей (вне основного прохода), s: "
              + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in run.get("model_setup_seconds", {}).items()), "",
              "## Ограничения", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Фиксация T12 и предусловия финального запуска
# --------------------------------------------------------------------------- #
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class EvaluationProtocol:
    """То, что T14 берёт из утверждённой фиксации T12 (через адаптер её формата)."""
    approved: bool
    classifier_backend: str
    classifier_report_path: str
    classifier_report_sha256: str
    classifier_model_path: str | None
    max_text_length: int | None
    speech_config_path: str
    speech_config: Mapping[str, Any]
    t11_report_sha256: str
    t11_content_sha256: str
    normalization: str
    asr_failure_wer_rule: str
    freeze_path: str = ""
    freeze_sha256: str = ""


@dataclass(frozen=True)
class T11Paths:
    xlsx: Path
    audio_dir: Path
    manifest: Path
    report: Path

    @classmethod
    def defaults(cls) -> T11Paths:
        vt = _validator()
        return cls(vt.DEFAULT_XLSX, vt.DEFAULT_AUDIO, vt.DEFAULT_MANIFEST, vt.DEFAULT_REPORT)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreconditionError(message)


def _repo_path(root: Path, value: Any, name: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"T12 protocol {name} must be a path")
    _require(not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
                  or ".." in re.split(r"[\\/]", value)),
             f"T12 protocol {name} must be relative to the repository")
    return root / value


def validate_protocol(protocol: EvaluationProtocol, root: Path,
                      normalizations: Mapping[str, Callable[[str], list[str]]]) -> None:
    _require(protocol.approved is True, "T12 freeze is not approved; the final voice test stays closed")
    _require(protocol.classifier_backend in BACKENDS, f"T12 protocol classifier_backend must be one of {BACKENDS}")
    _repo_path(root, protocol.classifier_report_path, "classifier_report_path")
    if protocol.classifier_model_path is not None:
        _repo_path(root, protocol.classifier_model_path, "classifier_model_path")
    _repo_path(root, protocol.speech_config_path, "speech_config_path")
    limit = protocol.max_text_length
    _require(limit is None or (isinstance(limit, int) and not isinstance(limit, bool) and limit > 0),
             "T12 protocol max_text_length must be a positive integer or null")
    for key in ("classifier_report_sha256", "t11_report_sha256", "t11_content_sha256"):
        _require(isinstance(getattr(protocol, key), str) and bool(_SHA256.fullmatch(getattr(protocol, key))),
                 f"T12 protocol {key} must be SHA-256")
    try:
        SpeechConfig(**protocol.speech_config)
    except (TypeError, ValueError) as exc:
        raise PreconditionError(f"T12 protocol speech_config is invalid: {exc}") from None
    _require(protocol.normalization in normalizations,
             f"WER normalization {protocol.normalization!r} is not an approved T12 rule set")
    _require(protocol.asr_failure_wer_rule in ASR_FAILURE_RULES,
             f"T12 protocol asr_failure_wer_rule must be one of {ASR_FAILURE_RULES}")


def load_protocol(path: Path, *, root: Path = ROOT,
                  adapters: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None,
                  normalizations: Mapping[str, Callable[[str], list[str]]] | None = None) -> EvaluationProtocol:
    """Прочитать фиксацию T12 через адаптер её схемы; ничего больше не читается."""
    adapters = PROTOCOL_ADAPTERS if adapters is None else adapters
    normalizations = APPROVED_NORMALIZATIONS if normalizations is None else normalizations
    shown = display_path(path, root)
    _require(path.is_file(), f"T12 freeze not found: {shown}; the final voice test stays closed")
    try:
        data = read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreconditionError(f"T12 freeze is not readable JSON ({type(exc).__name__})") from None
    schema = data.get("schema") if isinstance(data, dict) else None
    adapter = adapters.get(schema) if isinstance(schema, str) else None
    _require(adapter is not None, f"No approved T12 protocol adapter for schema {schema!r}; "
                                  "integration with the T12 freeze is pending")
    try:
        protocol = EvaluationProtocol(**dict(adapter(data)), freeze_path=shown, freeze_sha256=sha256_file(path))
    except (TypeError, KeyError, ValueError) as exc:
        raise PreconditionError(f"T12 freeze does not provide the T14 protocol ({type(exc).__name__})") from None
    validate_protocol(protocol, root, normalizations)
    return protocol


def git_state(root: Path) -> dict[str, Any]:
    """Commit и наличие незакоммиченных изменений отслеживаемых файлов."""
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return proc.stdout if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit.strip() if commit else None, "dirty": None if status is None else bool(status.strip())}


def check_frozen_inputs(protocol: EvaluationProtocol, root: Path, *, composition_path: Path,
                        private_dir: Path, git: Callable[[Path], Mapping[str, Any]] = git_state,
                        is_ignored: Callable[[Path, Path], bool | None] = git_ignore_status) -> dict[str, Any]:
    """Сверить публичные входы с фиксацией. Приватные данные T11 и модели не трогаются."""
    report = _repo_path(root, protocol.classifier_report_path, "classifier_report_path")
    _require(report.is_file() and sha256_file(report) == protocol.classifier_report_sha256,
             "Classifier run report differs from the T12 freeze")
    try:
        speech = SpeechConfig.load(_repo_path(root, protocol.speech_config_path, "speech_config_path"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PreconditionError(f"Speech configuration is not readable ({type(exc).__name__})") from None
    _require(sha256_json({f.name: getattr(speech, f.name) for f in fields(speech)})
             == sha256_json(dict(protocol.speech_config)),
             "Whisper model, revision or parameters differ from the T12 freeze")
    _require(composition_path.is_file() and sha256_file(composition_path) == protocol.t11_report_sha256,
             "T11 public composition report differs from the T12 freeze")
    composition = read_json(composition_path)
    _require(isinstance(composition, dict) and composition.get("content_sha256") == protocol.t11_content_sha256,
             "T11 content_sha256 differs from the T12 freeze")
    state = dict(git(root))
    _require(isinstance(state.get("commit"), str) and state.get("dirty") is False,
             "Final evaluation requires a git checkout without uncommitted changes to tracked files")
    _require(is_ignored(private_dir, root) is True,
             f"Private results directory {display_path(private_dir, root)} must be ignored by Git")
    return {"composition": composition, "git_commit": state["commit"]}


def voice_inputs(manifest: Mapping[str, Any], composition: Mapping[str, Any], *,
                 protocol: EvaluationProtocol, audio_dir: Path) -> list[VoiceInput]:
    """Входы оценки из проверенного манифеста T11: все 36 ID, 12/12/12, суммы аудио."""
    vt = _validator()
    expected_ids = [f"voice_{n:03d}" for n in range(1, vt.EXPECTED_RECORDS + 1)]
    if manifest.get("label_ids") != EXPECTED_LABEL_MAPPING:
        raise DataMismatchError("T11 label order differs from negative=0, neutral=1, positive=2")
    if not (manifest.get("content_sha256") == composition.get("content_sha256") == protocol.t11_content_sha256):
        raise DataMismatchError("T11 manifest content_sha256 differs from the frozen report")
    records = manifest.get("records")
    public = composition.get("records")
    if not isinstance(records, list) or not isinstance(public, list):
        raise DataMismatchError("T11 manifest or report has no record list")
    ids = [r.get("record_id") for r in records]
    if ids != expected_ids or [r.get("record_id") for r in public] != expected_ids:
        raise DataMismatchError(f"T11 records must be exactly {expected_ids[0]}–{expected_ids[-1]} in order; "
                                f"manifest {len(ids)}, report {len(public)}")
    inputs = []
    for record, frozen in zip(records, public):
        record_id = record["record_id"]
        if record.get("audio_sha256") != frozen.get("audio_sha256"):
            raise DataMismatchError(f"{record_id}: audio SHA-256 differs from the frozen report")
        if record.get("final_label") not in LABELS:
            raise DataMismatchError(f"{record_id}: invalid final_label")
        text = record.get("reference_transcript")
        if not isinstance(text, str) or not text.strip():
            raise DataMismatchError(f"{record_id}: missing reference transcript")
        inputs.append(VoiceInput(record_id, record["final_label"], text, audio_dir / f"{record_id}.ogg",
                                 record["audio_sha256"]))
    counts = Counter(item.true_label for item in inputs)
    if any(counts[label] != vt.EXPECTED_PER_CLASS for label in LABELS):
        raise DataMismatchError(f"T11 labels must be {vt.EXPECTED_PER_CLASS} per class")
    return inputs


def verify_t11(paths: T11Paths) -> dict[str, Any]:
    """Штатная проверка неизменности T11; возвращает проверенный манифест.

    Первое обращение к приватным данным: таблица и аудио декодируются валидатором
    (без ASR). Вызывается только после проверки фиксации T12.
    """
    vt = _validator()
    try:
        manifest, report, _warnings = vt.build(paths.xlsx, paths.audio_dir)
    except vt.ValidationError as exc:
        raise DataMismatchError(f"T11 validation failed:\n{exc}") from None
    problems = vt.verify(manifest, report, paths.manifest, paths.report)
    if problems:
        raise DataMismatchError("T11 differs from its freeze:\n" + "\n".join(problems))
    return manifest


def environment() -> dict[str, Any]:
    versions = library_versions()
    for name in ("faster-whisper", "ctranslate2", "av", "torch", "transformers"):
        versions[name] = package_version(name)
    return {"versions": versions, "machine": platform.machine(), "processor": platform.processor() or None,
            "cpu_count": os.cpu_count()}


def _timed(clock: Callable[[], float], func: Callable[[], Any]) -> tuple[Any, float]:
    start = clock()
    value = func()
    return value, clock() - start


def _default_classifier(protocol: EvaluationProtocol, root: Path) -> Any:
    model = None if protocol.classifier_model_path is None else root / protocol.classifier_model_path
    return create_classifier(protocol.classifier_backend, model, report_path=root / protocol.classifier_report_path)


def _default_recognizer(protocol: EvaluationProtocol, root: Path) -> Any:
    return SpeechRecognizer(SpeechConfig(**protocol.speech_config), allow_download=False)


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:  # существующие результаты не заменяются
        stream.write(text)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def run_final_evaluation(
    freeze_path: Path, *, root: Path = ROOT, t11_paths: T11Paths | None = None,
    private_dir: Path = DEFAULT_PRIVATE_DIR, public_dir: Path = DEFAULT_PUBLIC_DIR,
    warmup_audio: Path | None = None, check_only: bool = False,
    adapters: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None,
    normalizations: Mapping[str, Callable[[str], list[str]]] | None = None,
    t11_verifier: Callable[[T11Paths], Mapping[str, Any]] = verify_t11,
    make_classifier: Callable[[EvaluationProtocol, Path], Any] = _default_classifier,
    make_recognizer: Callable[[EvaluationProtocol, Path], Any] = _default_recognizer,
    git: Callable[[Path], Mapping[str, Any]] = git_state,
    is_ignored: Callable[[Path, Path], bool | None] = git_ignore_status,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Финальная оценка T14. Порядок шагов обязателен:

    1. фиксация T12 и публичные входы (без приватных данных и моделей);
    2. штатная проверка T11 и сверка 36 записей;
    3. загрузка и прогрев моделей отдельно от основного прохода;
    4. оба режима для всех записей, повторная проверка T11;
    5. приватные построчные результаты и публичные агрегаты, без перезаписи.
    """
    normalizations = APPROVED_NORMALIZATIONS if normalizations is None else normalizations
    protocol = load_protocol(freeze_path, root=root, adapters=adapters, normalizations=normalizations)
    paths = t11_paths or T11Paths.defaults()
    frozen = check_frozen_inputs(protocol, root, composition_path=paths.report, private_dir=private_dir,
                                 git=git, is_ignored=is_ignored)
    targets = [private_dir / PRIVATE_RECORDS_NAME, public_dir / PUBLIC_METRICS_NAME, public_dir / PUBLIC_REPORT_NAME]
    _require(not any(p.exists() for p in targets), "Final T14 results already exist; they are never replaced")
    if warmup_audio is not None:
        frozen_hashes = {r.get("audio_sha256") for r in frozen["composition"].get("records", [])}
        _require(warmup_audio.is_file() and sha256_file(warmup_audio) not in frozen_hashes,
                 "Warm-up audio must be an existing file outside the T11 set")

    inputs = voice_inputs(t11_verifier(paths), frozen["composition"], protocol=protocol, audio_dir=paths.audio_dir)
    expected_ids = [item.record_id for item in inputs]
    if check_only:
        return {"status": "READY", "records": len(inputs), "freeze_sha256": protocol.freeze_sha256}

    classifier = make_classifier(protocol, root)
    recognizer = make_recognizer(protocol, root)
    setup: dict[str, float | None] = {}
    try:
        # Same eager load as the bot (src/bot.py::load_models).
        _, setup["classifier_load"] = _timed(clock, classifier._get_bundle)
        _, setup["classifier_warmup"] = _timed(clock, lambda: classifier.predict_batch(
            _smoke_texts(), max_text_length=protocol.max_text_length))
        _, setup["asr_load"] = _timed(clock, recognizer.load)
    except Exception as exc:
        raise EvaluationError(f"Model setup failed: {_error_kind(exc)}") from exc
    setup["asr_warmup"] = None
    if warmup_audio is not None:
        warm, setup["asr_warmup"] = _timed(clock, lambda: recognizer.transcribe(warmup_audio))
        if warm.status == "error":
            raise EvaluationError(f"Whisper warm-up failed: {warm.error_code}")

    records = evaluate_records(inputs, recognizer=recognizer, classifier=classifier,
                               tokenize=normalizations[protocol.normalization],
                               max_text_length=protocol.max_text_length, clock=clock)
    # The set must be unchanged after predictions as well.
    after = voice_inputs(t11_verifier(paths), frozen["composition"], protocol=protocol, audio_dir=paths.audio_dir)
    if after != inputs:
        raise DataMismatchError("T11 inputs changed during the evaluation; nothing written")

    run = {"git_commit": frozen["git_commit"], "model_setup_seconds": setup,
           "asr_warmup_audio_sha256": None if warmup_audio is None else sha256_file(warmup_audio),
           "device": protocol.speech_config["device"], "environment": environment()}
    summary = build_public_summary(records, protocol=protocol, run=run, expected_ids=expected_ids)
    public_json, report = _dump(summary), render_report(summary)
    assert_no_private_text([public_json, report],
                           [item.reference_text for item in inputs] + [r["asr_text"] for r in records])
    private = {"schema": "t14-voice-eval-records/draft-1", "freeze_sha256": protocol.freeze_sha256,
               "git_commit": frozen["git_commit"],
               "fields": [*RECORD_FIELDS, *PRIVATE_RECORD_FIELDS], "records": records}
    _write_new(targets[0], _dump(private))
    _write_new(targets[1], public_json)
    _write_new(targets[2], report)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T14: final Whisper impact evaluation on the T11 voice test")
    parser.add_argument("--freeze", type=Path, required=True, help="approved T12 freeze")
    parser.add_argument("--check", action="store_true",
                        help="verify preconditions and T11 only; no models are loaded")
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--warmup-audio", type=Path, help="Whisper warm-up file outside T11")
    parser.add_argument("--xlsx", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        defaults = T11Paths.defaults()
        paths = T11Paths(args.xlsx or defaults.xlsx, args.audio_dir or defaults.audio_dir,
                         args.manifest or defaults.manifest, defaults.report)
        result = run_final_evaluation(args.freeze, t11_paths=paths, private_dir=args.private_dir,
                                      public_dir=args.public_dir, warmup_audio=args.warmup_audio,
                                      check_only=args.check)
    except (PreconditionError, DataMismatchError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except EvaluationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print(json.dumps(result))
    else:
        results = result["results"]
        print(json.dumps({"records": results["records"], "reference_macro_f1": results["reference"]["macro_f1"],
                          "asr_macro_f1": results["asr"]["macro_f1"], "delta_macro_f1": results["delta_macro_f1"],
                          "wer": results["wer"]["wer"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
