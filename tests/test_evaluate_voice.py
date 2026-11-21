"""T14 на искусственных данных: подменные Whisper и классификатор, без моделей и T11.

Ожидаемые значения посчитаны вручную и записаны константами, а не получены
тестируемыми функциями.
"""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import evaluate_voice as ev
from src.classifier import ClassifierError, TextTooLongError
from src.speech import SpeechResult
from tools.report_nlp_data_quality import word_error_rate, words

ROOT = Path(__file__).resolve().parents[1]
REAL_FREEZE = ROOT / "reports" / "freeze.json"
SPEECH = json.loads((ROOT / "configs" / "speech.json").read_text(encoding="utf-8"))
IDS = [f"voice_{n:03d}" for n in range(1, 37)]


# --------------------------------------------------------------------------- #
# Подменные компоненты
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeClassifier:
    """predict_batch общего интерфейса; метка задаётся по тексту, время — шагом часов."""

    def __init__(self, clock, labels, *, raises=None, step=0.25):
        self.clock, self.labels, self.raises, self.step = clock, labels, raises or {}, step
        self.calls, self.loads = [], 0

    def _get_bundle(self):
        self.loads += 1
        self.clock.now += 3.0

    def predict_batch(self, texts, *, max_text_length):
        self.calls.append((list(texts), max_text_length))
        self.clock.now += self.step
        if texts[0] in self.raises:
            raise self.raises[texts[0]]
        return [{"label": self.labels.get(text, "neutral")} for text in texts]


class FakeRecognizer:
    """transcribe по имени файла; файл читается, но не меняется."""

    def __init__(self, clock, results):
        self.clock, self.results, self.loads, self.seen = clock, results, 0, []

    def load(self):
        self.loads += 1
        self.clock.now += 5.0

    def transcribe(self, path):
        path = Path(path)
        self.seen.append(path.read_bytes())
        status, text, code, elapsed = self.results[path.name]
        self.clock.now += elapsed
        return SpeechResult(status=status, text=text, error_code=code, duration_seconds=10.0,
                            elapsed_seconds=elapsed, model_load_seconds=0.0, segment_count=1 if text else 0)


def make_audio(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# WER и утверждённая нормализация T12
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reference, hypothesis, expected", [
    ("фильм очень хороший", "фильм очень хороший", (0, 0, 0)),          # точное совпадение
    ("фильм очень хороший", "фильм очень плохой", (1, 0, 0)),           # замена
    ("фильм очень хороший", "фильм хороший", (0, 1, 0)),                # удаление
    ("фильм хороший", "фильм очень хороший", (0, 0, 1)),                # вставка
    ("а б в г", "а х в г д", (1, 0, 1)),                                # замена и вставка
    ("один два три четыре пять", "два три шесть", (1, 2, 0)),           # несколько ошибок
    ("скучный и затянутый фильм", "", (0, 4, 0)),                       # пустая гипотеза
    ("", "лишние слова", (0, 0, 2)),                                    # пустой эталон
    ("", "", (0, 0, 0)),
])
def test_wer_counts_known_alignments(reference, hypothesis, expected):
    assert ev.wer_counts(reference.split(), hypothesis.split()) == expected


@pytest.mark.parametrize("reference, hypothesis, expected", [
    ("Фильм, очень хороший!", "фильм очень хороший", (0, 0, 0)),        # регистр и пунктуация
    ("Ёлки зелёные", "елки зеленые", (2, 0, 0)),                        # ё и е различаются
    ("кое-как снято", "коекак снято", (1, 1, 0)),                       # дефис разделяет слова
    ("в 2024 году", "в две тысячи двадцать четвёртом году", (1, 0, 3)),  # цифры не переводятся в слова
    ("один два три четыре пять", "два три шесть", (1, 2, 0)),
    ("скучный и затянутый фильм", "", (0, 4, 0)),
])
def test_wer_counts_sum_equals_existing_word_error_rate(reference, hypothesis, expected):
    counts = ev.wer_counts(words(reference), words(hypothesis))
    assert counts == expected
    # S + D + I совпадает с расстоянием существующей word_error_rate по тем же словам.
    reference_check = word_error_rate(reference, hypothesis)
    assert sum(counts) == reference_check["word_edits"]
    assert len(words(reference)) == reference_check["reference_words"]


@pytest.mark.parametrize("text, expected", [
    ("Фильм ОЧЕНЬ хороший", ["фильм", "очень", "хороший"]),
    ("Ёлки и всё", ["ёлки", "и", "всё"]),                    # ё не заменяется на е
    ("Ёж", ["ёж"]),                                    # Unicode NFC
    ("Ну, фильм... так себе!", ["ну", "фильм", "так", "себе"]),
    ("«Отлично» — сказал он", ["отлично", "сказал", "он"]),  # кавычки и тире — категория P
    ("кое-как", ["кое", "как"]),
    ("в 2024 году, 5 из 5", ["в", "2024", "году", "5", "из", "5"]),
    ("  много \t пробелов\n", ["много", "пробелов"]),
    ("", []),
    ("?!…", []),
])
def test_approved_t12_normalization_edge_cases(text, expected):
    normalize = ev.APPROVED_NORMALIZATIONS[ev.T12_NORMALIZATION]
    assert normalize is words  # переиспользуется существующая функция, логика не дублируется
    assert normalize(text) == expected


def test_t12_wer_keeps_yo_and_digits_literal():
    clock = Clock()
    item = ev.VoiceInput("syn_1", "positive", "Пять звёзд из пяти", Path("unused"))
    recognizer = type("R", (), {"transcribe": lambda self, path: SpeechResult("ok", "5 звезд из 5")})()
    record = ev.evaluate_record(item, recognizer=recognizer, classifier=FakeClassifier(clock, {}),
                                tokenize=words, max_text_length=2000, clock=clock)
    # пять -> 5, звёзд -> звезд, пяти -> 5: три замены, «из» совпадает.
    assert (record["wer_substitutions"], record["wer_deletions"], record["wer_insertions"]) == (3, 0, 0)
    assert record["reference_word_count"] == 4


def wer_record(record_id, status, words_count, s, d, i):
    return {"record_id": record_id, "asr_status": status, "reference_word_count": words_count,
            "wer_substitutions": s, "wer_deletions": d, "wer_insertions": i}


def test_corpus_wer_uses_total_words_not_mean_of_records():
    records = [wer_record("a", "ok", 2, 1, 0, 0),   # WER записи 0.5
               wer_record("b", "ok", 8, 0, 0, 0)]   # WER записи 0.0
    result = ev.corpus_wer(records)
    assert result["wer"] == pytest.approx(0.1)      # 1 / 10, а не (0.5 + 0) / 2 = 0.25
    assert result["wer"] != pytest.approx(0.25)
    assert (result["word_errors"], result["reference_words"], result["records_counted"]) == (1, 10, 2)


def test_empty_and_failed_asr_are_empty_hypotheses_counted_separately():
    records = [wer_record("a", "ok", 6, 1, 1, 1),
               wer_record("b", "empty", 4, 0, 4, 0),
               wer_record("c", "error", 5, 0, 5, 0)]
    result = ev.corpus_wer(records)
    assert result["wer"] == pytest.approx(12 / 15)  # (1+1+1) + 4 + 5 удалений из 6+4+5 слов
    assert (result["deletions"], result["records_counted"]) == (10, 3)  # ни одна запись не исключена
    assert (result["asr_failures"], result["empty_transcripts"]) == (1, 1)
    assert result["asr_failure_rule"] == "empty_hypothesis"


def test_corpus_wer_may_exceed_one():
    assert ev.corpus_wer([wer_record("a", "ok", 2, 2, 0, 3)])["wer"] == pytest.approx(5 / 2)


def test_corpus_wer_requires_reference_words():
    with pytest.raises(ev.DataMismatchError):
        ev.corpus_wer([wer_record("a", "ok", 0, 0, 0, 0)])


# --------------------------------------------------------------------------- #
# Метрики классификации
# --------------------------------------------------------------------------- #
def test_mode_metrics_hand_computed_with_errors():
    true = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    pred = ["negative", "error", "neutral", "negative", "error", "error"]
    result = ev.mode_metrics(true, pred)
    # negative: TP 1, предсказан 2 раза, support 2 -> P 0.5, R 0.5, F1 0.5
    # neutral:  TP 1, предсказан 1 раз,  support 2 -> P 1.0, R 0.5, F1 2/3
    # positive: TP 0 -> 0; macro-F1 = (0.5 + 2/3 + 0) / 3
    assert result["macro_f1"] == pytest.approx(7 / 18)
    assert result["accuracy"] == pytest.approx(2 / 6)
    assert result["per_class"] == {
        "negative": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
        "neutral": {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2},
        "positive": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2},
    }
    assert result["confusion_matrix"] == {
        "rows": ["negative", "neutral", "positive"],
        "columns": ["negative", "neutral", "positive", "error"],
        "values": [[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 2]],
    }
    assert (result["records"], result["correct"], result["incorrect"], result["wrong_label"],
            result["no_prediction"]) == (6, 2, 4, 1, 3)


def test_macro_f1_keeps_class_that_is_never_predicted():
    true = ["negative", "neutral", "positive", "negative", "neutral", "positive"]
    pred = ["negative", "negative", "positive", "negative", "negative", "positive"]
    result = ev.mode_metrics(true, pred)
    # negative F1 = 2/3, neutral F1 = 0 (нет предсказаний), positive F1 = 1.
    assert result["macro_f1"] == pytest.approx(5 / 9)   # не (2/3 + 1) / 2 по двум классам
    assert result["per_class"]["neutral"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}


def test_error_is_a_miss_not_neutral_and_not_a_fourth_class():
    result = ev.mode_metrics(["neutral", "neutral", "positive"], ["error", "error", "positive"])
    assert set(result["per_class"]) == {"negative", "neutral", "positive"}
    assert result["per_class"]["neutral"]["recall"] == 0.0  # «нейтрально» дало бы recall 1.0
    # negative: нет support и предсказаний -> 0; neutral 0; positive 1 -> 1/3
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"]["values"] == [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0]]
    assert ev.ERROR_ID == -1  # служебное предсказание протокола T12


@pytest.mark.parametrize("true, pred", [
    (["error"], ["neutral"]),                 # служебный статус не может быть истинной меткой
    (["neutral"], ["noise"]),                 # четвёртого класса нет
    (["neutral", "positive"], ["neutral"]),   # разная длина
    ([], []),
])
def test_mode_metrics_rejects_invalid_labels(true, pred):
    with pytest.raises(ev.DataMismatchError):
        ev.mode_metrics(true, pred)


# --------------------------------------------------------------------------- #
# Статусы, ошибки и полнота
# --------------------------------------------------------------------------- #
REF = {
    "r1": "отличный фильм с хорошим финалом",   # 5 слов
    "r2": "скучный и затянутый фильм",          # 4
    "r3": "обычный фильм без впечатлений",      # 4
    "r4": "просто шедевр",                      # 2
    "r5": "ужасная игра актёров",               # 3
}
HYP = {"r1": "отличный фильм с плохим финалом", "r4": "просто шедевр года", "r5": "ужасная игра актеров вообще"}
TRUE = {"r1": "positive", "r2": "negative", "r3": "neutral", "r4": "positive", "r5": "negative"}


@pytest.fixture
def five_cases(tmp_path):
    """r1 успех, r2 пустая транскрипция, r3 сбой Whisper, r4 сбой классификатора на эталоне,
    r5 сбой классификатора на результате Whisper."""
    clock = Clock()
    audio = tmp_path / "audio"
    inputs = [ev.VoiceInput(rid, TRUE[rid], REF[rid], make_audio(audio, f"{rid}.ogg", f"syn-{rid}".encode()))
              for rid in REF]
    labels = {REF["r1"]: "positive", HYP["r1"]: "positive", REF["r2"]: "negative", REF["r3"]: "neutral",
              HYP["r4"]: "positive", REF["r5"]: "negative"}
    classifier = FakeClassifier(clock, labels, raises={
        REF["r4"]: ClassifierError("Baseline prediction failed"),
        HYP["r5"]: TextTooLongError("too long"),
    })
    recognizer = FakeRecognizer(clock, {
        "r1.ogg": ("ok", HYP["r1"], None, 1.0),
        "r2.ogg": ("empty", "", None, 2.0),
        "r3.ogg": ("error", "", "decode_failed", 4.0),
        "r4.ogg": ("ok", HYP["r4"], None, 1.5),
        "r5.ogg": ("ok", HYP["r5"], None, 1.5),
    })
    records = ev.evaluate_records(inputs, recognizer=recognizer, classifier=classifier,
                                  tokenize=words, max_text_length=2000, clock=clock)
    return {"inputs": inputs, "records": records, "classifier": classifier, "recognizer": recognizer}


def test_five_outcomes_are_distinguished(five_cases):
    by_id = {r["record_id"]: r for r in five_cases["records"]}
    picked = lambda rid: tuple(by_id[rid][k] for k in (  # noqa: E731
        "reference_status", "reference_prediction", "asr_status", "asr_error_code",
        "asr_classification_status", "asr_prediction"))
    assert picked("r1") == ("ok", "positive", "ok", None, "ok", "positive")
    assert picked("r2") == ("ok", "negative", "empty", None, "not_run", "error")
    assert picked("r3") == ("ok", "neutral", "error", "decode_failed", "not_run", "error")
    assert picked("r4") == ("classifier_error", "error", "ok", None, "ok", "positive")
    assert picked("r5") == ("ok", "negative", "ok", None, "classifier_error", "error")
    assert by_id["r4"]["reference_error_kind"] == "ClassifierError"
    assert by_id["r5"]["asr_classifier_error_kind"] == "TextTooLongError"
    # S/D/I: r1 замена; r2 и r3 — пустая гипотеза, все 4 слова удалены; r4 вставка;
    # r5 «актёров» -> «актеров» (ё ≠ е) и вставка: сбой классификатора расшифровку не обнуляет.
    wer = {rid: (r["wer_substitutions"], r["wer_deletions"], r["wer_insertions"], r["reference_word_count"])
           for rid, r in by_id.items()}
    assert wer == {"r1": (1, 0, 0, 5), "r2": (0, 4, 0, 4), "r3": (0, 4, 0, 4),
                   "r4": (0, 0, 1, 2), "r5": (1, 0, 1, 3)}


def test_failed_records_stay_in_the_main_evaluation(five_cases):
    summary = ev.summarize(five_cases["records"], expected_ids=["r1", "r2", "r3", "r4", "r5"])
    assert summary["records"] == 5 and summary["record_ids"] == ["r1", "r2", "r3", "r4", "r5"]
    ref, asr = summary["reference"], summary["asr"]
    assert ref["records"] == asr["records"] == 5
    # Эталон: [pos, neg, neu, error, neg] при истинных [pos, neg, neu, pos, neg]
    # negative F1 1, neutral F1 1, positive P 1 R 0.5 F1 2/3 -> macro 8/9
    assert ref["macro_f1"] == pytest.approx(8 / 9)
    assert (ref["correct"], ref["wrong_label"], ref["no_prediction"]) == (4, 0, 1)
    # Whisper: [pos, error, error, pos, error] -> negative 0, neutral 0, positive 1 -> macro 1/3
    assert asr["macro_f1"] == pytest.approx(1 / 3)
    assert (asr["correct"], asr["wrong_label"], asr["no_prediction"]) == (2, 0, 3)
    assert summary["delta_macro_f1"] == pytest.approx(1 / 3 - 8 / 9)
    # WER: S 1 + 1, D 4 + 4, I 1 + 1, N 18 -> 12/18
    assert summary["wer"]["wer"] == pytest.approx(12 / 18)
    assert summary["wer"]["records_counted"] == 5
    failures = summary["failures"]
    assert failures["asr_status"] == {"ok": 3, "empty": 1, "error": 1}
    assert failures["asr_error_codes"] == {"decode_failed": 1}
    assert (failures["asr_error_record_ids"], failures["asr_empty_record_ids"]) == (["r3"], ["r2"])
    assert (failures["reference_classifier_errors"], failures["asr_classifier_errors"]) == (1, 1)
    assert failures["classifier_error_kinds"] == {"ClassifierError": 1, "TextTooLongError": 1}


def test_both_modes_use_one_classifier_with_identical_settings(five_cases):
    calls = five_cases["classifier"].calls
    # Эталон каждой записи и каждый успешный результат Whisper: 5 + 3 вызовов.
    assert [texts[0] for texts, _ in calls] == [
        REF["r1"], HYP["r1"], REF["r2"], REF["r3"], REF["r4"], HYP["r4"], REF["r5"], HYP["r5"]]
    assert {limit for _, limit in calls} == {2000}
    assert all(len(texts) == 1 for texts, _ in calls)


def test_timing_is_per_stage_and_excludes_model_loading(five_cases):
    by_id = {r["record_id"]: r for r in five_cases["records"]}
    # Классификатор: 0.25 s на вызов; Whisper: elapsed из SpeechResult.
    assert by_id["r1"]["reference_classifier_seconds"] == pytest.approx(0.25)
    assert by_id["r1"]["asr_seconds"] == 1.0 and by_id["r1"]["asr_classifier_seconds"] == pytest.approx(0.25)
    assert by_id["r1"]["total_seconds"] == pytest.approx(1.25)
    assert by_id["r3"]["asr_classifier_seconds"] is None and by_id["r3"]["total_seconds"] == pytest.approx(4.0)
    assert all(r["asr_model_load_seconds"] == 0.0 for r in by_id.values())
    timing = ev.timing_summary(five_cases["records"])
    # [1, 2, 4, 1.5, 1.5]: p95 linear = 2 + 0.8 * (4 - 2) = 3.6
    assert timing["asr_seconds"] == {"count": 5, "total": 10.0, "mean": 2.0, "median": 1.5,
                                     "p95": pytest.approx(3.6)}
    assert timing["total_seconds"]["total"] == pytest.approx(10.0 + 3 * 0.25)
    assert five_cases["classifier"].loads == 0 and five_cases["recognizer"].loads == 0


def test_timing_main_aggregates_cover_all_records_in_mixed_set(five_cases):
    timing = ev.timing_summary(five_cases["records"])
    # Все этапы — по всем 5 записям; r2 (пустой ASR) и r3 (сбой) дают 0 классификации ASR:
    # [0.25, 0, 0, 0.25, 0.25] -> median 0.25, p95 linear 0.25.
    assert {key: stats["count"] for key, stats in timing.items()} == dict.fromkeys(ev.TIMING_FIELDS, 5)
    assert timing["asr_classifier_seconds"] == {
        "count": 5, "total": pytest.approx(0.75), "mean": pytest.approx(0.15), "median": 0.25,
        "p95": pytest.approx(0.25),
        "executed_calls": {"count": 3, "total": pytest.approx(0.75), "mean": pytest.approx(0.25),
                           "median": 0.25, "p95": pytest.approx(0.25)}}
    # Сбой классификатора (r4, r5) — время до отказа, этап выполнен; прочие этапы выполнены везде.
    assert all("executed_calls" not in timing[key] for key in ev.TIMING_FIELDS if key != "asr_classifier_seconds")
    # Построчно невыполненный этап остаётся пустым со статусом not_run.
    by_id = {r["record_id"]: r for r in five_cases["records"]}
    assert [(by_id[rid]["asr_classification_status"], by_id[rid]["asr_classifier_seconds"])
            for rid in ("r2", "r3")] == [("not_run", None), ("not_run", None)]


def render_for(results):
    """Отчёт по синтетической сводке с минимальными сведениями о фиксации и запуске."""
    protocol = {"classifier_backend": "baseline", "speech_config": {"model": "m", "revision": "r"},
                "freeze_path": "reports/freeze.json", "freeze_sha256": "0" * 64,
                "t11_content_sha256": "1" * 64, "normalization": "t12/1"}
    return ev.render_report({"results": results, "protocol": protocol, "limitations": [],
                             "run": {"git_commit": "c", "model_setup_seconds": {}}})


def timing_record(record_id, asr_status, asr_seconds, asr_classifier_seconds):
    failed = asr_status != "ok"
    return valid_record(
        record_id=record_id, asr_status=asr_status, asr_error_code="decode_failed" if asr_status == "error" else None,
        asr_prediction="error" if failed else "positive",
        asr_classification_status="not_run" if failed else "ok",
        wer_substitutions=0 if failed else 1, wer_deletions=3 if failed else 0,
        asr_seconds=asr_seconds, reference_classifier_seconds=0.1,
        asr_classifier_seconds=asr_classifier_seconds,
        total_seconds=asr_seconds + (asr_classifier_seconds or 0.0))


@pytest.mark.parametrize("asr_status", ["empty", "error"])
def test_timing_with_only_unexecuted_asr_classification(asr_status):
    # Все записи — пустой ASR или технический сбой: классификация ASR не выполнялась ни разу.
    records = [timing_record(f"syn_{n}", asr_status, seconds, None) for n, seconds in enumerate((2.0, 3.0, 7.0))]
    timing = ev.timing_summary(records)
    assert timing["asr_classifier_seconds"] == {
        "count": 3, "total": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0,
        "executed_calls": {"count": 0, "total": 0, "mean": None, "median": None, "p95": None}}
    # Время ASR до пустого ответа или отказа учитывается: [2, 3, 7] -> p95 = 3 + 0.9 * 4 = 6.6
    assert timing["asr_seconds"] == {"count": 3, "total": 12.0, "mean": 4.0, "median": 3.0,
                                     "p95": pytest.approx(6.6)}
    assert timing["total_seconds"]["count"] == 3 and timing["total_seconds"]["total"] == 12.0
    assert "executed_calls" not in timing["asr_seconds"]


def test_timing_mixed_set_zero_contributions_and_report_rows():
    records = [timing_record("ok_1", "ok", 1.0, 0.2), timing_record("empty_1", "empty", 2.0, None),
               timing_record("error_1", "error", 4.0, None), timing_record("ok_2", "ok", 3.0, 0.4)]
    timing = ev.timing_summary(records)
    # [0.2, 0, 0, 0.4]: median 0.1, p95 linear = 0.2 + 0.85 * 0.2 = 0.37
    main = timing["asr_classifier_seconds"]
    assert (main["count"], main["total"], main["mean"], main["median"], main["p95"]) == (
        4, pytest.approx(0.6), pytest.approx(0.15), pytest.approx(0.1), pytest.approx(0.37))
    executed = main["executed_calls"]
    assert (executed["count"], executed["median"], executed["p95"]) == (
        2, pytest.approx(0.3), pytest.approx(0.39))
    assert timing["total_seconds"]["total"] == pytest.approx(10.6)

    summary = ev.summarize(records)
    assert summary["timing"] == timing
    lines = render_for(summary).splitlines()
    assert "| asr_classifier_seconds | 4 | 0.600 | 0.150 | 0.100 | 0.370 |" in lines
    assert "| asr_classifier_seconds, только выполненные вызовы | 2 | 0.600 | 0.300 | 0.300 | 0.390 |" in lines
    assert any("нулевым временем" in line for line in lines)


def test_timing_all_successful_keeps_previous_shape():
    records = [timing_record(f"syn_{n}", "ok", 1.0 + n, 0.1 * (n + 1)) for n in range(3)]
    timing = ev.timing_summary(records)
    assert all(set(stats) == {"count", "total", "mean", "median", "p95"} for stats in timing.values())
    assert timing["asr_classifier_seconds"]["count"] == 3
    assert timing["asr_classifier_seconds"]["median"] == pytest.approx(0.2)
    report = render_for(ev.summarize(records))
    assert "только выполненные вызовы" not in report and "нулевым временем" not in report


def test_unexpected_classifier_exception_is_not_masked(tmp_path):
    clock = Clock()

    class Broken(FakeClassifier):
        def predict_batch(self, texts, *, max_text_length):
            raise RuntimeError("bug")

    item = ev.VoiceInput("syn_1", "neutral", "обычный фильм", make_audio(tmp_path, "a.ogg", b"x"))
    with pytest.raises(RuntimeError):
        ev.evaluate_record(item, recognizer=FakeRecognizer(clock, {"a.ogg": ("ok", "фильм", None, 1.0)}),
                           classifier=Broken(clock, {}), tokenize=words, max_text_length=2000, clock=clock)


def test_evaluation_does_not_change_inputs(five_cases, tmp_path):
    inputs = five_cases["inputs"]
    before = copy.deepcopy(inputs)
    payloads = [p.read_bytes() for p in (i.audio_path for i in inputs)]
    assert inputs == before
    assert five_cases["recognizer"].seen == payloads == [f"syn-{rid}".encode() for rid in REF]


def valid_record(**changes):
    record = {
        "record_id": "syn_1", "true_label": "neutral",
        "reference_prediction": "neutral", "reference_status": "ok", "reference_error_kind": None,
        "asr_status": "ok", "asr_error_code": None,
        "asr_prediction": "positive", "asr_classification_status": "ok", "asr_classifier_error_kind": None,
        "wer_substitutions": 1, "wer_deletions": 0, "wer_insertions": 0, "reference_word_count": 3,
        "asr_seconds": 1.0, "asr_model_load_seconds": 0.0, "reference_classifier_seconds": 0.1,
        "asr_classifier_seconds": 0.1, "total_seconds": 1.1, "audio_duration_seconds": 5.0,
    }
    record.update(changes)
    return record


@pytest.mark.parametrize("changes, field", [
    ({"asr_status": "empty", "asr_classification_status": "not_run", "asr_prediction": "neutral",
      "asr_classifier_seconds": None, "wer_substitutions": 0, "wer_deletions": 3}, "asr_prediction"),
    ({"asr_status": "empty", "asr_classification_status": "not_run", "asr_prediction": "error",
      "asr_classifier_seconds": None, "wer_substitutions": 0, "wer_deletions": 1}, "wer_deletions"),
    ({"asr_status": "error", "asr_classification_status": "not_run", "asr_prediction": "error",
      "asr_classifier_seconds": None}, "wer_deletions"),          # сбой обязан дать D = N
    ({"asr_status": "error", "asr_classification_status": "not_run", "asr_prediction": "error",
      "asr_classifier_seconds": None, "wer_substitutions": None, "wer_deletions": None,
      "wer_insertions": None}, "wer_substitutions"),             # сбой нельзя вывести из WER
    ({"asr_status": "error", "asr_classification_status": "ok"}, "asr_classification_status"),
    ({"reference_status": "classifier_error"}, "reference_prediction"),
    ({"reference_prediction": "error"}, "reference_prediction"),
    ({"true_label": "error"}, "true_label"),
    ({"asr_prediction": "noise"}, "asr_prediction"),
    ({"asr_status": "timeout"}, "asr_status"),
    ({"wer_substitutions": 2, "wer_deletions": 2}, "wer_substitutions"),
    ({"total_seconds": -1.0}, "total_seconds"),
    ({"asr_seconds": float("nan")}, "asr_seconds"),
    ({"reference_word_count": True}, "reference_word_count"),
    ({"reference_word_count": 0, "wer_substitutions": 0}, "reference_word_count"),  # пустой эталон
])
def test_validate_records_rejects_contradictions(changes, field):
    ev.validate_records([valid_record()])
    with pytest.raises(ev.DataMismatchError, match=f"syn_1: inconsistent {field}"):
        ev.validate_records([valid_record(**changes)])


def test_validate_records_requires_complete_matching_ids():
    record = valid_record()
    del record["asr_status"]
    with pytest.raises(ev.DataMismatchError, match="missing fields asr_status"):
        ev.validate_records([record])
    with pytest.raises(ev.DataMismatchError, match="Duplicate"):
        ev.validate_records([valid_record(), valid_record()])
    pair = [valid_record(record_id="a"), valid_record(record_id="b")]
    ev.validate_records(pair, ["a", "b"])
    with pytest.raises(ev.DataMismatchError, match="missing 1, extra 0"):
        ev.validate_records(pair, ["a", "b", "c"])
    with pytest.raises(ev.DataMismatchError, match="missing 0, extra 1"):
        ev.validate_records(pair, ["a"])
    with pytest.raises(ev.DataMismatchError, match="different order"):
        ev.validate_records(pair, ["b", "a"])
    with pytest.raises(ev.DataMismatchError):
        ev.validate_records([])


def test_public_guard_rejects_private_text_and_record_fields():
    texts = ["отличный фильм с хорошим финалом"]
    ev.assert_no_private_text(['{"wer": 0.1}', "Фильм", "record_id,reference_status,asr_status\n"], texts)
    with pytest.raises(ev.EvaluationError, match="private transcript"):
        ev.assert_no_private_text(["примечание: отличный  фильм с\nхорошим финалом"], texts)
    with pytest.raises(ev.EvaluationError, match="per-record"):
        ev.assert_no_private_text(['{"true_label": "neutral"}'], [])
    with pytest.raises(ev.EvaluationError, match="per-record"):
        ev.assert_no_private_text(["record_id,true_label\nvoice_001,1\n"], [])


def test_public_predictions_use_t12_ids_and_no_texts(five_cases):
    rows = list(csv.DictReader(io.StringIO(ev.public_predictions_csv(five_cases["records"]))))
    assert list(rows[0]) == list(ev.PUBLIC_RECORD_FIELDS)
    # Вариант В1: ни эталонных меток, ни текстов, ни индивидуальных предсказаний.
    assert not {"true_label", "asr_text", "reference_prediction", "asr_prediction"} & set(rows[0])
    assert [row["record_id"] for row in rows] == ["r1", "r2", "r3", "r4", "r5"]
    by_id = {row["record_id"]: row for row in rows}
    # Статусы, коды ошибок и время сохраняются.
    assert (by_id["r2"]["asr_status"], by_id["r2"]["asr_classification_status"]) == ("empty", "not_run")
    assert (by_id["r3"]["asr_status"], by_id["r3"]["asr_error_code"]) == ("error", "decode_failed")
    assert (by_id["r4"]["reference_status"], by_id["r4"]["reference_error_kind"]) == (
        "classifier_error", "ClassifierError")
    assert by_id["r5"]["asr_classifier_error_kind"] == "TextTooLongError"
    assert by_id["r3"]["asr_classifier_seconds"] == "" and float(by_id["r1"]["total_seconds"]) == 1.25


def test_public_guard_rejects_withheld_prediction_columns():
    with pytest.raises(ev.EvaluationError, match="per-record"):
        ev.assert_no_private_text(["record_id,reference_prediction\nvoice_001,1\n"], [])
    with pytest.raises(ev.EvaluationError, match="per-record"):
        ev.assert_no_private_text(["record_id,asr_status,asr_prediction\n"], [])


ORIGINAL_HEADER = ",".join(ev.ORIGINAL_PUBLIC_RECORD_FIELDS)


def test_redaction_only_drops_prediction_columns():
    original = (ORIGINAL_HEADER + "\n"
                "a,2,ok,,ok,,2,ok,,1.5,0.001,0.002,1.502\n"
                "b,-1,classifier_error,ClassifierError,error,decode_failed,-1,not_run,,4.0,0.003,,4.0\n")
    assert ev.redact_public_predictions_csv(original) == (
        ",".join(ev.PUBLIC_RECORD_FIELDS) + "\n"
        "a,ok,,ok,,ok,,1.5,0.001,0.002,1.502\n"
        "b,classifier_error,ClassifierError,error,decode_failed,not_run,,4.0,0.003,,4.0\n")
    with pytest.raises(ev.DataMismatchError, match="original T14 format"):
        ev.redact_public_predictions_csv(",".join(ev.PUBLIC_RECORD_FIELDS) + "\n")
    with pytest.raises(ev.DataMismatchError, match="width"):
        ev.redact_public_predictions_csv(ORIGINAL_HEADER + "\na,2\n")


# --------------------------------------------------------------------------- #
# Фиксация T12: адаптер и отказы
# --------------------------------------------------------------------------- #
def test_t12_adapter_and_normalization_are_registered(tmp_path):
    assert set(ev.PROTOCOL_ADAPTERS) == {"t13-final-text-eval-input/draft-1"}
    assert ev.APPROVED_NORMALIZATIONS == {"t12/1": words}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"schema": "t12-freeze/1", "status": "approved"}), encoding="utf-8")
    with pytest.raises(ev.PreconditionError, match="No approved T12 protocol adapter"):
        ev.load_protocol(freeze, root=tmp_path)


@pytest.mark.skipif(not REAL_FREEZE.is_file(), reason="reports/freeze.json (T12) is not in this checkout")
def test_adapter_reads_the_approved_t12_freeze():
    # Только публичная фиксация; приватные данные T11 и модели не читаются.
    protocol = ev.load_protocol(REAL_FREEZE, root=ROOT)
    assert protocol.freeze_sha256 == ev.ACCEPTED_FREEZE_SHA256
    assert protocol.approved is True and protocol.classifier_backend == "baseline"
    assert protocol.classifier_model_path == "models/nlp_baseline/tfidf_logreg_baseline.joblib"
    assert protocol.classifier_model_sha256 == "389a768bcd108721c79ebf3a7b9dc5212ec100f5c2e1b06cf519ed076cd9bee3"
    assert protocol.classifier_report_path == "reports/nlp/training/run_metadata.json"
    assert protocol.max_text_length == 2000
    assert dict(protocol.speech_config) == SPEECH
    assert set(protocol.speech_bundle_files) == {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
    assert protocol.t11_report_sha256 == sha256((ROOT / "reports/voice_test/composition.json").read_bytes())
    assert protocol.record_ids == tuple(IDS)
    assert protocol.normalization == "t12/1"
    assert protocol.warmup_audio_sha256 == "d6a29c51a3e22a08dafcbfdd045769bd4c6481f982e9ec03b7f91f382a0fa530"
    assert protocol.warmup_audio_bytes == 204813
    assert protocol.frozen_commit == "5f22d93a8cff63b30fc129bb68a1cfbd4230995d"


def spy(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called before the preconditions pass")
    return fail


def guarded_run(freeze, tmp_path, **overrides):
    kwargs = dict(
        root=tmp_path, t11_paths=ev.T11Paths(tmp_path / "t11.xlsx", tmp_path / "raw_audio",
                                            tmp_path / "manifest.json", tmp_path / "composition.json"),
        private_dir=tmp_path / "private", public_dir=tmp_path / "public",
        t11_verifier=spy("T11 validator"), make_classifier=spy("classifier"), make_recognizer=spy("Whisper"),
        git=spy("git"), is_ignored=spy("git check-ignore"), environment_check=spy("environment check"),
    )
    kwargs.update(overrides)
    return ev.run_final_evaluation(freeze, **kwargs)


def test_final_run_refuses_without_freeze_before_private_data_or_models(tmp_path):
    with pytest.raises(ev.PreconditionError, match="T12 freeze not found"):
        guarded_run(tmp_path / "missing.json", tmp_path)
    assert not (tmp_path / "private").exists() and not (tmp_path / "public").exists()


@pytest.mark.parametrize("content", ["not json", "[]", '{"schema": null}', '{"schema": "t12-freeze/1"}'])
def test_final_run_refuses_unreadable_or_unsupported_freeze(tmp_path, content):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(content, encoding="utf-8")
    with pytest.raises(ev.PreconditionError):
        guarded_run(freeze, tmp_path)


def test_cli_refuses_without_freeze(tmp_path, capsys):
    code = ev.main(["--freeze", str(tmp_path / "missing.json"), "--private-dir", str(tmp_path / "p"),
                    "--public-dir", str(tmp_path / "q")])
    assert code == 2
    assert "REFUSED: T12 freeze not found" in capsys.readouterr().err
    assert not (tmp_path / "p").exists() and not (tmp_path / "q").exists()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256(data)


@pytest.fixture
def repo(tmp_path):
    """Искусственный репозиторий с фиксацией в схеме T12, комплектами, записью прогрева
    и 36 записями, не связанными с T11."""
    root = tmp_path / "repo"
    audio_dir = root / "artifacts" / "t11" / "raw_audio"
    labels = ["negative", "neutral", "positive"]
    records = []
    for n, rid in enumerate(IDS, 1):
        payload = f"synthetic audio {n}".encode()
        make_audio(audio_dir, f"{rid}.ogg", payload)
        records.append({"record_id": rid, "final_label": labels[n % 3], "audio_sha256": sha256(payload),
                        "reference_transcript": f"искусственный отзыв номер {n} для проверки"})
    manifest = {"label_ids": {"negative": 0, "neutral": 1, "positive": 2}, "content_sha256": "c" * 64,
                "source_xlsx_sha256": "b" * 64, "records": records}
    composition = {"content_sha256": "c" * 64,
                   "records": [{"record_id": r["record_id"], "audio_sha256": r["audio_sha256"]} for r in records]}

    pinned = {}
    for rel in ev.T14_PINNED_INPUTS:
        data = {"configs/speech.json": (ROOT / "configs/speech.json").read_bytes(),
                "reports/voice_test/composition.json": json.dumps(composition).encode()}.get(
            rel, f"synthetic {rel}\n".encode())
        pinned[rel] = put(root / rel, data)
    report_rel, model_rel = "reports/nlp/training/run_metadata.json", "models/nlp_baseline/model.joblib"
    pinned[report_rel] = put(root / report_rel, b'{"bundle_sha256": "synthetic"}')
    model = b"synthetic baseline bundle"
    put(root / model_rel, model)
    snapshot = ev.whisper_snapshot_dir(SPEECH, root)
    bundle = {}
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        data = f"synthetic whisper {name}".encode()
        bundle[name] = {"sha256": put(snapshot / name, data), "bytes": len(data)}
    warmup_data = b"synthetic T16 debug recording"
    warmup = tmp_path / "t16" / "t16_debug.ogg"
    put(warmup, warmup_data)

    base = {
        "schema": "t13-final-text-eval-input/draft-1", "protocol_version": "t12/1",
        "status": "approved", "pending_approval": [], "approved_by": ["coordinator"],
        "confirmed_decisions": {"bot_model": "baseline"}, "bot_model": "baseline",
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2}, "frozen_commit": "f" * 40,
        "models": {"baseline": {"model_path": model_rel, "model_sha256": sha256(model), "bytes": len(model),
                                "report_path": report_rel, "report_sha256": pinned[report_rel]}},
        "runtime_environment_variables": {"OMP_NUM_THREADS": "1"},
        "environment": {"machine": {"python": "3.13.7"}, "packages": {"numpy": "2.3.3"}},
        "voice_test": {"composition_path": "reports/voice_test/composition.json",
                       "composition_sha256": pinned["reports/voice_test/composition.json"],
                       "content_sha256": "c" * 64, "xlsx_sha256": "b" * 64, "rows": 36, "record_ids": IDS,
                       "class_counts": {"negative": 12, "neutral": 12, "positive": 12}},
        "speech": {"config": SPEECH, "bundle_files": bundle},
        "voice_protocol": {
            "backend": "baseline", "modes": ["reference_text", "whisper_text"], "records": 36,
            "batch_size": 1, "max_text_length": 2000, "target_labels": [0, 1, 2], "error_prediction": -1,
            "zero_division": 0,
            "wer": {"implementation": "tools.report_nlp_data_quality.words / word_error_rate",
                    "normalization": ["Unicode NFC", "casefold", "Unicode punctuation category P → space",
                                      "split whitespace"],
                    "yo_equals_ye": False, "numbers": "literal", "may_exceed_one": True,
                    "empty_or_asr_failure": "Пустая гипотеза: все слова эталона — удаления."},
            "warmup": {"audio_sha256": sha256(warmup_data), "audio_bytes": len(warmup_data),
                       "include_in_metrics": False},
        },
        "file_sha256": pinned,
    }
    freeze = root / "reports" / "freeze.json"
    clock = Clock()
    asr_text = {r["record_id"]: r["reference_transcript"].replace("проверки", "проверка") for r in records}
    results = {f"{rid}.ogg": ("ok", text, None, 1.0) for rid, text in asr_text.items()}
    results["voice_005.ogg"] = ("error", "", "inference_failed", 2.0)
    results["voice_006.ogg"] = ("empty", "", None, 2.0)
    results[warmup.name] = ("ok", "отладочная запись", None, 0.5)
    true = {r["reference_transcript"]: r["final_label"] for r in records}
    classifier = FakeClassifier(clock, {**true, **{asr_text[r["record_id"]]: r["final_label"] for r in records}})
    recognizer = FakeRecognizer(clock, results)
    verifications, git_calls = [], []

    def verifier(paths):
        verifications.append(paths)
        return copy.deepcopy(manifest)

    def git(path, frozen_commit):
        git_calls.append(frozen_commit)
        return {"commit": "a" * 40, "dirty": False, "frozen_commit_is_ancestor": True}

    kwargs = dict(
        root=root,
        t11_paths=ev.T11Paths(root / "artifacts/t11/T11_dataset.xlsx", audio_dir,
                              root / "artifacts/t11/manifest.json", root / "reports/voice_test/composition.json"),
        private_dir=root / "artifacts" / "t14", public_dir=root / "reports" / "voice_eval",
        warmup_audio=warmup, t11_verifier=verifier,
        make_classifier=lambda p, r: classifier, make_recognizer=lambda p, r: recognizer,
        git=git, is_ignored=lambda path, r: True, environment_check=lambda protocol: None, clock=clock,
    )

    def write_freeze(change=None):
        data = copy.deepcopy(base)
        if change is not None:
            change(data)
        raw = json.dumps(data, ensure_ascii=False).encode()
        put(freeze, raw)
        kwargs["accepted_freeze_sha256"] = sha256(raw)  # принятой считается именно эта синтетическая версия
        return freeze

    write_freeze()
    return {"root": root, "freeze": freeze, "write_freeze": write_freeze, "kwargs": kwargs, "manifest": manifest,
            "verifications": verifications, "git_calls": git_calls, "classifier": classifier,
            "recognizer": recognizer, "warmup": warmup, "snapshot": snapshot, "model": root / model_rel,
            "texts": list(true) + list(asr_text.values())}


def no_private_access(repo, **overrides):
    """Аргументы запуска, при которых обращение к T11 или моделям — ошибка теста."""
    return {**repo["kwargs"], "t11_verifier": spy("T11 validator"),
            "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper"), **overrides}


def test_adapter_extracts_the_t12_voice_protocol(repo):
    protocol = ev.load_protocol(repo["freeze"], root=repo["root"],
                                accepted_sha256=repo["kwargs"]["accepted_freeze_sha256"])
    assert (protocol.classifier_backend, protocol.max_text_length, protocol.normalization) == (
        "baseline", 2000, "t12/1")
    assert protocol.classifier_model_path == "models/nlp_baseline/model.joblib"
    assert protocol.record_ids == tuple(IDS)
    assert protocol.warmup_audio_bytes == len(b"synthetic T16 debug recording")
    assert dict(protocol.runtime_environment) == {"OMP_NUM_THREADS": "1"}


def test_final_run_produces_private_records_and_public_results(repo):
    summary = ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])
    assert len(repo["verifications"]) == 2  # T11 проверяется до и после прогона
    assert repo["git_calls"] == ["f" * 40]  # происхождение от frozen_commit фиксации
    assert repo["classifier"].loads == 1 and repo["recognizer"].loads == 1
    # Загрузка (3 s / 5 s) и прогрев учтены отдельно от построчного времени.
    setup = summary["run"]["model_setup_seconds"]
    assert setup["classifier_load"] == 3.0 and setup["asr_load"] == 5.0
    assert setup["classifier_warmup"] == pytest.approx(0.25) and setup["asr_warmup"] == 0.5
    # Прогрев Whisper — один проход записи T16 до основного прохода, вне метрик.
    assert repo["recognizer"].seen[0] == repo["warmup"].read_bytes() and len(repo["recognizer"].seen) == 37
    assert summary["run"]["asr_warmup_audio_sha256"] == sha256(repo["warmup"].read_bytes())

    results = summary["results"]
    assert results["records"] == 36 and results["record_ids"] == IDS
    # Эталон: 36 верных; Whisper: 34 верных и 2 без предсказания (voice_005 positive, voice_006 negative).
    assert results["reference"]["macro_f1"] == 1.0
    assert (results["asr"]["correct"], results["asr"]["no_prediction"]) == (34, 2)
    # positive и negative: P 1, R 11/12 -> F1 22/23; neutral 1 -> macro (2*22/23 + 1)/3
    assert results["asr"]["macro_f1"] == pytest.approx((2 * 22 / 23 + 1) / 3)
    # WER: 34 замены «проверки -> проверка», 6 + 6 удалений (сбой и пустой ответ); по 6 слов в 36 записях
    assert results["wer"]["wer"] == pytest.approx((34 + 12) / 216)
    # p95 linear для 34 × 1.0 и 2 × 2.0 s
    assert results["timing"]["asr_seconds"]["p95"] == pytest.approx(1.25)

    private = json.loads((repo["root"] / "artifacts/t14/records.json").read_text(encoding="utf-8"))
    assert [r["record_id"] for r in private["records"]] == results["record_ids"]
    assert set(private["fields"]) == set(private["records"][0])
    public_dir = repo["root"] / "reports/voice_eval"
    public = (public_dir / "metrics.json").read_text(encoding="utf-8")
    predictions = (public_dir / "predictions.csv").read_text(encoding="utf-8")
    report = (public_dir / "report.md").read_text(encoding="utf-8")
    for text in repo["texts"]:
        assert text not in public and text not in report and text not in predictions
    for key in ("true_label", "asr_text", "reference_prediction", "asr_prediction"):
        assert key not in public
    rows = list(csv.DictReader(io.StringIO(predictions)))
    assert [row["record_id"] for row in rows] == IDS and list(rows[0]) == list(ev.PUBLIC_RECORD_FIELDS)
    assert (rows[4]["asr_status"], rows[4]["asr_classification_status"]) == ("error", "not_run")
    # Приватные результаты по-прежнему полны: метки и предсказания обоих режимов.
    assert {"true_label", "reference_prediction", "asr_prediction", "asr_text"} <= set(private["fields"])
    assert private["records"][4]["asr_prediction"] == "error"
    assert "Малый размер голосового теста" in report and "ΔF1" in report and "p95" in report
    assert json.loads(public)["protocol"]["normalization"] == "t12/1"

    # Повторный запуск не заменяет результаты.
    with pytest.raises(ev.PreconditionError, match="never replaced"):
        ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])


def test_final_run_blocks_public_files_when_a_failure_reveals_a_label(repo):
    # Единственный сбой Whisper (voice_005, positive): столбец error матрицы ASR занят одной
    # строкой, поэтому публичный статус сбоя раскрыл бы эталонную метку этой записи.
    repo["recognizer"].results["voice_006.ogg"] = ("ok", "искусственный отзыв номер 6 для проверка", None, 1.0)
    with pytest.raises(ev.PublicationBlockedError, match=r"label of 1 record\(s\)") as caught:
        ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])
    assert "voice_" not in str(caught.value)  # без ID и меток
    public_dir = repo["root"] / "reports/voice_eval"
    assert not any((public_dir / name).exists() for name in (
        ev.PUBLIC_METRICS_NAME, ev.PUBLIC_PREDICTIONS_NAME, ev.PUBLIC_REPORT_NAME))
    # Приватные результаты единственного прогона сохранены целиком; сбой не скрыт и запись не исключена.
    private = json.loads((repo["root"] / "artifacts/t14/records.json").read_text(encoding="utf-8"))
    assert [r["record_id"] for r in private["records"]] == IDS
    assert (private["records"][4]["asr_status"], private["records"][4]["asr_prediction"]) == ("error", "error")
    # Повторный запуск не обходит блокировку.
    with pytest.raises(ev.PreconditionError, match="never replaced"):
        ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])


def test_cli_reports_blocked_publication(monkeypatch, capsys):
    def blocked(*args, **kwargs):
        raise ev.PublicationBlockedError("Public results would reveal the reference label of 1 record(s)")

    monkeypatch.setattr(ev, "run_final_evaluation", blocked)
    assert ev.main([]) == 3
    assert "BLOCKED: Public results would reveal" in capsys.readouterr().err


def public_results(ref_values, asr_values, failures=None):
    columns = [*ev.LABELS, ev.ERROR]
    return {mode: {"confusion_matrix": {"rows": list(ev.LABELS), "columns": columns, "values": values}}
            for mode, values in (("reference", ref_values), ("asr", asr_values))} | {
        "failures": {"asr_error_record_ids": [], "asr_empty_record_ids": [], **(failures or {})}}


def public_row(record_id, reference_status="ok", asr_status="ok", asr_classification_status="ok"):
    return {"record_id": record_id, "reference_status": reference_status, "asr_status": asr_status,
            "asr_classification_status": asr_classification_status}


def public_csv(rows, fields=ev.PUBLIC_RECORD_FIELDS):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fields, restval="", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def test_disclosure_check_detects_original_format_mechanism():
    # a — negative, b — neutral; эталон верно даёт negative обоим, ASR путает b с positive.
    # Пара предсказаний по ID плюс две матрицы однозначно дают обе метки.
    rows = [{**public_row("a"), "reference_prediction": "0", "asr_prediction": "0"},
            {**public_row("b"), "reference_prediction": "0", "asr_prediction": "2"}]
    results = public_results([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                             [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    assert ev.disclosed_record_ids(results, public_csv(rows, ev.ORIGINAL_PUBLIC_RECORD_FIELDS)) == ["a", "b"]
    # Те же агрегаты в варианте В1: обе записи неразличимы.
    assert ev.disclosed_record_ids(results, public_csv([public_row("a"), public_row("b")])) == []


@pytest.mark.parametrize("rows, failures, ref, asr, expected", [
    # технический сбой Whisper, столбец error занят одной строкой
    ([public_row("a"), public_row("b", asr_status="error", asr_classification_status="not_run")],
     {"asr_error_record_ids": ["b"]},
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], ["a", "b"]),
    # пустой ответ Whisper известен только из списка ID в metrics.json
    ([public_row("a"), public_row("b"), public_row("c")], {"asr_empty_record_ids": ["c"]},
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], ["c"]),
    # сбой классификатора на эталоне
    ([public_row("a", reference_status="classifier_error"), public_row("b"), public_row("c")], {},
     [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], ["a"]),
    # два сбоя в разных классах и два успеха в тех же классах: ни одна метка не однозначна
    ([public_row("a"), public_row("b", asr_status="error", asr_classification_status="not_run"),
      public_row("c", asr_status="empty", asr_classification_status="not_run"), public_row("d")], {},
     [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]], [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1]], []),
])
def test_disclosure_check_on_failure_statuses(rows, failures, ref, asr, expected):
    predictions = public_csv(rows)
    results = public_results(ref, asr, failures)
    assert ev.disclosed_record_ids(results, predictions) == expected
    if expected:
        with pytest.raises(ev.PublicationBlockedError, match=f"label of {len(expected)} record"):
            ev.assert_no_label_disclosure(results, predictions)
    else:
        ev.assert_no_label_disclosure(results, predictions)


def test_disclosure_check_rejects_files_inconsistent_with_matrices():
    rows = [public_row("a", asr_status="error", asr_classification_status="not_run")]
    with pytest.raises(ev.DataMismatchError, match="contradict"):
        ev.disclosed_record_ids(public_results([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                               [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), public_csv(rows))


def test_check_mode_stops_before_models(repo):
    kwargs = {**repo["kwargs"], "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper")}
    result = ev.run_final_evaluation(repo["freeze"], check_only=True, **kwargs)
    assert result["status"] == "READY" and result["records"] == 36
    assert result["warmup_audio_sha256"] == sha256(repo["warmup"].read_bytes())
    assert len(repo["verifications"]) == 1
    assert not (repo["root"] / "artifacts/t14").exists()


def test_final_run_accepts_only_the_accepted_freeze_version(repo):
    with pytest.raises(ev.PreconditionError, match="accepted version"):
        ev.run_final_evaluation(repo["freeze"], **no_private_access(repo, accepted_freeze_sha256="0" * 64))


@pytest.mark.parametrize("change, message", [
    (lambda f: f.update(status="draft"), "not approved"),
    (lambda f: f.update(pending_approval=["coordinator"]), "not approved"),
    (lambda f: f.update(protocol_version="t12/2"), "header"),
    (lambda f: f.update(label_mapping={"negative": 0, "positive": 1, "neutral": 2}), "header"),
    (lambda f: f["confirmed_decisions"].update(bot_model="rubert"), "confirmed_decisions"),
    (lambda f: f["voice_protocol"].update(backend="rubert"), "voice_protocol"),
    (lambda f: f["voice_protocol"].update(error_prediction=3), "voice_protocol"),
    (lambda f: f["voice_protocol"].update(batch_size=8), "voice_protocol"),
    (lambda f: f["voice_protocol"]["wer"].update(yo_equals_ye=True), "voice_protocol.wer"),
    (lambda f: f["voice_protocol"]["wer"].update(numbers="words"), "voice_protocol.wer"),
    (lambda f: f["voice_protocol"]["warmup"].update(include_in_metrics=True), "voice_protocol.warmup"),
    (lambda f: f["voice_protocol"]["warmup"].update(audio_sha256="abc"), "warmup_audio_sha256"),
    (lambda f: f["voice_protocol"].update(max_text_length=None), "max_text_length"),
    (lambda f: f.pop("voice_protocol"), "voice_protocol"),
    (lambda f: f["voice_test"].update(rows=35), "voice_test"),
    (lambda f: f["voice_test"]["class_counts"].update(neutral=11), "voice_test"),
    (lambda f: f["voice_test"]["record_ids"].pop(), "record_ids"),
    (lambda f: f["models"]["baseline"].update(report_path="/etc/report.json"), "relative"),
    (lambda f: f["models"]["baseline"].update(model_path="../model.joblib"), "relative"),
    (lambda f: f["models"]["baseline"].update(report_sha256="abc"), "SHA-256"),
    (lambda f: f["speech"].update(config={"model": "x"}), "speech_config"),
    (lambda f: f["speech"]["bundle_files"]["model.bin"].update(bytes=0), "speech_bundle_files"),
    (lambda f: f.update(frozen_commit="abc"), "frozen_commit"),
    (lambda f: f["file_sha256"].pop("tools/report_nlp_data_quality.py"), "not pinned"),
    (lambda f: f["models"]["baseline"].update(report_sha256="0" * 64), "inconsistent"),
    (lambda f: f["voice_test"].update(composition_sha256="0" * 64), "inconsistent"),
    (lambda f: f["voice_test"].update(content_sha256="d" * 64), "content_sha256 differs"),
    (lambda f: f.pop("runtime_environment_variables"), "runtime_environment_variables"),
])
def test_final_run_refuses_invalid_or_mismatched_protocol(repo, change, message):
    repo["write_freeze"](change)
    with pytest.raises(ev.PreconditionError, match=message):
        ev.run_final_evaluation(repo["freeze"], **no_private_access(repo))


def test_final_run_refuses_changed_whisper_parameters(repo):
    for key, value in (("beam_size", 1), ("revision", "f" * 40), ("cpu_threads", 4)):
        repo["write_freeze"](lambda f: f["speech"]["config"].update({key: value}))
        with pytest.raises(ev.PreconditionError, match="Whisper model, revision or parameters differ"):
            ev.run_final_evaluation(repo["freeze"], **no_private_access(repo))


@pytest.mark.parametrize("rel", ["configs/speech.json", "src/speech.py", "tools/report_nlp_data_quality.py"])
def test_final_run_refuses_changed_pinned_file(repo, rel):
    path = repo["root"] / rel
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ev.PreconditionError, match=f"Frozen files differ from T12 or are missing: {rel}"):
        ev.run_final_evaluation(repo["freeze"], **no_private_access(repo))


def test_check_environment_compares_variables_python_and_packages(repo):
    protocol = ev.load_protocol(repo["freeze"], root=repo["root"],
                                accepted_sha256=repo["kwargs"]["accepted_freeze_sha256"])
    good = dict(environ={"OMP_NUM_THREADS": "1"}, version={"numpy": "2.3.3"}.get, python="3.13.7")
    ev.check_environment(protocol, **good)
    for change, message in (({"environ": {}}, "environment variables OMP_NUM_THREADS"),
                            ({"python": "3.12.3"}, "Python 3.12.3 instead of 3.13.7"),
                            ({"version": lambda name: "1.0"}, "package versions numpy")):
        with pytest.raises(ev.PreconditionError, match=message):
            ev.check_environment(protocol, **{**good, **change})
    with pytest.raises(ev.PreconditionError, match="Measurement environment differs"):
        ev.run_final_evaluation(repo["freeze"], **no_private_access(
            repo, environment_check=lambda p: ev.check_environment(p, **{**good, "python": "3.12.3"})))


@pytest.mark.parametrize("override, message", [
    ({"git": lambda root, frozen: {"commit": "a" * 40, "dirty": True, "frozen_commit_is_ancestor": True}},
     "uncommitted changes"),
    ({"git": lambda root, frozen: {"commit": None, "dirty": None, "frozen_commit_is_ancestor": False}},
     "git checkout"),
    ({"git": lambda root, frozen: {"commit": "a" * 40, "dirty": False, "frozen_commit_is_ancestor": False}},
     "does not descend from the T12 frozen commit"),
    ({"is_ignored": lambda path, root: False}, "must be ignored by Git"),
])
def test_final_run_refuses_unsafe_checkout(repo, override, message):
    with pytest.raises(ev.PreconditionError, match=message):
        ev.run_final_evaluation(repo["freeze"], **no_private_access(repo, **override))


def test_final_run_refuses_changed_or_missing_model_bundles(repo):
    def refused(message):
        with pytest.raises(ev.PreconditionError, match=message):
            ev.run_final_evaluation(repo["freeze"], **no_private_access(repo))

    model, original = repo["model"], repo["model"].read_bytes()
    model.write_bytes(original[::-1])  # тот же размер, другие байты
    refused("Classifier bundle differs from the T12 freeze")
    model.unlink()
    refused("Frozen classifier bundle not found")
    model.write_bytes(original)

    weights = repo["snapshot"] / "model.bin"
    data = weights.read_bytes()
    weights.write_bytes(data[::-1])
    refused("Whisper bundle checksum differs from the T12 freeze: model.bin")
    weights.write_bytes(data)
    extra = repo["snapshot"] / "preprocessor_config.json"
    extra.write_text("{}", encoding="utf-8")
    refused("Whisper bundle files differ")
    extra.unlink()
    snapshot = repo["snapshot"].rename(repo["snapshot"].with_name("other"))
    refused("Whisper bundle not found")
    snapshot.rename(repo["snapshot"])
    ev.run_final_evaluation(repo["freeze"], check_only=True, **{
        **repo["kwargs"], "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper")})


def test_t16_warmup_recording_is_required_and_checked(repo):
    def refused(warmup_audio, message):
        with pytest.raises(ev.PreconditionError, match=message):
            ev.run_final_evaluation(repo["freeze"], **no_private_access(repo, warmup_audio=warmup_audio))

    refused(None, "T16 warm-up recording is required")
    refused(repo["warmup"].with_name("missing.ogg"), "warm-up recording not found")
    t11_file = repo["kwargs"]["t11_paths"].audio_dir / "voice_001.ogg"
    refused(t11_file, "differs from the T16 recording")  # запись T11 не подменяет T16
    other = repo["warmup"].with_name("other.ogg")
    other.write_bytes(repo["warmup"].read_bytes()[::-1])  # тот же размер, другая сумма
    refused(other, "differs from the T16 recording")


def test_cli_refuses_without_t16_warmup(repo, monkeypatch, capsys):
    real = ev.run_final_evaluation

    def run(freeze, *, warmup_audio, check_only, **cli_paths):
        # Пути CLI заменяются путями искусственного репозитория; --warmup-audio не передан.
        return real(freeze, check_only=check_only, **no_private_access(repo, warmup_audio=warmup_audio))

    monkeypatch.setattr(ev, "run_final_evaluation", run)
    assert ev.main(["--freeze", str(repo["freeze"]), "--check"]) == 2
    assert "REFUSED: The T16 warm-up recording is required" in capsys.readouterr().err


def test_failed_warmup_stops_before_any_t11_record(repo):
    repo["recognizer"].results[repo["warmup"].name] = ("error", "", "inference_failed", 0.5)
    with pytest.raises(ev.EvaluationError, match="warm-up on the T16 recording failed"):
        ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])
    assert len(repo["recognizer"].seen) == 1  # только запись T16
    assert [texts for texts, _ in repo["classifier"].calls] == [list(ev._smoke_texts())]
    assert not (repo["root"] / "artifacts/t14").exists() and not (repo["root"] / "reports/voice_eval").exists()


def broken_manifest(manifest, change):
    manifest = copy.deepcopy(manifest)
    change(manifest)
    return manifest


@pytest.mark.parametrize("change, message", [
    (lambda m: m["records"].pop(), "exactly the 36 T12 IDs"),
    (lambda m: m["records"].append(copy.deepcopy(m["records"][0])), "exactly the 36 T12 IDs"),
    (lambda m: m["records"].reverse(), "exactly the 36 T12 IDs"),
    (lambda m: m["records"][3].update(audio_sha256="0" * 64), "voice_004: audio SHA-256"),
    (lambda m: m["records"][0].update(final_label="positive"), "12 per class"),  # было neutral
    (lambda m: m["records"][0].update(final_label="error"), "invalid final_label"),
    (lambda m: m["records"][1].update(reference_transcript=" "), "voice_002: missing reference"),
    (lambda m: m["records"][2].update(reference_transcript="?! …"), "voice_003: reference has no words"),
    (lambda m: m.update(label_ids={"negative": 0, "positive": 1, "neutral": 2}), "label order"),
    (lambda m: m.update(content_sha256="e" * 64), "content_sha256"),
    (lambda m: m.update(source_xlsx_sha256="0" * 64), "table SHA-256"),
])
def test_final_run_refuses_incomplete_or_inconsistent_t11(repo, change, message):
    manifest = broken_manifest(repo["manifest"], change)
    kwargs = {**repo["kwargs"], "t11_verifier": lambda paths: manifest,
              "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper")}
    with pytest.raises(ev.DataMismatchError, match=message):
        ev.run_final_evaluation(repo["freeze"], **kwargs)


def test_final_run_refuses_if_t11_changes_during_run(repo):
    calls = []

    def verifier(paths):
        calls.append(paths)
        if len(calls) == 1:
            return copy.deepcopy(repo["manifest"])
        return broken_manifest(repo["manifest"], lambda m: m["records"][0].update(reference_transcript="другое"))

    with pytest.raises(ev.DataMismatchError, match="changed during the evaluation"):
        ev.run_final_evaluation(repo["freeze"], **{**repo["kwargs"], "t11_verifier": verifier})
    assert not (repo["root"] / "artifacts/t14").exists() and not (repo["root"] / "reports/voice_eval").exists()


def test_verify_t11_reuses_the_standard_validator(monkeypatch, tmp_path):
    vt = ev._validator()
    paths = ev.T11Paths(tmp_path / "a.xlsx", tmp_path / "audio", tmp_path / "m.json", tmp_path / "r.json")
    calls = []
    monkeypatch.setattr(vt, "build", lambda xlsx, audio: calls.append((xlsx, audio)) or ({"m": 1}, {"r": 1}, []))
    monkeypatch.setattr(vt, "verify", lambda m, r, mp, rp: calls.append((mp, rp)) or [])
    assert ev.verify_t11(paths) == {"m": 1}
    assert calls == [(paths.xlsx, paths.audio_dir), (paths.manifest, paths.report)]

    monkeypatch.setattr(vt, "verify", lambda *args: ["Манифест: отличается content_sha256"])
    with pytest.raises(ev.DataMismatchError, match="differs from its freeze"):
        ev.verify_t11(paths)

    def invalid(*args):
        raise vt.ValidationError("Таблица не найдена")

    monkeypatch.setattr(vt, "build", invalid)
    with pytest.raises(ev.DataMismatchError, match="T11 validation failed"):
        ev.verify_t11(paths)
