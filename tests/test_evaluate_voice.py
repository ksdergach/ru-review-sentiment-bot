"""T14 на искусственных данных: подменные Whisper и классификатор, без моделей и T11.

Ожидаемые значения посчитаны вручную и записаны константами, а не получены
тестируемыми функциями.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src import evaluate_voice as ev
from src.classifier import ClassifierError, TextTooLongError
from src.speech import SpeechResult

ROOT = Path(__file__).resolve().parents[1]
TEST_SCHEMA = "test-only-t14-protocol"
TEST_RULES = "test-rules"


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


# --------------------------------------------------------------------------- #
# WER
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


def wer_record(record_id, status, words, s=None, d=None, i=None):
    return {"record_id": record_id, "asr_status": status, "reference_word_count": words,
            "wer_substitutions": s, "wer_deletions": d, "wer_insertions": i}


def test_corpus_wer_uses_total_words_not_mean_of_records():
    records = [wer_record("a", "ok", 2, 1, 0, 0),   # WER записи 0.5
               wer_record("b", "ok", 8, 0, 0, 0)]   # WER записи 0.0
    result = ev.corpus_wer(records, "all_deletions")
    assert result["wer"] == pytest.approx(0.1)      # 1 / 10, а не (0.5 + 0) / 2 = 0.25
    assert result["wer"] != pytest.approx(0.25)
    assert (result["word_errors"], result["reference_words"], result["records_counted"]) == (1, 10, 2)


def test_corpus_wer_failure_rule_is_explicit_and_counted_separately():
    records = [wer_record("a", "ok", 6, 1, 1, 1),
               wer_record("b", "empty", 4, 0, 4, 0),
               wer_record("c", "error", 5)]
    deletions = ev.corpus_wer(records, "all_deletions")
    assert deletions["wer"] == pytest.approx(12 / 15)  # (1+1+1) + 4 + 5 удалений из 6+4+5 слов
    assert (deletions["deletions"], deletions["asr_failures"], deletions["empty_transcripts"]) == (10, 1, 1)
    excluded = ev.corpus_wer(records, "excluded")
    assert excluded["wer"] == pytest.approx(7 / 10)
    assert (excluded["reference_words"], excluded["records_counted"], excluded["asr_failures"]) == (10, 2, 1)
    with pytest.raises(ValueError):
        ev.corpus_wer(records, "ignore")
    with pytest.raises(TypeError):
        ev.corpus_wer(records)  # правило берётся из фиксации, значения по умолчанию нет


def test_corpus_wer_requires_reference_words():
    with pytest.raises(ev.DataMismatchError):
        ev.corpus_wer([wer_record("a", "error", 3)], "excluded")


@pytest.mark.parametrize("text, expected", [
    ("Фильм ОЧЕНЬ хороший", ["фильм", "очень", "хороший"]),
    ("Ёлки и всё", ["елки", "и", "все"]),
    ("Ну, фильм... так себе!", ["ну", "фильм", "так", "себе"]),
    ("кое-как", ["кое", "как"]),
    ("в 2024 году", ["в", "2024", "году"]),
    ("  много \t пробелов\n", ["много", "пробелов"]),
    ("", []),
    ("?!…", []),
])
def test_provisional_words_edge_cases(text, expected):
    # Черновые правила разработки, не утверждённая нормализация T12.
    assert ev.provisional_words(text) == expected


def test_normalization_is_pluggable_without_changing_wer():
    clock = Clock()
    item = ev.VoiceInput("syn_1", "positive", "Пять звёзд из пяти", Path("unused"))
    recognizer = type("R", (), {"transcribe": lambda self, path: SpeechResult("ok", "5 звезд из 5")})()
    kwargs = dict(recognizer=recognizer, classifier=FakeClassifier(clock, {}), max_text_length=2000, clock=clock)

    draft = ev.evaluate_record(item, tokenize=ev.provisional_words, **kwargs)
    assert (draft["wer_substitutions"], draft["wer_deletions"], draft["wer_insertions"]) == (2, 0, 0)

    numbers = {"5": "пять"}
    custom = ev.evaluate_record(
        item, tokenize=lambda t: [numbers.get(w, w) for w in ev.provisional_words(t)], **kwargs)
    # «пяти» и «пять» остаются разными словами: 1 замена.
    assert (custom["wer_substitutions"], custom["wer_deletions"], custom["wer_insertions"]) == (1, 0, 0)


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
                                  tokenize=ev.provisional_words, max_text_length=2000, clock=clock)
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
    # S/D/I: r1 замена, r2 все 4 слова удалены, r3 не измерено, r4 вставка, r5 вставка (ё = е)
    wer = {rid: (r["wer_substitutions"], r["wer_deletions"], r["wer_insertions"], r["reference_word_count"])
           for rid, r in by_id.items()}
    assert wer == {"r1": (1, 0, 0, 5), "r2": (0, 4, 0, 4), "r3": (None, None, None, 4),
                   "r4": (0, 0, 1, 2), "r5": (0, 0, 1, 3)}


def test_failed_records_stay_in_the_main_evaluation(five_cases):
    summary = ev.summarize(five_cases["records"], failure_rule="all_deletions",
                           expected_ids=["r1", "r2", "r3", "r4", "r5"])
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
    # WER: S 1, D 4 + 4 (сбой r3 как удаления), I 2, N 18 -> 11/18
    assert summary["wer"]["wer"] == pytest.approx(11 / 18)
    assert ev.summarize(five_cases["records"], failure_rule="excluded")["wer"]["wer"] == pytest.approx(7 / 14)
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
    assert timing["asr_seconds"] == {"count": 5, "total": 10.0, "mean": 2.0, "median": 1.5}
    assert timing["asr_classifier_seconds"]["count"] == 3
    assert timing["total_seconds"]["total"] == pytest.approx(10.0 + 3 * 0.25)
    assert five_cases["classifier"].loads == 0 and five_cases["recognizer"].loads == 0


def test_unexpected_classifier_exception_is_not_masked(tmp_path):
    clock = Clock()

    class Broken(FakeClassifier):
        def predict_batch(self, texts, *, max_text_length):
            raise RuntimeError("bug")

    item = ev.VoiceInput("syn_1", "neutral", "обычный фильм", make_audio(tmp_path, "a.ogg", b"x"))
    with pytest.raises(RuntimeError):
        ev.evaluate_record(item, recognizer=FakeRecognizer(clock, {"a.ogg": ("ok", "фильм", None, 1.0)}),
                           classifier=Broken(clock, {}), tokenize=ev.provisional_words,
                           max_text_length=2000, clock=clock)


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
      "asr_classifier_seconds": None}, "wer_substitutions"),
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
    ev.assert_no_private_text(['{"wer": 0.1}', "Фильм"], texts)
    with pytest.raises(ev.EvaluationError, match="private transcript"):
        ev.assert_no_private_text(["примечание: отличный  фильм с\nхорошим финалом"], texts)
    with pytest.raises(ev.EvaluationError, match="per-record"):
        ev.assert_no_private_text(['{"true_label": "neutral"}'], [])


# --------------------------------------------------------------------------- #
# Фиксация T12 и финальный запуск
# --------------------------------------------------------------------------- #
def test_stage_a_has_no_approved_t12_integration(tmp_path):
    # До утверждения T12 нет ни адаптера фиксации, ни утверждённой нормализации.
    assert ev.PROTOCOL_ADAPTERS == {} and ev.APPROVED_NORMALIZATIONS == {}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"schema": "t12-freeze/1", "status": "approved"}), encoding="utf-8")
    with pytest.raises(ev.PreconditionError, match="No approved T12 protocol adapter"):
        ev.load_protocol(freeze, root=tmp_path)


def spy(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called before the T12 freeze is approved")
    return fail


def guarded_run(freeze, tmp_path, **overrides):
    kwargs = dict(
        root=tmp_path, t11_paths=ev.T11Paths(tmp_path / "t11.xlsx", tmp_path / "raw_audio",
                                            tmp_path / "manifest.json", tmp_path / "composition.json"),
        private_dir=tmp_path / "private", public_dir=tmp_path / "public",
        t11_verifier=spy("T11 validator"), make_classifier=spy("classifier"), make_recognizer=spy("Whisper"),
        git=spy("git"), is_ignored=spy("git check-ignore"),
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


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repo(tmp_path):
    """Искусственный репозиторий: фиксация, отчёт классификатора, 36 записей, не связанных с T11."""
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "speech.json", root / "configs" / "speech.json")
    report = root / "reports" / "classifier_run.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"bundle_sha256": "synthetic"}', encoding="utf-8")

    audio_dir = root / "artifacts" / "t11" / "raw_audio"
    labels = ["negative", "neutral", "positive"]
    records = []
    for n in range(1, 37):
        rid = f"voice_{n:03d}"
        payload = f"synthetic audio {n}".encode()
        make_audio(audio_dir, f"{rid}.ogg", payload)
        records.append({"record_id": rid, "final_label": labels[n % 3], "audio_sha256": sha256(payload),
                        "reference_transcript": f"искусственный отзыв номер {n} для проверки"})
    manifest = {"label_ids": {"negative": 0, "neutral": 1, "positive": 2}, "content_sha256": "c" * 64,
                "records": records}
    composition_path = root / "reports" / "voice_test" / "composition.json"
    composition_path.parent.mkdir(parents=True)
    composition = {"content_sha256": "c" * 64,
                   "records": [{"record_id": r["record_id"], "audio_sha256": r["audio_sha256"]} for r in records]}
    composition_path.write_text(json.dumps(composition), encoding="utf-8")

    speech = json.loads((root / "configs" / "speech.json").read_text(encoding="utf-8"))
    protocol = {
        "approved": True, "classifier_backend": "baseline",
        "classifier_report_path": "reports/classifier_run.json", "classifier_report_sha256": sha256(report.read_bytes()),
        "classifier_model_path": None, "max_text_length": 2000,
        "speech_config_path": "configs/speech.json", "speech_config": speech,
        "t11_report_sha256": sha256(composition_path.read_bytes()), "t11_content_sha256": "c" * 64,
        "normalization": TEST_RULES, "asr_failure_wer_rule": "all_deletions",
    }
    freeze = root / "reports" / "freeze.json"

    def write_freeze(**changes):
        freeze.write_text(json.dumps({"schema": TEST_SCHEMA, "protocol": {**protocol, **changes}}), encoding="utf-8")
        return freeze

    write_freeze()
    clock = Clock()
    asr_text = {r["record_id"]: r["reference_transcript"].replace("проверки", "проверка") for r in records}
    results = {f"{rid}.ogg": ("ok", text, None, 1.0) for rid, text in asr_text.items()}
    results["voice_005.ogg"] = ("error", "", "inference_failed", 2.0)
    results["voice_006.ogg"] = ("empty", "", None, 2.0)
    true = {r["reference_transcript"]: r["final_label"] for r in records}
    classifier = FakeClassifier(clock, {**true, **{asr_text[r["record_id"]]: r["final_label"] for r in records}})
    recognizer = FakeRecognizer(clock, results)
    verifications = []

    def verifier(paths):
        verifications.append(paths)
        return copy.deepcopy(manifest)

    kwargs = dict(
        root=root,
        t11_paths=ev.T11Paths(root / "artifacts/t11/T11_dataset.xlsx", audio_dir,
                              root / "artifacts/t11/manifest.json", composition_path),
        private_dir=root / "artifacts" / "t14", public_dir=root / "reports" / "voice_eval",
        adapters={TEST_SCHEMA: lambda data: data["protocol"]}, normalizations={TEST_RULES: ev.provisional_words},
        t11_verifier=verifier, make_classifier=lambda p, r: classifier, make_recognizer=lambda p, r: recognizer,
        git=lambda r: {"commit": "a" * 40, "dirty": False}, is_ignored=lambda path, r: True, clock=clock,
    )
    return {"root": root, "freeze": freeze, "write_freeze": write_freeze, "kwargs": kwargs, "manifest": manifest,
            "verifications": verifications, "classifier": classifier, "recognizer": recognizer,
            "texts": list(true) + list(asr_text.values())}


def test_final_run_produces_private_records_and_public_aggregates(repo):
    summary = ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])
    assert len(repo["verifications"]) == 2  # T11 проверяется до и после прогона
    assert repo["classifier"].loads == 1 and repo["recognizer"].loads == 1
    # Загрузка (3 s / 5 s) и прогрев учтены отдельно от построчного времени.
    setup = summary["run"]["model_setup_seconds"]
    assert setup["classifier_load"] == 3.0 and setup["asr_load"] == 5.0 and setup["asr_warmup"] is None
    assert setup["classifier_warmup"] == pytest.approx(0.25)

    results = summary["results"]
    assert results["records"] == 36 and results["record_ids"] == [f"voice_{n:03d}" for n in range(1, 37)]
    # Эталон: 36 верных; Whisper: 34 верных и 2 без предсказания (voice_005 positive, voice_006 negative).
    assert results["reference"]["macro_f1"] == 1.0
    assert (results["asr"]["correct"], results["asr"]["no_prediction"]) == (34, 2)
    # positive и negative: P 1, R 11/12 -> F1 22/23; neutral 1 -> macro (2*22/23 + 1)/3
    assert results["asr"]["macro_f1"] == pytest.approx((2 * 22 / 23 + 1) / 3)
    # WER: 34 замены «проверки -> проверка», 6 + 6 удалений; по 6 слов в 36 записях
    assert results["wer"]["wer"] == pytest.approx((34 + 12) / 216)

    private = json.loads((repo["root"] / "artifacts/t14/records.json").read_text(encoding="utf-8"))
    assert [r["record_id"] for r in private["records"]] == results["record_ids"]
    assert set(private["fields"]) == set(private["records"][0])
    public = (repo["root"] / "reports/voice_eval/metrics.json").read_text(encoding="utf-8")
    report = (repo["root"] / "reports/voice_eval/report.md").read_text(encoding="utf-8")
    for text in repo["texts"]:
        assert text not in public and text not in report
    for key in ("true_label", "asr_text", "reference_prediction", "asr_prediction"):
        assert key not in public
    assert "Малый размер голосового теста" in report and "ΔF1" in report

    # Повторный запуск не заменяет результаты.
    with pytest.raises(ev.PreconditionError, match="never replaced"):
        ev.run_final_evaluation(repo["freeze"], **repo["kwargs"])


def test_check_mode_stops_before_models(repo):
    kwargs = {**repo["kwargs"], "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper")}
    result = ev.run_final_evaluation(repo["freeze"], check_only=True, **kwargs)
    assert result["status"] == "READY" and result["records"] == 36
    assert len(repo["verifications"]) == 1
    assert not (repo["root"] / "artifacts/t14").exists()


@pytest.mark.parametrize("changes, message", [
    ({"approved": False}, "not approved"),
    ({"normalization": "draft"}, "not an approved T12 rule set"),
    ({"asr_failure_wer_rule": "skip"}, "asr_failure_wer_rule"),
    ({"classifier_backend": "gpt"}, "classifier_backend"),
    ({"classifier_report_path": "/etc/report.json"}, "relative"),
    ({"classifier_report_path": "../report.json"}, "relative"),
    ({"classifier_report_sha256": "abc"}, "SHA-256"),
    ({"max_text_length": 0}, "max_text_length"),
    ({"speech_config": {"model": "x"}}, "speech_config"),
    ({"classifier_report_sha256": "0" * 64}, "Classifier run report differs"),
    ({"t11_report_sha256": "0" * 64}, "composition report differs"),
    ({"t11_content_sha256": "d" * 64}, "content_sha256 differs"),
])
def test_final_run_refuses_invalid_or_mismatched_protocol(repo, changes, message):
    repo["write_freeze"](**changes)
    kwargs = {**repo["kwargs"], "t11_verifier": spy("T11 validator"),
              "make_classifier": spy("classifier"), "make_recognizer": spy("Whisper")}
    with pytest.raises(ev.PreconditionError, match=message):
        ev.run_final_evaluation(repo["freeze"], **kwargs)


def test_final_run_refuses_changed_whisper_parameters(repo):
    speech = json.loads((repo["root"] / "configs/speech.json").read_text(encoding="utf-8"))
    for key, value in (("beam_size", 1), ("revision", "f" * 40), ("cpu_threads", 4)):
        repo["write_freeze"](speech_config={**speech, key: value})
        kwargs = {**repo["kwargs"], "t11_verifier": spy("T11 validator")}
        with pytest.raises(ev.PreconditionError, match="Whisper model, revision or parameters differ"):
            ev.run_final_evaluation(repo["freeze"], **kwargs)


@pytest.mark.parametrize("override, message", [
    ({"git": lambda root: {"commit": "a" * 40, "dirty": True}}, "uncommitted changes"),
    ({"git": lambda root: {"commit": None, "dirty": None}}, "git checkout"),
    ({"is_ignored": lambda path, root: False}, "must be ignored by Git"),
])
def test_final_run_refuses_unsafe_checkout(repo, override, message):
    kwargs = {**repo["kwargs"], "t11_verifier": spy("T11 validator"), **override}
    with pytest.raises(ev.PreconditionError, match=message):
        ev.run_final_evaluation(repo["freeze"], **kwargs)


def broken_manifest(manifest, change):
    manifest = copy.deepcopy(manifest)
    change(manifest)
    return manifest


@pytest.mark.parametrize("change, message", [
    (lambda m: m["records"].pop(), "exactly voice_001"),
    (lambda m: m["records"].append(copy.deepcopy(m["records"][0])), "exactly voice_001"),
    (lambda m: m["records"].reverse(), "exactly voice_001"),
    (lambda m: m["records"][3].update(audio_sha256="0" * 64), "voice_004: audio SHA-256"),
    (lambda m: m["records"][0].update(final_label="positive"), "12 per class"),  # было neutral
    (lambda m: m["records"][0].update(final_label="error"), "invalid final_label"),
    (lambda m: m["records"][1].update(reference_transcript=" "), "voice_002: missing reference"),
    (lambda m: m.update(label_ids={"negative": 0, "positive": 1, "neutral": 2}), "label order"),
    (lambda m: m.update(content_sha256="e" * 64), "content_sha256"),
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


def test_warmup_audio_must_not_be_from_t11(repo):
    t11_file = repo["kwargs"]["t11_paths"].audio_dir / "voice_001.ogg"
    with pytest.raises(ev.PreconditionError, match="outside the T11 set"):
        ev.run_final_evaluation(repo["freeze"], warmup_audio=t11_file,
                                **{**repo["kwargs"], "t11_verifier": spy("T11 validator")})


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
