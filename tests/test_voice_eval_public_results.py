"""Опубликованные результаты T14 (`reports/voice_eval/`): неизменность, арифметика и приватность.

Используются только публичные файлы рабочего дерева и искусственные данные; приватные
данные T11, история Git, модели и инференс не нужны. ID и метки отдельных записей
опубликованного набора в сообщениях тестов не выводятся — только количества.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import evaluate_voice as ev

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "reports" / "voice_eval"
FREEZE = json.loads((ROOT / "reports" / "freeze.json").read_text(encoding="utf-8"))
# Оригинальные файлы единственного финального запуска.
METRICS_SHA256 = "cac6208afbdf38cdf9af3fecfdba68220c29c960a8cde7e1009b576add852a7a"
REPORT_SHA256 = "d0e8ca525d3e766c6a6dcf9536b41960d1503d9f65885d022ed5f16b8df132c6"
# Вариант В1: тот же CSV без reference_prediction и asr_prediction.
PREDICTIONS_SHA256 = "ced8b2439ba08cfdd35e6dc5242df19c306db0caaea12df5ae0d4e42d2398843"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def published():
    metrics = json.loads((PUBLIC / "metrics.json").read_text(encoding="utf-8"))
    predictions = (PUBLIC / "predictions.csv").read_text(encoding="utf-8")
    return {"metrics": metrics, "results": metrics["results"], "predictions": predictions,
            "rows": list(csv.DictReader(io.StringIO(predictions))),
            "report": (PUBLIC / "report.md").read_text(encoding="utf-8")}


# Искусственный набор первоначального формата, не связанный с T11:
# (ID, истинная метка, предсказание эталона, предсказание ASR, время ASR).
SYNTHETIC = [("s1", "negative", 0, 0, 1.25), ("s2", "negative", 0, 2, 2.5), ("s3", "neutral", 0, 2, 3.75),
             ("s4", "neutral", 1, 1, 1.5), ("s5", "positive", 2, 2, 2.25), ("s6", "positive", 2, 2, 4.0)]


def synthetic_original():
    """CSV первоначального формата и согласованные с ним матрицы ошибок 3×4."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(ev.ORIGINAL_PUBLIC_RECORD_FIELDS)
    values = {"reference": [[0] * 4 for _ in ev.LABELS], "asr": [[0] * 4 for _ in ev.LABELS]}
    for rid, label, ref, asr, seconds in SYNTHETIC:
        row = {"record_id": rid, "reference_prediction": ref, "reference_status": "ok", "reference_error_kind": "",
               "asr_status": "ok", "asr_error_code": "", "asr_prediction": asr, "asr_classification_status": "ok",
               "asr_classifier_error_kind": "", "asr_seconds": seconds, "reference_classifier_seconds": 0.001,
               "asr_classifier_seconds": 0.002, "total_seconds": seconds + 0.002}
        writer.writerow([row[key] for key in ev.ORIGINAL_PUBLIC_RECORD_FIELDS])
        values["reference"][ev.LABELS.index(label)][ref] += 1
        values["asr"][ev.LABELS.index(label)][asr] += 1
    results = {mode: {"confusion_matrix": {"rows": list(ev.LABELS), "columns": [*ev.LABELS, ev.ERROR],
                                           "values": values[mode]}} for mode in values}
    results["failures"] = {"asr_error_record_ids": [], "asr_empty_record_ids": []}
    return stream.getvalue(), results


def counts(candidates):
    sizes = [len(labels) for labels in candidates.values()]
    return {"disclosed": sizes.count(1), "narrowed_to_two": sizes.count(2), "undetermined": sizes.count(3)}


# --------------------------------------------------------------------------- #
# Неизменность и формат
# --------------------------------------------------------------------------- #
def test_metrics_and_report_are_the_original_files():
    assert sha256((PUBLIC / "metrics.json").read_bytes()) == METRICS_SHA256
    assert sha256((PUBLIC / "report.md").read_bytes()) == REPORT_SHA256


def test_public_csv_is_the_approved_b1_representation(published):
    assert sha256((PUBLIC / "predictions.csv").read_bytes()) == PREDICTIONS_SHA256
    rows = published["rows"]
    assert list(rows[0]) == list(ev.PUBLIC_RECORD_FIELDS)
    assert not set(ev.WITHHELD_PUBLIC_FIELDS) & set(rows[0])
    assert [row["record_id"] for row in rows] == FREEZE["voice_test"]["record_ids"]
    assert published["results"]["record_ids"] == FREEZE["voice_test"]["record_ids"]
    ev.assert_no_private_text([published["predictions"], json.dumps(published["metrics"]), published["report"]],
                              [])


def test_b1_csv_is_the_deterministic_redaction_of_the_original():
    original, _ = synthetic_original()
    redacted = ev.redact_public_predictions_csv(original)
    assert ev.redact_public_predictions_csv(original) == redacted
    assert redacted.splitlines()[0] == ",".join(ev.PUBLIC_RECORD_FIELDS)
    before, after = (list(csv.DictReader(io.StringIO(text))) for text in (original, redacted))
    # Все ID, порядок, статусы и время — без изменений; удалены только два столбца.
    assert [{key: row[key] for key in ev.PUBLIC_RECORD_FIELDS} for row in before] == after
    keep = [i for i, key in enumerate(ev.ORIGINAL_PUBLIC_RECORD_FIELDS) if key not in ev.WITHHELD_PUBLIC_FIELDS]
    assert redacted.splitlines() == [",".join(line.split(",")[i] for i in keep) for line in original.splitlines()]


# --------------------------------------------------------------------------- #
# Приватность: восстановление эталонных меток по совокупности публичных файлов
# --------------------------------------------------------------------------- #
def test_original_public_format_reveals_labels_and_b1_does_not():
    original, results = synthetic_original()
    rows = list(csv.DictReader(io.StringIO(original)))
    # s1: ASR negative бывает только у negative; s4: эталон neutral только у neutral;
    # s5, s6: эталон positive только у positive; s2/s3 — один negative и один neutral.
    candidates = ev.public_label_candidates(results, rows)
    assert {rid for rid, labels in candidates.items() if len(labels) == 1} == {"s1", "s4", "s5", "s6"}
    assert candidates["s2"] == candidates["s3"] == {"negative", "neutral"}
    with pytest.raises(ev.PublicationBlockedError, match=r"label of 4 record\(s\)"):
        ev.assert_no_label_disclosure(results, original)
    # Тот же набор в формате В1, с теми же статусами и различимым построчным временем.
    redacted = ev.redact_public_predictions_csv(original)
    b1 = ev.public_label_candidates(results, list(csv.DictReader(io.StringIO(redacted))))
    assert counts(b1) == {"disclosed": 0, "narrowed_to_two": 0, "undetermined": 6}
    ev.assert_no_label_disclosure(results, redacted)


def test_b1_public_results_reveal_no_individual_label(published):
    # Модель противника — по каждой записи отдельно, с учётом оставшихся статусов, кодов
    # ошибок, построчного времени и списков ID сбоев из metrics.json.
    found = counts(ev.public_label_candidates(published["results"], published["rows"]))
    assert found == {"disclosed": 0, "narrowed_to_two": 0, "undetermined": 36}
    ev.assert_no_label_disclosure(published["results"], published["predictions"])


def test_no_public_aggregate_is_broken_down_by_class_except_the_matrices(published):
    results = published["results"]
    assert set(results["timing"]) == set(ev.TIMING_FIELDS)
    assert all(set(stats) == {"count", "total", "mean", "median", "p95"} for stats in results["timing"].values())
    assert not set(ev.LABELS) & set(results["wer"]) and not set(ev.LABELS) & set(results["failures"])


# --------------------------------------------------------------------------- #
# Арифметическая согласованность опубликованных агрегатов
# --------------------------------------------------------------------------- #
def test_classification_metrics_follow_from_the_confusion_matrices(published):
    results = published["results"]
    macro = {}
    for mode in ("reference", "asr"):
        m = results[mode]
        values = m["confusion_matrix"]["values"]
        assert [sum(row) for row in values] == [FREEZE["voice_test"]["class_counts"][label] for label in ev.LABELS]
        f1s = []
        for t, label in enumerate(ev.LABELS):
            tp, support, predicted = values[t][t], sum(values[t]), sum(row[t] for row in values)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            stats = m["per_class"][label]
            assert (stats["precision"], stats["recall"], stats["f1"]) == pytest.approx((precision, recall, f1))
            assert stats["support"] == support
            f1s.append(f1)
        correct = sum(values[t][t] for t in range(len(ev.LABELS)))
        no_prediction = sum(row[-1] for row in values)
        assert (m["records"], m["correct"], m["no_prediction"]) == (36, correct, no_prediction)
        assert m["wrong_label"] == m["incorrect"] - no_prediction == 36 - correct - no_prediction
        assert m["accuracy"] == pytest.approx(correct / 36)
        assert m["macro_f1"] == pytest.approx(sum(f1s) / 3)
        macro[mode] = m["macro_f1"]
    assert results["delta_macro_f1"] == pytest.approx(macro["asr"] - macro["reference"])


def test_wer_failures_and_timing_are_consistent(published):
    results, rows = published["results"], published["rows"]
    wer = results["wer"]
    errors = wer["substitutions"] + wer["deletions"] + wer["insertions"]
    assert wer["word_errors"] == errors and wer["wer"] == pytest.approx(errors / wer["reference_words"])
    assert wer["records_counted"] == 36
    failures = results["failures"]
    assert failures["asr_status"] == {status: sum(r["asr_status"] == status for r in rows)
                                      for status in ev.ASR_STATUSES}
    assert (wer["asr_failures"], wer["empty_transcripts"]) == (failures["asr_status"]["error"],
                                                                failures["asr_status"]["empty"])
    assert failures["reference_classifier_errors"] == sum(r["reference_status"] != "ok" for r in rows)
    assert failures["asr_classifier_errors"] == sum(r["asr_classification_status"] == "classifier_error"
                                                    for r in rows)
    for mode, failed in (("reference", failures["reference_classifier_errors"]),
                         ("asr", failures["asr_classifier_errors"] + failures["asr_status"]["error"]
                          + failures["asr_status"]["empty"])):
        assert results[mode]["no_prediction"] == failed
    # Успешный сценарий: прежние значения агрегатов времени и побайтно тот же отчёт.
    timing_rows = [{key: None if row[key] == "" else float(row[key]) for key in ev.TIMING_FIELDS} for row in rows]
    assert ev.timing_summary(timing_rows) == results["timing"]
    assert ev.render_report(published["metrics"]) == published["report"]
