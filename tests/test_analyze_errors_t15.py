"""T15: отбор и журнал ошибок текстового test; переходы «эталон → Whisper» по искусственным записям T14.

Модели и инференс не нужны. Проверки реальных файлов используют только опубликованные
результаты T13/T14 и исходный test Parquet, уже находящийся в репозитории;
приватные результаты T14 не требуются и не читаются.
"""
import copy
import csv
import io
import json
from pathlib import Path
import re
import shutil

import pytest

from src import train_baseline as b
from tools import analyze_errors_t15 as t
from tools import analyze_voice_transitions_t15 as v
from tools import evaluate_voice as ev

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Текстовый test: ошибки, отбор, распределения
# --------------------------------------------------------------------------- #
def synthetic(per_cell=20, correct=5):
    rows = [{"record_id": f"rec-{a}{c}-{i:02d}", "true_label_id": a, "predicted_label_id": c}
            for a in b.CLASS_IDS for c in b.CLASS_IDS for i in range(per_cell if a != c else correct)]
    return rows


def test_errors_are_exact_label_mismatches():
    rows = synthetic(per_cell=3, correct=4)
    errors = t.find_errors(rows)
    assert len(errors) == 6 * 3
    assert all(r["true_label_id"] != r["predicted_label_id"] for r in errors)
    assert len(rows) - len(errors) == 3 * 4


def test_parse_predictions_rejects_bad_labels_and_duplicates():
    good = "record_id,label_id,baseline_label_id,rubert_label_id\nx,0,1,1\ny,2,2,0\n"
    assert [r["predicted_label_id"] for r in t.parse_predictions(good)] == [1, 2]
    with pytest.raises(ValueError, match="label id"):
        t.parse_predictions(good.replace("y,2,2,0", "y,2,3,0"))
    with pytest.raises(ValueError, match="Duplicate"):
        t.parse_predictions(good.replace("y,", "x,"))


def test_selection_is_deterministic_stratified_and_order_independent():
    errors = t.find_errors(synthetic())
    selected = t.select_errors(errors)
    assert len(selected) == len({r["record_id"] for r in selected}) == 40
    by_pair = t.distribution(selected)["by_direction"]
    # 40 = 6·6 + 4: остаток получают четыре направления с участием neutral.
    assert by_pair == {"negative->neutral": 7, "positive->neutral": 7, "neutral->negative": 7,
                       "neutral->positive": 7, "negative->positive": 6, "positive->negative": 6}
    assert t.select_errors(list(reversed(errors))) == selected
    assert t.select_errors(errors, seed=43) != selected
    # Внутри направления — по SHA-256(seed:record_id), а не по порядку строк.
    cell = [r for r in selected if (r["true_label_id"], r["predicted_label_id"]) == (0, 1)]
    assert cell == sorted(cell, key=lambda r: t.rank(r["record_id"]))


@pytest.mark.parametrize("size", [30, 40, 50])
def test_selection_size_within_issue_range(size):
    selected = t.select_errors(t.find_errors(synthetic()), size=size)
    assert len(selected) == size and len({r["record_id"] for r in selected}) == size


@pytest.mark.parametrize("count", [0, 1, 7, 39])
def test_selection_takes_all_errors_when_fewer_than_requested(count):
    errors = t.find_errors(synthetic())[:count]
    assert {r["record_id"] for r in t.select_errors(errors)} == {r["record_id"] for r in errors}


def test_empty_direction_is_filled_by_the_others():
    errors = [r for r in t.find_errors(synthetic()) if (r["true_label_id"], r["predicted_label_id"]) != (1, 0)]
    selected = t.select_errors(errors)
    assert len(selected) == 40 and t.distribution(selected)["by_direction"]["neutral->negative"] == 0


def test_selection_rejects_invalid_input():
    errors = t.find_errors(synthetic())
    with pytest.raises(ValueError, match="Duplicate"):
        t.select_errors([errors[0], errors[0]])
    with pytest.raises(ValueError, match="misclassified"):
        t.select_errors(synthetic(per_cell=1, correct=1))
    with pytest.raises(ValueError, match="six"):
        t.select_errors(errors, order=t.PAIR_ORDER[:5])


def test_distribution_denominators():
    rows = [{"record_id": "a", "true_label_id": 0, "predicted_label_id": 1},
            {"record_id": "b", "true_label_id": 2, "predicted_label_id": 0},
            {"record_id": "c", "true_label_id": 2, "predicted_label_id": 1}]
    d = t.distribution(rows)
    assert d["total"] == 3 == sum(d["by_true_class"].values()) == sum(d["by_predicted_class"].values())
    assert sum(d["by_direction"].values()) == 3
    assert d["by_true_class"] == {"negative": 1, "neutral": 0, "positive": 2}
    assert d["by_predicted_class"] == {"negative": 1, "neutral": 2, "positive": 0}
    assert d["involving_neutral"] == 2


# --------------------------------------------------------------------------- #
# Разметка и проверка участником
# --------------------------------------------------------------------------- #
def annotated(selected, **entry_changes):
    entries = [{"record_id": r["record_id"], "true_label_id": r["true_label_id"],
                "predicted_label_id": r["predicted_label_id"], "text_sha256": r["text_sha256"],
                "categories": ["mixed", "neutral_confusion"], "confidence": "medium",
                "explanation": "Интерпретация.", "review_status": "pending", **entry_changes} for r in selected]
    return {"author": t.AUTHOR, "analysis_date": "2026-11-21", "human_review_status": "pending", "errors": entries}


@pytest.fixture
def small_selection():
    chosen = t.select_errors(t.find_errors(synthetic()), size=6)
    return t.text_facts(chosen, {r["record_id"]: f"текст {r['record_id']}" for r in chosen})


def test_several_categories_per_case_are_counted_separately(small_selection):
    data = annotated(small_selection)
    data["errors"][0]["categories"] = ["sarcasm"]
    t.validate_annotations(data, small_selection)
    counts = t.category_counts(data["errors"])
    assert counts["mixed"] == counts["neutral_confusion"] == 5 and counts["sarcasm"] == 1
    assert sum(counts.values()) == 11  # теги пересекаются: сумма больше числа случаев


@pytest.mark.parametrize("changes, message", [
    ({"categories": []}, "categories"),
    ({"categories": ["noise"]}, "categories"),
    ({"categories": ["mixed", "mixed"]}, "categories"),
    ({"categories": ["unknown", "mixed"]}, "unknown"),
    ({"confidence": "certain"}, "confidence"),
    ({"explanation": " "}, "explanation"),
    ({"true_label_id": 9}, "true_label_id"),
    ({"text_sha256": "x"}, "text_sha256"),
])
def test_invalid_annotations_are_rejected(small_selection, changes, message):
    data = annotated(small_selection)
    data["errors"][2].update(changes)
    with pytest.raises(ValueError, match=message):
        t.validate_annotations(data, small_selection)


def test_annotations_must_follow_selection_order(small_selection):
    data = annotated(small_selection)
    data["errors"].reverse()
    with pytest.raises(ValueError, match="order"):
        t.validate_annotations(data, small_selection)


def evidence_for(selected, data, verdicts=None):
    verdicts = verdicts or ["agree"] * len(data["errors"])
    reviews = [{"position": i, "record_id": e["record_id"], "verdict": v,
                **({"note": "Замечание."} if v == "partial" else {})}
               for i, (e, v) in enumerate(zip(data["errors"], verdicts), 1)]
    return {"selected_ids_sha256": b.sha256_json([r["record_id"] for r in selected]),
            "annotations_sha256": t.entries_sha256(data["errors"]),
            "source": {"decisions_sha256": t.decisions_sha256(reviews)},
            "reviewed_count": len(reviews),
            "verdict_counts": {k: verdicts.count(k) for k in t.VERDICTS},
            "partial_positions": [i for i, v in enumerate(verdicts, 1) if v == "partial"],
            "reviews": reviews}


def test_confirmed_requires_evidence_for_every_case(small_selection):
    data = annotated(small_selection, review_status="reviewed")
    data["human_review_status"] = "confirmed"
    with pytest.raises(ValueError, match="evidence"):
        t.validate_annotations(data, small_selection)
    evidence = evidence_for(small_selection, data)
    t.validate_annotations(data, small_selection, evidence)
    partial = copy.deepcopy(evidence)
    partial["reviews"].pop()
    with pytest.raises(ValueError, match="every"):
        t.validate_annotations(data, small_selection, partial)
    edited = copy.deepcopy(data)
    edited["errors"][0]["explanation"] = "Изменено после проверки."
    with pytest.raises(ValueError, match="every"):
        t.validate_annotations(edited, small_selection, evidence)
    counted = {**evidence, "verdict_counts": {"agree": len(small_selection) - 1, "partial": 1, "disagree": 0}}
    with pytest.raises(ValueError, match="counts"):
        t.validate_annotations(data, small_selection, counted)


@pytest.fixture
def confirmed(small_selection):
    data = annotated(small_selection, review_status="reviewed")
    data["human_review_status"] = "confirmed"
    verdicts = ["partial" if i == 1 else "agree" for i in range(len(small_selection))]
    evidence = evidence_for(small_selection, data, verdicts)
    t.validate_annotations(data, small_selection, evidence, evidence["source"]["decisions_sha256"])
    return data, evidence


def _swap_verdicts(ev_):
    r = ev_["reviews"]
    r[0]["verdict"], r[1]["verdict"] = r[1]["verdict"], r[0]["verdict"]
    r[0]["note"] = r[1].pop("note")


@pytest.mark.parametrize("tamper, message", [
    (_swap_verdicts, "decisions"),  # счётчики, позиции и замечания согласованы, сумма — нет
    (lambda e: e["source"].update(decisions_sha256="0" * 64), "decisions"),
    (lambda e: e.pop("verdict_counts"), "counts"),
    (lambda e: e.update(partial_positions=[1]), "Partial positions"),
    (lambda e: e["reviews"][0].update(note="Чужое замечание."), "note"),
    (lambda e: e["reviews"][1].pop("note"), "note"),
    (lambda e: e["reviews"][1].update(position=5), "position"),
    (lambda e: e["reviews"].pop(), "every"),
    (lambda e: e["reviews"].__setitem__(1, dict(e["reviews"][0])), "every"),
    (lambda e: e.update(recorder_flags=[{"position": 1, "record_id": e["reviews"][1]["record_id"]}]), "flag"),
])
def test_tampered_review_evidence_is_rejected(small_selection, confirmed, tamper, message):
    data, evidence = confirmed
    accepted = evidence["source"]["decisions_sha256"]
    broken = copy.deepcopy(evidence)
    tamper(broken)
    if tamper is _swap_verdicts:  # подделка согласована во всём, кроме исходной суммы решений
        broken["partial_positions"] = [1]
        broken["source"]["decisions_sha256"] = t.decisions_sha256(broken["reviews"])
    with pytest.raises(ValueError, match=message):
        t.validate_annotations(data, small_selection, broken, accepted)


def test_published_evidence_matches_original_decisions_file():
    evidence = json.loads((ROOT / t.REVIEW_EVIDENCE).read_text(encoding="utf-8"))
    assert t.decisions_sha256(evidence["reviews"]) == t.ACCEPTED_DECISIONS_SHA256
    assert evidence["source"]["decisions_sha256"] == t.ACCEPTED_DECISIONS_SHA256
    assert evidence["verdict_counts"] == {"agree": 31, "partial": 9, "disagree": 0}


def test_pending_review_cannot_mark_single_cases_reviewed(small_selection):
    data = annotated(small_selection)
    data["errors"][0]["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="pending"):
        t.validate_annotations(data, small_selection)


def test_text_guard_detects_copied_reviews():
    texts = {"a": "Фильм  очень\nпонравился всем", "b": "да"}
    t.assert_no_texts(["ID a, 2 слова: да"], texts)  # одно-два слова — не утечка
    with pytest.raises(ValueError, match="source review"):
        t.assert_no_texts(["...фильм: Фильм очень понравился всем!"], texts)


def test_review_packet_shows_texts_and_never_claims_confirmation(small_selection):
    texts = {r["record_id"]: f"текст {r['record_id']}" for r in small_selection}
    packet = t.review_packet(small_selection, texts, annotated(small_selection))
    assert all(text in packet for text in texts.values())
    assert packet.count("[ ] согласен  [ ] частично согласен  [ ] не согласен") == len(small_selection)
    assert "confirmed" not in packet and "[x]" not in packet


def test_tampered_t13_predictions_are_rejected(tmp_path):
    for rel in (t.PREDICTIONS, t.FINAL_METRICS, t.FREEZE):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    t.load_predictions(tmp_path)
    path = tmp_path / t.PREDICTIONS
    path.write_text(path.read_text(encoding="utf-8").replace(",0,0,0\n", ",0,1,0\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the T13 run"):
        t.load_predictions(tmp_path)


# --------------------------------------------------------------------------- #
# Опубликованный журнал T15 (реальные файлы рабочего дерева)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real():
    return t.analyze(ROOT)


def test_published_ids_are_real_baseline_errors_from_t13(real):
    records, metrics = t.load_predictions(ROOT)
    by_id = {r["record_id"]: r for r in records}
    rows = list(csv.DictReader(io.StringIO((ROOT / t.ERRORS_CSV).read_text(encoding="utf-8"))))
    assert len(rows) == 40 and len({r["record_id"] for r in rows}) == 40
    for row in rows:
        source = by_id[row["record_id"]]
        assert source["true_label_id"] != source["predicted_label_id"]
        assert b.CLASS_NAMES[source["true_label_id"]] == row["true_label"]
        assert b.CLASS_NAMES[source["predicted_label_id"]] == row["predicted_label"]
    assert [r["record_id"] for r in rows] == [r["record_id"] for r in real["selected"]]
    errors = real["summary"]["errors"]
    assert errors["total"] == metrics["models"]["baseline"]["metrics"]["errors"]["total"] == 1019
    assert errors["by_true_class"] == metrics["models"]["baseline"]["metrics"]["errors"]["by_true_class"]
    assert errors["by_predicted_class"] == metrics["models"]["baseline"]["metrics"]["errors"]["by_predicted_class"]


def test_published_files_match_recomputation(real):
    for relative, text in real["public"].items():
        assert (ROOT / relative).read_text(encoding="utf-8") == text, relative
    assert real["annotations"]["human_review_status"] in ("pending", "confirmed")
    if real["annotations"]["human_review_status"] == "pending":
        assert not (ROOT / t.REVIEW_EVIDENCE).exists()
        assert all(e["review_status"] == "pending" for e in real["annotations"]["errors"])


def test_public_t15_files_contain_no_source_texts_or_voice_records(real):
    public_dir = ROOT / t.PUBLIC_DIR
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(public_dir.iterdir()) if p.is_file()}
    t.assert_no_texts(list(files.values()), real["texts"])
    voice_ids = json.loads((ROOT / "reports/freeze.json").read_text(encoding="utf-8"))["voice_test"]["record_ids"]
    for payload in files.values():  # метки текстового test публичны, голосовые ID и метки — нет
        assert not any(rid in payload for rid in voice_ids)
    forbidden = re.compile(r"(?<!\w)(?:" + "|".join(ev.FORBIDDEN_PUBLIC_KEYS) + r")(?!\w)")
    if (voice := files.get(Path(v.PUBLIC_AGGREGATES).name)) is not None:
        assert not forbidden.search(voice)


def test_local_review_packet_is_git_ignored():
    assert b.git_ignore_status(ROOT / t.LOCAL_PACKET, ROOT) in (True, None)
    assert b.git_ignore_status(ROOT / v.PRIVATE_JOURNAL, ROOT) in (True, None)
    assert b.git_ignore_status(ROOT / v.PRIVATE_FULL_AGGREGATES, ROOT) in (True, None)


# --------------------------------------------------------------------------- #
# Голосовой набор: переходы по искусственным приватным записям
# --------------------------------------------------------------------------- #
def voice_record(rid, true, ref, asr, *, asr_status="ok", s=1, d=0, i=0, n=10):
    failed = asr_status != "ok"
    return {
        "record_id": rid, "true_label": true,
        "reference_prediction": ref, "reference_status": "ok", "reference_error_kind": None,
        "asr_status": asr_status, "asr_error_code": "decode_failed" if asr_status == "error" else None,
        "asr_prediction": "error" if failed else asr,
        "asr_classification_status": "not_run" if failed else "ok", "asr_classifier_error_kind": None,
        "wer_substitutions": 0 if failed else s, "wer_deletions": n if failed else d,
        "wer_insertions": 0 if failed else i, "reference_word_count": n,
        "asr_seconds": 1.0, "asr_model_load_seconds": 0.0, "reference_classifier_seconds": 0.01,
        "asr_classifier_seconds": None if failed else 0.01, "total_seconds": 1.01, "audio_duration_seconds": 4.0,
        "asr_text": "" if failed else f"распознанный текст номер {rid}",
    }


def six_records():
    return [
        voice_record("voice_001", "negative", "negative", "negative", s=0),            # stayed_correct
        voice_record("voice_002", "negative", "negative", "neutral", s=3, n=10),       # worsened
        voice_record("voice_003", "neutral", "positive", "neutral", s=1, n=10),        # improved
        voice_record("voice_004", "neutral", "positive", "positive", s=2, i=1, n=20),  # stayed_wrong_same
        voice_record("voice_005", "positive", "negative", "neutral", d=2, n=10),       # stayed_wrong_changed
        voice_record("voice_006", "positive", "positive", None, asr_status="empty", n=5),  # worsened: empty
    ]


def test_transition_kinds():
    kinds = [v.transition(r) for r in six_records()]
    assert kinds == ["stayed_correct", "worsened", "improved", "stayed_wrong_same",
                     "stayed_wrong_changed", "worsened"]


def test_aggregates_count_transitions_failures_and_wer():
    agg = v.aggregate(six_records())
    assert agg["transitions"] == {"stayed_correct": 1, "worsened": 2, "improved": 1,
                                  "stayed_wrong_same": 1, "stayed_wrong_changed": 1}
    assert agg["net_correct_change"] == -1
    assert agg["prediction_changed"] == 4
    assert agg["prediction_change_directions"] == {"negative->neutral": 2, "positive->error": 1,
                                                   "positive->neutral": 1}
    assert agg["failures"] == {"asr_empty": 1, "asr_error": 0, "reference_classifier_errors": 0,
                               "asr_classifier_errors": 0, "worsened_with_asr_failure": 1}
    worsened = agg["wer_by_transition"]["worsened"]
    assert (worsened["substitutions"], worsened["deletions"], worsened["reference_words"]) == (3, 5, 15)
    assert worsened["wer"] == pytest.approx(8 / 15)
    assert agg["wer_by_transition"]["stayed_correct"]["records_without_word_errors"] == 1
    assert agg["wer_by_prediction_change"]["unchanged"]["reference_words"] == 30
    # ID сбоя публичен в T14: разбивка по истинному классу не публикуется.
    assert isinstance(agg["by_true_class"], str) and agg["by_true_class"].startswith("withheld")


def test_by_true_class_published_only_without_failures():
    records = six_records()[:5]
    agg = v.aggregate(records)
    assert agg["by_true_class"]["neutral"] == {"stayed_correct": 0, "worsened": 0, "improved": 1,
                                               "stayed_wrong_same": 1, "stayed_wrong_changed": 0}
    assert sum(sum(row.values()) for row in agg["by_true_class"].values()) == 5


def matrices(records):
    s = ev.summarize(records)
    return {"reference": s["reference"], "asr": s["asr"]}


def test_bounds_from_published_matrices_contain_actual_counts():
    records = six_records()
    bounds = v.public_transition_bounds(matrices(records))
    agg = v.aggregate(records)
    for key in ("worsened", "improved"):
        assert bounds[key][0] <= agg["transitions"][key] <= bounds[key][1]


def test_bounds_for_published_t14_matrices():
    public = v.load_public_metrics(ROOT)
    bounds = v.public_transition_bounds(public["results"])
    assert bounds["by_true_class"] == {"negative": {"worsened": [0, 3], "improved": [0, 3]},
                                       "neutral": {"worsened": [4, 7], "improved": [0, 3]},
                                       "positive": {"worsened": [0, 0], "improved": [0, 0]}}
    assert bounds["worsened"] == [4, 10] and bounds["improved"] == [0, 6]


def test_control_values_match_published_t14():
    results = v.load_public_metrics(ROOT)["results"]
    observed = {"reference_macro_f1": results["reference"]["macro_f1"], "asr_macro_f1": results["asr"]["macro_f1"],
                "delta_macro_f1": results["delta_macro_f1"], "wer": results["wer"]["wer"]}
    for key, value in v.CONTROL.items():
        assert abs(observed[key] - value) <= v.CONTROL_TOLERANCE, key


def test_consistency_rejects_impossible_transitions():
    records = six_records()
    agg = v.aggregate(records)
    v.check_consistency(agg, ev.summarize(records))
    broken = copy.deepcopy(agg)
    broken["transitions"]["worsened"] += 1
    broken["transitions"]["stayed_correct"] -= 1
    broken["transitions"]["improved"] += 1
    broken["transitions"]["stayed_wrong_same"] -= 1
    with pytest.raises(ev.DataMismatchError):
        v.check_consistency(broken, ev.summarize(records))


def test_public_guard_blocks_ids_and_transcripts():
    records = six_records()
    v.assert_public_safe('{"worsened": 2}', records)
    with pytest.raises(ev.PublicationBlockedError):
        v.assert_public_safe('{"ids": ["voice_002"]}', records)
    with pytest.raises(ev.EvaluationError):
        v.assert_public_safe(json.dumps({"x": records[1]["asr_text"]}, ensure_ascii=False), records)


def thirty_records():
    """Шесть видов исхода по пять раз: группы WER не меньше порога публикации."""
    base = six_records()
    return [{**r, "record_id": f"voice_{i * 6 + j + 1:03d}",
             "asr_text": r["asr_text"] and f"распознанный текст номер {i * 6 + j + 1}"}
            for i in range(5) for j, r in enumerate(base)]


@pytest.fixture
def voice_project(tmp_path, monkeypatch):
    records = thirty_records()
    ids = [r["record_id"] for r in records]
    results = json.loads(json.dumps(ev.summarize(records, expected_ids=ids)))
    metrics = json.dumps({"results": results}).encode()
    (tmp_path / "reports/voice_eval").mkdir(parents=True)
    (tmp_path / v.PUBLIC_METRICS).write_bytes(metrics)
    (tmp_path / "reports/freeze.json").write_text(json.dumps({"voice_test": {"record_ids": ids}}))
    private = tmp_path / v.PRIVATE_RECORDS
    private.parent.mkdir(parents=True)
    private.write_text(json.dumps({"schema": v.RECORDS_SCHEMA, "freeze_sha256": "f" * 64, "records": records},
                                  ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(v.baseline, "git_ignore_status", lambda path, root: True)
    control = {"reference_macro_f1": results["reference"]["macro_f1"], "asr_macro_f1": results["asr"]["macro_f1"],
               "delta_macro_f1": results["delta_macro_f1"], "wer": results["wer"]["wer"]}
    kwargs = {"metrics_sha256": __import__("hashlib").sha256(metrics).hexdigest(),
              "freeze_sha256": "f" * 64, "control": control}
    return tmp_path, kwargs, records


def test_run_writes_private_journal_and_public_aggregates_only(voice_project):
    root, kwargs, records = voice_project
    summary = v.run(root, write=True, **kwargs)
    public = (root / v.PUBLIC_AGGREGATES).read_text(encoding="utf-8")
    assert json.loads(public) == v.public_view(summary) and summary["source"]["inference_run"] is False
    assert "wer_by_transition" in summary and "wer_by_prediction_change" in summary
    assert not {"wer_by_transition", "wer_by_prediction_change"} & set(json.loads(public))
    assert "records_without_word_errors" not in public
    assert json.loads((root / v.PRIVATE_FULL_AGGREGATES).read_text(encoding="utf-8")) == summary
    assert not any(r["record_id"] in public for r in records)
    assert not any(r["asr_text"] and r["asr_text"] in public for r in records)
    journal = json.loads((root / v.PRIVATE_JOURNAL).read_text(encoding="utf-8"))
    assert [r["transition"] for r in journal["records"]] == [v.transition(r) for r in records]
    assert all("asr_text" not in r for r in journal["records"])
    with pytest.raises(ev.PreconditionError, match="already exist"):
        v.run(root, write=True, **kwargs)


def test_run_refuses_mismatching_or_missing_private_records(voice_project):
    root, kwargs, records = voice_project
    private = root / v.PRIVATE_RECORDS
    data = json.loads(private.read_text(encoding="utf-8"))
    data["records"][2]["asr_prediction"] = "negative"
    private.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ev.DataMismatchError, match="differ from published"):
        v.run(root, **kwargs)
    private.unlink()
    with pytest.raises(ev.PreconditionError, match="not found"):
        v.run(root, **kwargs)
    assert not any((root / p).exists() for p in (v.PUBLIC_AGGREGATES, v.PRIVATE_JOURNAL, v.PRIVATE_FULL_AGGREGATES))


def test_run_refuses_wrong_control_values(voice_project):
    root, kwargs, _ = voice_project
    kwargs["control"] = {**kwargs["control"], "wer": kwargs["control"]["wer"] + 1e-6}
    with pytest.raises(ev.DataMismatchError, match="control"):
        v.run(root, **kwargs)


def test_run_without_private_results_in_this_repository():
    if (ROOT / v.PRIVATE_RECORDS).exists():
        pytest.skip("private T14 results present: run the tool itself on this machine")
    assert v.main(["--check"]) == 2


def test_validate_aggregates_rejects_inconsistent_values(voice_project):
    root, kwargs, _ = voice_project
    summary = v.run(root, **kwargs)
    public = json.loads((root / v.PUBLIC_METRICS).read_text())
    args = {"metrics_sha256": kwargs["metrics_sha256"], "control": kwargs["control"]}
    v.validate_aggregates(summary, public, **args)
    v.validate_aggregates(v.public_view(summary), public, **args)
    for path, delta in ((("prediction_change_directions", "negative->neutral"), 1),
                        (("wer_by_prediction_change", "changed", "deletions"), -1),
                        (("wer_by_transition", "improved", "substitutions"), 1),
                        (("failures", "asr_empty"), 1), (("net_correct_change",), 1)):
        broken = copy.deepcopy(summary)
        node = broken
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += delta
        with pytest.raises(ev.DataMismatchError):
            v.validate_aggregates(broken, public, **args)
    renamed = {**v.public_view(summary), "withheld": {"wer_by_transition": "x"}}
    with pytest.raises(ev.DataMismatchError, match="withheld"):
        v.validate_aggregates(renamed, public, **args)


def test_by_true_class_must_add_up_to_published_matrices():
    records = six_records()[:5]
    ids = [r["record_id"] for r in records]
    results = json.loads(json.dumps(ev.summarize(records, expected_ids=ids)))
    control = {"reference_macro_f1": results["reference"]["macro_f1"], "asr_macro_f1": results["asr"]["macro_f1"],
               "delta_macro_f1": results["delta_macro_f1"], "wer": results["wer"]["wer"]}
    summary = {"task": "T15", "source": {"t14_metrics": v.PUBLIC_METRICS, "t14_metrics_sha256": "0" * 64,
                                         "private_records": v.PRIVATE_RECORDS, "inference_run": False},
               "t14_control": control, "public_bounds": v.public_transition_bounds(results), **v.aggregate(records)}
    args = {"metrics_sha256": "0" * 64, "control": control}
    v.validate_aggregates(summary, {"results": results}, **args)
    swapped = copy.deepcopy(summary)
    swapped["by_true_class"]["neutral"]["improved"] -= 1
    swapped["by_true_class"]["negative"]["improved"] += 1
    with pytest.raises(ev.DataMismatchError, match="by_true_class"):
        v.validate_aggregates(swapped, {"results": results}, **args)


def _five_record_summary():
    records = six_records()[:5]  # без сбоев
    ids = [r["record_id"] for r in records]
    results = json.loads(json.dumps(ev.summarize(records, expected_ids=ids)))
    control = {"reference_macro_f1": results["reference"]["macro_f1"], "asr_macro_f1": results["asr"]["macro_f1"],
               "delta_macro_f1": results["delta_macro_f1"], "wer": results["wer"]["wer"]}
    summary = {"task": "T15", "source": {"t14_metrics": v.PUBLIC_METRICS, "t14_metrics_sha256": "0" * 64,
                                         "private_records": v.PRIVATE_RECORDS, "inference_run": False},
               "t14_control": control, "public_bounds": v.public_transition_bounds(results), **v.aggregate(records)}
    return summary, {"results": results}, {"metrics_sha256": "0" * 64, "control": control}


def test_public_view_has_no_group_wer_and_no_unknown_nested_fields():
    summary, public, args = _five_record_summary()
    noisy = copy.deepcopy(summary)
    noisy["transitions"]["note"] = "приватное"
    noisy["source"]["note"] = "приватное"
    noisy["wer_by_prediction_change"]["changed"]["note"] = "приватное"
    view = v.public_view(noisy)
    assert view == v.public_view(summary)
    assert "приватное" not in json.dumps(view, ensure_ascii=False)
    assert "wer_by_transition" not in view and "wer_by_prediction_change" not in view
    assert "records_without_word_errors" not in json.dumps(view)
    assert set(view["withheld"]) == {"wer_by_transition", "wer_by_prediction_change"}
    v.validate_aggregates(view, public, **args)
    assert view["transitions"] == summary["transitions"] and view["t14_control"] == summary["t14_control"]


@pytest.mark.parametrize("tamper", [
    lambda s: s["failures"].update(worsened_with_asr_failure=1),  # ухудшение «со сбоем» при нуле сбоев
    lambda s: s["transitions"].update(improved=True),  # bool вместо счётчика
    lambda s: s["by_true_class"]["neutral"].update(improved=True),
    lambda s: s["failures"].update(note=0),
    lambda s: s["wer_by_prediction_change"]["changed"].update(note=0),
])
def test_validate_aggregates_rejects_impossible_counts(tamper):
    summary, public, args = _five_record_summary()
    v.validate_aggregates(summary, public, **args)
    broken = copy.deepcopy(summary)
    tamper(broken)
    with pytest.raises(ev.DataMismatchError):
        v.validate_aggregates(broken, public, **args)


def test_validate_aggregates_rejects_negative_wer_compensated_by_another_group():
    summary, public, args = _five_record_summary()
    broken = copy.deepcopy(summary)
    groups = broken["wer_by_transition"]
    for key, delta in (("improved", -1), ("stayed_wrong_same", 1)):  # сумма удалений прежняя
        g = groups[key]
        g["deletions"] += delta
        g["wer"] = (g["substitutions"] + g["deletions"] + g["insertions"]) / g["reference_words"]
    assert groups["improved"]["deletions"] < 0
    with pytest.raises(ev.DataMismatchError, match="wer_by_transition"):
        v.validate_aggregates(broken, public, **args)


def test_publish_from_full_writes_only_the_public_view(voice_project):
    root, kwargs, _ = voice_project
    args = {"metrics_sha256": kwargs["metrics_sha256"], "control": kwargs["control"]}
    with pytest.raises(ev.PreconditionError, match="not found"):
        v.publish_from_full(root, **args)
    summary = v.run(root, **kwargs)
    full = root / v.PRIVATE_FULL_AGGREGATES
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(v.public_view(summary)), encoding="utf-8")
    with pytest.raises(ev.DataMismatchError, match="not a full"):
        v.publish_from_full(root, **args)
    tampered = copy.deepcopy(summary)
    tampered["transitions"]["worsened"] += 1
    tampered["transitions"]["stayed_correct"] -= 1
    full.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ev.DataMismatchError):
        v.publish_from_full(root, **args)
    assert not (root / v.PUBLIC_AGGREGATES).exists()
    extra = copy.deepcopy(summary)
    extra["wer_by_prediction_change"]["changed"]["note"] = "приватное"
    full.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ev.DataMismatchError, match="wer_by_prediction_change"):
        v.publish_from_full(root, **args)
    assert not (root / v.PUBLIC_AGGREGATES).exists()
    full.write_text(json.dumps(summary), encoding="utf-8")
    view = v.publish_from_full(root, **args)
    assert json.loads((root / v.PUBLIC_AGGREGATES).read_text(encoding="utf-8")) == view == v.public_view(summary)
    with pytest.raises(ev.PreconditionError, match="already exist"):
        v.publish_from_full(root, **args)


def test_published_voice_aggregates_agree_with_t14():
    path = ROOT / v.PUBLIC_AGGREGATES
    if not path.exists():
        pytest.skip("voice aggregates not published yet")
    agg = json.loads(path.read_text(encoding="utf-8"))
    v.validate_aggregates(agg, v.load_public_metrics(ROOT))
    assert agg["withheld"] == v.PUBLIC_WITHHELD
    assert not {"wer_by_transition", "wer_by_prediction_change"} & set(agg)
    assert agg["records"] == 36 and agg["prediction_changed"] == 5 and agg["net_correct_change"] == -4
    assert all(n == 0 for n in agg["failures"].values())
    assert agg["t14_control"]["wer"] == 126 / 803
    assert agg["transitions"] == {"stayed_correct": 26, "worsened": 4, "improved": 0,
                                  "stayed_wrong_same": 5, "stayed_wrong_changed": 1}
    assert path.read_text(encoding="utf-8") == json.dumps(agg, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_public_t15_files_contain_no_group_wer_or_unsupported_causal_claims():
    public_dir = ROOT / t.PUBLIC_DIR
    texts = {p.name: p.read_text(encoding="utf-8") for p in sorted(public_dir.iterdir()) if p.is_file()}
    for name, payload in texts.items():
        assert "records_without_word_errors" not in payload, name
        assert not re.search(r'"wer_by_(?:transition|prediction_change)"\s*:\s*\{', payload), name
    report = texts["report.md"]
    for phrase in ("35 из 36", "без единой ошибки", "0,1466", "0,1587", "0,147 против", "17 / 116",
                   "ничего не раскрывают", "потеря слабого нейтрального сигнала", "не отличаются"):
        assert phrase.lower() not in report.lower(), phrase
    assert "гипотеза, не проверенная причинным экспериментом" in report.lower()
