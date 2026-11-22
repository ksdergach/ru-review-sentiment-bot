"""T15: переходы классификации «эталон → Whisper» по приватным результатам T14 — без инференса.

Запускает уполномоченный участник там, где уже лежит приватный
``artifacts/t14/records.json`` единственного финального запуска T14 (машина измерений).
Модели, Whisper, аудио и тексты T11 не нужны и не читаются::

    python tools/analyze_voice_transitions_t15.py --check   # сверка с T14 и агрегаты, без записи
    python tools/analyze_voice_transitions_t15.py           # + приватные журнал и полные агрегаты, публичные агрегаты
    python tools/analyze_voice_transitions_t15.py --from-full  # публичные агрегаты из готовых полных

Перед расчётом приватные записи сверяются с опубликованным ``reports/voice_eval/metrics.json``:
пересчитанная сводка T14 должна совпасть полностью. Публично пишутся только агрегаты
без ID и меток отдельных записей и без WER по группам переходов; построчные переходы и
полные агрегаты — только в ``artifacts/t15/``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as baseline  # noqa: E402
from tools import evaluate_voice as ev  # noqa: E402

RECORDS_SCHEMA = "t14-voice-eval-records/draft-1"
PRIVATE_RECORDS = "artifacts/t14/records.json"
PRIVATE_JOURNAL = "artifacts/t15/voice_transitions.json"
PRIVATE_FULL_AGGREGATES = "artifacts/t15/voice_transitions_full.json"
PUBLIC_AGGREGATES = "reports/error_analysis/voice_transitions.json"
PUBLIC_METRICS = "reports/voice_eval/metrics.json"
# Оригинальный metrics.json единственного финального запуска T14 (tests/test_voice_eval_public_results.py).
PUBLIC_METRICS_SHA256 = "cac6208afbdf38cdf9af3fecfdba68220c29c960a8cde7e1009b576add852a7a"
# Контрольные значения T14 (10 знаков) — сверяются, а не вычисляются заново как новый эксперимент.
CONTROL = {"reference_macro_f1": 0.8298931342, "asr_macro_f1": 0.7018518519,
           "delta_macro_f1": -0.1280412824, "wer": 0.1569115816}
CONTROL_TOLERANCE = 5e-11
TRANSITIONS = ("stayed_correct", "worsened", "improved", "stayed_wrong_same", "stayed_wrong_changed")
LABELS = ev.LABELS
# Первоначальный публичный CSV T14 (история PR #45) содержит предсказания по каждому ID, поэтому
# состав группы «ответ изменился» известен. Группы переходов малы (в T14 одна — из одной записи), и
# их разность с wer_by_prediction_change дала бы WER и длину эталона отдельной записи с известным ID.
PUBLIC_WITHHELD = {"wer_by_transition": "withheld: small transition groups; together with wer_by_prediction_change "
                                        "and the original T14 CSV of PR #45 they reveal per-record WER. "
                                        f"Full aggregates stay private in {PRIVATE_FULL_AGGREGATES}"}
MIN_PUBLIC_WER_GROUP = 5  # обычный минимальный размер публикуемой группы; пустая группа допустима
AGGREGATE_KEYS = frozenset({"task", "source", "t14_control", "public_bounds", "records", "transitions",
                            "net_correct_change", "prediction_changed", "prediction_change_directions",
                            "failures", "wer_by_transition", "wer_by_prediction_change", "by_true_class"})


# --------------------------------------------------------------------------- #
# Переходы
# --------------------------------------------------------------------------- #
def transition(record: Mapping[str, Any]) -> str:
    """``error`` (пустой ответ, сбой Whisper или классификатора) — неверное предсказание."""
    true, ref, asr = record["true_label"], record["reference_prediction"], record["asr_prediction"]
    if ref == true:
        return "stayed_correct" if asr == true else "worsened"
    if asr == true:
        return "improved"
    return "stayed_wrong_same" if asr == ref else "stayed_wrong_changed"


def _wer_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    s, d, i, n = (sum(r[k] for r in records) for k in
                  ("wer_substitutions", "wer_deletions", "wer_insertions", "reference_word_count"))
    return {"records": len(records), "substitutions": s, "deletions": d, "insertions": i, "reference_words": n,
            "wer": (s + d + i) / n if n else None,
            "records_without_word_errors": sum(r["wer_substitutions"] + r["wer_deletions"] + r["wer_insertions"] == 0
                                               for r in records)}


def has_failures(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(r["asr_status"] != "ok" or r["reference_status"] != "ok"
               or r["asr_classification_status"] != "ok" for r in records)


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Только суммы. Разбивка по истинному классу не публикуется при сбоях: ID сбоев публичны в T14."""
    kinds = [transition(r) for r in records]
    groups = {t: [r for r, k in zip(records, kinds) if k == t] for t in TRANSITIONS}
    changed = [r for r in records if r["reference_prediction"] != r["asr_prediction"]]
    result = {
        "records": len(records),
        "transitions": {t: len(groups[t]) for t in TRANSITIONS},
        "net_correct_change": len(groups["improved"]) - len(groups["worsened"]),
        "prediction_changed": len(changed),
        "prediction_change_directions": dict(sorted(Counter(
            f"{r['reference_prediction']}->{r['asr_prediction']}" for r in changed).items())),
        "failures": {
            "asr_empty": sum(r["asr_status"] == "empty" for r in records),
            "asr_error": sum(r["asr_status"] == "error" for r in records),
            "reference_classifier_errors": sum(r["reference_status"] == "classifier_error" for r in records),
            "asr_classifier_errors": sum(r["asr_classification_status"] == "classifier_error" for r in records),
            "worsened_with_asr_failure": sum(r["asr_status"] != "ok" for r in groups["worsened"]),
        },
        "wer_by_transition": {t: _wer_group(groups[t]) for t in TRANSITIONS},
        "wer_by_prediction_change": {
            "changed": _wer_group(changed),
            "unchanged": _wer_group([r for r in records if r["reference_prediction"] == r["asr_prediction"]]),
        },
    }
    if has_failures(records):
        result["by_true_class"] = "withheld: failures present, their record IDs are public in T14"
    else:
        result["by_true_class"] = {label: {t: sum(r["true_label"] == label for r in groups[t]) for t in TRANSITIONS}
                                   for label in LABELS}
    return result


def public_transition_bounds(results: Mapping[str, Any]) -> dict[str, Any]:
    """Границы числа ухудшений/улучшений, следующие только из двух опубликованных матриц T14."""
    bounds, low_w, high_w, low_i, high_i = {}, 0, 0, 0, 0
    for t, label in enumerate(LABELS):
        ref, asr = (results[m]["confusion_matrix"]["values"][t] for m in ("reference", "asr"))
        n, c_ref, c_asr = sum(ref), ref[t], asr[t]
        lo, hi = max(0, c_ref - c_asr), min(c_ref, n - c_asr)
        bounds[label] = {"worsened": [lo, hi], "improved": [lo - (c_ref - c_asr), hi - (c_ref - c_asr)]}
        low_w, high_w = low_w + lo, high_w + hi
        low_i, high_i = low_i + lo - (c_ref - c_asr), high_i + hi - (c_ref - c_asr)
    return {"by_true_class": bounds, "worsened": [low_w, high_w], "improved": [low_i, high_i]}


# --------------------------------------------------------------------------- #
# Входы и сверка с T14
# --------------------------------------------------------------------------- #
def load_public_metrics(root: Path, expected_sha256: str = PUBLIC_METRICS_SHA256) -> dict[str, Any]:
    raw = (root / PUBLIC_METRICS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ev.DataMismatchError(f"{PUBLIC_METRICS} differs from the original T14 run")
    return json.loads(raw)


def load_private_records(path: Path, expected_ids: Sequence[str],
                         freeze_sha256: str = ev.ACCEPTED_FREEZE_SHA256) -> list[dict[str, Any]]:
    if not path.exists():
        raise ev.PreconditionError(f"Private T14 results not found: {PRIVATE_RECORDS}; run on the measurement machine")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != RECORDS_SCHEMA or data.get("freeze_sha256") != freeze_sha256:
        raise ev.DataMismatchError("Private records are not the accepted T14 schema/freeze")
    records = data.get("records")
    ev.validate_records(records, expected_ids)
    return records


def check_against_t14(records: Sequence[Mapping[str, Any]], public: Mapping[str, Any],
                      control: Mapping[str, float] = CONTROL) -> dict[str, Any]:
    """Сводка, пересчитанная из приватных записей, совпадает с опубликованной; контрольные значения T14."""
    results = public["results"]
    recomputed = json.loads(json.dumps(ev.summarize(records, expected_ids=results["record_ids"])))
    if recomputed != results:
        differing = sorted(k for k in results if recomputed.get(k) != results[k])
        raise ev.DataMismatchError(f"Private records differ from published T14 results: {', '.join(differing)}")
    return _observed_control(results, control)


def _observed_control(results: Mapping[str, Any], control: Mapping[str, float]) -> dict[str, float]:
    observed = {"reference_macro_f1": results["reference"]["macro_f1"], "asr_macro_f1": results["asr"]["macro_f1"],
                "delta_macro_f1": results["delta_macro_f1"], "wer": results["wer"]["wer"]}
    for key, value in control.items():
        if not math.isclose(observed[key], value, rel_tol=0, abs_tol=CONTROL_TOLERANCE):
            raise ev.DataMismatchError(f"T14 control value differs: {key}")
    return observed


def check_consistency(agg: Mapping[str, Any], results: Mapping[str, Any]) -> None:
    t = agg["transitions"]
    if (t["stayed_correct"] + t["worsened"] != results["reference"]["correct"]
            or t["stayed_correct"] + t["improved"] != results["asr"]["correct"] or sum(t.values()) != agg["records"]):
        raise ev.DataMismatchError("Transitions do not add up to T14 correct counts")
    bounds = public_transition_bounds(results)
    for key in ("worsened", "improved"):
        if not bounds[key][0] <= t[key] <= bounds[key][1]:
            raise ev.DataMismatchError(f"Transitions outside bounds implied by T14 matrices: {key}")


def _check_wer_groups(groups: Mapping[str, Any], sizes: Mapping[str, int], totals: Mapping[str, Any]) -> bool:
    if set(groups) != set(sizes):
        return False
    for key, g in groups.items():
        errors = g["substitutions"] + g["deletions"] + g["insertions"]
        if (g["records"] != sizes[key] or not 0 <= g["records_without_word_errors"] <= g["records"]
                or g["wer"] != (errors / g["reference_words"] if g["reference_words"] else None)):
            return False
    fields = ("substitutions", "deletions", "insertions", "reference_words")
    return all(sum(g[f] for g in groups.values()) == totals[f] for f in fields)


def validate_aggregates(agg: Mapping[str, Any], public: Mapping[str, Any], *,
                        metrics_sha256: str = PUBLIC_METRICS_SHA256,
                        control: Mapping[str, float] = CONTROL) -> None:
    """Полные или публичные агрегаты согласованы с опубликованной сводкой T14; приватные записи не нужны."""
    results = public["results"]

    def fail(what: str) -> None:
        raise ev.DataMismatchError(f"T15 voice aggregates disagree with T14: {what}")

    full = "withheld" not in agg
    if set(agg) != (AGGREGATE_KEYS if full else AGGREGATE_KEYS - set(PUBLIC_WITHHELD) | {"withheld"}):
        fail("keys")
    if not full and agg["withheld"] != PUBLIC_WITHHELD:
        fail("withheld")
    if agg["task"] != "T15" or agg["source"] != {"t14_metrics": PUBLIC_METRICS, "t14_metrics_sha256": metrics_sha256,
                                                  "private_records": PRIVATE_RECORDS, "inference_run": False}:
        fail("source")
    if agg["t14_control"] != _observed_control(results, control):
        fail("t14_control")
    if agg["public_bounds"] != public_transition_bounds(results) or agg["records"] != results["records"]:
        fail("public_bounds/records")
    t = agg["transitions"]
    if set(t) != set(TRANSITIONS):
        fail("transitions")
    check_consistency(agg, results)
    if agg["net_correct_change"] != t["improved"] - t["worsened"]:
        fail("net_correct_change")
    directions = agg["prediction_change_directions"]
    changed = t["worsened"] + t["improved"] + t["stayed_wrong_changed"]
    if agg["prediction_changed"] != changed or sum(directions.values()) != changed:
        fail("prediction_changed")
    columns = results["reference"]["confusion_matrix"]["columns"]
    ref_cols, asr_cols = ([sum(col) for col in zip(*results[m]["confusion_matrix"]["values"])] for m in ("reference", "asr"))
    flow = Counter()
    for key, n in directions.items():
        source, _, target = key.partition("->")
        if source == target or not {source, target} <= set(columns) or n <= 0:
            fail("prediction_change_directions")
        flow[source] -= n
        flow[target] += n
    if any(r + flow[c] != a for c, r, a in zip(columns, ref_cols, asr_cols)):
        fail("prediction_change_directions")
    f, pf = agg["failures"], results["failures"]
    if (f["asr_empty"], f["asr_error"], f["reference_classifier_errors"], f["asr_classifier_errors"]) != (
            pf["asr_status"]["empty"], pf["asr_status"]["error"], pf["reference_classifier_errors"],
            pf["asr_classifier_errors"]) or not 0 <= f["worsened_with_asr_failure"] <= t["worsened"]:
        fail("failures")
    by_class, failures_present = agg["by_true_class"], any(f[k] for k in f if k != "worsened_with_asr_failure")
    if failures_present or isinstance(by_class, str):
        if not (failures_present and isinstance(by_class, str) and by_class.startswith("withheld")):
            fail("by_true_class")
    else:
        if set(by_class) != set(LABELS) or any(set(row) != set(TRANSITIONS) for row in by_class.values()):
            fail("by_true_class")
        for i, label in enumerate(LABELS):
            row, ref, asr = (by_class[label], results["reference"]["confusion_matrix"]["values"][i],
                             results["asr"]["confusion_matrix"]["values"][i])
            if (sum(row.values()) != sum(ref) or row["stayed_correct"] + row["worsened"] != ref[i]
                    or row["stayed_correct"] + row["improved"] != asr[i]):
                fail(f"by_true_class: {label}")
        if any(sum(by_class[label][k] for label in LABELS) != t[k] for k in TRANSITIONS):
            fail("by_true_class totals")
    if not _check_wer_groups(agg["wer_by_prediction_change"],
                             {"changed": changed, "unchanged": agg["records"] - changed}, results["wer"]):
        fail("wer_by_prediction_change")
    if full and not _check_wer_groups(agg["wer_by_transition"], t, results["wer"]):
        fail("wer_by_transition")


def public_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Публичное представление полных агрегатов: без групп WER, разность которых выделяет запись."""
    small = [k for k, g in summary["wer_by_prediction_change"].items() if 0 < g["records"] < MIN_PUBLIC_WER_GROUP]
    if small:
        raise ev.PublicationBlockedError(f"WER group smaller than {MIN_PUBLIC_WER_GROUP} records: "
                                         f"{', '.join(small)}; nothing written")
    view = {k: v for k, v in summary.items() if k not in PUBLIC_WITHHELD}
    return {**view, "withheld": dict(PUBLIC_WITHHELD)}


def _dump(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


_RECORD_ID = re.compile(r"voice_\d{3}")


def assert_public_safe(payload: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Публичные агрегаты: без ID записей, построчных полей и текстов Whisper."""
    if _RECORD_ID.search(payload) or any(r["record_id"] in payload for r in records):
        raise ev.PublicationBlockedError("Public aggregates contain a record ID; nothing written")
    ev.assert_no_private_text([payload], [r.get("asr_text") or "" for r in records])


def private_journal(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keep = ("record_id", "true_label", "reference_prediction", "asr_prediction", "asr_status",
            "reference_status", "asr_classification_status", "wer_substitutions", "wer_deletions",
            "wer_insertions", "reference_word_count")
    return {"schema": "t15-voice-transitions/draft-1", "source": PRIVATE_RECORDS,
            "records": [{**{k: r[k] for k in keep}, "transition": transition(r),
                         "content_note": None} for r in records]}


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:  # существующие результаты не перезаписываются
        fh.write(text)


def run(root: Path = ROOT, *, records_path: Path | None = None, write: bool = False,
        metrics_sha256: str = PUBLIC_METRICS_SHA256, freeze_sha256: str = ev.ACCEPTED_FREEZE_SHA256,
        control: Mapping[str, float] = CONTROL) -> dict[str, Any]:
    """Параметры после ``write`` — только для тестов на искусственных данных."""
    public = load_public_metrics(root, metrics_sha256)
    freeze_ids = baseline.read_json(root / "reports/freeze.json")["voice_test"]["record_ids"]
    if public["results"]["record_ids"] != freeze_ids:
        raise ev.DataMismatchError("Published T14 record IDs differ from the T12 freeze")
    records = load_private_records(records_path or root / PRIVATE_RECORDS, freeze_ids, freeze_sha256)
    observed = check_against_t14(records, public, control)
    agg = aggregate(records)
    check_consistency(agg, public["results"])
    summary = {"task": "T15", "source": {"t14_metrics": PUBLIC_METRICS, "t14_metrics_sha256": metrics_sha256,
                                         "private_records": PRIVATE_RECORDS, "inference_run": False},
               "t14_control": observed, "public_bounds": public_transition_bounds(public["results"]), **agg}
    validate_aggregates(summary, public, metrics_sha256=metrics_sha256, control=control)
    payload = _dump(public_view(summary))
    assert_public_safe(payload, records)
    if write:
        targets = [root / PRIVATE_JOURNAL, root / PRIVATE_FULL_AGGREGATES, root / PUBLIC_AGGREGATES]
        if existing := [str(p.relative_to(root)) for p in targets if p.exists()]:
            raise ev.PreconditionError(f"Results already exist: {', '.join(existing)}")
        for private in targets[:2]:
            if baseline.git_ignore_status(private, root) is not True:
                raise ev.PreconditionError(f"{private.relative_to(root)} must be ignored by Git")
        _write_new(targets[0], json.dumps(private_journal(records), ensure_ascii=False, indent=2) + "\n")
        _write_new(targets[1], _dump(summary))
        _write_new(targets[2], payload)
    return summary


def publish_from_full(root: Path = ROOT, *, metrics_sha256: str = PUBLIC_METRICS_SHA256,
                      control: Mapping[str, float] = CONTROL) -> dict[str, Any]:
    """Публичные агрегаты из уже посчитанных полных агрегатов (``artifacts/t15/``) — без приватных записей."""
    source = root / PRIVATE_FULL_AGGREGATES
    if not source.exists():
        raise ev.PreconditionError(f"Full aggregates not found: {PRIVATE_FULL_AGGREGATES}")
    if baseline.git_ignore_status(source, root) is not True:
        raise ev.PreconditionError(f"{PRIVATE_FULL_AGGREGATES} must be ignored by Git")
    public = load_public_metrics(root, metrics_sha256)
    full = json.loads(source.read_text(encoding="utf-8"))
    if "withheld" in full:
        raise ev.DataMismatchError(f"{PRIVATE_FULL_AGGREGATES} is not a full aggregates file")
    validate_aggregates(full, public, metrics_sha256=metrics_sha256, control=control)
    view = public_view(full)
    validate_aggregates(view, public, metrics_sha256=metrics_sha256, control=control)
    payload = _dump(view)
    assert_public_safe(payload, [])
    target = root / PUBLIC_AGGREGATES
    if target.exists():
        raise ev.PreconditionError(f"Results already exist: {PUBLIC_AGGREGATES}")
    _write_new(target, payload)
    return view


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify and print aggregates, write nothing")
    mode.add_argument("--from-full", action="store_true",
                      help=f"write {PUBLIC_AGGREGATES} from {PRIVATE_FULL_AGGREGATES}, no private records needed")
    args = parser.parse_args(argv)
    try:
        summary = publish_from_full(ROOT) if args.from_full else run(ROOT, write=not args.check)
    except (ev.PreconditionError, ev.DataMismatchError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except ev.PublicationBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"transitions": summary["transitions"], "failures": summary["failures"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
