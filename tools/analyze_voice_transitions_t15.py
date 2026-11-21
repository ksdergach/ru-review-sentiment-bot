"""T15: переходы классификации «эталон → Whisper» по приватным результатам T14 — без инференса.

Запускает уполномоченный участник там, где уже лежит приватный
``artifacts/t14/records.json`` единственного финального запуска T14 (машина измерений).
Модели, Whisper, аудио и тексты T11 не нужны и не читаются::

    python tools/analyze_voice_transitions_t15.py --check   # сверка с T14 и агрегаты, без записи
    python tools/analyze_voice_transitions_t15.py           # + приватный журнал и публичные агрегаты

Перед расчётом приватные записи сверяются с опубликованным ``reports/voice_eval/metrics.json``:
пересчитанная сводка T14 должна совпасть полностью. Публично пишутся только агрегаты
без ID и меток отдельных записей; построчные переходы — только в ``artifacts/t15/``.
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
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert_public_safe(payload, records)
    if write:
        targets = [root / PRIVATE_JOURNAL, root / PUBLIC_AGGREGATES]
        if existing := [str(p.relative_to(root)) for p in targets if p.exists()]:
            raise ev.PreconditionError(f"Results already exist: {', '.join(existing)}")
        if baseline.git_ignore_status(targets[0], root) is not True:
            raise ev.PreconditionError(f"{PRIVATE_JOURNAL} must be ignored by Git")
        _write_new(targets[0], json.dumps(private_journal(records), ensure_ascii=False, indent=2) + "\n")
        _write_new(targets[1], payload)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify and print aggregates, write nothing")
    args = parser.parse_args(argv)
    try:
        summary = run(ROOT, write=not args.check)
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
