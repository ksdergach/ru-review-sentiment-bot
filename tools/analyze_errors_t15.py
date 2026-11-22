"""T15: разбор ошибок модели бота на финальном текстовом test — без инференса и обучения.

Источники — результаты T13, уже полученные по протоколу T12:
``reports/evaluation/predictions.csv`` (сверяется с SHA-256 в ``final_metrics.json``)
и исходный ``NLP_dataset/test-00000-of-00001.parquet`` (сверяется с SHA-256 из T13).
Тексты читаются только для хеша, длины и локального пакета ручной проверки;
в публичные файлы ``reports/error_analysis/`` они не попадают::

    python tools/analyze_errors_t15.py --review-packet   # локально: artifacts/t15/text_review.md
    python tools/analyze_errors_t15.py --check           # сверка опубликованных файлов, без записи
    python tools/analyze_errors_t15.py                   # журнал и сводка по разметке

Модель, метки, предсказания и состав test не меняются.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as baseline  # noqa: E402

MODEL = "baseline"  # модель бота из T12 (freeze.json: bot_model)
SEED = 42  # тот же ключ хеш-ранжирования, что в N02 (tools/report_nlp_baseline.py)
SAMPLE_SIZE = 40
# Сначала четыре направления с участием neutral: при 40 = 6·6 + 4 остаток достаётся им.
PAIR_ORDER = ((0, 1), (2, 1), (1, 0), (1, 2), (0, 2), (2, 0))
CATEGORIES = ("mixed", "unclear_label", "sarcasm", "short", "neutral_confusion",
              "context", "negation", "unknown")
CONFIDENCE = ("high", "medium", "low")
VERDICTS = ("agree", "partial", "disagree")
AUTHOR = "AI agent (draft for participant review)"

PREDICTIONS = "reports/evaluation/predictions.csv"
FINAL_METRICS = "reports/evaluation/final_metrics.json"
FREEZE = "reports/freeze.json"
SOURCE_TEST = "NLP_dataset/test-00000-of-00001.parquet"
PUBLIC_DIR = "reports/error_analysis"
ANNOTATIONS = f"{PUBLIC_DIR}/text_error_annotations.json"
REVIEW_EVIDENCE = f"{PUBLIC_DIR}/human_review_evidence.json"
ERRORS_CSV = f"{PUBLIC_DIR}/text_errors.csv"
SUMMARY_JSON = f"{PUBLIC_DIR}/text_summary.json"
LOCAL_PACKET = "artifacts/t15/text_review.md"
MIN_GUARDED_WORDS = 3  # как в T14: одно-два слова могут совпасть со служебным текстом


def pair_name(pair: tuple[int, int]) -> str:
    return f"{baseline.CLASS_NAMES[pair[0]]}->{baseline.CLASS_NAMES[pair[1]]}"


# --------------------------------------------------------------------------- #
# Предсказания T13
# --------------------------------------------------------------------------- #
def parse_predictions(text: str, model: str = MODEL) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    column = f"{model}_label_id"
    if not rows or not {"record_id", "label_id", column} <= set(rows[0]):
        raise ValueError(f"predictions.csv lacks record_id/label_id/{column}")
    records = []
    for row in rows:
        true, pred = int(row["label_id"]), int(row[column])
        if true not in baseline.CLASS_IDS or pred not in baseline.CLASS_IDS:
            raise ValueError("Unexpected label id in predictions.csv")
        records.append({"record_id": row["record_id"], "true_label_id": true, "predicted_label_id": pred})
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Duplicate record_id in predictions.csv")
    return records


def confusion(records: list[dict]) -> list[list[int]]:
    matrix = [[0] * len(baseline.CLASS_IDS) for _ in baseline.CLASS_IDS]
    for r in records:
        matrix[r["true_label_id"]][r["predicted_label_id"]] += 1
    return matrix


def load_predictions(root: Path = ROOT) -> tuple[list[dict], dict]:
    """Предсказания модели бота; сумма файла и матрица ошибок совпадают с T13."""
    metrics = baseline.read_json(root / FINAL_METRICS)
    freeze = baseline.read_json(root / FREEZE)
    if not (metrics.get("bot_model") == freeze.get("bot_model") == MODEL):
        raise ValueError("Bot model differs from the T12 freeze")
    if metrics.get("bot_model_changed_by_test") is not False:
        raise ValueError("T13 must record that the test did not change the bot model")
    raw = (root / PREDICTIONS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metrics["outputs_sha256"]["predictions.csv"]:
        raise ValueError("predictions.csv differs from the T13 run")
    records = parse_predictions(raw.decode("utf-8"))
    published = metrics["models"][MODEL]["metrics"]
    if not len(records) == published["rows"] == metrics["test"]["rows"] or confusion(records) != published["confusion_matrix"]:
        raise ValueError("Recomputed confusion matrix differs from T13")
    return records, metrics


def find_errors(records: list[dict]) -> list[dict]:
    """Ошибка — несовпадение истинной и предсказанной метки."""
    return [r for r in records if r["true_label_id"] != r["predicted_label_id"]]


# --------------------------------------------------------------------------- #
# Детерминированный отбор
# --------------------------------------------------------------------------- #
def rank(record_id: str, seed: int = SEED) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest(), record_id


def select_errors(errors: list[dict], size: int = SAMPLE_SIZE, seed: int = SEED,
                  order: tuple[tuple[int, int], ...] = PAIR_ORDER) -> list[dict]:
    """По очереди из шести направлений ошибок, внутри — по SHA-256(seed:record_id).

    Не зависит от порядка строк и содержания текстов. Если ошибок меньше ``size`` —
    берутся все. Пустое направление пропускается, его место занимают остальные.
    """
    if size < 1:
        raise ValueError("Sample size must be positive")
    if len(order) != len(PAIR_ORDER) or set(order) != set(PAIR_ORDER):
        raise ValueError("Order must list the six confusion directions once")
    if any(r["true_label_id"] == r["predicted_label_id"] for r in errors):
        raise ValueError("Only misclassified records can be selected")
    if len({r["record_id"] for r in errors}) != len(errors):
        raise ValueError("Duplicate record_id in errors")
    groups = {pair: sorted((r for r in errors if (r["true_label_id"], r["predicted_label_id"]) == pair),
                           key=lambda r: rank(r["record_id"], seed)) for pair in order}
    target, selected, position = min(size, len(errors)), [], 0
    while len(selected) < target:
        for pair in order:
            if position < len(groups[pair]) and len(selected) < target:
                selected.append(groups[pair][position])
        position += 1
    return selected


def distribution(rows: list[dict]) -> dict:
    by_pair = Counter((r["true_label_id"], r["predicted_label_id"]) for r in rows)
    return {
        "total": len(rows),
        "by_true_class": {name: sum(r["true_label_id"] == i for r in rows) for i, name in enumerate(baseline.CLASS_NAMES)},
        "by_predicted_class": {name: sum(r["predicted_label_id"] == i for r in rows)
                               for i, name in enumerate(baseline.CLASS_NAMES)},
        "by_direction": {pair_name(pair): by_pair[pair] for pair in PAIR_ORDER},
        "involving_neutral": sum(1 in (r["true_label_id"], r["predicted_label_id"]) for r in rows),
    }


# --------------------------------------------------------------------------- #
# Тексты исходного test (только локально)
# --------------------------------------------------------------------------- #
def source_row(record_id: str) -> int:
    name, sep, row = record_id.partition("#row=")
    if name != Path(SOURCE_TEST).name or not sep or not row.isdigit():
        raise ValueError("record_id is not a row of the source test parquet")
    return int(row)


def load_texts(root: Path, selected: list[dict], expected_sha256: str) -> dict[str, str]:
    """Тексты выбранных строк исходного Parquet; метки сверяются с predictions.csv."""
    import pyarrow.parquet as pq

    path = root / SOURCE_TEST
    if baseline.sha256_file(path) != expected_sha256:
        raise ValueError("Source test parquet differs from T13")
    table = pq.read_table(path, columns=["review_text", "review_sentiment"]).to_pydict()
    texts = {}
    for r in selected:
        i = source_row(r["record_id"])
        label = baseline.EXPECTED_LABEL_MAPPING[table["review_sentiment"][i].strip().casefold()]
        if label != r["true_label_id"]:
            raise ValueError("Source label differs from predictions.csv")
        texts[r["record_id"]] = table["review_text"][i]
    return texts


def text_facts(selected: list[dict], texts: dict[str, str]) -> list[dict]:
    return [{**r, "text_sha256": baseline.sha256_text(texts[r["record_id"]]),
             "word_count": len(texts[r["record_id"]].split())} for r in selected]


def assert_no_texts(payloads: list[str], texts: dict[str, str]) -> None:
    """Публичные файлы не содержат исходных отзывов (фрагменты от MIN_GUARDED_WORDS слов целиком)."""
    joined = [" ".join(p.split()) for p in payloads]
    for text in texts.values():
        text = " ".join(text.split())
        if len(text.split()) >= MIN_GUARDED_WORDS and any(text in p for p in joined):
            raise ValueError("Public output contains a source review text")


# --------------------------------------------------------------------------- #
# Разметка и проверка участником
# --------------------------------------------------------------------------- #
def entries_sha256(entries: list[dict]) -> str:
    return baseline.sha256_json([{k: e[k] for k in ("record_id", "categories", "explanation", "confidence")}
                                 for e in entries])


def validate_annotations(data: dict, selected: list[dict], evidence: dict | None = None) -> dict:
    """Разметка относится к этой выборке; ``confirmed`` — только с записанной проверкой участника."""
    entries = data.get("errors", [])
    if [e.get("record_id") for e in entries] != [r["record_id"] for r in selected]:
        raise ValueError("Annotation IDs/order do not match the deterministic selection")
    for entry, record in zip(entries, selected):
        for key in ("true_label_id", "predicted_label_id", "text_sha256"):
            if entry.get(key) != record[key]:
                raise ValueError(f"Annotation mismatch: {key}")
        cats = entry.get("categories")
        if not cats or len(set(cats)) != len(cats) or not set(cats) <= set(CATEGORIES):
            raise ValueError("Each error needs one or more known categories")
        if "unknown" in cats and len(cats) > 1:
            raise ValueError("'unknown' cannot be combined with other categories")
        if entry.get("confidence") not in CONFIDENCE or not str(entry.get("explanation", "")).strip():
            raise ValueError("Each error needs an explanation and a confidence level")
        if entry.get("review_status") not in ("pending", "reviewed"):
            raise ValueError("Unknown review status")
    status = data.get("human_review_status")
    if status == "pending":
        if any(e["review_status"] != "pending" for e in entries):
            raise ValueError("Per-case review recorded while overall review is pending")
        return data
    if status != "confirmed":
        raise ValueError("Missing human review status")
    if evidence is None:
        raise ValueError("Human confirmation requires recorded evidence")
    reviews = evidence.get("reviews", [])
    if (evidence.get("selected_ids_sha256") != baseline.sha256_json([r["record_id"] for r in selected])
            or evidence.get("annotations_sha256") != entries_sha256(entries)
            or [v.get("record_id") for v in reviews] != [e["record_id"] for e in entries]
            or any(v.get("verdict") not in VERDICTS for v in reviews)
            or any(e["review_status"] != "reviewed" for e in entries)):
        raise ValueError("Human review evidence does not cover every selected case")
    counts = Counter(v["verdict"] for v in reviews)
    if "verdict_counts" in evidence and evidence["verdict_counts"] != {k: counts[k] for k in VERDICTS}:
        raise ValueError("Human review verdict counts do not match the reviews")
    return data


def category_counts(entries: list[dict]) -> dict:
    counts = Counter(c for e in entries for c in e["categories"])
    return {c: counts[c] for c in CATEGORIES}


def category_by_direction(entries: list[dict]) -> dict:
    table = {}
    for pair in PAIR_ORDER:
        rows = [e for e in entries if (e["true_label_id"], e["predicted_label_id"]) == pair]
        counts = Counter(c for e in rows for c in e["categories"])
        table[pair_name(pair)] = {"cases": len(rows), **{c: counts[c] for c in CATEGORIES if counts[c]}}
    return table


# --------------------------------------------------------------------------- #
# Публичные файлы
# --------------------------------------------------------------------------- #
def build_summary(records: list[dict], errors: list[dict], selected: list[dict], metrics: dict,
                  annotations: dict | None) -> dict:
    entries = annotations["errors"] if annotations else []
    return {
        "task": "T15",
        "model": MODEL,
        "final_test_used_for_tuning": False,
        "inference_run": False,
        "source": {"predictions": PREDICTIONS, "predictions_sha256": metrics["outputs_sha256"]["predictions.csv"],
                   "final_metrics": FINAL_METRICS, "test_rows": len(records),
                   "test_ids_sha256": metrics["test"]["ids_sha256"],
                   "source_parquet_sha256": metrics["test"]["source_parquet_sha256"]},
        "errors": distribution(errors),
        "selection": {
            "rule": "round-robin over six confusion directions in the listed order; inside a direction "
                    "ascending SHA-256('<seed>:<record_id>'), ties by record_id; all errors if fewer than size",
            "seed": SEED, "requested_size": SAMPLE_SIZE, "selected_size": len(selected),
            "direction_order": [pair_name(p) for p in PAIR_ORDER],
            "representative_of_error_causes": False,
            "selected_ids_sha256": baseline.sha256_json([r["record_id"] for r in selected]),
            "selected": distribution(selected),
        },
        "annotation": None if annotations is None else {
            "author": annotations["author"], "analysis_date": annotations["analysis_date"],
            "human_review_status": annotations["human_review_status"],
            "annotations_sha256": entries_sha256(entries),
            "category_counts": category_counts(entries),
            "cases_with_several_categories": sum(len(e["categories"]) > 1 for e in entries),
            "confidence_counts": {c: sum(e["confidence"] == c for e in entries) for c in CONFIDENCE},
            "categories_by_direction": category_by_direction(entries),
        },
    }


CSV_FIELDS = ("record_id", "true_label", "predicted_label", "text_sha256", "word_count",
              "categories", "confidence", "explanation", "review_status")


def errors_csv(selected: list[dict], annotations: dict) -> str:
    notes = {e["record_id"]: e for e in annotations["errors"]}
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for r in selected:
        note = notes[r["record_id"]]
        writer.writerow({"record_id": r["record_id"], "true_label": baseline.CLASS_NAMES[r["true_label_id"]],
                         "predicted_label": baseline.CLASS_NAMES[r["predicted_label_id"]],
                         "text_sha256": r["text_sha256"], "word_count": r["word_count"],
                         "categories": ";".join(note["categories"]), "confidence": note["confidence"],
                         "explanation": note["explanation"], "review_status": note["review_status"]})
    return out.getvalue()


def dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def review_packet(selected: list[dict], texts: dict[str, str], annotations: dict | None) -> str:
    """Локальный пакет для участника: тексты, метки, черновик разбора и поля решения."""
    notes = {e["record_id"]: e for e in (annotations or {}).get("errors", [])}
    lines = ["# T15 — проверка 40 ошибок baseline на test (локально, вне Git)", "",
             "Исходная метка и прогноз показаны — это не слепая разметка. Метки test не исправляются.",
             "Для каждого случая отметьте одно решение и при необходимости допишите замечание.", ""]
    for i, r in enumerate(selected, 1):
        note = notes.get(r["record_id"], {})
        lines += [f"## {i}. `{r['record_id']}`", "",
                  f"Исходная метка: **{baseline.CLASS_NAMES[r['true_label_id']]}** → прогноз: "
                  f"**{baseline.CLASS_NAMES[r['predicted_label_id']]}**; слов: {r['word_count']}", "",
                  "> " + "\n> ".join(texts[r["record_id"]].splitlines() or [""]), "",
                  f"- Категории: `{';'.join(note.get('categories', [])) or 'ожидается'}`; "
                  f"уверенность: {note.get('confidence', '—')}",
                  f"- Объяснение: {note.get('explanation', 'ожидается')}",
                  f"- Вопрос участнику: {note.get('question', 'Согласны ли вы с категориями и объяснением?')}",
                  "- Решение: [ ] согласен  [ ] частично согласен  [ ] не согласен",
                  "- Замечание:", ""]
    return "\n".join(lines)


def analyze(root: Path = ROOT, *, need_annotations: bool = True) -> dict:
    records, metrics = load_predictions(root)
    errors = find_errors(records)
    if len(errors) != metrics["models"][MODEL]["metrics"]["errors"]["total"]:
        raise ValueError("Error count differs from T13")
    chosen = select_errors(errors)
    texts = load_texts(root, chosen, metrics["test"]["source_parquet_sha256"])
    selected = text_facts(chosen, texts)
    annotations = None
    if (root / ANNOTATIONS).exists():
        evidence_path = root / REVIEW_EVIDENCE
        evidence = baseline.read_json(evidence_path) if evidence_path.exists() else None
        annotations = validate_annotations(baseline.read_json(root / ANNOTATIONS), selected, evidence)
    elif need_annotations:
        raise FileNotFoundError(f"Missing {ANNOTATIONS}; run --review-packet to prepare the draft")
    summary = build_summary(records, errors, selected, metrics, annotations)
    public = {SUMMARY_JSON: dump(summary)}
    if annotations:
        public[ERRORS_CSV] = errors_csv(selected, annotations)
    assert_no_texts(list(public.values()), texts)
    return {"summary": summary, "selected": selected, "texts": texts, "annotations": annotations, "public": public}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--review-packet", action="store_true", help=f"write the local packet {LOCAL_PACKET}")
    mode.add_argument("--check", action="store_true", help="compare published files, write nothing")
    args = parser.parse_args(argv)  # Намеренно без путей к другим данным.
    try:
        result = analyze(ROOT, need_annotations=not args.review_packet)
        if args.review_packet:
            target = ROOT / LOCAL_PACKET
            if baseline.git_ignore_status(target, ROOT) is not True:
                raise ValueError(f"{LOCAL_PACKET} must be ignored by Git")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(review_packet(result["selected"], result["texts"], result["annotations"]),
                              encoding="utf-8")
            print(f"Локальный пакет: {LOCAL_PACKET}; проверка участником этим не подтверждается.")
            return 0
        if args.check:
            stale = [p for p, text in result["public"].items()
                     if not (ROOT / p).exists() or (ROOT / p).read_text(encoding="utf-8") != text]
            if stale:
                raise ValueError(f"Published files differ: {', '.join(stale)}")
        else:
            for relative, text in result["public"].items():
                (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / relative).write_text(text, encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        print(f"T15: {exc}", file=sys.stderr)
        return 1
    s = result["summary"]
    print(json.dumps({"errors": s["errors"]["total"], "selected": s["selection"]["selected_size"],
                      "human_review_status": (s["annotation"] or {}).get("human_review_status", "pending")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
