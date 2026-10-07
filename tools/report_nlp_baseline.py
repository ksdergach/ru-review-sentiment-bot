"""N02: evaluate the frozen baseline on full validation, without fitting or test input."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as baseline

SEED = 42
SAMPLE_SIZE = 40
PAIRS = [(a, b) for a in baseline.CLASS_IDS for b in baseline.CLASS_IDS if a != b]
TAGS = {"mixed", "negation", "sarcasm", "short", "neutral", "context", "unclear"}


def select_errors(records: list[dict], size: int = SAMPLE_SIZE, seed: int = SEED) -> list[dict]:
    """Up to four shortest errors, then round-robin over all six confusion cells.

    Hash ranking within each cell; independent of row order and Python random.
    This is a qualitative coverage sample, not a random estimate of error causes.
    """
    if size < 1:
        raise ValueError("Sample size must be positive")
    errors = [r for r in records if r["true_label_id"] != r["predicted_label_id"]]
    if len({r["record_id"] for r in errors}) != len(errors):
        raise ValueError("Duplicate record_id in errors")
    rank = lambda r: (hashlib.sha256(f"{seed}:{r['record_id']}".encode()).hexdigest(), r["record_id"])
    target = min(size, len(errors))
    selected = sorted(errors, key=lambda r: (r["word_count"], *rank(r)))[:min(4, target)]
    used = {r["record_id"] for r in selected}
    groups = {pair: sorted([r for r in errors if (r["true_label_id"], r["predicted_label_id"]) == pair
                           and r["record_id"] not in used], key=rank) for pair in PAIRS}
    positions = dict.fromkeys(PAIRS, 0)
    while len(selected) < target:
        for pair in PAIRS:
            pos = positions[pair]
            if pos < len(groups[pair]):
                selected.append(groups[pair][pos])
                positions[pair] += 1
                if len(selected) == target:
                    break
    return selected


def check_hash(root: Path, relative: str, expected: str) -> str:
    path = baseline.resolve_path(root, relative)
    actual = baseline.sha256_file(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {relative}")
    return actual


def model_state_sha256(bundle: dict) -> str:
    """Hash inference state, independent of joblib compression and array layout."""
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    params = vectorizer.get_params(deep=False)
    params["dtype"] = np.dtype(params["dtype"]).str
    return baseline.sha256_json({
        "format": "nlp-inference-state-v1",
        "tfidf_state_sha256": baseline._vectorizer_state_sha256(vectorizer),
        "tfidf_params": params,
        "label_mapping": bundle["label_mapping"],
        "classes": classifier.classes_.tolist(),
        "coef": classifier.coef_.tolist(),
        "coef_dtype": classifier.coef_.dtype.str,
        "intercept": classifier.intercept_.tolist(),
        "intercept_dtype": classifier.intercept_.dtype.str,
    })


def load_annotations(path: Path, selected: list[dict], prediction_hash: str,
                     model_hash: str, *, allow_missing: bool = False) -> dict | None:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(
            "Missing error annotations; reports were not written. "
            "Use --prepare-errors only to prepare a local review packet."
        )
    data = baseline.read_json(path)
    if data.get("model_state_sha256") != model_hash:
        raise ValueError("Annotations belong to a different model state")
    if data.get("validation_predictions_sha256") != prediction_hash:
        raise ValueError("Annotations belong to different model predictions")
    if data.get("author") != "Codex (AI agent)":
        raise ValueError("Analysis author must be declared explicitly")
    if data.get("human_review_status") not in ("pending", "confirmed"):
        raise ValueError("Missing human review status")
    if data["human_review_status"] == "confirmed" and not data.get("human_review_evidence"):
        raise ValueError("Human confirmation requires recorded evidence")
    entries = data.get("errors", [])
    if [e["record_id"] for e in entries] != [r["record_id"] for r in selected]:
        raise ValueError("Annotation IDs/order do not match deterministic selection")
    for entry, record in zip(entries, selected):
        for key in ("true_label_id", "predicted_label_id", "text_sha256"):
            if entry.get(key) != record[key]:
                raise ValueError(f"Annotation mismatch: {key}")
        if not entry.get("tags") or not set(entry["tags"]) <= TAGS or not entry.get("explanation", "").strip():
            raise ValueError("Each error needs valid tags and an explanation")
    return data


def evaluate(root: Path = ROOT, *, allow_missing_annotations: bool = False) -> tuple[dict, list[dict], dict | None]:
    config_path = root / "configs/nlp_baseline.json"
    config = baseline.load_config(config_path)
    metadata_rel = f"{config['report_dir']}/run_metadata.json"
    meta = baseline.read_json(root / metadata_rel)
    if meta.get("final_test_used") is not False:
        raise ValueError("Training must explicitly exclude final test")
    if meta["config_canonical_sha256"] != baseline.sha256_json(config):
        raise ValueError("Training configuration changed")
    check_hash(root, "src/train_baseline.py", meta["script_sha256"])
    check_hash(root, config["data_prep_audit_path"], meta["t03_audit_sha256"])
    check_hash(root, config["data_prep_split_ids_path"], meta["t03_split_ids_sha256"])
    bundle_rel = f"{config['model_dir']}/{config['bundle_name']}"
    check_hash(root, bundle_rel, meta["bundle_sha256"])
    for split in ("train", "validation"):
        check_hash(root, config[f"{split}_path"], meta[f"{split}_parquet_sha256"])
    bundle = baseline.load_bundle(root / bundle_rel)
    for key in ("train_ids_sha256", "validation_ids_sha256", "tfidf_state_sha256",
                "train_parquet_sha256", "validation_parquet_sha256", "winner_C"):
        if bundle["training"][key] != meta[key]:
            raise ValueError(f"Bundle/metadata mismatch: {key}")
    current_versions = baseline.library_versions()
    for key, value in bundle["library_versions"].items():
        if key != "platform" and current_versions[key] != value:
            raise ValueError(f"Library version differs from training: {key}")
    if baseline._vectorizer_state_sha256(bundle["vectorizer"]) != meta["tfidf_state_sha256"]:
        raise ValueError("TF-IDF state differs from training")
    frame = baseline.load_prepared_split(root / config["validation_path"], split_name="validation",
                                        text_column=config["text_column"], label_column=config["label_column"],
                                        id_column=config["id_column"])
    if (len(frame) != meta["validation_rows"] or baseline.ids_sha256(
            frame, id_column=config["id_column"], label_column=config["label_column"]) != meta["validation_ids_sha256"]):
        raise ValueError("Full validation IDs do not match training")
    texts = frame[config["text_column"]].tolist()
    pred = baseline.predict_texts(bundle, texts)
    truth = frame[config["label_column"]].to_numpy(dtype=int)
    labels = np.asarray(pred["label_ids"], dtype=int)
    scores = baseline.validation_metrics(truth, labels)
    winner = baseline.read_json(root / config["report_dir"] / "winner_config.json")
    for key, value in scores.items():
        if value != winner["winner"][key]:
            raise ValueError(f"Recomputed metric differs from training: {key}")
    matrix = baseline.validation_confusion_matrix(truth, labels)
    if matrix != winner["validation_confusion_matrix"]["matrix"]:
        raise ValueError("Confusion matrix differs from training")
    records = []
    for i, (rid, text) in enumerate(zip(frame[config["id_column"]], texts)):
        records.append({"record_id": str(rid), "true_label_id": int(truth[i]),
                        "predicted_label_id": int(labels[i]), "text_sha256": baseline.sha256_text(text),
                        "word_count": len(text.split()), "review_text": text})
    prediction_hash = baseline.sha256_json([{k: r[k] for k in ("record_id", "true_label_id", "predicted_label_id")}
                                            for r in records])
    selected = select_errors(records)
    if selected:
        features = bundle["vectorizer"].transform([r["review_text"] for r in selected])
        decisions = bundle["classifier"].decision_function(features)
        for i, row in enumerate(selected):
            row["nonzero_features"] = int(features.getrow(i).nnz)
            row["predicted_minus_true_score"] = float(decisions[i, row["predicted_label_id"]] - decisions[i, row["true_label_id"]])
    state_hash = model_state_sha256(bundle)
    annotations = load_annotations(root / "reports/nlp/baseline_error_annotations.json", selected, prediction_hash,
                                   state_hash, allow_missing=allow_missing_annotations)
    evidence = ["configs/nlp_baseline.json", "src/train_baseline.py", "tools/report_nlp_baseline.py",
                "tools/restore_nlp_splits.py", metadata_rel, f"{config['report_dir']}/winner_config.json",
                f"{config['report_dir']}/bundle_roundtrip.json", config["data_prep_audit_path"], config["data_prep_split_ids_path"]]
    if annotations:
        evidence.append("reports/nlp/baseline_error_annotations.json")
    for optional in ("tools/diagnose_nlp_tfidf.py", "reports/nlp/tfidf_boundary_diagnostic.json",
                     "reports/nlp/retraining_verification.json", "reports/nlp/baseline_reproducibility.md"):
        if (root / optional).exists():
            evidence.append(optional)
    result = {"split": "validation", "independent_test": False, "final_test_used": False,
              "validation_used_for_C_selection": True, "rows": len(frame),
              "correct": int((truth == labels).sum()), "errors": int((truth != labels).sum()),
              "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
              "per_class": {name: {metric: scores[f"{name}_{metric}"] for metric in ("precision", "recall", "f1", "support")}
                            for name in baseline.CLASS_NAMES},
              "confusion_matrix": {"rows": "true", "columns": "predicted", "class_order": baseline.CLASS_NAMES, "matrix": matrix},
              "model": {"path": bundle_rel, "sha256": meta["bundle_sha256"], "tfidf_state_sha256": meta["tfidf_state_sha256"],
                        "state_sha256": state_hash,
                        "C": meta["winner_C"], "config": config, "training_versions": bundle["library_versions"]},
              "data": {key: meta[key] for key in ("train_rows", "validation_rows", "train_ids_sha256", "validation_ids_sha256",
                                                   "train_parquet_sha256", "validation_parquet_sha256")},
              "evaluation_versions": current_versions, "validation_predictions_sha256": prediction_hash,
              "selection": {"seed": SEED, "requested_size": SAMPLE_SIZE, "selected_size": len(selected),
                            "algorithm": "four shortest errors (whitespace words), then round-robin six confusion cells; SHA-256(seed:record_id) ranking",
                            "representative_sample": False,
                            "selected_ids_sha256": baseline.sha256_json([r["record_id"] for r in selected])},
              "analysis_author": annotations["author"] if annotations else None,
              "human_review_status": annotations["human_review_status"] if annotations else "pending",
              "evidence_sha256": {path: baseline.sha256_file(root / path) for path in evidence}}
    return result, selected, annotations


def render(result: dict, selected: list[dict], annotations: dict | None) -> str:
    r = result
    lines = ["# N02 — метрики baseline и анализ ошибок", "",
             "Оценка на полной **validation**, уже использованной для выбора C. Это промежуточная оценка; "
             "она не является независимым test и не гарантирует балл преподавателя. Финальный test не открыт.", "",
             f"Комплект Mac: `{r['model']['path']}`, SHA-256 `{r['model']['sha256']}`. "
             f"Обучение: {r['data']['train_rows']} train-строк; C={r['model']['C']:g}. "
             "Параметры и объяснение выбора — [отчёт обучения](training/validation_summary.md). "
             "Данные, окружение, конфиг и суммы связаны в [baseline_metrics.json](baseline_metrics.json).", "",
             "Решение по Windows/Mac и ограничения — [baseline_reproducibility.md](baseline_reproducibility.md). "
             "Сохранённые Windows 89,024625% не подставлены вместо результата этого комплекта.", "",
             "## Метрики", "", f"Правильных ответов: **{r['correct']} / {r['rows']}**; ошибок: **{r['errors']}**. "
             f"Accuracy: **{100*r['accuracy']:.4f}%**; macro-F1: **{r['macro_f1']:.6f}**.", "",
             "| Класс | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name, m in r["per_class"].items():
        lines.append(f"| {name} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['support']} |")
    lines += ["", "Macro-F1 усредняет F1 трёх классов с равным весом. "
              f"Neutral содержит только {r['per_class']['neutral']['support']} / {r['rows']} строк; "
              "accuracy при преобладании positive скрывает его слабое распознавание.", "",
              "Шкала задания относится к accuracy:", "", "| Порог accuracy | Баллы за классификацию |", "|---|---:|",
              "| ≥90% | 10 |", "| ≥85% | 8 |", "| ≥80% | 5 |", "| ≥75% | 3 |", "",
              f"Текущая validation accuracy {100*r['accuracy']:.4f}% попадает в диапазон "
              f"{next((str(points) for threshold, points in [(0.90,10),(0.85,8),(0.80,5),(0.75,3)] if r['accuracy'] >= threshold), 'ниже указанных порогов')} "
              "баллов по числовой шкале. Это сопоставление, не обещание оценки за работу.", "",
              "## Матрица ошибок", "", "Строки — исходная метка, столбцы — прогноз; порядок negative=0, neutral=1, positive=2.", "",
              "| Истина / прогноз | negative | neutral | positive |", "|---|---:|---:|---:|"]
    for name, row in zip(baseline.CLASS_NAMES, r["confusion_matrix"]["matrix"]):
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    lines += ["", "## Протокол анализа", "",
              f"После фиксации комплекта выбраны {len(selected)} реальных ошибок: четыре самых коротких "
              "по числу слов, затем по очереди шесть направлений путаницы классов. В каждой группе порядок "
              "задаёт SHA-256 от `42:record_id`; равенства разрешаются по ID. Уже выбранные строки исключаются. "
              "Если ошибок меньше 40, берутся все. Выбор не зависит от порядка строк.", "",
              "Выборка специально покрывает разные классы и короткие тексты; частоты причин нельзя переносить "
              "на все ошибки корпуса. Названия причин — интерпретация текста агентом, не доказательство "
              "внутреннего механизма модели. Исходные метки сохраняются, даже когда они неоднозначны.", "",
              "Автор анализа: Codex (AI agent). " + ("Проверка человеком: ожидается; выполнение не заявляется." if r["human_review_status"] == "pending"
                                                        else "Проверка человеком подтверждена в журнале с указанием основания."), "",
              "Полные тексты доступны только локально: `artifacts/nlp/baseline_errors.html`. "
              "[Журнал](baseline_errors.csv) содержит ID, метки, хеш текста и объяснение без исходных текстов.", ""]
    if annotations:
        lines += ["## Наблюдения", ""]
        for finding in annotations.get("findings", []):
            lines += [finding, ""]
        lines += [
                  "Теги могут пересекаться; это количества внутри выбранных случаев:", ""]
        counts = Counter(tag for e in annotations["errors"] for tag in e["tags"])
        lines += [f"- `{tag}`: {counts[tag]}" for tag in sorted(TAGS)]
        lines += ["", "## Разбор выбранных ошибок", "", "| № | ID | Истина → прогноз | Интерпретация агента |", "|---:|---|---|---|"]
        for i, e in enumerate(annotations["errors"], 1):
            explanation = e["explanation"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | `{e['record_id']}` | {baseline.CLASS_NAMES[e['true_label_id']]} → "
                         f"{baseline.CLASS_NAMES[e['predicted_label_id']]} | {explanation} |")
    else:
        lines += ["Содержательный анализ ещё не заполнен; наличие выбранных ID не означает завершение N02."]
    lines += ["", "## Воспроизведение", "", "Из корня репозитория в окружении requirements.txt:", "",
              "```bash", "python tools/report_nlp_baseline.py", "```", "",
              "Команда загружает сохранённый комплект, проверяет суммы и версии, заново вычисляет метрики "
              "и те же ID ошибок; обучение не запускается. Подготовка на чистой машине и повтор обучения — "
              "в [протоколе воспроизводимости](baseline_reproducibility.md).", "",
              "После просмотра ошибок C, предобработка, веса классов и пороги для этой сдачи не менялись. "
              "Незакоммиченный `src/text_preprocessing.py` не подключён к этому baseline. "
              "Материалы пригодны для T15 (#18), но не заменяют будущий анализ независимого test и ошибок ASR.", ""]
    return "\n".join(lines)


def write_outputs(root: Path, result: dict, selected: list[dict], annotations: dict | None) -> None:
    if annotations is None:
        raise ValueError("Error annotations are required before writing final reports")
    public = root / "reports/nlp"
    local = root / "artifacts/nlp"
    public.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    baseline.write_json(public / "baseline_metrics.json", result)
    baseline.write_json(local / "baseline_errors.json", selected)
    notes = {r["record_id"]: r for r in annotations["errors"]} if annotations else {}
    fields = ["record_id", "true_label_id", "predicted_label_id", "text_sha256", "word_count", "nonzero_features", "predicted_minus_true_score", "tags", "explanation", "author", "human_review_status"]
    with (public / "baseline_errors.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            note = notes.get(row["record_id"], {})
            writer.writerow({**{k: row[k] for k in fields[:7]}, "tags": ";".join(note.get("tags", [])),
                             "explanation": note.get("explanation", ""), "author": result["analysis_author"],
                             "human_review_status": result["human_review_status"]})
    cards = []
    for i, row in enumerate(selected, 1):
        note = notes.get(row["record_id"], {})
        cards.append(f"<article><h2>{i}. {html.escape(row['record_id'])}</h2><p>Исходная метка: "
                     f"{baseline.CLASS_NAMES[row['true_label_id']]} → прогноз: {baseline.CLASS_NAMES[row['predicted_label_id']]}</p>"
                     f"<div class='review'>{html.escape(row['review_text'])}</div>"
                     f"<p><b>Разбор агента:</b> {html.escape(note.get('explanation', 'ожидается'))}</p></article>")
    (local / "baseline_errors.html").write_text(
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>N02 — ошибки validation</title>"
        "<style>body{max-width:950px;margin:36px auto;padding:0 20px;font:18px/1.6 sans-serif}"
        "article{border-top:1px solid #bbb;margin:28px 0}h2{font-size:16px}.review{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>N02 — 40 ошибок baseline на validation</h1><p>Полные тексты — только локально. "
        "Исходная разметка и предсказания показаны; это не слепая проверка. "
        "Просмотр человеком не заявляется автоматически.</p>" + "".join(cards) + "</html>\n", encoding="utf-8")
    (public / "baseline_analysis.md").write_text(render(result, selected, annotations), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-errors", action="store_true",
                        help="Prepare a local review packet only; do not write final reports")
    args = parser.parse_args(argv)  # Deliberately no arbitrary dataset/test arguments.
    try:
        result, selected, annotations = evaluate(ROOT, allow_missing_annotations=args.prepare_errors)
    except (FileNotFoundError, ValueError) as exc:
        print(f"N02: {exc}", file=sys.stderr)
        return 1
    if args.prepare_errors:
        baseline.write_json(ROOT / "artifacts/nlp/baseline_error_selection.json", {
            "model_state_sha256": result["model"]["state_sha256"],
            "validation_predictions_sha256": result["validation_predictions_sha256"],
            "errors": selected,
        })
        print("Подготовлена локальная выборка; итоговые отчёты не изменены. Анализ ошибок этим не подтверждается.")
        return 0
    write_outputs(ROOT, result, selected, annotations)
    print(json.dumps({key: result[key] for key in ("split", "rows", "accuracy", "macro_f1", "errors", "human_review_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
