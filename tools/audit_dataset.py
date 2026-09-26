"""T02: verify raw data provenance and prepare a train-only human review packet."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"negative": 0, "neutral": 1, "positive": 2}
LANGUAGES = ("ru", "kk", "cs")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_remote(source: dict) -> None:
    """Read only public metadata; never accept access terms or download reviews."""
    with urlopen(source["api_url"], timeout=30) as response:
        remote = json.load(response)
    if remote["sha"] != source["revision"] or remote["id"] != source["repository"]:
        raise ValueError("Remote repository/revision differs from the source manifest")
    if remote["cardData"].get("license") != source["license_declared"]:
        raise ValueError("Remote license declaration differs from the source manifest")
    pc = next(x for x in remote["cardData"]["dataset_info"] if x["config_name"] == "pc")
    if pc["features"] != source["features"]:
        raise ValueError("Remote schema differs from the source manifest")
    for split, expected in source["files"].items():
        item = next(x for x in remote["siblings"] if x["rfilename"] == expected["remote_path"])
        rows = next(x["num_examples"] for x in pc["splits"] if x["name"] == split)
        if (item["lfs"]["sha256"], item["size"], rows) != (
            expected["sha256"], expected["size_bytes"], expected["rows"]
        ):
            raise ValueError(f"Remote evidence mismatch: {split}")


def inspect_file(path: Path, expected: dict, features: list) -> tuple[dict, list[dict]]:
    sha = sha256_file(path)
    if sha != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"Source bytes do not match the pinned corpus: {path.name}")
    table = pq.read_table(path)
    schema = [{"name": field.name, "dtype": str(field.type)} for field in table.schema]
    if schema != features or table.num_rows != expected["rows"]:
        raise ValueError(f"Source schema/row count mismatch: {path.name}")
    rows = table.to_pylist()
    languages = {key: 0 for key in LANGUAGES}
    counts = {lang: {label: 0 for label in LABELS} for lang in LANGUAGES}
    for index, row in enumerate(rows):
        if any(not isinstance(row[key], str) or not row[key].strip() for key in table.column_names):
            raise ValueError(f"Null/blank/invalid field: {path.name}, row {index}")
        lang = row["review_language"]
        label = row["review_sentiment"].lower()
        if lang not in LANGUAGES or label not in LABELS:
            raise ValueError(f"Unknown language/label: {path.name}, row {index}")
        languages[lang] += 1
        counts[lang][label] += 1
        row["source_row"] = index
        row["record_id"] = f"{path.name}#row={index:08d}"
        row["label_name"] = label
    return {
        "sha256": sha, "size_bytes": path.stat().st_size,
        "matches_upstream": True, "rows": len(rows), "schema": schema,
        "language_counts": languages, "class_counts_by_language": counts,
        "class_counts_all": {label: sum(counts[lang][label] for lang in LANGUAGES) for label in LABELS},
        "null_or_blank_fields": 0, "unique_movies": len({row["movie_id"] for row in rows}),
    }, rows


def select_sample(train_rows: list[dict], seed: int, per_class: dict) -> list[dict]:
    """Hash ranking avoids RNG/library-version changes; source IDs are zero-based."""
    selected = []
    for label in LABELS:
        candidates = [r for r in train_rows if r["review_language"] == "ru" and r["label_name"] == label]
        candidates.sort(key=lambda r: (
            hashlib.sha256(f"{seed}:{r['record_id']}".encode()).hexdigest(), r["record_id"]
        ))
        count = per_class[label]
        if count <= 0 or len(candidates) < count:
            raise ValueError(f"Not enough Russian train rows for {label}: need {count}")
        selected.extend(candidates[:count])
    return selected


def sample_metadata(rows: list[dict]) -> list[dict]:
    return [{
        "record_id": r["record_id"], "source_row": r["source_row"],
        "label_name": r["label_name"], "label_id": LABELS[r["label_name"]],
    } for r in rows]


def write_review_packet(rows: list[dict], directory: Path, seed: int = 42) -> None:
    """Text stays in ignored artifacts; never overwrite participant notes."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata = sample_metadata(rows)
    manifest = directory / "sample_ids.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing review packet has different IDs; use a separate directory")
    notes = directory / "notes.md"
    if not notes.exists():
        notes.write_text(
            "# Ручной просмотр T02 — не выполнен\n\n"
            f"Участник: \nДата: \nПросмотрены все {len(rows)} записей: нет\n\n"
            "Для 10–15 пограничных случаев укажите ID, исходную метку, трактовку "
            "участника и аргументацию. Не копируйте исходные отзывы в Git. "
            "Не меняйте исходные метки и test.\n\n"
            "| ID | Исходная метка | Трактовка участника | Почему случай пограничный |\n"
            "|---|---|---|---|\n", encoding="utf-8"
        )
    cards = []
    for i, row in enumerate(rows, 1):
        cards.append(
            f"<article><h2>{i}. {html.escape(row['record_id'])}</h2>"
            f"<p>Исходная метка: <b>{row['label_name']}</b></p>"
            f"<div class='review'>{html.escape(row['review_text'])}</div></article>"
        )
    (directory / "review.html").write_text(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        f"<title>T02 — {len(rows)} train-отзывов</title><style>"
        "body{max-width:850px;margin:40px auto;padding:0 20px;font:18px/1.6 sans-serif}"
        "article{border-top:1px solid #ccc;padding:20px 0}h2{font-size:16px}"
        ".review{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        f"<h1>Ручной просмотр train</h1><p>{len(rows)} русскоязычных отзывов; seed={seed}. "
        "Прочитайте все записи, отметьте 10–15 пограничных случаев в notes.md. "
        "Исходные метки показаны для аудита корпуса; ответов модели здесь нет. "
        "Эта страница сама по себе не подтверждает выполненный просмотр.</p>"
        + "".join(cards) + "</html>\n", encoding="utf-8"
    )
    write_json(manifest, metadata)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit(root: Path, source: dict) -> tuple[dict, list[dict]]:
    summaries, movie_ids, train = {}, {}, None
    for split, expected in source["files"].items():
        summary, rows = inspect_file(root / "NLP_dataset" / expected["local_file"], expected, source["features"])
        summary["plan_counts_match"] = source["plan_raw_counts"][split] == {
            "all": summary["rows"], "ru": summary["language_counts"]["ru"],
            **summary["class_counts_by_language"]["ru"],
        }
        summaries[split] = summary
        movie_ids[split] = {r["movie_id"] for r in rows}
        if split == "train":
            train = rows
    overlaps = {
        f"{a}/{b}": len(movie_ids[a] & movie_ids[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    sample = select_sample(train, **source["manual_sample"])
    report = {
        "source_revision": source["revision"], "source_config": source["config"],
        "class_mapping": LABELS, "splits": summaries, "movie_overlap": overlaps,
        "all_match_plan": all(s["plan_counts_match"] for s in summaries.values()),
        "manual_review_status": "pending_participant",
        "sampling": {**source["manual_sample"], "algorithm": "SHA-256(seed:record_id), ascending within each class", "split": "train", "language": "ru"},
    }
    return report, sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-remote", action="store_true", help="Also verify pinned public Hugging Face metadata (network required)")
    args = parser.parse_args()
    source = json.loads((ROOT / "configs/dataset_source.json").read_text(encoding="utf-8"))
    if args.verify_remote:
        verify_remote(source)
    report, sample = audit(ROOT, source)
    write_review_packet(sample, ROOT / "artifacts/dataset_audit", source["manual_sample"]["seed"])
    write_json(ROOT / "reports/dataset/audit.json", report)
    write_json(ROOT / "reports/dataset/sample_ids.json", sample_metadata(sample))
    print(json.dumps({
        "provenance": "PASS", "remote_metadata_checked": args.verify_remote,
        "plan_counts_match": report["all_match_plan"], "movie_overlap": report["movie_overlap"],
        "manual_review": "PENDING", "sample_size": len(sample),
        "review_packet": "artifacts/dataset_audit/review.html",
    }, ensure_ascii=False, indent=2))
    return 0 if report["all_match_plan"] and not any(report["movie_overlap"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
