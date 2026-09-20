"""Независимая перепроверка полноты поиска near-duplicate кандидатов.

Скрипт заново, своим кодом, строит блоки и исчерпывающе сравнивает все пары
внутри каждого блока, после чего сличает полученный набор кандидатов с тем,
что записал конвейер в reports/data_prep/near_duplicate_candidates.csv.

Ничего не оценивается и не семплируется: если набор совпал, поиск полон в
пределах выбранной схемы блокирования. Если конвейер что-то упустил, скрипт
это покажет и выгрузит недостающие пары с текстами для ручного аудита.

Скрипт не изменяет ни исходные Parquet, ни выходные данные конвейера.
Он пишет:

  reports/data_prep/blocking_analysis.json          — агрегаты, без текстов
  reports/data_prep/blocking_large_blocks.csv       — крупнейшие блоки
  artifacts/data_prep/new_near_duplicate_review.csv — недостающие пары с текстами
                                                       (только если они есть)

Запуск из корня проекта:

    python tools/analyze_blocking_local.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prepare_data import (  # noqa: E402
    SPLITS,
    _block_keys,
    char_ngrams,
    exact_deduplicate,
    filter_and_map_labels,
    load_config,
    load_source_splits,
    stable_pair_id,
)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently re-check near-duplicate search completeness")
    parser.add_argument("--config", default="configs/data_prep.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config_path = (root / args.config).resolve()
    config = load_config(config_path)

    report_dir = root / config["report_dir"]
    artifact_dir = root / config["artifact_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    near = config["near_duplicates"]
    threshold = float(near["threshold"])
    ngram_size = int(near["ngram_size"])
    min_chars = int(near["min_chars"])
    prefix_chars = int(near["prefix_chars"])
    suffix_chars = int(near["suffix_chars"])
    length_ratio_min = float(near["length_ratio_min"])
    max_block_pairs = int(near["max_block_pairs"])

    print("=" * 78)
    print("T03 — INDEPENDENT NEAR-DUPLICATE SEARCH RE-CHECK")
    print("=" * 78)
    print(f"threshold={threshold}  ngram_size={ngram_size}  min_chars={min_chars}")
    print(f"prefix_chars={prefix_chars}  suffix_chars={suffix_chars}")
    print(f"length_ratio_min={length_ratio_min}  max_block_pairs={max_block_pairs}")

    # Воспроизводим ровно то состояние, на котором работает поиск кандидатов.
    source_frames, source_files = load_source_splits(config, root)
    ru_frames = filter_and_map_labels(source_frames, config)
    exact_clean, _, _ = exact_deduplicate(ru_frames)

    records: dict[str, dict] = {}
    blocks: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        frame = exact_clean[split]
        for rid, text, label, movie_id, review_text in zip(
            frame["record_id"], frame["_near_key"], frame["label_name"],
            frame["movie_id"], frame["review_text"],
        ):
            if len(text) < min_chars:
                continue
            records[rid] = {
                "split": split,
                "text": text,
                "chars": len(text),
                "label": label,
                "movie_id": str(movie_id),
                "review_text": review_text,
            }
            for key in _block_keys(text, prefix_chars, suffix_chars):
                blocks[key].append(rid)

    print(f"\nrecords considered: {len(records)}")
    print(f"blocking keys built: {len(blocks)}")

    # -------------------------------------------------- 1. Размеры блоков
    sizes: list[tuple[str, int]] = []
    for key, ids in blocks.items():
        n = len(set(ids))
        if n >= 2:
            sizes.append((key, n))
    sizes.sort(key=lambda kv: (-kv[1], kv[0]))
    largest = sizes[0][1] if sizes else 0
    total_pairs = sum(n * (n - 1) // 2 for _, n in sizes)
    over_budget = [(k, n) for k, n in sizes if n * (n - 1) // 2 > max_block_pairs]

    print(f"\n--- 1. Блоки ---")
    print(f"  блоков с двумя и более записями: {len(sizes)}")
    print(f"  самый большой блок: {largest} записей")
    print(f"  всего пар внутри блоков: {total_pairs}")
    print(f"  блоков сверх бюджета max_block_pairs: {len(over_budget)}")
    print("  двадцать крупнейших блоков:")
    for key, n in sizes[:20]:
        kind = {"p": "prefix", "s": "suffix", "e": "edge"}.get(key[0], key[0])
        body = key[2:]
        preview = (body[:52] + "…") if len(body) > 52 else body
        print(f"    size={n:6d}  type={kind:6s}  key={preview!r}")

    pd.DataFrame(
        [
            {
                "block_size": n,
                "key_type": {"p": "prefix", "s": "suffix", "e": "edge"}.get(k[0], k[0]),
                "pairs": n * (n - 1) // 2,
            }
            for k, n in sizes[:200]
        ]
    ).to_csv(report_dir / "blocking_large_blocks.csv", index=False, encoding="utf-8", lineterminator="\n")

    # ------------------------- 2. Независимый исчерпывающий набор кандидатов
    print(f"\n--- 2. Независимый исчерпывающий пересчёт ---")
    ngram_cache: dict[str, set[str]] = {}

    def grams(rid: str) -> set[str]:
        cached = ngram_cache.get(rid)
        if cached is None:
            cached = char_ngrams(records[rid]["text"], ngram_size)
            ngram_cache[rid] = cached
        return cached

    eligible_pairs: set[tuple[str, str]] = set()
    for key, _ in sizes:
        ids = sorted(set(blocks[key]))
        for left, right in itertools.combinations(ids, 2):
            a, b = records[left], records[right]
            if a["text"] == b["text"]:
                continue
            ratio = min(a["chars"], b["chars"]) / max(a["chars"], b["chars"])
            if ratio < length_ratio_min:
                continue
            eligible_pairs.add((left, right))
    print(f"  пар после фильтра длины: {len(eligible_pairs)}")

    independent: dict[str, dict] = {}
    for i, (left, right) in enumerate(sorted(eligible_pairs), 1):
        if i % 500000 == 0:
            print(f"    ... {i}/{len(eligible_pairs)}")
        sim = jaccard(grams(left), grams(right))
        if sim >= threshold:
            a, b = records[left], records[right]
            pid = stable_pair_id(left, right)
            independent[pid] = {
                "pair_id": pid,
                "left_record_id": left,
                "left_split": a["split"],
                "right_record_id": right,
                "right_split": b["split"],
                "similarity": round(sim, 6),
                "left_chars": a["chars"],
                "right_chars": b["chars"],
                "left_label": a["label"],
                "right_label": b["label"],
                "labels_conflict": a["label"] != b["label"],
                "left_movie_id": a["movie_id"],
                "right_movie_id": b["movie_id"],
                "same_movie_id": a["movie_id"] == b["movie_id"],
            }
    print(f"  независимо найдено пар выше порога: {len(independent)}")

    # --------------------------------- 3. Сличение с набором конвейера
    candidates_path = report_dir / "near_duplicate_candidates.csv"
    pipeline_ids: set[str] = set()
    if candidates_path.exists():
        pipeline_ids = set(pd.read_csv(candidates_path)["pair_id"])
    missing = sorted(set(independent) - pipeline_ids)
    extra = sorted(pipeline_ids - set(independent))

    print(f"\n--- 3. Сличение с конвейером ---")
    print(f"  кандидатов у конвейера: {len(pipeline_ids)}")
    print(f"  независимо найдено: {len(independent)}")
    print(f"  ПРОПУЩЕНО конвейером: {len(missing)}")
    print(f"  лишних у конвейера: {len(extra)}")

    new_review_path = artifact_dir / "new_near_duplicate_review.csv"
    if missing:
        rows = []
        for pid in missing:
            row = dict(independent[pid])
            row["left_text"] = records[row["left_record_id"]]["review_text"]
            row["right_text"] = records[row["right_record_id"]]["review_text"]
            row["decision"] = ""
            row["note"] = ""
            rows.append(row)
        pd.DataFrame(rows).sort_values("similarity", ascending=False, kind="mergesort").to_csv(
            new_review_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        print(f"  недостающие пары с текстами выгружены: {new_review_path}")
        for pid in missing[:20]:
            p = independent[pid]
            print(
                f"    {pid} sim={p['similarity']} {p['left_split']}/{p['right_split']} "
                f"labels={p['left_label']}/{p['right_label']} same_movie={p['same_movie_id']}"
            )

    # ------------------------------------------------------------- Вердикт
    ok = not missing and not extra and not over_budget
    print("\n" + "=" * 78)
    if ok:
        print("ВЕРДИКТ: набор кандидатов конвейера совпал с независимым исчерпывающим пересчётом.")
        print("Ни один блок не отброшен, ни одна пара выше порога не потеряна.")
    else:
        if over_budget:
            print(f"ВЕРДИКТ: {len(over_budget)} блок(ов) превышают max_block_pairs.")
        if missing:
            print(f"ВЕРДИКТ: конвейер пропустил {len(missing)} пар(у) выше порога — нужен ручной аудит.")
        if extra:
            print(f"ВЕРДИКТ: у конвейера {len(extra)} пар(ы), которых нет в независимом пересчёте.")
    print("=" * 78)

    analysis = {
        "config_near_duplicates": near,
        "source_sha256": {sf.split: sf.sha256_before for sf in source_files},
        "records_considered": len(records),
        "blocking_keys_total": len(blocks),
        "blocks_with_pairs": len(sizes),
        "largest_block_size": largest,
        "total_block_pairs": total_pairs,
        "blocks_over_budget": len(over_budget),
        "eligible_pairs_after_length_filter": len(eligible_pairs),
        "independent_pairs_above_threshold": len(independent),
        "pipeline_candidate_pairs": len(pipeline_ids),
        "lost_pairs_above_threshold": len(missing),
        "new_pairs_vs_current_candidates": len(missing),
        "extra_pairs_in_pipeline": len(extra),
        "verdict": "complete" if ok else "incomplete",
    }
    out = report_dir / "blocking_analysis.json"
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"Saved: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
