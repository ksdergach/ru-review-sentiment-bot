"""Локальная проверка T03 в среде с pandas + pyarrow.

Скрипт ничего не изменяет в исходных данных. Он запускает штатный конвейер
подготовки дважды и независимо перепроверяет результат.

Запуск из корня проекта:

    python tools/verify_t03_local.py

Результат печатается в stdout и дублируется в
reports/data_prep/verification_local.json (только агрегаты и идентификаторы,
без текстов отзывов).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prepare_data import exact_normalize  # noqa: E402

SPLITS = ("train", "validation", "test")
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: object = "") -> bool:
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail != "" else ""))
    return ok


def info(name: str, value: object) -> None:
    CHECKS.append({"check": name, "status": "INFO", "detail": str(value)})
    print(f"[INFO] {name}: {value}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent local verification of T03")
    parser.add_argument("--config", default="configs/data_prep.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config_path = (root / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    input_dir = root / config["input_dir"]
    report_dir = root / config["report_dir"]
    output_dir = root / config["output_dir"]
    source_paths = {s: input_dir / config["files"][s] for s in SPLITS}

    print("=" * 78)
    print("T03 LOCAL VERIFICATION")
    print("=" * 78)
    info("python", sys.version.split()[0])
    try:
        import pyarrow

        info("pyarrow", pyarrow.__version__)
    except ImportError:
        check("pyarrow is available", False, "pyarrow is required")
        return 2
    info("pandas", pd.__version__)

    # ---------------------------------------------------------------- 1. Источники
    print("\n--- 1. Исходные Parquet ---")
    sha_before = {}
    raw = {}
    for split, path in source_paths.items():
        if not check(f"source file exists: {path.name}", path.exists()):
            return 2
        sha_before[split] = sha256_file(path)
        info(f"sha256 {split}", sha_before[split])
        try:
            raw[split] = pd.read_parquet(path)
            check(f"source readable: {split}", True, f"{len(raw[split])} rows")
        except Exception as exc:
            check(f"source readable: {split}", False, exc)
            return 2

    required = ["movie_id", "review_text", "review_sentiment", "review_language"]
    for split, df in raw.items():
        info(f"schema {split}", dict(df.dtypes.astype(str)))
        check(f"required columns present: {split}", set(required) <= set(df.columns),
              sorted(set(required) - set(df.columns)))
        nulls = {c: int(df[c].isna().sum()) for c in required if int(df[c].isna().sum())}
        check(f"no nulls in required columns: {split}", not nulls, nulls)
        non_str = int((~df["review_text"].map(lambda v: isinstance(v, str))).sum())
        check(f"review_text all strings: {split}", non_str == 0, f"{non_str} non-string")
        empty = int(df["review_text"].map(lambda v: isinstance(v, str) and v == "").sum())
        blank = int(df["review_text"].map(lambda v: isinstance(v, str) and v.strip() == "" and v != "").sum())
        check(f"no empty review_text: {split}", empty == 0, f"{empty} empty")
        check(f"no whitespace-only review_text: {split}", blank == 0, f"{blank} whitespace-only")
        info(f"languages {split}", df["review_language"].value_counts().to_dict())
        info(f"raw review_sentiment values {split}", sorted(df["review_sentiment"].astype(str).unique()))
        non_nfc = int(df["review_text"].map(lambda v: unicodedata.normalize("NFC", v) != v).sum())
        info(f"rows whose review_text is not NFC-normalised: {split}", non_nfc)

    # --------------------------------------------------- 2. Русскоязычная часть
    print("\n--- 2. Русскоязычные части и классы ---")
    ru = {}
    for split, df in raw.items():
        df = df.reset_index(drop=True)
        positions = list(df.index[df["review_language"].eq(config["target_language"])])
        part = df.loc[positions].copy()
        part["source_row"] = positions
        part["label_name"] = part["review_sentiment"].astype(str).str.strip().str.casefold()
        part["_key"] = part["review_text"].map(exact_normalize)
        part["record_id"] = [
            f"{source_paths[split].name}#row={i:08d}" for i in part["source_row"]
        ]
        ru[split] = part
        info(f"ru rows {split}", len(part))
        info(f"class distribution {split}", part["label_name"].value_counts().to_dict())
        info(f"unique movie_id {split}", int(part["movie_id"].nunique()))

    for split, part in ru.items():
        extra = int(part.duplicated("_key", keep="first").sum())
        groups = int((part.groupby("_key").size() > 1).sum())
        info(f"exact duplicate groups within {split}", f"{groups} groups / {extra} extra rows")

    combined = pd.concat(
        [p[["record_id", "_key", "label_name"]].assign(split=s) for s, p in ru.items()], ignore_index=True
    )
    per_key = combined.groupby("_key")["label_name"].nunique()
    conflict_keys = set(per_key.loc[per_key > 1].index)
    info("label conflict groups (same normalised text, different label)", len(conflict_keys))
    if conflict_keys:
        sample = combined.loc[combined["_key"].isin(list(conflict_keys)[:5])]
        info("label conflict sample (ids only)",
             sample[["record_id", "split", "label_name"]].to_dict("records")[:10])

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = len(set(ru[a]["movie_id"].astype(str)) & set(ru[b]["movie_id"].astype(str)))
        info(f"raw movie_id overlap {a}/{b}", overlap)

    # ------------------------------------------------------ 3. Два прогона
    print("\n--- 3. Два последовательных запуска конвейера ---")
    cmd = [sys.executable, "src/prepare_data.py", "--config", args.config]
    first = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(first.stdout.strip())
    if first.stderr.strip():
        print("stderr:", first.stderr.strip())
    if not check("pipeline run #1 exit code 0", first.returncode == 0, first.returncode):
        (report_dir / "verification_local.json").write_text(
            json.dumps({"checks": CHECKS, "stderr": first.stderr}, ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n",
        )
        return 2

    tracked = ["split_ids.csv", "excluded_ids.csv", "audit.json", "audit_report.md",
               "near_duplicate_candidates.csv", "label_conflicts.csv"]
    snapshot = {n: (report_dir / n).read_bytes() for n in tracked if (report_dir / n).exists()}
    prepared_snapshot = {s: (output_dir / f"{s}.parquet").read_bytes() for s in SPLITS}

    second = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(second.stdout.strip())
    if second.stderr.strip():
        print("stderr:", second.stderr.strip())
    check("pipeline run #2 exit code 0", second.returncode == 0, second.returncode)

    print("\n--- 4. Воспроизводимость ---")
    for name, content in snapshot.items():
        check(f"byte-identical across runs: {name}", (report_dir / name).read_bytes() == content)
    for split, content in prepared_snapshot.items():
        same = (output_dir / f"{split}.parquet").read_bytes() == content
        # Байтовое различие Parquet может быть вызвано метаданными библиотеки;
        # решающая проверка — совпадение record_id ниже.
        info(f"prepared parquet byte-identical: {split}", same)

    audit = json.loads((report_dir / "audit.json").read_text(encoding="utf-8"))
    check("audit status is completed", audit.get("status") == "completed", audit.get("status"))

    # --------------------------------------------- 5. Целостность источников
    print("\n--- 5. Неизменность исходных Parquet ---")
    for split, path in source_paths.items():
        after = sha256_file(path)
        check(f"source unchanged: {split}", after == sha_before[split], after)
        check(f"audit reports source unchanged: {split}", audit["source_files"][split]["unchanged"] is True)
        check(f"audit sha256 matches measured: {split}",
              audit["source_files"][split]["sha256_after"] == after)

    # ------------------------------------------------- 6. Итоговые Parquet
    print("\n--- 6. Итоговые data/processed ---")
    prepared = {}
    for split in SPLITS:
        path = output_dir / f"{split}.parquet"
        if not check(f"prepared file exists: {split}", path.exists()):
            continue
        prepared[split] = pd.read_parquet(path)
        check(f"prepared readable: {split}", True, f"{len(prepared[split])} rows")

    for split, df in prepared.items():
        internal = [c for c in df.columns if str(c).startswith("_")]
        check(f"no internal columns: {split}", not internal, internal)
        info(f"prepared columns {split}", list(df.columns))
        check(f"rows match audit: {split}", len(df) == audit["final_summary"][split]["rows"],
              f"{len(df)} vs {audit['final_summary'][split]['rows']}")
        check(f"only target language: {split}", df["review_language"].eq(config["target_language"]).all())
        check(f"label_id within {{0,1,2}}: {split}", set(df["label_id"].unique()) <= {0, 1, 2})
        mapping_ok = df.groupby("label_name")["label_id"].nunique().eq(1).all() and set(
            zip(df["label_name"], df["label_id"])
        ) <= {("negative", 0), ("neutral", 1), ("positive", 2)}
        check(f"class mapping negative=0/neutral=1/positive=2: {split}", mapping_ok)
        counts = df["label_name"].value_counts().to_dict()
        info(f"prepared class distribution {split}", counts)
        check(f"record_id unique: {split}", df["record_id"].is_unique)

    # split_ids соответствие
    split_ids = pd.read_csv(report_dir / "split_ids.csv")
    for split, df in prepared.items():
        ids_csv = set(split_ids.loc[split_ids["split"].eq(split), "record_id"])
        check(f"split_ids matches prepared parquet: {split}", ids_csv == set(df["record_id"]),
              f"csv={len(ids_csv)} parquet={len(df)}")

    # review_text сохранён буквально
    print("\n--- 7. Сохранность review_text и record_id ---")
    for split, df in prepared.items():
        src = raw[split].copy()
        src["record_id"] = [f"{source_paths[split].name}#row={i:08d}" for i in range(len(src))]
        merged = df[["record_id", "review_text", "source_row"]].merge(
            src[["record_id", "review_text"]], on="record_id", how="left", suffixes=("_out", "_src")
        )
        check(f"every prepared record_id exists in source: {split}", merged["review_text_src"].notna().all())
        identical = merged["review_text_out"].equals(merged["review_text_src"])
        check(f"review_text identical to source: {split}", identical)
        rebuilt = merged["record_id"].eq(
            df["source_row"].map(lambda i: f"{source_paths[split].name}#row={i:08d}").values
        ).all()
        check(f"record_id consistent with source_row: {split}", bool(rebuilt))

    # ------------------------------------------------ 8. Утечки после очистки
    print("\n--- 8. Утечки после очистки ---")
    keys = {s: set(df["review_text"].map(exact_normalize)) for s, df in prepared.items()}
    movies = {s: set(df["movie_id"].astype(str)) for s, df in prepared.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        check(f"no normalised exact-text overlap {a}/{b}", not (keys[a] & keys[b]), len(keys[a] & keys[b]))
        check(f"no movie_id overlap {a}/{b}", not (movies[a] & movies[b]), len(movies[a] & movies[b]))

    # test неизменен относительно ru-части источника
    ru_test_ids = set(ru["test"]["record_id"])
    check("test keeps every russian source row", set(prepared["test"]["record_id"]) == ru_test_ids,
          f"{len(prepared['test'])} vs {len(ru_test_ids)}")

    # --------------------------------------- 9. Соблюдение ручных решений
    print("\n--- 9. Соблюдение ручных near-duplicate решений ---")
    candidates = pd.read_csv(report_dir / "near_duplicate_candidates.csv")
    decision_path = root / config["near_duplicates"]["decision_file"]
    decisions: dict[str, str] = {}
    if decision_path.exists():
        dec_df = pd.read_csv(decision_path, encoding="utf-8-sig", dtype=str).fillna("")
        decisions = dict(
            zip(dec_df["pair_id"].str.strip(), dec_df["decision"].str.strip().str.casefold())
        )
    info("manual decisions", f"{len(decisions)} total for {len(candidates)} candidates")

    surviving = set().union(*(set(df["record_id"]) for df in prepared.values()))
    excluded = pd.read_csv(report_dir / "excluded_ids.csv", dtype=str).fillna("")
    near_removed = {
        r.record_id: r.related_record_id
        for r in excluded.itertuples(index=False)
        if str(r.reason).startswith("near_duplicate_")
    }

    decided = candidates.assign(decision=candidates["pair_id"].map(decisions))
    confirmed = decided.loc[decided["decision"].eq("duplicate")]
    rejected = decided.loc[decided["decision"].eq("not_duplicate")]
    info("confirmed duplicate pairs", len(confirmed))
    info("not_duplicate pairs", len(rejected))

    # (a) каждое решение duplicate действительно применено
    violations_a = []
    for pair in confirmed.itertuples(index=False):
        sides = (pair.left_record_id, pair.right_record_id)
        alive = [rid for rid in sides if rid in surviving]
        both_test = pair.left_split == "test" and pair.right_split == "test"
        if both_test:
            if len(alive) != 2:
                violations_a.append((pair.pair_id, "test row removed", alive))
        elif len(alive) > 1:
            violations_a.append((pair.pair_id, "both sides survived", alive))
    check("every 'duplicate' decision is reflected in the final splits",
          not violations_a, violations_a[:5])

    # (b) решение not_duplicate не приводит к удалению именно near-механизмом
    violations_b = []
    for pair in rejected.itertuples(index=False):
        for rid, other in ((pair.left_record_id, pair.right_record_id),
                           (pair.right_record_id, pair.left_record_id)):
            if near_removed.get(rid) == other:
                violations_b.append((pair.pair_id, rid))
    check("no 'not_duplicate' pair was removed by the near-duplicate mechanism",
          not violations_b, violations_b[:5])

    # (c) подтверждённых near-связей между split'ами не осталось
    cross = confirmed.loc[confirmed["left_split"].ne(confirmed["right_split"])]
    violations_c = [
        pair.pair_id
        for pair in cross.itertuples(index=False)
        if pair.left_record_id in surviving and pair.right_record_id in surviving
    ]
    check("no confirmed near-duplicate link remains between train/validation/test",
          not violations_c, f"{len(cross)} cross-split confirmed pairs, violations: {violations_c[:5]}")

    # (d) каждый кандидат должен иметь ручное решение
    undecided_ids = sorted(set(candidates["pair_id"]) - set(decisions))
    check("every candidate pair has a manual decision", not undecided_ids,
          f"{len(undecided_ids)} undecided: {undecided_ids[:5]}")
    check("audit reports the number of undecided candidates",
          audit["near_duplicates"].get("candidates_without_decision") == len(undecided_ids),
          audit["near_duplicates"].get("candidates_without_decision"))

    # (e) поиск не завершился с непроанализированными блоками без явного статуса
    print("\n--- 10. Полнота поиска кандидатов ---")
    nd = audit["near_duplicates"]
    for field in ("near_blocks_skipped_large", "near_largest_block_size",
                  "near_block_pairs_evaluated", "near_candidates_truncated"):
        check(f"audit exposes search-completeness field: {field}", field in nd)
    check("candidate list was not truncated by max_candidates",
          nd.get("near_candidates_truncated") is False, nd.get("near_candidates_truncated"))
    check("no blocking key was skipped", nd.get("near_blocks_skipped_large") == 0,
          nd.get("near_blocks_skipped_large"))
    info("largest block / pairs evaluated",
         f"{nd.get('near_largest_block_size')} records / {nd.get('near_block_pairs_evaluated')} pairs")

    analysis_path = report_dir / "blocking_analysis.json"
    has_analysis = analysis_path.exists()
    check("an independent search re-check report is present", has_analysis,
          "run: python tools/analyze_blocking_local.py" if not has_analysis else "")
    if has_analysis:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        check("re-check was produced for the current source files",
              analysis.get("source_sha256") == {s: audit["source_files"][s]["sha256_after"] for s in SPLITS})
        check("re-check was produced for the current near-duplicate config",
              analysis.get("config_near_duplicates") == config["near_duplicates"])
        check("re-check finds no pair above threshold missing from the pipeline",
              analysis.get("lost_pairs_above_threshold") == 0,
              analysis.get("lost_pairs_above_threshold"))
        check("re-check finds no extra pair in the pipeline",
              analysis.get("extra_pairs_in_pipeline") == 0,
              analysis.get("extra_pairs_in_pipeline"))
        check("re-check reports no block over the pair budget",
              analysis.get("blocks_over_budget") == 0, analysis.get("blocks_over_budget"))
        check("independent pair count equals the pipeline candidate count",
              analysis.get("independent_pairs_above_threshold") == len(candidates),
              f"{analysis.get('independent_pairs_above_threshold')} vs {len(candidates)}")

    print("\n" + "=" * 78)
    failed = [c for c in CHECKS if c["status"] == "FAIL"]
    passed = [c for c in CHECKS if c["status"] == "PASS"]
    print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
    for c in failed:
        print(f"  FAIL: {c['check']} — {c['detail']}")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "verification_local.json").write_text(
        json.dumps({"passed": len(passed), "failed": len(failed), "checks": CHECKS},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Saved: {report_dir / 'verification_local.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
