"""T11: validate the private final voice test and freeze its composition.

Reads the coordinator's Excel table and the OGG/Opus files, never runs ASR or a
classifier. ``--write`` stores the private manifest (with transcripts and
labels, outside Git) and the public report (IDs, checksums, aggregates only).
Without ``--write`` the same inputs are rebuilt and compared with both files:
the set is unchanged only if everything except the freeze date matches.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "artifacts/t11/T11_dataset.xlsx"
DEFAULT_AUDIO = ROOT / "artifacts/t11/raw_audio"
DEFAULT_MANIFEST = ROOT / "artifacts/t11/manifest.json"
DEFAULT_REPORT = ROOT / "reports/voice_test/composition.json"
DEBUG_REPORT = ROOT / "reports/speech/verification.json"
LIMITS_CONFIG = ROOT / "configs/resource_check.json"
HUMAN_CONFIRMATION = "reports/voice_test/human_confirmation.json"

LABELS = ("negative", "neutral", "positive")  # ids 0, 1, 2 as in docs/team_plan.md
# Stable pseudonyms of the two real annotators, as confirmed by the coordinator:
# the same person filled each label column for all records.
ANNOTATORS = {"label_a": "R01", "label_b": "R02"}
EXPECTED_RECORDS = 36
EXPECTED_PER_CLASS = 12
MIN_SPEAKERS = 3
YES, NO = "да", "нет"
EMPTY_REASON = {"", "-", "—"}

RECORD_SHEET = "02_Записи"
LABEL_SHEET = "03_Сведение_меток"
RECORD_COLUMNS = {
    "record_id": "ID записи",
    "speaker": "Участник",
    "script": "Текст для записи",
    "file": "Имя файла (предложение)",
    "words": "Слов",
    "noise": "Шум, да/нет",
    "transcript": "Дословная транскрипция ПОСЛЕ аудио",
    "transcript_checked": "Сверена человеком, да/нет",
}
LABEL_COLUMNS = {
    "record_id": "ID",
    "speaker": "Участник",
    "label_a": "Метка А",
    "label_b": "Метка Б",
    "final": "Согласованная метка",
    "dispute": "Есть спор, да/нет",
    "ambiguity": "Причина неоднозначности",
    "confirmed": "Подтверждено человеком, да/нет",
}
UNCHANGED_CONDITIONS = [
    "В каталоге аудио ровно 36 файлов voice_001.ogg–voice_036.ogg с SHA-256 из records.",
    "content_sha256 приватного манифеста совпадает: эталоны, метки R01/R02, согласованные метки, "
    "шум, неоднозначность и обезличенные дикторы не менялись.",
    "Повторный запуск валидатора без --write завершается с кодом 0.",
    "Состав и метки не меняются после получения любых модельных предсказаний; "
    "трудные записи не исключаются.",
]

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class ValidationError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "")).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower().replace("ё", "е"))


# --- xlsx (standard library only) -------------------------------------------------

def _column(ref: str) -> int:
    number = 0
    for char in re.match(r"[A-Z]+", ref).group():
        number = number * 26 + ord(char) - 64
    return number - 1


def read_xlsx(path: Path) -> tuple[dict[str, list[list[str]]], dict]:
    """Return sheet name -> rows of cell strings, and safe workbook metadata."""
    try:
        book = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValidationError(f"Таблица не читается как .xlsx: {error.__class__.__name__}") from None
    with book:
        names = set(book.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            for item in ET.fromstring(book.read("xl/sharedStrings.xml")).findall(f"{{{MAIN_NS}}}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{MAIN_NS}}}t")))
        rels = {
            rel.get("Id"): rel.get("Target")
            for rel in ET.fromstring(book.read("xl/_rels/workbook.xml.rels")).findall(f"{{{PKG_REL_NS}}}Relationship")
        }
        sheets = {}
        for sheet in ET.fromstring(book.read("xl/workbook.xml")).iter(f"{{{MAIN_NS}}}sheet"):
            target = rels[sheet.get(f"{{{REL_NS}}}id")].lstrip("/")
            part = target if target.startswith("xl/") else str(PurePosixPath("xl") / target)
            rows = []
            for row in ET.fromstring(book.read(part)).iter(f"{{{MAIN_NS}}}row"):
                cells = {}
                for cell in row.findall(f"{{{MAIN_NS}}}c"):
                    kind, value = cell.get("t"), cell.find(f"{{{MAIN_NS}}}v")
                    if kind == "s" and value is not None:
                        text = shared[int(value.text)]
                    elif kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{{{MAIN_NS}}}t"))
                    else:
                        text = value.text if value is not None else ""
                    cells[_column(cell.get("r"))] = text or ""
                width = max(cells, default=-1) + 1
                rows.append([cells.get(i, "") for i in range(width)])
            sheets[sheet.get("name")] = rows
        modified = None
        if "docProps/core.xml" in names:
            match = re.search(rb"<dcterms:modified[^>]*>([^<]+)<", book.read("docProps/core.xml"))
            modified = match.group(1).decode() if match else None
    return sheets, {"modified": modified, "sheets": list(sheets)}


def read_table(sheets: dict, sheet: str, columns: dict[str, str]) -> list[dict[str, str]]:
    if sheet not in sheets:
        raise ValidationError(f"В таблице нет листа «{sheet}»")
    rows = sheets[sheet]
    wanted = list(columns.values())
    for index, row in enumerate(rows):
        header = [clean(value) for value in row]
        if all(name in header for name in wanted):
            positions = {key: header.index(name) for key, name in columns.items()}
            break
    else:
        raise ValidationError(f"На листе «{sheet}» нет строки заголовков: {', '.join(wanted)}")
    records = []
    for row in rows[index + 1:]:
        values = {key: clean(row[pos]) if pos < len(row) else "" for key, pos in positions.items()}
        if not values["record_id"]:
            if any(values.values()):
                raise ValidationError(f"На листе «{sheet}» есть строка без ID")
            continue
        records.append(values)
    return records


# --- audio ---------------------------------------------------------------------------

def opus_vendor(path: Path) -> str:
    """Encoder name from the OpusTags vendor field, e.g. ``libopus`` or ``Lavf61.7.103``."""
    head = path.read_bytes()[:4096]
    start = head.find(b"OpusTags")
    if start < 0 or len(head) < start + 12:
        return "unknown"
    length = int.from_bytes(head[start + 8:start + 12], "little")
    vendor = head[start + 12:start + 12 + length].decode("utf-8", "replace").split()
    return vendor[0] if vendor else "unknown"


def probe_audio(path: Path) -> dict:
    """Fully decode one file with PyAV; no ASR involved."""
    import av
    import numpy as np

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValidationError(f"{path.name}: нет аудиопотока")
            stream = container.streams.audio[0]
            info = {
                "container": container.format.name,
                "codec": stream.codec_context.name,
                "channels": stream.codec_context.channels,
                "sample_rate": stream.codec_context.sample_rate,
                "encoder": opus_vendor(path),
            }
            samples, peak, clipped, energy = 0, 0.0, 0, 0.0
            for frame in container.decode(stream):
                data = np.abs(frame.to_ndarray().astype(np.float64))
                samples += frame.samples
                peak = max(peak, float(data.max(initial=0.0)))
                clipped += int((data >= 0.999).sum())
                energy += float((data ** 2).sum()) / max(data.shape[0], 1)
    except ValidationError:
        raise
    except Exception as error:  # PyAV raises several decoder-specific errors
        raise ValidationError(f"{path.name}: не декодируется ({error.__class__.__name__})") from None
    if samples == 0 or not info["sample_rate"]:
        raise ValidationError(f"{path.name}: декодировано 0 сэмплов")
    rms = math.sqrt(energy / samples)
    info.update(
        decoded_samples=samples,
        duration_seconds=round(samples / info["sample_rate"], 3),
        peak_dbfs=round(20 * math.log10(peak), 1) if peak > 0 else None,
        rms_dbfs=round(20 * math.log10(rms), 1) if rms > 0 else None,
        clipped_samples=clipped,
    )
    return info


def _collect_hashes(value, key: str) -> set[str]:
    if isinstance(value, dict):
        found = {value[key]} if isinstance(value.get(key), str) else set()
        return found.union(*(_collect_hashes(item, key) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_collect_hashes(item, key) for item in value))
    return set()


def debug_hashes(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        return set(), set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return _collect_hashes(data, "audio_sha256"), _collect_hashes(data, "text_sha256")


def load_limits(path: Path) -> dict:
    audio = json.loads(path.read_text(encoding="utf-8"))["debug_audio"]
    return {"max_seconds": float(audio["max_seconds"]), "max_bytes": int(audio["max_bytes"])}


# --- build ---------------------------------------------------------------------------

def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build(xlsx: Path, audio_dir: Path, *, debug_report: Path | None = None,
          limits_config: Path | None = None) -> tuple[dict, dict, list[str]]:
    """Return (private manifest, public report, warnings) or raise with all errors."""
    debug_report = debug_report or DEBUG_REPORT
    limits_config = limits_config or LIMITS_CONFIG
    errors: list[str] = []
    warnings: list[str] = []
    if not xlsx.is_file():
        raise ValidationError("Таблица не найдена")
    if not audio_dir.is_dir():
        raise ValidationError("Каталог аудио не найден")
    sheets, book_meta = read_xlsx(xlsx)
    plan = read_table(sheets, RECORD_SHEET, RECORD_COLUMNS)
    merge = read_table(sheets, LABEL_SHEET, LABEL_COLUMNS)
    limits = load_limits(limits_config)

    ids = [row["record_id"] for row in plan]
    expected_ids = [f"voice_{n:03d}" for n in range(1, EXPECTED_RECORDS + 1)]
    if sorted(ids) != expected_ids:
        errors.append(f"ID записей: ожидалось voice_001–voice_{EXPECTED_RECORDS:03d} без повторов, "
                      f"получено {len(ids)} строк, {len(set(ids))} уникальных")
    if [row["record_id"] for row in merge] != ids:
        errors.append("ID и порядок строк на листах записей и сведения меток различаются")
    merge_by_id = {row["record_id"]: row for row in merge}

    expected_files = {f"{record_id}.ogg" for record_id in ids}
    present = {path.name for path in audio_dir.iterdir()}
    if missing := sorted(expected_files - present):
        errors.append(f"Нет аудиофайлов: {', '.join(missing)}")
    if extra := sorted(present - expected_files):
        errors.append(f"Лишние файлы в каталоге аудио: {', '.join(extra)}")

    debug_audio, debug_text = debug_hashes(debug_report)
    pseudonyms: dict[str, str] = {}
    records, audio_info, seen_audio, seen_text = [], {}, {}, {}
    for row in plan:
        record_id = row["record_id"]
        labels = merge_by_id.get(record_id)
        if labels is None:
            continue
        where = f"{record_id}:"
        if row["file"] != f"{record_id}.ogg":
            errors.append(f"{where} имя файла в таблице не совпадает с ID")
        if not row["speaker"]:
            errors.append(f"{where} не указан диктор")
        elif labels["speaker"] != row["speaker"]:
            errors.append(f"{where} диктор различается на двух листах")
        speaker_id = pseudonyms.setdefault(row["speaker"], f"S{len(pseudonyms) + 1:02d}")
        if not row["transcript"]:
            errors.append(f"{where} нет эталонной транскрипции")
        if row["transcript_checked"] != YES:
            errors.append(f"{where} транскрипция не отмечена как сверенная человеком")
        if row["noise"] not in (YES, NO):
            errors.append(f"{where} шум должен быть «да» или «нет»")
        for key in ("label_a", "label_b", "final"):
            if labels[key] not in LABELS:
                errors.append(f"{where} {LABEL_COLUMNS[key]} должна быть одной из {', '.join(LABELS)}")
        if labels["dispute"] not in (YES, NO):
            errors.append(f"{where} «Есть спор» должно быть «да» или «нет»")
        if labels["label_a"] != labels["label_b"] and labels["dispute"] != YES:
            errors.append(f"{where} метки А и Б различаются, но спор не отмечен")
        if labels["label_a"] == labels["label_b"] and labels["final"] != labels["label_a"]:
            errors.append(f"{where} согласованная метка отличается от совпавших меток А и Б")
        if labels["confirmed"] != YES:
            errors.append(f"{where} согласование не подтверждено человеком")
        transcript_hash = sha256_bytes(row["transcript"].encode())
        if transcript_hash in seen_text:
            errors.append(f"{where} транскрипция совпадает с {seen_text[transcript_hash]}")
        seen_text.setdefault(transcript_hash, record_id)
        if transcript_hash in debug_text:
            errors.append(f"{where} транскрипция совпадает с отладочной T08")

        path = audio_dir / f"{record_id}.ogg"
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size == 0:
            errors.append(f"{where} пустой аудиофайл")
            continue
        digest = sha256_file(path)
        if digest in seen_audio:
            errors.append(f"{where} аудио побайтово совпадает с {seen_audio[digest]}")
        seen_audio.setdefault(digest, record_id)
        if digest in debug_audio:
            errors.append(f"{where} аудио совпадает с отладочной записью T08")
        try:
            info = probe_audio(path)
        except ValidationError as error:
            errors.append(str(error))
            continue
        if (info["container"], info["codec"]) != ("ogg", "opus"):
            errors.append(f"{where} формат {info['container']}/{info['codec']}, ожидался ogg/opus")
        if info["duration_seconds"] > limits["max_seconds"] or size > limits["max_bytes"]:
            errors.append(f"{where} превышен лимит голоса ({limits['max_seconds']:g} s / {limits['max_bytes']} B)")
        if info["clipped_samples"]:
            warnings.append(f"{where} {info['clipped_samples']} сэмплов на уровне клиппинга")
        audio_info[record_id] = info
        records.append({
            "record_id": record_id,
            "speaker_id": speaker_id,
            "audio_path": _display_path(path),
            "audio_sha256": digest,
            "audio_bytes": size,
            "duration_seconds": info["duration_seconds"],
            "reference_transcript": row["transcript"],
            "transcript_sha256": transcript_hash,
            "annotator_a": ANNOTATORS["label_a"],
            "label_a": labels["label_a"],
            "annotator_b": ANNOTATORS["label_b"],
            "label_b": labels["label_b"],
            "final_label": labels["final"],
            "dispute": labels["dispute"] == YES,
            "ambiguous": labels["ambiguity"] not in EMPTY_REASON,
            "ambiguity_reason": "" if labels["ambiguity"] in EMPTY_REASON else labels["ambiguity"],
            "noise": row["noise"] == YES,
        })
        # The planning column counts whitespace-separated tokens, a dash included.
        if row["words"] and row["words"] != str(len(row["script"].split())):
            warnings.append(f"{where} в столбце «Слов» {row['words']}, в сценарии {len(row['script'].split())}")

    final = Counter(row["final"] for row in merge)
    if any(final.get(label, 0) != EXPECTED_PER_CLASS for label in LABELS) or len(merge) != EXPECTED_RECORDS:
        errors.append(f"Согласованные метки: ожидалось по {EXPECTED_PER_CLASS} на класс, получено {dict(final)}")
    if len(pseudonyms) < MIN_SPEAKERS:
        errors.append(f"Дикторов {len(pseudonyms)}, нужно не меньше {MIN_SPEAKERS}")
    if errors:
        raise ValidationError("\n".join(errors))

    same_as_script = sum(words(row["script"]) == words(row["transcript"]) for row in plan)
    if same_as_script:
        warnings.append(f"{same_as_script} из {len(plan)} эталонов совпадают со сценарием с точностью "
                        "до регистра и пунктуации; дословность подтверждает только человек")
    content = [{key: value for key, value in record.items() if key != "audio_path"} for record in records]
    manifest = {
        "schema": "t11-voice-manifest/1",
        "label_ids": {label: index for index, label in enumerate(LABELS)},
        "annotators": {ANNOTATORS["label_a"]: LABEL_COLUMNS["label_a"],
                       ANNOTATORS["label_b"]: LABEL_COLUMNS["label_b"],
                       "confirmation": HUMAN_CONFIRMATION},
        "source_xlsx_sha256": sha256_file(xlsx),
        "content_sha256": canonical_sha256(content),
        "records": records,
    }
    debug_compared = {"audio_sha256": len(debug_audio), "transcript_sha256": len(debug_text), "matches": 0}
    report = public_report(manifest, audio_info, book_meta, limits, same_as_script, debug_compared, warnings)
    return manifest, report, warnings


def public_report(manifest: dict, audio_info: dict, book_meta: dict, limits: dict,
                  same_as_script: int, debug_compared: dict, warnings: list[str]) -> dict:
    records = manifest["records"]
    durations = [record["duration_seconds"] for record in records]
    sizes = [record["audio_bytes"] for record in records]
    lengths = [len(words(record["reference_transcript"])) for record in records]
    by_class = lambda key: {label: sum(r[key] == label for r in records) for label in LABELS}  # noqa: E731
    return {
        "schema": "t11-voice-composition/1",
        "task": "T11",
        "frozen_at": None,
        "rebuilt_at": None,
        "human_confirmation": HUMAN_CONFIRMATION,
        "source": {
            "xlsx_sha256": manifest["source_xlsx_sha256"],
            "xlsx_modified": book_meta["modified"],
            "sheets": {RECORD_SHEET: list(RECORD_COLUMNS.values()), LABEL_SHEET: list(LABEL_COLUMNS.values())},
            "audio_layout": "raw_audio/voice_NNN.ogg",
        },
        "content_sha256": manifest["content_sha256"],
        "labeling_rules": "docs/team_plan.md, разделы 3 и 8: negative=0, neutral=1, positive=2; "
                          "две независимые оценки эталонного текста, затем согласование без ответов моделей",
        "aggregates": {
            "records": len(records),
            "speakers": len({r["speaker_id"] for r in records}),
            "records_per_speaker": dict(sorted(Counter(r["speaker_id"] for r in records).items())),
            "annotators": sorted(ANNOTATORS.values()),
            "final_labels": by_class("final_label"),
            "labels_" + ANNOTATORS["label_a"]: by_class("label_a"),
            "labels_" + ANNOTATORS["label_b"]: by_class("label_b"),
            "a_equals_b": sum(r["label_a"] == r["label_b"] for r in records),
            "disputes": sum(r["dispute"] for r in records),
            "ambiguous": sum(r["ambiguous"] for r in records),
            "noise": sum(r["noise"] for r in records),
            "noise_by_class": {label: sum(r["noise"] and r["final_label"] == label for r in records)
                               for label in LABELS},
            "reference_words": {"min": min(lengths), "max": max(lengths)},
            "reference_equals_script": same_as_script,
            "duration_seconds": {"min": min(durations), "median": round(statistics.median(durations), 3),
                                 "max": max(durations), "total": round(sum(durations), 3)},
            "audio_bytes": {"min": min(sizes), "max": max(sizes), "total": sum(sizes)},
            "formats": dict(sorted(Counter(
                f"{i['container']}/{i['codec']}, {i['channels']} ch, {i['sample_rate']} Hz"
                for i in audio_info.values()).items())),
            "encoders": dict(sorted(Counter(i["encoder"] for i in audio_info.values()).items())),
            "files_with_clipping": sum(i["clipped_samples"] > 0 for i in audio_info.values()),
            "voice_limits": limits,
            "debug_hashes_compared": debug_compared,
        },
        # Per-record transcript hashes stay private: short known scripts could be matched to them.
        "records": [{key: r[key] for key in ("record_id", "audio_sha256", "audio_bytes", "duration_seconds")}
                    for r in records],
        "warnings": warnings,
        "unchanged_conditions": UNCHANGED_CONDITIONS,
    }


# --- CLI -----------------------------------------------------------------------------

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


DATE_FIELDS = ("frozen_at", "rebuilt_at")  # not part of the frozen content


def compare(manifest: dict, report: dict, manifest_path: Path, report_path: Path) -> list[str]:
    problems = []
    if not report_path.is_file():
        return [f"Нет зафиксированного отчёта {report_path.name}; сначала --write"]
    frozen = json.loads(report_path.read_text(encoding="utf-8"))
    report = {**report, **{key: frozen.get(key) for key in DATE_FIELDS}}
    if frozen != report:
        changed = sorted(key for key in set(frozen) | set(report) if frozen.get(key) != report.get(key))
        problems.append(f"Состав отличается от зафиксированного: {', '.join(changed)}")
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("content_sha256") != manifest["content_sha256"]:
            problems.append("content_sha256 приватного манифеста отличается")
    else:
        problems.append("Приватный манифест не найден")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write", action="store_true", help="записать манифест и зафиксировать отчёт")
    parser.add_argument("--frozen-at", type=date.fromisoformat,
                        help="дата первоначальной фиксации; по умолчанию сохраняется из отчёта или сегодня")
    args = parser.parse_args(argv)
    try:
        manifest, report, warnings = build(args.xlsx, args.audio_dir)
    except ValidationError as error:
        print(f"FAIL\n{error}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARN {warning}", file=sys.stderr)
    if args.write:
        previous = json.loads(args.report.read_text(encoding="utf-8")) if args.report.is_file() else {}
        today = date.today().isoformat()
        report["frozen_at"] = args.frozen_at.isoformat() if args.frozen_at else previous.get("frozen_at") or today
        report["rebuilt_at"] = today
        write_json(args.manifest, manifest)
        write_json(args.report, report)
        print(json.dumps({"status": "WRITTEN", "records": len(manifest["records"]),
                          "content_sha256": manifest["content_sha256"]}))
        return 0
    problems = compare(manifest, report, args.manifest, args.report)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    print(json.dumps({"status": "CHANGED" if problems else "UNCHANGED",
                      "records": len(manifest["records"]), "content_sha256": manifest["content_sha256"]}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
