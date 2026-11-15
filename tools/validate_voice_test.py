"""T11: validate the private final voice test and freeze its composition.

Reads the coordinator's Excel table and the OGG/Opus files, never runs ASR or a
classifier. ``--write`` creates the freeze once: the private manifest (with
transcripts and labels, outside Git) and the public report (IDs, checksums,
aggregates only). It never replaces an existing freeze. Without ``--write`` the
inputs are rebuilt and compared with the stored manifest content and the report:
the set is unchanged only if everything except audio paths and dates matches.
Diagnostics name records and fields, never cell values.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
import os
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
ID_PATTERN = re.compile(r"voice_\d{3}")
WORDS_PATTERN = re.compile(r"\d{1,3}")

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

# Private manifest layout; content_sha256 covers every record field except audio_path.
MANIFEST_KEYS = ("schema", "label_ids", "annotators", "source_xlsx_sha256", "content_sha256", "records")
RECORD_KEYS = ("record_id", "speaker_id", "audio_path", "audio_sha256", "audio_bytes", "duration_seconds",
               "reference_transcript", "transcript_sha256", "annotator_a", "label_a", "annotator_b", "label_b",
               "final_label", "dispute", "ambiguous", "ambiguity_reason", "noise")
LOCATION_KEYS = ("audio_path",)  # may differ between checkouts
DATE_FIELDS = ("frozen_at", "rebuilt_at")  # report metadata, not frozen content

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


def content_of(records: list[dict]) -> list[dict]:
    return [{key: value for key, value in record.items() if key not in LOCATION_KEYS} for record in records]


def load_json(path: Path, name: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{name} не читается как JSON ({error.__class__.__name__})") from None


# --- xlsx (standard library only) -------------------------------------------------

def _column(ref: str | None) -> int:
    match = re.fullmatch(r"([A-Z]{1,3})[1-9]\d*", ref or "")
    if not match:
        raise ValueError("bad cell reference")
    number = 0
    for char in match.group(1):
        number = number * 26 + ord(char) - 64
    return number - 1


def read_xlsx(path: Path) -> tuple[dict[str, list[list[str]]], dict]:
    """Return sheet name -> rows of cell strings, and safe workbook metadata."""
    try:
        return _read_xlsx(path)
    except ValidationError:
        raise
    # Missing parts, broken XML, bad references or indexes: report the kind only.
    except (OSError, zipfile.BadZipFile, KeyError, IndexError, ValueError, TypeError, AttributeError,
            ET.ParseError) as error:
        raise ValidationError(f"Таблица повреждена или не является .xlsx ({error.__class__.__name__})") from None


def _read_xlsx(path: Path) -> tuple[dict[str, list[list[str]]], dict]:
    with zipfile.ZipFile(path) as book:
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
    for number, row in enumerate(rows[index + 1:], start=1):
        values = {key: clean(row[pos]) if pos < len(row) else "" for key, pos in positions.items()}
        if not values["record_id"]:
            if any(values.values()):
                raise ValidationError(f"На листе «{sheet}» строка данных {number} без ID")
            continue
        # Safe label for diagnostics: a well-formed ID or the row position, never cell text.
        values["where"] = (f"{values['record_id']}:" if ID_PATTERN.fullmatch(values["record_id"])
                           else f"лист «{sheet}», строка данных {number}:")
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
    # Without T08 data the overlap check would silently compare against nothing.
    if not path.is_file():
        raise ValidationError("Нет сведений об отладочных материалах T08 (reports/speech/verification.json)")
    data = load_json(path, "Сведения T08")
    audio = _collect_hashes(data, "audio_sha256")
    if not audio:
        raise ValidationError("В сведениях T08 нет SHA-256 отладочного аудио")
    return audio, _collect_hashes(data, "text_sha256")


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
    for row in plan + merge:
        if not ID_PATTERN.fullmatch(row["record_id"]):
            errors.append(f"{row['where']} ID не в формате voice_NNN")
    if sorted(ids) != expected_ids:
        errors.append(f"ID записей: ожидалось voice_001–voice_{EXPECTED_RECORDS:03d} без повторов, "
                      f"получено {len(ids)} строк, {len(set(ids))} уникальных")
    if [row["record_id"] for row in merge] != ids:
        errors.append("ID и порядок строк на листах записей и сведения меток различаются")
    merge_by_id = {row["record_id"]: row for row in merge}

    expected_files = {f"{record_id}.ogg" for record_id in expected_ids}
    present = {path.name for path in audio_dir.iterdir()}
    if missing := sorted(expected_files - present):
        errors.append(f"Нет аудиофайлов: {', '.join(missing)}")
    if extra := sorted(present - expected_files):
        errors.append(f"Лишние элементы в каталоге аудио: {len(extra)}")

    debug_audio, debug_text = debug_hashes(debug_report)
    pseudonyms: dict[str, str] = {}
    records, audio_info, seen_audio, seen_text = [], {}, {}, {}
    for row in plan:
        record_id, where = row["record_id"], row["where"]
        labels = merge_by_id.get(record_id)
        if labels is None or not ID_PATTERN.fullmatch(record_id):
            continue
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
        # The planning column counts whitespace-separated tokens, a dash included.
        if row["words"] and not WORDS_PATTERN.fullmatch(row["words"]):
            errors.append(f"{where} «Слов» должно быть целым неотрицательным числом")
        elif row["words"] and int(row["words"]) != len(row["script"].split()):
            warnings.append(f"{where} число в столбце «Слов» не совпадает с числом слов сценария")
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
        if not path.exists() and not path.is_symlink():
            continue  # already reported as missing
        if path.is_symlink() or not path.is_file():
            errors.append(f"{where} ожидаемый путь аудио не является обычным файлом")
            continue
        size = path.stat().st_size
        if size == 0:
            errors.append(f"{where} пустой аудиофайл")
            continue
        if size > limits["max_bytes"]:  # checked before decoding
            errors.append(f"{where} превышен лимит размера голоса ({limits['max_bytes']} B)")
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
        if info["duration_seconds"] > limits["max_seconds"]:
            errors.append(f"{where} превышен лимит длительности голоса ({limits['max_seconds']:g} s)")
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

    final = Counter(row["final"] for row in merge)
    if any(final.get(label, 0) != EXPECTED_PER_CLASS for label in LABELS) or len(merge) != EXPECTED_RECORDS:
        counts = ", ".join(f"{label}={final.get(label, 0)}" for label in LABELS)
        other = sum(count for label, count in final.items() if label not in LABELS)
        errors.append(f"Согласованные метки: ожидалось по {EXPECTED_PER_CLASS} на класс, "
                      f"получено {counts}, прочих значений {other}")
    if len(pseudonyms) < MIN_SPEAKERS:
        errors.append(f"Дикторов {len(pseudonyms)}, нужно не меньше {MIN_SPEAKERS}")
    # The freeze itself must hold exactly the expected, fully checked records.
    checked = Counter(record["final_label"] for record in records)
    if len(records) != EXPECTED_RECORDS or any(checked[label] != EXPECTED_PER_CLASS for label in LABELS):
        errors.append(f"Проверено записей {len(records)} из {EXPECTED_RECORDS}; "
                      "фиксация требует полного набора 12/12/12")
    if errors:
        raise ValidationError("\n".join(errors))

    same_as_script = sum(words(row["script"]) == words(row["transcript"]) for row in plan)
    if same_as_script:
        warnings.append(f"{same_as_script} из {len(plan)} эталонов совпадают со сценарием с точностью "
                        "до регистра и пунктуации; дословность подтверждает только человек")
    content = content_of(records)
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

def _record_name(rebuilt: list[dict], index: int) -> str:
    return rebuilt[index]["record_id"] if index < len(rebuilt) else f"запись {index + 1}"


def check_manifest(stored, rebuilt: dict) -> list[str]:
    """Compare a stored private manifest with the rebuilt one; name fields, never values."""
    if not isinstance(stored, dict):
        return ["Манифест: ожидался JSON-объект"]
    problems = []
    if missing := [key for key in MANIFEST_KEYS if key not in stored]:
        problems.append(f"Манифест: нет полей {', '.join(missing)}")
    if extra := [key for key in stored if key not in MANIFEST_KEYS]:
        problems.append(f"Манифест: лишних полей {len(extra)}")
    records = stored.get("records")
    if not isinstance(records, list):
        return [*problems, "Манифест: records не является списком"]
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(RECORD_KEYS):
            problems.append(f"Манифест, {_record_name(rebuilt['records'], index)}: неверный набор полей записи")
    if problems:
        return problems
    content = content_of(records)
    if canonical_sha256(content) != stored["content_sha256"]:
        problems.append("Манифест: content_sha256 не соответствует его записям")
    expected = content_of(rebuilt["records"])
    if len(content) != len(expected):
        problems.append(f"Манифест: записей {len(content)}, в проверенном наборе {len(expected)}")
    for index, (saved, actual) in enumerate(zip(content, expected)):
        if fields := [key for key in RECORD_KEYS if key not in LOCATION_KEYS and saved[key] != actual[key]]:
            problems.append(f"Манифест, {_record_name(rebuilt['records'], index)}: отличаются {', '.join(fields)}")
    for key in MANIFEST_KEYS:
        if key != "records" and stored[key] != rebuilt[key]:
            problems.append(f"Манифест: отличается {key}")
    return problems


def _is_date(value) -> bool:
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def check_report(frozen, rebuilt: dict) -> list[str]:
    """Compare the frozen public report with the rebuilt one, ignoring only its dates."""
    if not isinstance(frozen, dict):
        return ["Отчёт: ожидался JSON-объект"]
    problems = []
    if not _is_date(frozen.get("frozen_at")):
        problems.append("Отчёт: frozen_at не является датой YYYY-MM-DD")
    if frozen.get("rebuilt_at") is not None and not _is_date(frozen["rebuilt_at"]):
        problems.append("Отчёт: rebuilt_at не является датой YYYY-MM-DD")
    if extra := [key for key in frozen if key not in rebuilt]:
        problems.append(f"Отчёт: лишних полей {len(extra)}")
    if changed := [key for key in rebuilt if key not in DATE_FIELDS and frozen.get(key) != rebuilt[key]]:
        problems.append(f"Состав отличается от зафиксированного: {', '.join(changed)}")
    return problems


def verify(manifest: dict, report: dict, manifest_path: Path, report_path: Path) -> list[str]:
    """All differences between the rebuilt set and the stored freeze."""
    problems = []
    stored_manifest = frozen_report = None
    for path, name in ((manifest_path, "Приватный манифест"), (report_path, "Зафиксированный отчёт")):
        if not path.is_file():
            problems.append(f"{name} не найден")
            continue
        try:
            value = load_json(path, name)
        except ValidationError as error:
            problems.append(str(error))
            continue
        if path == manifest_path:
            stored_manifest = value
        else:
            frozen_report = value
    if stored_manifest is not None:
        problems += check_manifest(stored_manifest, manifest)
    if frozen_report is not None:
        problems += check_report(frozen_report, report)
    if isinstance(stored_manifest, dict) and isinstance(frozen_report, dict) \
            and stored_manifest.get("content_sha256") != frozen_report.get("content_sha256"):
        problems.append("content_sha256 манифеста и отчёта не совпадают")
    return problems


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_new_freeze(files: list[tuple[Path, dict]]) -> None:
    """Create all files or none; never replaces an existing file."""
    temporary, created = [], []
    try:
        for path, value in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            with tmp.open("x", encoding="utf-8") as stream:
                stream.write(_dump(value))
            temporary.append((tmp, path))
        for tmp, path in temporary:
            os.link(tmp, path)  # fails if the target already exists
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for tmp, _ in temporary:
            tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write", action="store_true",
                        help="создать фиксацию, если её ещё нет; существующая не заменяется")
    parser.add_argument("--frozen-at", type=date.fromisoformat,
                        help="дата первоначальной фиксации для нового набора; по умолчанию сегодня")
    args = parser.parse_args(argv)
    try:
        manifest, report, warnings = build(args.xlsx, args.audio_dir)
    except ValidationError as error:
        print(f"FAIL\n{error}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARN {warning}", file=sys.stderr)
    summary = {"records": len(manifest["records"]), "content_sha256": manifest["content_sha256"]}

    if args.write and not args.manifest.exists() and not args.report.exists():
        report["frozen_at"] = (args.frozen_at or date.today()).isoformat()
        try:
            write_new_freeze([(args.manifest, manifest), (args.report, report)])
        except OSError as error:
            print(f"FAIL Фиксация не записана ({error.__class__.__name__}); файлы не созданы", file=sys.stderr)
            return 1
        print(json.dumps({"status": "WRITTEN", **summary}))
        return 0

    problems = verify(manifest, report, args.manifest, args.report)
    if args.write and args.frozen_at and not problems:
        frozen = load_json(args.report, "Зафиксированный отчёт")
        if frozen.get("frozen_at") != args.frozen_at.isoformat():
            problems.append("Дата первоначальной фиксации не меняется")
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if args.write and problems:
        print("FAIL Фиксация уже существует; --write её не заменяет, файлы не изменены", file=sys.stderr)
    status = "CHANGED" if problems else ("ALREADY_FROZEN" if args.write else "UNCHANGED")
    print(json.dumps({"status": status, **summary}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
