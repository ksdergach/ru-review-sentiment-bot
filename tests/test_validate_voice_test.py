import json
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape
import zipfile

import av
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_voice_test as vt  # noqa: E402

SPEAKERS = ["Анна", "Борис", "Вера"]  # fake names; must never reach the outputs
LABELS = ["negative", "neutral", "positive"]
NOISY = {"voice_002", "voice_010", "voice_020", "voice_027", "voice_033"}


def write_opus(path, frequency, seconds=0.3):
    with av.open(str(path), "w", format="ogg") as output:
        stream = output.add_stream("libopus", rate=48000, layout="mono")
        t = np.arange(int(48000 * seconds)) / 48000
        frame = av.AudioFrame.from_ndarray((0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)[None, :],
                                           format="flt", layout="mono")
        frame.sample_rate = 48000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            output.mux(packet)


def sheet_xml(rows):
    def cell(col, row, value):
        return f'<c r="{chr(65 + col)}{row}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
    body = "".join(f'<row r="{r}">' + "".join(cell(c, r, v) for c, v in enumerate(values) if v != "") + "</row>"
                   for r, values in enumerate(rows, start=1))
    return f'<worksheet xmlns="{vt.MAIN_NS}"><sheetData>{body}</sheetData></worksheet>'


def write_xlsx(path, plan, merge):
    sheets = [(vt.RECORD_SHEET, plan, vt.RECORD_COLUMNS), (vt.LABEL_SHEET, merge, vt.LABEL_COLUMNS)]
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/workbook.xml", f'<workbook xmlns="{vt.MAIN_NS}" xmlns:r="{vt.REL_NS}"><sheets>' + "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _, _) in enumerate(sheets, 1))
            + "</sheets></workbook>")
        book.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{vt.PKG_REL_NS}">' + "".join(
            f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 3))
            + "</Relationships>")
        for i, (name, rows, columns) in enumerate(sheets, 1):
            table = [[name], ["Служебная строка"], [], list(columns.values())]
            table += [[row[key] for key in columns] for row in rows]
            book.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(table))


def make_rows():
    plan, merge = [], []
    for n in range(1, 37):
        record_id = f"voice_{n:03d}"
        speaker, label = SPEAKERS[(n - 1) % 3], LABELS[(n - 1) // 12]
        script = f"Сценарий номер {n} про фильм — уникальная фраза {n * 7}"
        plan.append({"record_id": record_id, "speaker": speaker, "script": script, "file": f"{record_id}.ogg",
                     "words": str(len(script.split())), "noise": "да" if record_id in NOISY else "нет",
                     "transcript": script, "transcript_checked": "да"})
        merge.append({"record_id": record_id, "speaker": speaker, "label_a": label, "label_b": label,
                      "final": label, "dispute": "нет", "ambiguity": "-", "confirmed": "да"})
    return plan, merge


@pytest.fixture(scope="module")
def audio_template(tmp_path_factory):
    folder = tmp_path_factory.mktemp("template")
    for n in range(1, 37):
        write_opus(folder / f"voice_{n:03d}.ogg", 200 + 25 * n)
    return folder


@pytest.fixture
def dataset(tmp_path, audio_template):
    audio = tmp_path / "raw_audio"
    shutil.copytree(audio_template, audio)
    plan, merge = make_rows()
    xlsx = tmp_path / "T11.xlsx"
    write_xlsx(xlsx, plan, merge)
    debug = tmp_path / "debug.json"
    debug.write_text(json.dumps({"results": {"x": {"audio_sha256": "0" * 64, "text_sha256": "1" * 64}}}))
    return {"root": tmp_path, "xlsx": xlsx, "audio": audio, "plan": plan, "merge": merge, "debug": debug}


def build(data, **kwargs):
    return vt.build(data["xlsx"], data["audio"], debug_report=data["debug"], **kwargs)


def rewrite(data):
    write_xlsx(data["xlsx"], data["plan"], data["merge"])


def test_valid_set_gives_pseudonymized_manifest_and_text_free_report(dataset):
    manifest, report, _ = build(dataset)

    assert len(manifest["records"]) == 36
    assert {r["speaker_id"] for r in manifest["records"]} == {"S01", "S02", "S03"}
    first = manifest["records"][0]
    assert first["audio_sha256"] == vt.sha256_file(dataset["audio"] / "voice_001.ogg")
    assert first["reference_transcript"] == dataset["plan"][0]["transcript"]
    assert first["transcript_sha256"] == vt.sha256_bytes(first["reference_transcript"].encode())
    assert {(r["annotator_a"], r["annotator_b"]) for r in manifest["records"]} == {("R01", "R02")}
    assert set(report["records"][0]) == {"record_id", "audio_sha256", "audio_bytes", "duration_seconds"}
    agg = report["aggregates"]
    assert agg["annotators"] == ["R01", "R02"]
    assert agg["labels_R01"] == agg["labels_R02"] == {"negative": 12, "neutral": 12, "positive": 12}
    assert agg["final_labels"] == {"negative": 12, "neutral": 12, "positive": 12}
    assert (agg["speakers"], agg["noise"], agg["a_equals_b"], agg["ambiguous"]) == (3, 5, 36, 0)
    assert agg["formats"] == {"ogg/opus, 1 ch, 48000 Hz": 36}
    assert agg["reference_equals_script"] == 36
    published = json.dumps(report, ensure_ascii=False)
    private = json.dumps(manifest, ensure_ascii=False)
    for name in SPEAKERS:
        assert name not in published and name not in private
    assert "Сценарий" not in published
    assert first["transcript_sha256"] not in published
    assert report["content_sha256"] == manifest["content_sha256"]


def test_content_hash_ignores_location_but_tracks_labels(dataset, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(dataset["audio"], moved)
    original = build(dataset)[0]["content_sha256"]
    assert vt.build(dataset["xlsx"], moved, debug_report=dataset["debug"])[0]["content_sha256"] == original

    dataset["merge"][0].update(label_b="neutral", dispute="да")
    rewrite(dataset)
    assert build(dataset)[0]["content_sha256"] != original


def break_distribution(d):
    for row in d["merge"][:1]:
        row.update(label_a="neutral", label_b="neutral", final="neutral")


def break_audio_format(d):
    with __import__("wave").open(str(d["audio"] / "voice_005.ogg"), "wb") as wav:
        wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 1600)


@pytest.mark.parametrize(("mutate", "message"), [
    (break_distribution, "по 12 на класс"),
    (lambda d: d["merge"][3].update(label_b="positive"), "спор не отмечен"),
    (lambda d: d["merge"][3].update(final="positive"), "отличается от совпавших"),
    (lambda d: d["merge"][4].update(final="mixed"), "должна быть одной из"),
    (lambda d: d["merge"][6].update(confirmed="нет"), "не подтверждено человеком"),
    (lambda d: d["plan"][2].update(transcript=""), "нет эталонной транскрипции"),
    (lambda d: d["plan"][2].update(transcript_checked="нет"), "не отмечена как сверенная"),
    (lambda d: d["plan"][8].update(transcript=d["plan"][9]["transcript"]), "voice_010: транскрипция совпадает с voice_009"),
    (lambda d: d["plan"][1].update(noise="чуть-чуть"), "шум должен быть"),
    (lambda d: d["plan"][1].update(file="voice_099.ogg"), "имя файла"),
    (lambda d: d["merge"][1].update(speaker="Кто-то"), "диктор различается"),
    (lambda d: [r.update(speaker="Анна") for r in d["plan"] + d["merge"] if r["speaker"] == "Вера"], "Дикторов 2"),
    (lambda d: d["plan"].pop(), "ID записей"),
    (lambda d: (d["audio"] / "voice_007.ogg").unlink(), "Нет аудиофайлов: voice_007.ogg"),
    (lambda d: shutil.copy(d["audio"] / "voice_001.ogg", d["audio"] / "voice_037.ogg"), "Лишние файлы"),
    (lambda d: shutil.copy(d["audio"] / "voice_001.ogg", d["audio"] / "voice_002.ogg"), "побайтово совпадает"),
    (lambda d: (d["audio"] / "voice_003.ogg").write_bytes(b""), "пустой аудиофайл"),
    (lambda d: (d["audio"] / "voice_004.ogg").write_bytes(b"OggS not really"), "не декодируется"),
    (break_audio_format, "ожидался ogg/opus"),
])
def test_invalid_sets_are_rejected_not_fixed(dataset, mutate, message):
    mutate(dataset)
    rewrite(dataset)
    with pytest.raises(vt.ValidationError, match=message):
        build(dataset)


def test_overlap_with_debug_material_is_rejected(dataset):
    dataset["debug"].write_text(json.dumps({"results": [
        {"audio_sha256": vt.sha256_file(dataset["audio"] / "voice_011.ogg")},
        {"text_sha256": vt.sha256_bytes(dataset["plan"][12]["transcript"].encode())},
    ]}))
    with pytest.raises(vt.ValidationError) as error:
        build(dataset)
    assert "voice_011: аудио совпадает с отладочной" in str(error.value)
    assert "voice_013: транскрипция совпадает с отладочной" in str(error.value)


def test_voice_limits_come_from_config(dataset, tmp_path):
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"debug_audio": {"max_seconds": 0.1, "max_bytes": 5242880}}))
    with pytest.raises(vt.ValidationError, match="превышен лимит"):
        build(dataset, limits_config=limits)


def test_cli_freezes_then_detects_any_change(dataset, monkeypatch):
    monkeypatch.setattr(vt, "DEBUG_REPORT", dataset["debug"])
    root = dataset["root"]
    args = ["--xlsx", str(dataset["xlsx"]), "--audio-dir", str(dataset["audio"]),
            "--manifest", str(root / "private/manifest.json"), "--report", str(root / "public/composition.json")]

    assert vt.main(args) == 1  # nothing frozen yet
    assert vt.main([*args, "--write", "--frozen-at", "2026-11-15"]) == 0
    report = json.loads((root / "public/composition.json").read_text(encoding="utf-8"))
    assert report["frozen_at"] == "2026-11-15"
    assert vt.main(args) == 0

    # A later rebuild keeps the original freeze date and records its own date separately.
    assert vt.main([*args, "--write"]) == 0
    report = json.loads((root / "public/composition.json").read_text(encoding="utf-8"))
    assert report["frozen_at"] == "2026-11-15"
    assert report["rebuilt_at"] == vt.date.today().isoformat()
    assert vt.main(args) == 0

    write_opus(dataset["audio"] / "voice_001.ogg", 999)
    assert vt.main(args) == 1
