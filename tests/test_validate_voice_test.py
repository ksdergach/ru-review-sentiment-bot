import hashlib
import json
from pathlib import Path
import shutil
import sys
import wave
from xml.sax.saxutils import escape
import zipfile

import av
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_voice_test as vt  # noqa: E402

# Expected values are written out here, not taken from the validator.
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
RECORD_SHEET, LABEL_SHEET = "02_Записи", "03_Сведение_меток"
RECORD_HEADERS = {
    "record_id": "ID записи", "speaker": "Участник", "script": "Текст для записи",
    "file": "Имя файла (предложение)", "words": "Слов", "noise": "Шум, да/нет",
    "transcript": "Дословная транскрипция ПОСЛЕ аудио", "transcript_checked": "Сверена человеком, да/нет",
}
LABEL_HEADERS = {
    "record_id": "ID", "speaker": "Участник", "label_a": "Метка А", "label_b": "Метка Б",
    "final": "Согласованная метка", "dispute": "Есть спор, да/нет", "ambiguity": "Причина неоднозначности",
    "confirmed": "Подтверждено человеком, да/нет",
}
SPEAKERS = ["Анна", "Борис", "Вера"]  # fake names; must never reach the outputs
LABELS = ["negative", "neutral", "positive"]
NOISY = {"voice_002", "voice_010", "voice_020", "voice_027", "voice_033"}
MARKER = "ИСКУССТВЕННЫЙ-ПРИВАТНЫЙ-МАРКЕР-7319"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True, scope="module")
def real_freeze_untouched():
    """Tests use temporary data only; the real T11 freeze must stay byte-identical."""
    paths = [ROOT / "artifacts/t11/manifest.json", ROOT / "reports/voice_test/composition.json"]
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    yield
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == before


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
    return f'<worksheet xmlns="{MAIN_NS}"><sheetData>{body}</sheetData></worksheet>'


def xlsx_parts(plan, merge):
    sheets = [(RECORD_SHEET, plan, RECORD_HEADERS), (LABEL_SHEET, merge, LABEL_HEADERS)]
    parts = {
        "xl/workbook.xml": f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><sheets>' + "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _, _) in enumerate(sheets, 1))
        + "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{PKG_REL_NS}">' + "".join(
            f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 3))
        + "</Relationships>",
    }
    for i, (name, rows, headers) in enumerate(sheets, 1):
        table = [[name], ["Служебная строка"], [], list(headers.values())]
        table += [[row[key] for key in headers] for row in rows]
        parts[f"xl/worksheets/sheet{i}.xml"] = sheet_xml(table)
    return parts


def write_xlsx(path, plan, merge, **override):
    parts = {**xlsx_parts(plan, merge), **override}
    with zipfile.ZipFile(path, "w") as book:
        for name, data in parts.items():
            if data is not None:
                book.writestr(name, data)


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
def dataset(tmp_path, audio_template, monkeypatch):
    audio = tmp_path / "raw_audio"
    shutil.copytree(audio_template, audio)
    plan, merge = make_rows()
    xlsx = tmp_path / "T11.xlsx"
    write_xlsx(xlsx, plan, merge)
    debug = tmp_path / "debug.json"
    debug.write_text(json.dumps({"results": {"x": {"audio_sha256": "0" * 64, "text_sha256": "1" * 64}}}))
    monkeypatch.setattr(vt, "DEBUG_REPORT", debug)  # CLI runs never read the real T08 report
    manifest, report = tmp_path / "private/manifest.json", tmp_path / "public/composition.json"
    args = ["--xlsx", str(xlsx), "--audio-dir", str(audio), "--manifest", str(manifest), "--report", str(report)]
    return {"root": tmp_path, "xlsx": xlsx, "audio": audio, "plan": plan, "merge": merge, "debug": debug,
            "manifest": manifest, "report": report, "args": args}


def build(data, **kwargs):
    return vt.build(data["xlsx"], data["audio"], debug_report=data["debug"], **kwargs)


def rewrite(data):
    write_xlsx(data["xlsx"], data["plan"], data["merge"])


def run(data, capsys, *extra):
    code = vt.main([*data["args"], *extra])
    out, err = capsys.readouterr()
    return code, out, err


def freeze(data, capsys, frozen_at="2020-01-02"):
    code, out, _ = run(data, capsys, "--write", "--frozen-at", frozen_at)
    assert code == 0 and '"WRITTEN"' in out
    return data["manifest"].read_bytes(), data["report"].read_bytes()


# --- building a valid set ---------------------------------------------------------------

def test_valid_set_gives_pseudonymized_manifest_and_text_free_report(dataset):
    manifest, report, _ = build(dataset)

    assert len(manifest["records"]) == 36
    assert {r["speaker_id"] for r in manifest["records"]} == {"S01", "S02", "S03"}
    first = manifest["records"][0]
    assert first["audio_sha256"] == sha256((dataset["audio"] / "voice_001.ogg").read_bytes())
    assert first["reference_transcript"] == dataset["plan"][0]["transcript"]
    assert first["transcript_sha256"] == sha256(dataset["plan"][0]["transcript"].encode())
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
    content = [{k: v for k, v in r.items() if k != "audio_path"} for r in manifest["records"]]
    expected_hash = sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    assert manifest["content_sha256"] == report["content_sha256"] == expected_hash


def test_content_hash_ignores_location_but_tracks_labels(dataset, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(dataset["audio"], moved)
    original = build(dataset)[0]["content_sha256"]
    assert vt.build(dataset["xlsx"], moved, debug_report=dataset["debug"])[0]["content_sha256"] == original

    dataset["merge"][0].update(label_b="neutral", dispute="да")
    rewrite(dataset)
    assert build(dataset)[0]["content_sha256"] != original


# --- invalid inputs are rejected, never fixed --------------------------------------------

def break_distribution(d):
    d["merge"][0].update(label_a="neutral", label_b="neutral", final="neutral")


def break_audio_format(d):
    with wave.open(str(d["audio"] / "voice_005.ogg"), "wb") as wav:
        wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * 1600)


def directory_instead_of_audio(d):
    (d["audio"] / "voice_007.ogg").unlink()
    (d["audio"] / "voice_007.ogg").mkdir()


def symlink_instead_of_audio(d):
    target = d["root"] / "outside.ogg"
    (d["audio"] / "voice_008.ogg").rename(target)
    (d["audio"] / "voice_008.ogg").symlink_to(target)


@pytest.mark.parametrize(("mutate", "message"), [
    (break_distribution, "по 12 на класс"),
    (lambda d: d["merge"][3].update(label_b="positive"), "спор не отмечен"),
    (lambda d: d["merge"][3].update(final="positive"), "отличается от совпавших"),
    (lambda d: d["merge"][4].update(final="mixed"), "должна быть одной из"),
    (lambda d: d["merge"][4].update(dispute="возможно"), "«Есть спор» должно быть"),
    (lambda d: d["merge"][6].update(confirmed="нет"), "не подтверждено человеком"),
    (lambda d: d["plan"][2].update(transcript=""), "нет эталонной транскрипции"),
    (lambda d: d["plan"][2].update(transcript_checked="нет"), "не отмечена как сверенная"),
    (lambda d: d["plan"][8].update(transcript=d["plan"][9]["transcript"]),
     "voice_010: транскрипция совпадает с voice_009"),
    (lambda d: d["plan"][1].update(noise="чуть-чуть"), "шум должен быть"),
    (lambda d: d["plan"][1].update(file="voice_099.ogg"), "имя файла"),
    (lambda d: d["merge"][1].update(speaker="Кто-то"), "диктор различается"),
    (lambda d: [r.update(speaker="Анна") for r in d["plan"] + d["merge"] if r["speaker"] == "Вера"], "Дикторов 2"),
    (lambda d: d["plan"].pop(), "ID записей"),
    (lambda d: (d["audio"] / "voice_007.ogg").unlink(), "Нет аудиофайлов: voice_007.ogg"),
    (lambda d: shutil.copy(d["audio"] / "voice_001.ogg", d["audio"] / "voice_037.ogg"), "Лишние элементы"),
    (lambda d: shutil.copy(d["audio"] / "voice_001.ogg", d["audio"] / "voice_002.ogg"), "побайтово совпадает"),
    (lambda d: (d["audio"] / "voice_003.ogg").write_bytes(b""), "пустой аудиофайл"),
    (lambda d: (d["audio"] / "voice_004.ogg").write_bytes(b"OggS not really"), "не декодируется"),
    (break_audio_format, "ожидался ogg/opus"),
    (directory_instead_of_audio, "voice_007: ожидаемый путь аудио не является обычным файлом"),
    (symlink_instead_of_audio, "voice_008: ожидаемый путь аудио не является обычным файлом"),
])
def test_invalid_sets_are_rejected_not_fixed(dataset, mutate, message):
    mutate(dataset)
    rewrite(dataset)
    with pytest.raises(vt.ValidationError, match=message):
        build(dataset)


def test_directory_in_place_of_audio_cannot_be_frozen_as_35_records(dataset, capsys):
    directory_instead_of_audio(dataset)
    code, out, err = run(dataset, capsys, "--write")
    assert code == 1
    assert "Проверено записей 35 из 36" in err
    assert not dataset["manifest"].exists() and not dataset["report"].exists()


@pytest.mark.parametrize("value", ["12.5", "-3", "двадцать", "0x10", "1e2"])
def test_word_count_must_be_an_integer(dataset, value):
    dataset["plan"][5]["words"] = value
    rewrite(dataset)
    with pytest.raises(vt.ValidationError, match="voice_006: «Слов» должно быть целым"):
        build(dataset)


@pytest.mark.parametrize(("sheet", "row", "field"), [
    ("plan", 4, "words"), ("merge", 4, "final"), ("merge", 4, "label_a"), ("merge", 4, "dispute"),
    ("plan", 4, "noise"), ("plan", 4, "file"), ("plan", 4, "record_id"), ("merge", 4, "record_id"),
])
def test_private_cell_values_never_reach_output(dataset, capsys, sheet, row, field):
    dataset[sheet][row][field] = f"Как будто транскрипция {MARKER} попала не в тот столбец"
    rewrite(dataset)
    code, out, err = run(dataset, capsys, "--write")
    assert code == 1 and "FAIL" in err
    assert MARKER not in out + err
    assert not dataset["report"].exists() and not dataset["manifest"].exists()


def test_overlap_with_debug_material_is_rejected(dataset):
    dataset["debug"].write_text(json.dumps({"results": [
        {"audio_sha256": sha256((dataset["audio"] / "voice_011.ogg").read_bytes())},
        {"text_sha256": sha256(dataset["plan"][12]["transcript"].encode())},
    ]}))
    with pytest.raises(vt.ValidationError) as error:
        build(dataset)
    assert "voice_011: аудио совпадает с отладочной" in str(error.value)
    assert "voice_013: транскрипция совпадает с отладочной" in str(error.value)


@pytest.mark.parametrize(("content", "message"), [
    (None, "Нет сведений об отладочных материалах T08"),
    ("{not json", "Сведения T08 не читается как JSON"),
    (json.dumps({"results": [{"text_sha256": "1" * 64}]}), "нет SHA-256 отладочного аудио"),
])
def test_missing_debug_data_is_an_error_not_an_empty_check(dataset, content, message):
    if content is None:
        dataset["debug"].unlink()
    else:
        dataset["debug"].write_text(content)
    with pytest.raises(vt.ValidationError, match=message):
        build(dataset)


def test_duration_limit_comes_from_config(dataset, tmp_path):
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"debug_audio": {"max_seconds": 0.1, "max_bytes": 5242880}}))
    with pytest.raises(vt.ValidationError, match="превышен лимит длительности"):
        build(dataset, limits_config=limits)


def test_size_limit_is_checked_before_decoding(dataset, tmp_path, monkeypatch):
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"debug_audio": {"max_seconds": 30, "max_bytes": 100}}))
    decoded = []
    monkeypatch.setattr(vt, "probe_audio", lambda path: decoded.append(path))
    with pytest.raises(vt.ValidationError, match="превышен лимит размера"):
        build(dataset, limits_config=limits)
    assert decoded == []


# --- damaged workbooks --------------------------------------------------------------------

def bad_cell_reference(parts):
    return {"xl/worksheets/sheet1.xml": parts["xl/worksheets/sheet1.xml"].replace('r="A4"', 'r="4A"', 1)}


def bad_shared_string(parts):
    sheet = parts["xl/worksheets/sheet1.xml"].replace(
        '<c r="A1" t="inlineStr"><is><t>02_Записи</t></is></c>', '<c r="A1" t="s"><v>99</v></c>')
    return {"xl/worksheets/sheet1.xml": sheet,
            "xl/sharedStrings.xml": f'<sst xmlns="{MAIN_NS}"><si><t>x</t></si></sst>'}


@pytest.mark.parametrize("damage", [
    lambda parts: {"xl/_rels/workbook.xml.rels": None},
    lambda parts: {"xl/worksheets/sheet2.xml": None},
    lambda parts: {"xl/worksheets/sheet1.xml": "<worksheet><sheetData><row>"},
    lambda parts: {"xl/workbook.xml": parts["xl/workbook.xml"].replace('r:id="rId2"', 'r:id="rId9"')},
    bad_cell_reference,
    bad_shared_string,
])
def test_damaged_xlsx_is_a_validation_error(dataset, capsys, damage):
    parts = xlsx_parts(dataset["plan"], dataset["merge"])
    write_xlsx(dataset["xlsx"], dataset["plan"], dataset["merge"], **damage(parts))
    with pytest.raises(vt.ValidationError, match="Таблица повреждена"):
        build(dataset)
    code, out, err = run(dataset, capsys)
    assert code == 1 and "Traceback" not in err
    assert str(dataset["root"]) not in err


def test_file_that_is_not_a_zip_is_a_validation_error(dataset):
    dataset["xlsx"].write_bytes(b"not a workbook")
    with pytest.raises(vt.ValidationError, match="Таблица повреждена"):
        build(dataset)


# --- verifying a stored freeze ------------------------------------------------------------

def test_unchanged_freeze_passes(dataset, capsys):
    freeze(dataset, capsys)
    code, out, _ = run(dataset, capsys)
    assert code == 0 and '"UNCHANGED"' in out


def keep_hash(edit):
    def mutate(manifest):
        edit(manifest)
        return manifest
    return mutate


@pytest.mark.parametrize(("tamper", "message"), [
    (keep_hash(lambda m: m["records"][2].update(reference_transcript=MARKER)),
     "voice_003: отличаются reference_transcript"),
    (keep_hash(lambda m: m["records"][2].update(final_label="positive")), "voice_003: отличаются final_label"),
    (keep_hash(lambda m: m.pop("label_ids")), "нет полей label_ids"),
    (keep_hash(lambda m: m["records"][5].pop("noise")), "voice_006: неверный набор полей"),
    (keep_hash(lambda m: m["records"].pop()), "записей 35"),
    (keep_hash(lambda m: m["records"][0].update(audio_path=f"/elsewhere/{MARKER}.ogg")), None),
    (lambda m: {"content_sha256": m["content_sha256"]}, "нет полей schema"),
    (lambda m: [m], "ожидался JSON-объект"),
])
def test_tampered_manifest_is_detected_without_leaking_values(dataset, capsys, tamper, message):
    freeze(dataset, capsys)
    manifest = json.loads(dataset["manifest"].read_text(encoding="utf-8"))
    dataset["manifest"].write_text(json.dumps(tamper(manifest), ensure_ascii=False), encoding="utf-8")
    code, out, err = run(dataset, capsys)
    if message is None:  # only the location differs: allowed
        assert code == 0 and '"UNCHANGED"' in out
    else:
        assert code == 1 and '"CHANGED"' in out and message in err
    assert MARKER not in out + err


@pytest.mark.parametrize("target", ["manifest", "report"])
def test_corrupted_or_missing_freeze_files_are_changes(dataset, capsys, target):
    freeze(dataset, capsys)
    dataset[target].write_text('{"content_sha256": ', encoding="utf-8")
    code, out, err = run(dataset, capsys)
    assert code == 1 and '"CHANGED"' in out and "не читается как JSON" in err
    dataset[target].unlink()
    code, out, err = run(dataset, capsys)
    assert code == 1 and "не найден" in err


@pytest.mark.parametrize(("edit", "message"), [
    (lambda r: r.update(content_sha256="0" * 64), "манифеста и отчёта не совпадают"),
    (lambda r: r["aggregates"].update(noise=4), "Состав отличается от зафиксированного: aggregates"),
    (lambda r: r.update(frozen_at="15.11.2026"), "frozen_at не является датой"),
    (lambda r: r.update(extra="x"), "лишних полей 1"),
])
def test_tampered_report_is_detected(dataset, capsys, edit, message):
    freeze(dataset, capsys)
    report = json.loads(dataset["report"].read_text(encoding="utf-8"))
    edit(report)
    dataset["report"].write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    code, out, err = run(dataset, capsys)
    assert code == 1 and message in err


# --- writing a freeze ---------------------------------------------------------------------

def test_first_write_creates_freeze_with_given_date(dataset, capsys):
    freeze(dataset, capsys, frozen_at="2020-01-02")
    report = json.loads(dataset["report"].read_text(encoding="utf-8"))
    assert (report["frozen_at"], report["rebuilt_at"]) == ("2020-01-02", None)
    assert not list(dataset["root"].rglob(".*.tmp"))


def test_repeated_write_of_identical_set_changes_nothing(dataset, capsys):
    before = freeze(dataset, capsys, frozen_at="2020-01-02")
    code, out, _ = run(dataset, capsys, "--write")
    assert code == 0 and '"ALREADY_FROZEN"' in out
    assert (dataset["manifest"].read_bytes(), dataset["report"].read_bytes()) == before
    assert json.loads(before[1])["frozen_at"] == "2020-01-02"


def change_xlsx(d):
    d["merge"][0].update(label_b="neutral", dispute="да")
    rewrite(d)


def change_audio(d):
    write_opus(d["audio"] / "voice_001.ogg", 999)


@pytest.mark.parametrize("change", [change_xlsx, change_audio])
def test_write_never_replaces_an_existing_freeze(dataset, capsys, change):
    before = freeze(dataset, capsys, frozen_at="2020-01-02")
    change(dataset)
    code, out, err = run(dataset, capsys, "--write")
    assert code == 1 and "не заменяет" in err and '"CHANGED"' in out
    assert (dataset["manifest"].read_bytes(), dataset["report"].read_bytes()) == before
    assert run(dataset, capsys)[0] == 1


def test_write_does_not_change_the_original_freeze_date(dataset, capsys):
    before = freeze(dataset, capsys, frozen_at="2020-01-02")
    code, _, err = run(dataset, capsys, "--write", "--frozen-at", "2021-03-04")
    assert code == 1 and "Дата первоначальной фиксации не меняется" in err
    assert (dataset["manifest"].read_bytes(), dataset["report"].read_bytes()) == before


def test_write_refuses_when_only_part_of_a_freeze_exists(dataset, capsys):
    _, report_bytes = freeze(dataset, capsys)
    dataset["manifest"].unlink()
    code, _, err = run(dataset, capsys, "--write")
    assert code == 1 and "Приватный манифест не найден" in err
    assert not dataset["manifest"].exists() and dataset["report"].read_bytes() == report_bytes


def test_failed_write_leaves_no_partial_freeze(dataset, capsys, monkeypatch):
    real_link, calls = vt.os.link, []

    def link(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk full")
        real_link(source, target)

    monkeypatch.setattr(vt.os, "link", link)
    code, _, err = run(dataset, capsys, "--write")
    assert code == 1 and "файлы не созданы" in err
    assert not dataset["manifest"].exists() and not dataset["report"].exists()
    assert not list(dataset["root"].rglob(".*.tmp"))
