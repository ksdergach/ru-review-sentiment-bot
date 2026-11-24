"""Презентация защиты T18 (`docs/defense/T18_presentation.pptx`): целостность и числа.

Только стандартная библиотека: файл читается как ZIP с XML, python-pptx не нужен.
Числа на слайдах сверяются с принятыми JSON T03/T13/T14/T15 и с итоговым отчётом T17.
"""

import json
import re
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "defense" / "T18_presentation.pptx"
EXPECTED_SLIDES = 16


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def dec(value, digits=4):
    return ("−" if value < 0 else "") + f"{abs(value):.{digits}f}".replace(".", ",")


def num(value):
    return f"{value:,}".replace(",", " ")


def slide_number(name):
    return int(re.search(r"(\d+)\.xml$", name).group(1))


@pytest.fixture(scope="module")
def deck():
    with zipfile.ZipFile(DECK) as z:
        assert z.testzip() is None
        names = z.namelist()
        slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=slide_number)
        charts = [n for n in names if re.fullmatch(r"ppt/charts/chart\d+\.xml", n)]
        texts = {n: re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8")) for n in slides + charts}
        presentation = z.read("ppt/presentation.xml").decode("utf-8")
    flat = " \n".join(t for parts in texts.values() for t in parts).replace(" ", " ")
    return {"slides": slides, "charts": charts, "texts": texts, "all": flat, "presentation": presentation}


@pytest.fixture(scope="module")
def reference():
    return ((ROOT / "reports" / "final" / "report.md").read_text(encoding="utf-8")
            + (ROOT / "README.md").read_text(encoding="utf-8"))


def test_structure_16x9_and_no_empty_slides(deck):
    assert len(deck["slides"]) == EXPECTED_SLIDES
    size = re.search(r'<p:sldSz cx="(\d+)" cy="(\d+)"', deck["presentation"])
    assert abs(int(size.group(1)) / int(size.group(2)) - 16 / 9) < 0.01
    for name in deck["slides"]:
        assert "".join(deck["texts"][name]).strip(), name
    assert len(deck["charts"]) >= 7


def test_every_number_on_slides_is_in_final_report(deck, reference):
    decimals = set(re.findall(r"[−-]?\d+,\d+", deck["all"]))
    thousands = set(re.findall(r"\d{1,3}(?: \d{3})+", deck["all"]))
    assert decimals and thousands
    assert sorted(t for t in decimals | thousands if t not in reference) == []


def test_key_results_match_accepted_sources(deck, reference):
    t13 = load("reports/evaluation/final_metrics.json")["models"]
    t14 = load("reports/voice_eval/metrics.json")["results"]
    t15 = load("reports/error_analysis/voice_transitions.json")
    base, rubert = t13["baseline"]["metrics"], t13["rubert"]["metrics"]
    expected = [
        num(t13["baseline"]["metrics"]["rows"]),
        dec(base["accuracy"]), dec(base["macro_f1"]), dec(rubert["accuracy"]), dec(rubert["macro_f1"]),
        dec(base["per_class"]["neutral"]["f1"]), dec(rubert["per_class"]["neutral"]["f1"]),
        num(base["errors"]["total"]), num(rubert["errors"]["total"]),
        dec(t14["reference"]["macro_f1"]), dec(t14["asr"]["macro_f1"]), dec(t14["delta_macro_f1"]),
        dec(t14["wer"]["wer"]),
        f"Все {t15['prediction_changed']} смен ответа",
    ]
    for value in expected:
        assert value in deck["all"], value
        assert value in reference or value.startswith("Все "), value
    # Слайд 10: распределение четырёх состояний «эталон → Whisper» — из публичных агрегатов T15.
    tr = t15["transitions"]
    states = [tr[k] for k in ("stayed_correct", "worsened", "stayed_wrong_same", "stayed_wrong_changed")]
    assert states == [26, 4, 5, 1]
    for value in states:
        assert f"({value})" in deck["all"], value
    assert t15["records"] == t14["records"] == 36
    assert t15["prediction_changed"] == 5
    assert sum(t15["by_true_class"][c]["stayed_wrong_changed"] + t15["by_true_class"][c]["worsened"]
               for c in ("negative", "positive")) == 0


def test_honest_scope_statements(deck):
    text = deck["all"]
    assert "не реализованы, не обязательны" in text  # noise и TTS — только перспективы
    assert "Значимость разницы не проверялась" in text
    assert "T09 и T16 выполнены с RuBERT" in text
    assert re.search(r"\b(lorem|ipsum|TODO)\b", text, re.IGNORECASE) is None
    assert re.search(r"#\d+", text) is None  # без внутренних номеров задач на слайдах
