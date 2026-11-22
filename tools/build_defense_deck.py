"""T18: презентация защиты по принятым результатам T03–T17 — без обучения и инференса.

Все числа на слайдах читаются из опубликованных файлов результатов (T03, T13, T14, T15)
и форматируются так же, как в ``reports/final/report.md``. Значения вручную не набираются;
соответствие отчёту проверяет ``tests/test_defense_presentation.py``.

Нужен только ``python-pptx`` (в ``requirements.txt`` не входит: для бота и оценок не нужен)::

    python -m pip install python-pptx==1.0.2
    python tools/build_defense_deck.py                    # → docs/defense/T18_presentation.pptx
    python tools/build_defense_deck.py --out other.pptx

Приватные данные T11/T14/T15 (аудио, транскрипции, построчные предсказания) не читаются.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "defense" / "T18_presentation.pptx"

SOURCES = {
    "t03": "reports/data_prep/audit.json",
    "t04": "reports/nlp/baseline_metrics.json",
    "t06": "reports/transformer/run_metadata.json",
    "t11": "reports/voice_test/composition.json",
    "t13": "reports/evaluation/final_metrics.json",
    "t14": "reports/voice_eval/metrics.json",
    "t15_text": "reports/error_analysis/text_summary.json",
    "t15_voice": "reports/error_analysis/voice_transitions.json",
}

# Палитра: тёмный «кинозал» для титула и выводов, светлые слайды с данными.
INK = RGBColor(0x1B, 0x22, 0x38)
INK_SOFT = RGBColor(0x2B, 0x35, 0x52)
TEXT = RGBColor(0x1F, 0x24, 0x33)
MUTED = RGBColor(0x5B, 0x64, 0x78)
TINT = RGBColor(0xF1, 0xF3, 0xF8)
LINE = RGBColor(0xD5, 0xDA, 0xE5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ON_DARK = RGBColor(0xCF, 0xD6, 0xE6)
# Три класса — одинаково на всех слайдах.
NEG = RGBColor(0xC8, 0x41, 0x4B)
NEU = RGBColor(0xD9, 0x9A, 0x1E)
POS = RGBColor(0x2E, 0x9E, 0x6E)
NEU_TINT = RGBColor(0xFB, 0xF1, 0xDC)
# Две модели / два входа: тёмный — выбранная модель и эталон, светлый — сравнение.
SERIES_MAIN = INK
SERIES_ALT = RGBColor(0x9A, 0xA5, 0xC0)
WORSE = RGBColor(0xE4, 0x57, 0x2E)
STEEL_MID = RGBColor(0x6F, 0x7B, 0x9C)

FONT = "Calibri"
NBSP = "\u00a0"
SLIDE_W, SLIDE_H = 13.333, 7.5
MX = 0.6
CW = SLIDE_W - 2 * MX
BASELINE_NAME = "TF-IDF + LR"
RUBERT_NAME = "RuBERT-tiny2"
DECK_TITLE = "Анализ тональности голосовых киноотзывов"


# ---------- данные ----------

def load_facts(root: Path = ROOT) -> dict[str, Any]:
    raw = {key: json.loads((root / rel).read_text(encoding="utf-8")) for key, rel in SOURCES.items()}
    t13 = raw["t13"]["models"]
    t14 = raw["t14"]["results"]
    return {
        "split": raw["t03"]["final_summary"],
        "excluded": raw["t03"]["excluded_rows"],
        "overlap_movie": raw["t03"]["final_movie_id_overlap"],
        "overlap_text": raw["t03"]["final_exact_text_overlap"],
        "val_base": raw["t04"]["macro_f1"],
        "val_rubert": raw["t06"]["winner"]["macro_f1"],
        "rubert_step": raw["t06"]["winner"]["step"],
        "base": t13["baseline"],
        "rubert": t13["rubert"],
        "test_rows": raw["t13"]["test"]["rows"],
        "voice": t14,
        "voice_comp": raw["t11"]["aggregates"],
        "text_err": raw["t15_text"],
        "trans": raw["t15_voice"],
    }


def dec(value: float, digits: int = 4) -> str:
    text = f"{abs(value):.{digits}f}".replace(".", ",")
    return ("−" if value < 0 else "") + text


def num(value: int) -> str:
    """Разряды через неразрывный пробел: число не разрывается переносом строки."""
    return f"{value:,}".replace(",", NBSP)


def pct(part: int, whole: int, digits: int = 1) -> str:
    return dec(100 * part / whole, digits) + NBSP + "%"


# ---------- примитивы оформления ----------

def _run(paragraph, text: str, size: float, color: RGBColor, bold: bool = False, italic: bool = False):
    run = paragraph.add_run()
    run.text = text
    font = run.font
    font.name = FONT
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.color.rgb = color
    return run


def _bullet(paragraph, color: RGBColor, indent: float = 0.22) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    ppr.set("marL", str(Inches(indent)))
    ppr.set("indent", str(-Inches(indent)))
    for tag in ("a:buNone", "a:buChar", "a:buClr", "a:buFont"):
        for old in ppr.findall(qn(tag)):
            ppr.remove(old)
    clr = ppr.makeelement(qn("a:buClr"), {})
    srgb = clr.makeelement(qn("a:srgbClr"), {"val": str(color)})
    clr.append(srgb)
    ppr.append(clr)
    ppr.append(ppr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    ppr.append(ppr.makeelement(qn("a:buChar"), {"char": "•"}))


def text(slide, x, y, w, h, paras: Sequence[Any], *, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0.0, space_after=6, bullets=False, bullet_color=None, line_spacing=None,
         name=None):
    """paras: строка или список прогонов [(текст, {size,color,bold,italic})] на абзац."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for chunk, opts in runs:
            _run(p, chunk, opts.get("size", size), opts.get("color", color), opts.get("bold", bold),
                 opts.get("italic", False))
        if bullets:
            _bullet(p, bullet_color or color)
    return box


def card(slide, x, y, w, h, fill=TINT, radius=0.08, line=None, name=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.25)
    shape.shadow.inherit = False
    if name:
        shape.name = name
    shape.text_frame.text = ""
    return shape


def badge(slide, x, y, label: str, fill=INK, color=WHITE, d=0.46, size=16):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, label, size, color, bold=True)
    return shape


def chip(slide, x, y, w, label: str, fill=INK, color=WHITE, size=12, h=0.34, line=None):
    shape = card(slide, x, y, w, h, fill=fill, radius=0.5, line=line)
    tf = shape.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, label, size, color, bold=True)
    return shape


def legend_key(slide, x, y, color: RGBColor, label: str, size=13):
    """Ключ легенды: цветной квадрат-фигура и подпись (без символов, зависящих от шрифта)."""
    key = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.07), Inches(0.18), Inches(0.18))
    key.fill.solid()
    key.fill.fore_color.rgb = color
    key.line.fill.background()
    key.shadow.inherit = False
    text(slide, x + 0.28, y, 3.0, 0.32, [label], size=size, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)


def arrow(slide, x, y, w=0.32, h=0.34, direction=MSO_SHAPE.RIGHT_ARROW, fill=MUTED):
    shape = slide.shapes.add_shape(direction, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def wave(slide, x, y, heights: Sequence[float], colors: Sequence[RGBColor], bar=0.11, gap=0.09):
    """Мотив презентации: звуковая волна в цветах трёх классов."""
    base = y + max(heights)
    for i, hh in enumerate(heights):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + i * (bar + gap)),
                                       Inches(base - hh / 2 - max(heights) / 2), Inches(bar), Inches(hh))
        shape.adjustments[0] = 0.5
        shape.fill.solid()
        shape.fill.fore_color.rgb = colors[i % len(colors)]
        shape.line.fill.background()
        shape.shadow.inherit = False
        shape.name = f"Волна {i + 1}"


def set_title(slide, title: str, dark=False, size=30):
    shape = slide.shapes.title
    shape.left, shape.top = Inches(MX), Inches(0.38)
    shape.width, shape.height = Inches(CW), Inches(0.95)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.text = ""
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    _run(p, title, size, WHITE if dark else INK, bold=True)


def footer(slide, number: int, total: int, source: str | None = None, dark=False):
    color = ON_DARK if dark else MUTED
    if source:
        text(slide, MX, 6.88, 10.4, 0.32, [source], size=10, color=color, anchor=MSO_ANCHOR.MIDDLE,
             name="Источник")
    text(slide, SLIDE_W - MX - 1.6, 6.88, 1.6, 0.32, [f"{number} / {total}"], size=10, color=color,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, name="Номер слайда")


def dark_background(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = INK


def white_background(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def alt_text(graphic_frame, description: str) -> None:
    graphic_frame._element.nvGraphicFramePr.cNvPr.set("descr", description)


def style_chart(chart, legend=True, size=13):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = TEXT
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.name = FONT
    va = chart.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.has_minor_gridlines = False
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.name = FONT
    ca.tick_labels.font.color.rgb = TEXT
    ca.format.line.color.rgb = LINE
    ca.has_major_gridlines = False


def label_points(series, labels: Sequence[str | None], position=XL_LABEL_POSITION.OUTSIDE_END, size=12,
                 color=TEXT, bold=False):
    """Свои подписи точек; None — без подписи (подписи серии не включаются)."""
    for i, label in enumerate(labels):
        if label is None:
            continue
        dl = series.points[i].data_label
        dl.has_text_frame = True
        dl.position = position
        tf = dl.text_frame
        tf.text = ""
        _run(tf.paragraphs[0], label, size, color, bold=bold)


def fill_series(series, color: RGBColor):
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = color
    series.format.line.fill.background()


def table(slide, x, y, w, rows: Sequence[Sequence[str]], col_w: Sequence[float], row_h=0.42, size=13,
          header_fill=INK, zebra=TINT, bold_first_col=False, highlight_rows: Sequence[int] = (),
          highlight_fill=NEU_TINT, name=None, align_left=False):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * len(rows)))
    if name:
        shape.name = name
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            elif i in highlight_rows:
                cell.fill.fore_color.rgb = highlight_fill
            else:
                cell.fill.fore_color.rgb = zebra if i % 2 == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 or align_left else PP_ALIGN.CENTER
            _run(p, value, size, WHITE if i == 0 else TEXT, bold=(i == 0) or (bold_first_col and j == 0))
    return shape


# ---------- слайды ----------

def build(facts: dict[str, Any], out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    # Шаблон по умолчанию рассчитан на 4:3; растягиваем заполнители макетов под 16:9.
    k = SLIDE_W / 10
    for holder in [*prs.slide_master.placeholders,
                   *(ph for layout in prs.slide_layouts for ph in layout.placeholders)]:
        holder.left, holder.width = Emu(int(holder.left * k)), Emu(int(holder.width * k))
    layout = prs.slide_layouts[5]  # «Только заголовок»: у каждого слайда настоящий заголовок
    prs.core_properties.title = DECK_TITLE
    prs.core_properties.subject = "Итоговая защита учебного NLP-проекта (T18)"
    prs.core_properties.author = "Команда ru-review-sentiment-bot"
    prs.core_properties.keywords = "Whisper; TF-IDF; RuBERT-tiny2; тональность; Telegram"

    builders = [s_title, s_task, s_arch, s_data, s_models, s_t13, s_neutral, s_t14, s_t15_text,
                s_t15_voice, s_bot, s_readiness, s_answers, s_limits, s_backup_matrices, s_backup_resources]
    total = len(builders)
    for number, builder in enumerate(builders, start=1):
        slide = prs.slides.add_slide(layout)
        notes = builder(slide, facts, number, total)
        slide.notes_slide.notes_text_frame.text = notes
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def s_title(slide, f, n, total):
    dark_background(slide)
    wave(slide, MX, 0.85, [0.35, 0.7, 1.1, 0.55, 0.9, 1.25, 0.6, 0.95, 0.4, 0.75, 0.3],
         [NEG, NEU, POS])
    set_title(slide, DECK_TITLE, dark=True, size=44)
    slide.shapes.title.top = Inches(2.35)
    slide.shapes.title.height = Inches(1.5)
    text(slide, MX, 3.95, CW, 0.6,
         ["Голосовой отзыв в Telegram → Whisper → классификатор текста → тональность"],
         size=22, color=ON_DARK)
    text(slide, MX, 4.75, 7.4, 1.1, [
        [("Классы: ", {"color": ON_DARK}), ("negative = 0", {"color": RGBColor(0xF0, 0x8A, 0x92), "bold": True}),
         ("  ·  ", {"color": ON_DARK}), ("neutral = 1", {"color": RGBColor(0xF2, 0xC2, 0x5C), "bold": True}),
         ("  ·  ", {"color": ON_DARK}), ("positive = 2", {"color": RGBColor(0x6F, 0xD3, 0xA3), "bold": True})],
        "Учебный проект курса «Обработка естественного языка» (Нетология)",
    ], size=16, color=ON_DARK, space_after=8)
    text(slide, MX, 6.0, 7.6, 0.45, [[("Команда: ", {"color": ON_DARK, "bold": True}),
                                      ("впишите участников перед защитой", {"color": NEU, "italic": True})]],
         size=16, name="Команда — заполнить")
    text(slide, 8.4, 5.55, 4.33, 0.9, ["Итоговая защита · материалы T18", "Результаты — по итоговому отчёту T17"],
         size=12, color=ON_DARK, align=PP_ALIGN.RIGHT, space_after=2)
    return ("Представить тему, команду и цель. Система принимает голосовой киноотзыв, распознаёт речь "
            "Whisper и определяет тональность текста: negative, neutral, positive. Перед защитой вписать "
            "участников на этом слайде.")


def s_task(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Задача: определить тональность голосового киноотзыва")
    rows = [
        ("Вход", f"Голосовое сообщение Telegram с русскоязычным отзывом о фильме (до 30{NBSP}с)", INK),
        ("Обработка", "Готовый Whisper распознаёт речь; классификатор оценивает содержание текста", INK_SOFT),
        ("Выход", "Распознанный текст, тональность negative / neutral / positive и уверенность модели", INK),
    ]
    y = 1.65
    for label, body, color in rows:
        card(slide, MX, y, 6.25, 1.12)
        chip(slide, MX + 0.25, y + 0.36, 1.45, label, fill=color, size=13, h=0.4)
        text(slide, MX + 1.95, y + 0.12, 4.1, 0.9, [body], size=16, anchor=MSO_ANCHOR.MIDDLE)
        y += 1.3
    text(slide, MX, 5.62, 6.25, 1.1, [
        [("Условие кейса: ", {"bold": True}),
         ("классифицировать аудиосообщения по тональности. По пояснению преподавателя классификатор "
          "обучается на текстах, аудио нужно для проверки через Telegram.", {})]],
        size=14, color=MUTED)

    x = 7.25
    w = SLIDE_W - MX - x
    text(slide, x, 1.6, w, 0.45, ["Два исследовательских вопроса"], size=20, bold=True, color=INK)
    questions = [
        ("1", "Даёт ли дообученный RuBERT преимущество перед TF-IDF + Logistic Regression?",
         "Одинаковые train и validation, один неизменный test"),
        ("2", "Насколько ошибки распознавания речи меняют классификацию?",
         "Одна модель: эталонная транскрипция против текста Whisper"),
    ]
    y = 2.12
    for label, q, method in questions:
        card(slide, x, y, w, 1.62)
        badge(slide, x + 0.25, y + 0.25, label)
        text(slide, x + 0.9, y + 0.18, w - 1.1, 0.8, [q], size=16, bold=True, color=INK)
        text(slide, x + 0.9, y + 0.98, w - 1.1, 0.56, [method], size=14, color=MUTED)
        y += 1.78
    card(slide, x, 5.72, w, 0.98, fill=NEU_TINT)
    text(slide, x + 0.25, 5.8, w - 0.5, 0.85, [
        "Область обучения и проверки — киноотзывы. Работа на произвольных сообщениях Telegram "
        "не проверялась; интонация не анализируется."], size=14, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: docs/team_plan.md, разделы 1–2; итоговый отчёт T17, раздел 1")
    return ("Объяснить вход и выход системы и два исследовательских вопроса. Подчеркнуть границу области: "
            "киноотзывы, а не любые сообщения.")


def s_arch(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Архитектура: голос и текст идут в один классификатор")
    steps = [
        ("Голосовое сообщение", f"Telegram, aiogram{NBSP}3"),
        ("Лимиты голоса", f"≤{NBSP}30{NBSP}с и ≤{NBSP}5{NBSP}МиБ"),
        ("Whisper small", "CPU, int8, русский язык"),
        ("Лимит текста", f"≤{NBSP}2{NBSP}000 символов"),
        ("Классификатор", "TF-IDF + LR, общий predict"),
        ("Ответ", "транскрипция, метка, уверенность"),
    ]
    bw, gap, y = 1.68, 0.41, 1.85
    for i, (head, sub) in enumerate(steps):
        x = MX + i * (bw + gap)
        fill = INK if i in (2, 4) else TINT
        card(slide, x, y, bw, 1.45, fill=fill)
        color = WHITE if fill == INK else INK
        text(slide, x + 0.12, y + 0.15, bw - 0.24, 0.6, [head], size=15, bold=True, color=color,
             align=PP_ALIGN.CENTER)
        text(slide, x + 0.12, y + 0.72, bw - 0.24, 0.65, [sub], size=13,
             color=ON_DARK if fill == INK else MUTED, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            arrow(slide, x + bw + 0.05, y + 0.56, w=0.31, h=0.33)
    # Вспомогательный текстовый вход — в тот же лимит текста.
    tx = MX + 3 * (bw + gap)
    arrow(slide, tx + bw / 2 - 0.17, y + 1.55, w=0.34, h=0.4, direction=MSO_SHAPE.UP_ARROW)
    card(slide, tx - 0.35, y + 2.05, bw + 0.7, 0.85, line=INK, fill=WHITE)
    text(slide, tx - 0.25, y + 2.1, bw + 0.5, 0.75,
         [[("Текстовый отзыв", {"bold": True, "color": INK})], "вспомогательный режим"],
         size=14, color=MUTED, align=PP_ALIGN.CENTER, space_after=0, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, MX, y + 2.15, 6.0, 0.75, [
        "Одинаковый текст получает одинаковую метку: распознанная речь и обычное сообщение "
        "проходят через один экземпляр классификатора."], size=14, color=MUTED)

    facts = [
        ("Один процесс", "Long polling, без сервера и базы данных. Модели загружаются один раз при старте."),
        ("Один отзыв за раз", f"Остальным — просьба повторить позже. Через 60{NBSP}с — одно уведомление о задержке."),
        ("Приватность", "Аудио удаляется после обработки, транскрипции не пишутся в журналы."),
    ]
    fw = (CW - 2 * 0.3) / 3
    for i, (head, body) in enumerate(facts):
        x = MX + i * (fw + 0.3)
        card(slide, x, 5.25, fw, 1.45)
        text(slide, x + 0.25, 5.37, fw - 0.5, 0.4, [head], size=16, bold=True, color=INK)
        text(slide, x + 0.25, 5.8, fw - 0.5, 0.85, [body], size=14)
    footer(slide, n, total, "Источник: README (T09, T10), configs/speech.json, итоговый отчёт T17, раздел 4")
    return ("Пройти цепочку слева направо. Whisper не дообучался, классификатор видит только текст. "
            "Текстовый режим входит в тот же лимит и тот же predict.")


def s_data(slide, f, n, total):
    white_background(slide)
    split = f["split"]
    total_rows = sum(split[s]["rows"] for s in ("train", "validation", "test"))
    set_title(slide, f"Данные: {num(total_rows)} русскоязычный киноотзыв, neutral — около 4 %")
    names = [("train", "Train"), ("validation", "Validation"), ("test", "Test")]
    data = CategoryChartData()
    data.categories = [f"{label}\n{num(split[key]['rows'])}" for key, label in names]
    for cls in ("negative", "neutral", "positive"):
        data.add_series(cls, [split[key][cls] / split[key]["rows"] for key, _ in names])
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(MX), Inches(1.6), Inches(6.9),
                                   Inches(3.3), data)
    frame.name = "Распределение классов"
    alt_text(frame, "Доля классов negative, neutral, positive в train, validation и test после подготовки T03")
    chart = frame.chart
    style_chart(chart)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.overlap = 100
    for series, color, cls in zip(plot.series, (NEG, NEU, POS), ("negative", "neutral", "positive")):
        fill_series(series, color)
        labels = [num(split[key][cls]) if cls != "neutral" else None for key, _ in names]
        label_points(series, labels, position=XL_LABEL_POSITION.CENTER, size=13, color=WHITE, bold=True)
    neu = " / ".join(num(split[key]["neutral"]) for key, _ in names)
    card(slide, MX, 5.1, 6.9, 1.58, fill=NEU_TINT)
    text(slide, MX + 0.25, 5.2, 6.4, 1.4, [
        [("neutral: ", {"bold": True}), (f"{neu} записей. ", {}),
         ("В train ", {}), (pct(split['train']['neutral'], split['train']['rows'], 2), {"bold": True}),
         (" neutral и ", {}), (pct(split['train']['positive'], split['train']['rows'], 2), {"bold": True}),
         (" positive.", {})],
        "Поэтому основная метрика — macro-F1: все три класса весят одинаково.",
    ], size=15, anchor=MSO_ANCHOR.MIDDLE)

    x = 7.85
    w = SLIDE_W - MX - x
    text(slide, x, 1.6, w, 0.4, ["Источник"], size=18, bold=True, color=INK)
    text(slide, x, 2.0, w, 1.0, [
        "100,000+ Movie Reviews from Kazakhstan (kino.kz), конфигурация pc, закреплённая ревизия",
        [("Лицензия ", {}), ("CC BY 4.0", {"bold": True}), (" · только русский язык", {})],
    ], size=14, space_after=3)
    text(slide, x, 3.1, w, 0.4, ["Независимость выборок"], size=18, bold=True, color=INK)
    movie = sum(f["overlap_movie"].values())
    texts = sum(f["overlap_text"].values())
    items = [
        "Исходное разбиение по фильмам сохранено, нового случайного нет",
        f"Исключено {f['excluded']} записей: 9 точных и 49 близких повторов (триграммы, Jaccard ≥ 0,9)",
        f"Общих movie_id между выборками: {movie}; совпадающих нормализованных текстов: {texts}",
        "Test не использовался для подбора; финальная оценка — после фиксации T12",
    ]
    text(slide, x, 3.5, w, 3.2, items, size=14, bullets=True, bullet_color=INK, space_after=7)
    footer(slide, n, total, "Источник: reports/data_prep/audit.json (T03), NLP_dataset/README.md")
    return ("Источник, лицензия и распределение. Neutral — редкий класс, поэтому accuracy недостаточна. "
            "Утечки: сохранено разбиение по фильмам, удалены повторы, test не участвовал в настройке.")


def s_models(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Модели: TF-IDF + LR против дообученного RuBERT-tiny2")
    cw = (CW - 0.4) / 2
    models = [
        (BASELINE_NAME, "word-level TF-IDF + Logistic Regression", SERIES_MAIN, WHITE, [
            f"Слова и биграммы, до 100{NBSP}000 признаков, sublinear TF",
            "Словарь, IDF и веса классов — только по train",
            "C из 0,5 / 1 / 2 — выбран C = 2 по validation macro-F1",
        ], f["val_base"]),
        (RUBERT_NAME, "cointegrated/rubert-tiny2, дообучение энкодера", SERIES_ALT, INK, [
            f"Одна эпоха на тех же {num(f['split']['train']['rows'])} train ID, CPU",
            "Вход до 64 токенов, веса классов по train, seed 42",
            f"Контрольная точка — шаг {f['rubert_step']} из двух, по validation macro-F1",
        ], f["val_rubert"]),
    ]
    for i, (name, sub, color, on, items, val) in enumerate(models):
        x = MX + i * (cw + 0.4)
        card(slide, x, 1.6, cw, 3.65)
        chip(slide, x + 0.3, 1.85, 2.3, name, fill=color, color=on, size=15, h=0.44)
        text(slide, x + 0.3, 2.42, cw - 0.6, 0.4, [sub], size=14, color=MUTED)
        text(slide, x + 0.3, 2.92, cw - 2.6, 2.2, items, size=15, bullets=True, bullet_color=INK, space_after=8)
        text(slide, x + cw - 2.15, 3.2, 1.9, 0.4, ["Validation macro-F1"], size=12, color=MUTED,
             align=PP_ALIGN.RIGHT)
        text(slide, x + cw - 2.15, 3.55, 1.9, 0.8, [dec(val)], size=36, bold=True, color=INK,
             align=PP_ALIGN.RIGHT)
    card(slide, MX, 5.5, CW, 1.2, fill=INK)
    text(slide, MX + 0.35, 5.58, CW - 0.7, 1.05, [
        [("Выбор модели бота (T12): ", {"bold": True, "color": WHITE}),
         ("baseline — по validation и пригодности для Mac на CPU. ", {"color": ON_DARK}),
         ("Модели, параметры и протокол зафиксированы до открытия test; после результатов test "
          "обучение и выбор не менялись.", {"color": ON_DARK})]],
        size=16, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: configs/nlp_baseline.json, configs/transformer.json, reports/freeze.json; отчёт T17, разделы 4–5")
    return ("Две модели на одном train. Гиперпараметры и контрольная точка выбраны только по validation. "
            "Validation — результат разработки, не финальная оценка.")


def _metric_rows(model):
    m = model["metrics"]
    pc = m["per_class"]
    return [m["accuracy"], m["macro_f1"], pc["negative"]["f1"], pc["neutral"]["f1"], pc["positive"]["f1"]]


def s_t13(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Финальный test: baseline лучше RuBERT-tiny2 по всем метрикам")
    base, rub = _metric_rows(f["base"]), _metric_rows(f["rubert"])
    data = CategoryChartData()
    data.categories = ["Accuracy", "Macro-F1", "F1 negative", "F1 neutral", "F1 positive"]
    data.add_series(BASELINE_NAME, base)
    data.add_series(RUBERT_NAME, rub)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MX), Inches(1.55), Inches(8.2),
                                   Inches(5.15), data)
    frame.name = "Метрики T13"
    alt_text(frame, "Accuracy, macro-F1 и F1 по классам на test T13 для TF-IDF + LR и RuBERT-tiny2")
    chart = frame.chart
    style_chart(chart)
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 1.08
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.overlap = -8
    for series, color, values in zip(plot.series, (SERIES_MAIN, SERIES_ALT), (base, rub)):
        fill_series(series, color)
        label_points(series, [dec(v) for v in values], size=12)

    x = 9.15
    w = SLIDE_W - MX - x
    errors_b = f["base"]["metrics"]["errors"]["total"]
    errors_r = f["rubert"]["metrics"]["errors"]["total"]
    tiles = [
        (num(f["test_rows"]), "записи test: одни и те же ID, один запуск"),
        (f"{num(errors_b)} / {num(errors_r)}", f"ошибок: {BASELINE_NAME} / {RUBERT_NAME}"),
        (dec(f["rubert"]["metrics"]["macro_f1"] - f["base"]["metrics"]["macro_f1"]),
         "разница macro-F1 (RuBERT минус baseline)"),
    ]
    y = 1.6
    for big, small in tiles:
        card(slide, x, y, w, 1.12)
        text(slide, x + 0.25, y + 0.08, w - 0.5, 0.55, [big], size=26, bold=True, color=INK)
        text(slide, x + 0.25, y + 0.64, w - 0.5, 0.45, [small], size=12, color=MUTED)
        y += 1.27
    card(slide, x, 5.45, w, 1.25, fill=NEU_TINT)
    text(slide, x + 0.2, 5.5, w - 0.4, 1.15, [
        "Вывод — для rubert-tiny2: одна эпоха, 64 токена, CPU. Значимость разницы не проверялась."],
        size=13, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: reports/evaluation/final_metrics.json (T13, 21.11.2026)")
    return ("Главный слайд первого вопроса. Baseline выше по accuracy, macro-F1 и F1 каждого класса; порядок "
            "совпадает с validation. Не обобщать на все трансформеры.")


def s_neutral(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Neutral — самый трудный класс для обеих моделей")
    true_neu = f["base"]["metrics"]["per_class"]["neutral"]["support"]
    pred_b = f["base"]["metrics"]["predicted_counts"]["neutral"]
    pred_r = f["rubert"]["metrics"]["predicted_counts"]["neutral"]
    data = CategoryChartData()
    data.categories = ["Истинных neutral", BASELINE_NAME, RUBERT_NAME]
    data.add_series("Ответов neutral", [true_neu, pred_b, pred_r])
    text(slide, MX, 1.55, 5.9, 0.4, ["Сколько раз на test встречается и предсказывается neutral"], size=15,
         bold=True, color=INK)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MX), Inches(1.95), Inches(5.9),
                                   Inches(4.75), data)
    frame.name = "Ответы neutral"
    alt_text(frame, "Число истинных neutral в test и число ответов neutral у двух моделей")
    chart = frame.chart
    style_chart(chart, legend=False)
    plot = chart.plots[0]
    plot.gap_width = 70
    series = plot.series[0]
    fill_series(series, SERIES_MAIN)
    for point, color in zip(series.points, (NEU, SERIES_MAIN, SERIES_ALT)):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = color
    label_points(series, [num(true_neu), num(pred_b), num(pred_r)], size=16, bold=True)

    x = 7.0
    w = SLIDE_W - MX - x
    pb, pr = f["base"]["metrics"]["per_class"]["neutral"], f["rubert"]["metrics"]["per_class"]["neutral"]
    rows = [["neutral на test", "Precision", "Recall", "F1"],
            [BASELINE_NAME, dec(pb["precision"]), dec(pb["recall"]), dec(pb["f1"])],
            [RUBERT_NAME, dec(pr["precision"]), dec(pr["recall"]), dec(pr["f1"])]]
    table(slide, x, 1.6, w, rows, [w - 3.6, 1.2, 1.2, 1.2], row_h=0.46, size=14, bold_first_col=True,
          name="Neutral P/R/F1")
    err = f["text_err"]["errors"]
    involved = err["involving_neutral"]
    test_rows = f["test_rows"]
    text(slide, x, 3.2, w, 2.3, [
        [("RuBERT ", {"bold": True}), ("чаще отвечает neutral: recall выше, но precision намного ниже.", {})],
        [("Neutral участвует в ", {}), (f"{num(involved)} из {num(err['total'])}", {"bold": True}),
         (f" ошибок baseline ({pct(involved, err['total'])}), хотя составляет ", {}),
         (pct(true_neu, test_rows), {"bold": True}), (" test.", {})],
        [("Accuracy ", {}), (dec(f["base"]["metrics"]["accuracy"]), {"bold": True}),
         (" почти не показывает провал на редком классе — поэтому смотрим macro-F1.", {})],
    ], size=15, bullets=True, bullet_color=NEU, space_after=10)
    card(slide, x, 5.6, w, 1.1, fill=NEU_TINT)
    text(slide, x + 0.25, 5.65, w - 0.5, 1.0, [
        "Истинный neutral неоднороден: смешанные отзывы без итога и тексты без оценки фильма."],
        size=14, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: reports/evaluation/final_metrics.json (T13), reports/error_analysis/text_summary.json (T15)")
    return ("Neutral редкий и неоднородный. RuBERT предсказывает neutral 1 436 раз при 411 истинных. "
            "Связать с выбором macro-F1 как основной метрики.")


def s_t14(slide, f, n, total):
    white_background(slide)
    v = f["voice"]
    set_title(slide, f"Голос: после Whisper macro-F1 снизилась на {dec(abs(v['delta_macro_f1']), 3)}")
    tiles = [
        (f"{dec(v['reference']['macro_f1'])} → {dec(v['asr']['macro_f1'])}", "macro-F1: эталон → Whisper"),
        (dec(v["delta_macro_f1"]), "ΔF1 = F1 Whisper − F1 эталон"),
        (dec(v["wer"]["wer"]), "корпусный WER =\n" + NBSP.join(
            ["(" + str(v["wer"]["substitutions"]), "+", str(v["wer"]["deletions"]), "+",
             str(v["wer"]["insertions"]) + ")", "/", str(v["wer"]["reference_words"])])),
        (f"{v['reference']['correct']} → {v['asr']['correct']}", "верных ответов из 36; сбоев и пустых — 0"),
    ]
    tw = 2.85
    y = 1.6
    for i, (big, small) in enumerate(tiles):
        tx = MX + (i % 2) * (tw + 0.25)
        ty = y + (i // 2) * 1.55
        card(slide, tx, ty, tw, 1.38, fill=INK if i == 1 else TINT)
        text(slide, tx + 0.22, ty + 0.1, tw - 0.44, 0.6, [big], size=26 if i else 24, bold=True,
             color=WHITE if i == 1 else INK)
        text(slide, tx + 0.22, ty + 0.72, tw - 0.44, 0.6, small.split("\n"), size=12, space_after=0,
             color=ON_DARK if i == 1 else MUTED)
    comp = f["voice_comp"]
    text(slide, MX, 4.9, 2 * tw + 0.25, 1.8, [
        f"{comp['records']} записей OGG/Opus, {comp['speakers']} диктора, по 12 на класс",
        "Два разметчика независимо, без разногласий; чтение подготовленных сценариев",
        f"{comp['noise']} записей с бытовым шумом; классификатор — принятый baseline",
    ], size=14, bullets=True, bullet_color=INK, space_after=6)

    data = CategoryChartData()
    data.categories = ["negative", "neutral", "positive", "Macro-F1"]
    ref = [v["reference"]["per_class"][c]["f1"] for c in ("negative", "neutral", "positive")] + [
        v["reference"]["macro_f1"]]
    asr = [v["asr"]["per_class"][c]["f1"] for c in ("negative", "neutral", "positive")] + [v["asr"]["macro_f1"]]
    data.add_series("Эталонная транскрипция", ref)
    data.add_series("После Whisper", asr)
    x = 6.95
    text(slide, x, 1.55, SLIDE_W - MX - x, 0.4, ["F1 по классам, 36 записей"], size=15, bold=True, color=INK)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(1.95),
                                   Inches(SLIDE_W - MX - x), Inches(4.75), data)
    frame.name = "F1 T14"
    alt_text(frame, "F1 по классам и macro-F1 baseline на эталонных транскрипциях и после Whisper, T14")
    chart = frame.chart
    style_chart(chart)
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 1.05
    plot = chart.plots[0]
    plot.gap_width = 40
    plot.overlap = -12
    for series, color, values in zip(plot.series, (SERIES_MAIN, SERIES_ALT), (ref, asr)):
        fill_series(series, color)
        label_points(series, [dec(val) for val in values], size=11)
    footer(slide, n, total, "Источник: reports/voice_eval/metrics.json (T14), reports/voice_test/composition.json (T11)")
    return ("Второй вопрос. Одна модель на двух входах для каждой записи. Сильнее всего падает neutral. "
            "Выборка мала: одна запись меняет accuracy на 2,8 п. п.")


def s_t15_text(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Ошибки текста: смешанные оценки и граница с neutral")
    err = f["text_err"]["errors"]["by_direction"]
    order = sorted(err.items(), key=lambda kv: -kv[1])
    data = CategoryChartData()
    data.categories = [k.replace("->", " → ") for k, _ in order]
    data.add_series("Ошибок", [v for _, v in order])
    total_err = f["text_err"]["errors"]["total"]
    cw = 5.75
    text(slide, MX, 1.55, cw, 0.4, [f"Все {num(total_err)} ошибок baseline на test: эталон → ответ"], size=15,
         bold=True, color=INK)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MX), Inches(1.95), Inches(cw),
                                   Inches(3.55), data)
    frame.name = "Направления ошибок"
    alt_text(frame, "Число ошибок baseline на test по шести направлениям; направления с neutral выделены")
    chart = frame.chart
    style_chart(chart, legend=False, size=13)
    chart.category_axis.reverse_order = True
    plot = chart.plots[0]
    plot.gap_width = 45
    series = plot.series[0]
    fill_series(series, SERIES_ALT)
    for point, (key, _) in zip(series.points, order):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = NEU if "neutral" in key else SERIES_ALT
    label_points(series, [num(v) for _, v in order], size=13, bold=True)
    legend_key(slide, MX, 5.6, NEU, "направления с участием neutral")

    ann = f["text_err"]["annotation"]["category_counts"]
    names = {"mixed": "Смешанная оценка", "context": "Оценка не фильма / не отзыв",
             "unclear_label": "Спорная исходная метка", "neutral_confusion": "Размытая граница с neutral",
             "negation": "Отрицание", "sarcasm": "Сарказм", "short": "Короткий текст", "unknown": "Не определено"}
    cats = sorted(ann.items(), key=lambda kv: (-kv[1], list(names).index(kv[0])))
    data2 = CategoryChartData()
    data2.categories = [names[k] for k, _ in cats]
    data2.add_series("Случаев из 40", [v for _, v in cats])
    x = MX + cw + 0.5
    w = SLIDE_W - MX - x
    sample = f["text_err"]["selection"]["selected_size"]
    text(slide, x, 1.55, w, 0.4, [f"Категории в выборке из {sample} ошибок (пересекаются)"], size=15, bold=True,
         color=INK)
    frame2 = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(1.95), Inches(w),
                                    Inches(3.55), data2)
    frame2.name = "Категории ошибок"
    alt_text(frame2, "Категории ручного разбора 40 ошибок baseline, T15")
    chart2 = frame2.chart
    style_chart(chart2, legend=False, size=13)
    chart2.category_axis.reverse_order = True
    plot2 = chart2.plots[0]
    plot2.gap_width = 40
    fill_series(plot2.series[0], SERIES_MAIN)
    label_points(plot2.series[0], [str(v) for _, v in cats], size=13, bold=True)

    card(slide, MX, 6.08, CW, 0.64, fill=NEU_TINT)
    text(slide, MX + 0.25, 6.1, CW - 0.5, 0.6, [
        "Выборка почти равномерна по направлениям; черновик категорий — ИИ, проверил один участник "
        "(31 согласие, 9 частичных, 0 несогласий). Причинность не проверялась."],
        size=13, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: reports/error_analysis/text_summary.json, human_review_evidence.json (T15)")
    return ("Слева — все ошибки, справа — разобранная выборка. Категории — гипотезы о причинах, не доказанная "
            "причинность. Доли выборки не переносятся на все 1 019 ошибок.")


def s_t15_voice(slide, f, n, total):
    white_background(slide)
    t = f["trans"]
    set_title(slide, f"Все {t['prediction_changed']} смен ответа после Whisper — у истинного neutral")
    classes = ("negative", "neutral", "positive")
    by = t["by_true_class"]
    statuses = [("stayed_correct", "Верно в обоих режимах", SERIES_MAIN, WHITE),
                ("worsened", "Ухудшение после Whisper", WORSE, WHITE),
                ("stayed_wrong_same", "Неверно в обоих, тот же ответ", SERIES_ALT, INK),
                ("stayed_wrong_changed", "Неверно в обоих, ответ сменился", STEEL_MID, WHITE)]
    data = CategoryChartData()
    data.categories = [f"{c} (12)" for c in classes]
    for key, label, _, _ in statuses:
        data.add_series(label, [by[c][key] for c in classes])
    cw = 6.6
    text(slide, MX, 1.55, cw, 0.4, ["36 записей по истинному классу: эталон → Whisper"], size=15, bold=True,
         color=INK)
    for i, (_, label, color, _) in enumerate(statuses):
        legend_key(slide, MX + (i % 2) * 3.3, 2.0 + (i // 2) * 0.36, color, label)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(MX), Inches(2.75), Inches(cw),
                                   Inches(3.95), data)
    frame.name = "Переходы T15"
    alt_text(frame, "Публичные агрегаты T15: переходы классификации эталон → Whisper по истинному классу")
    chart = frame.chart
    style_chart(chart, legend=False, size=13)
    chart.category_axis.reverse_order = True
    chart.value_axis.maximum_scale = 12
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.overlap = 100
    for series, (key, _, color, on) in zip(plot.series, statuses):
        fill_series(series, color)
        label_points(series, [str(by[c][key]) if by[c][key] else None for c in classes],
                     position=XL_LABEL_POSITION.CENTER, size=14, color=on, bold=True)

    x = MX + cw + 0.45
    w = SLIDE_W - MX - x
    d = t["prediction_change_directions"]
    v = f["voice"]
    rec_ref = dec(v["reference"]["per_class"]["neutral"]["recall"], 2)
    rec_asr = dec(v["asr"]["per_class"]["neutral"]["recall"], 2)
    text(slide, x, 1.6, w, 3.1, [
        [("Смены ответа: ", {"bold": True}),
         (f"neutral → negative {d['neutral->negative']}, neutral → positive {d['neutral->positive']}, "
          f"negative → positive {d['negative->positive']} (неверно в обоих режимах)", {})],
        "Единого сдвига в одну полярность нет",
        [("Recall neutral: ", {"bold": True}), (f"{rec_ref} → {rec_asr}", {}),
         ("; у negative и positive ответы не изменились", {})],
        "Precision negative и positive снизилась: часть нейтральных записей ушла в полярные классы",
    ], size=15, bullets=True, bullet_color=NEU, space_after=9)
    card(slide, x, 4.85, w, 1.85, fill=NEU_TINT)
    text(slide, x + 0.25, 4.92, w - 0.5, 1.72, [
        [("WER ≠ причина. ", {"bold": True}),
         ("Ошибки слов частые (около 16 на 100 слов), смена класса редкая. Какие отличия текста "
          "повлияли на ответ, не установлено.", {})],
        "Построчные ответы и транскрипции не публикуются (решения T14/T15).",
    ], size=13, space_after=4, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: reports/error_analysis/voice_transitions.json (T15), reports/voice_eval/metrics.json (T14)")
    return ("Только публичные агрегаты. Все изменения — у истинного neutral; это наблюдение, а не доказанная "
            "причина. Гипотеза о слабых сигналах neutral не проверялась.")


def s_bot(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Telegram-бот: голос и текст в одном сценарии")
    w1 = 5.2
    card(slide, MX, 1.6, w1, 3.15, fill=INK)
    text(slide, MX + 0.3, 1.72, w1 - 0.6, 0.4, ["Формат ответа на голосовой отзыв"], size=15, bold=True,
         color=WHITE)
    text(slide, MX + 0.3, 2.2, w1 - 0.6, 2.75, [
        [("Распознанный текст:", {"bold": True, "color": WHITE})],
        [("<транскрипция Whisper>", {"italic": True, "color": ON_DARK})],
        [("Тональность: ", {"bold": True, "color": WHITE}), ("положительная (positive)", {"color": RGBColor(0x6F, 0xD3, 0xA3)})],
        [("Уверенность модели: ", {"bold": True, "color": WHITE}), ("<процент>", {"italic": True, "color": ON_DARK})],
        [("Уверенность модели не является гарантией правильности ответа.", {"color": ON_DARK, "size": 13})],
    ], size=15, space_after=7)
    text(slide, MX, 4.85, w1, 0.4, ["Шаблон ответа из src/bot.py; угловые скобки и метка — пример"],
         size=11, color=MUTED)

    x = MX + w1 + 0.45
    w = SLIDE_W - MX - x
    rows = [["Ситуация", "Ответ бота"],
            [f"Голос > 30{NBSP}с или > 5{NBSP}МиБ", "Отказ до распознавания"],
            [f"Текст > 2{NBSP}000 символов", "«Сообщение слишком длинное»"],
            ["Тишина, нет речи", "Просьба записать ещё раз, без метки"],
            ["Бот занят другим отзывом", "«Повторите запрос чуть позже»"],
            [f"Ответ не готов за 60{NBSP}с", "Одно уведомление о задержке"],
            ["Файл, стикер, кружок", "Поддерживаются голос и текст"]]
    table(slide, x, 1.6, w, rows, [w * 0.45, w * 0.55], row_h=0.49, size=13, name="Ограничения бота",
          align_left=True)
    card(slide, MX, 5.6, CW, 1.1)
    text(slide, MX + 0.3, 5.66, CW - 0.6, 1.0, [
        [("Проверено: ", {"bold": True}),
         ("T09 — ручной голосовой прогон; T16 — независимый участник на Mac: голос, текст, лимиты, занятость, "
          "825 passed / 1 skipped. T09 и T16 выполнены с RuBERT; сквозной показ с выбранным baseline — "
          "отдельная проверка T18.", {})]],
        size=14, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: src/bot.py, README (T09, T10), reports/bot/t16_independent_check_macos_2026-11-19.md")
    return ("Перед живым показом: что умеет бот и как отказывает. Не обещать конкретную метку. Показ: /start, "
            "голосовой отзыв, тот же текст сообщением, голос длиннее 30 с.")


def s_readiness(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Пять критериев готовности проекта и их подтверждение")
    rows = [
        ("Источник и подготовка датасета документированы, независимые сплиты сохранены",
         "NLP_dataset/README.md, аудит T03: 0 общих фильмов и текстов", "Выполнено"),
        ("TF-IDF и RuBERT сравнены на одном test по трём классам без настройки по нему",
         f"Фиксация T12, финальный запуск T13 на {num(f['test_rows'])} ID", "Выполнено"),
        ("Бот принимает голосовое сообщение и возвращает транскрипцию и тональность",
         "Ручные прогоны T09 и T16 (оба с RuBERT)", "Выполнено"),
        ("Проверка на человеческих голосовых киноотзывах с разбором влияния Whisper и ошибок",
         "36 записей T11, оценка T14, анализ T15", "Выполнено"),
        ("Другой участник повторил запуск; готовы отчёт, презентация и демонстрация",
         "Запуск T16, отчёт T17; презентация и показ — T18", "Показ на защите"),
    ]
    y = 1.6
    rh = 0.97
    for i, (crit, evidence, status) in enumerate(rows, start=1):
        card(slide, MX, y, CW, rh - 0.12)
        badge(slide, MX + 0.22, y + (rh - 0.12 - 0.46) / 2, str(i))
        text(slide, MX + 0.9, y + 0.06, 6.55, rh - 0.24, [crit], size=15, bold=True, color=INK,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, MX + 7.6, y + 0.06, 2.6, rh - 0.24, [evidence], size=12, color=MUTED,
             anchor=MSO_ANCHOR.MIDDLE)
        done = status == "Выполнено"
        chip(slide, MX + CW - 1.75, y + (rh - 0.12 - 0.4) / 2, 1.55, status, fill=INK if done else WHITE,
             color=WHITE if done else INK, size=12, h=0.4, line=None if done else INK)
        y += rh
    footer(slide, n, total, "Критерии: docs/team_plan.md, раздел 12. noise, TTS и обработка любых сообщений в них не входят")
    return ("Пройти пять критериев плана. Пятый закрывается самой защитой: презентация и показ. "
            "T16 проверял бота с RuBERT — сказать об этом прямо.")


def s_answers(slide, f, n, total):
    dark_background(slide)
    set_title(slide, "Ответы на исследовательские вопросы", dark=True)
    b, r = f["base"]["metrics"], f["rubert"]["metrics"]
    v = f["voice"]
    cw = (CW - 0.4) / 2
    blocks = [
        ("1", "Даёт ли RuBERT преимущество перед baseline?", "Нет — в исследованной конфигурации", [
            f"Macro-F1 {dec(r['macro_f1'])} против {dec(b['macro_f1'])}, accuracy {dec(r['accuracy'])} против "
            f"{dec(b['accuracy'])}; тот же порядок на validation",
            "Корректно измеренный отрицательный результат для rubert-tiny2 (одна эпоха, 64 токена), "
            "а не превосходство TF-IDF над всеми трансформерами",
        ]),
        ("2", "Насколько ошибки распознавания меняют классификацию?",
         f"Macro-F1 {dec(v['reference']['macro_f1'], 3)} → {dec(v['asr']['macro_f1'], 3)}", [
             f"ΔF1 = {dec(v['delta_macro_f1'])} при WER {dec(v['wer']['wer'])}; верных ответов "
             f"{v['reference']['correct']} → {v['asr']['correct']}; сбоев нет",
             f"Все 5 смен ответа — у истинного neutral. 36 записей: одна запись меняет accuracy на 2,8{NBSP}п.{NBSP}п.",
         ]),
    ]
    for i, (label, q, answer, items) in enumerate(blocks):
        x = MX + i * (cw + 0.4)
        card(slide, x, 1.6, cw, 4.25, fill=INK_SOFT)
        badge(slide, x + 0.3, 1.85, label, fill=NEU, color=INK)
        text(slide, x + 0.95, 1.8, cw - 1.25, 0.65, [q], size=16, color=ON_DARK, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.3, 2.6, cw - 0.6, 0.95, [answer], size=24, bold=True, color=WHITE,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 0.3, 3.7, cw - 0.6, 2.05, items, size=15, color=ON_DARK, bullets=True, bullet_color=NEU,
             space_after=10)
    wave(slide, MX, 6.08, [0.2, 0.42, 0.62, 0.32, 0.5, 0.26], [NEG, NEU, POS], bar=0.08, gap=0.07)
    text(slide, MX + 1.0, 6.02, CW - 1.0, 0.7, [
        [("Главное слабое место — neutral: ", {"bold": True, "color": WHITE}),
         ("редкий и неоднородный класс, худший F1 у обеих моделей на test.", {"color": ON_DARK})]],
        size=16, anchor=MSO_ANCHOR.MIDDLE)
    footer(slide, n, total, "Источник: итоговый отчёт T17, раздел 10", dark=True)
    return ("Сформулировать ответы на оба вопроса с ограничениями выводов. Значимость не проверялась; "
            "перенос на спонтанную речь и другие темы не проверялся.")


def s_limits(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Ограничения и возможные улучшения")
    cw = 6.9
    limits = [
        ("Домен", "только киноотзывы одного сайта; произвольные сообщения не проверялись"),
        ("Голосовой набор", "36 записей, 4 диктора, чтение подготовленных сценариев"),
        ("Статистика", "без тестов значимости и доверительных интервалов, по одному прогону"),
        ("Разметка", "исходные метки не исправлялись; часть ошибок — спорные метки"),
        ("Только текст", "интонация не анализируется, калибровка уверенности не проверялась"),
        ("Бот", "ручные прогоны T09 и T16 выполнены с RuBERT, не с выбранным baseline"),
    ]
    text(slide, MX, 1.55, cw, 0.4, ["Ограничения исследования"], size=18, bold=True, color=INK)
    text(slide, MX, 2.05, cw, 4.65, [[(head + ": ", {"bold": True, "color": INK}), (body, {})] for head, body in limits],
         size=15, space_after=13)
    x = MX + cw + 0.45
    w = SLIDE_W - MX - x
    card(slide, x, 1.55, w, 4.6)
    text(slide, x + 0.3, 1.7, w - 0.6, 0.4, ["Возможные улучшения"], size=18, bold=True, color=INK)
    chip(slide, x + 0.3, 2.18, 3.2, "не реализованы, не обязательны", fill=WHITE, color=INK, size=12, line=INK)
    text(slide, x + 0.3, 2.75, w - 0.6, 3.9, [
        "Другие длины контекста и режимы обучения трансформера",
        "Больше дикторов и спонтанная речь в голосовом тесте",
        "Доверительные интервалы и тесты значимости",
        "Класс noise для нерелевантных сообщений (предложение преподавателя)",
        "TTS-аугментация train (вариант из задания)",
        "Проверка сарказма и чужой критики (задача #44)",
    ], size=14, bullets=True, bullet_color=INK, space_after=7)
    footer(slide, n, total, "Источник: итоговый отчёт T17, раздел 11; docs/team_plan.md, раздел 9")
    return ("Честно назвать ограничения. noise и TTS — только перспективы, не реализованные функции и не "
            "условие сдачи.")


def s_backup_matrices(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Резерв: матрицы ошибок на текстовом test (T13)")
    classes = ("negative", "neutral", "positive")
    cw = (CW - 0.6) / 2
    for i, (name, model) in enumerate(((BASELINE_NAME, f["base"]), (RUBERT_NAME, f["rubert"]))):
        x = MX + i * (cw + 0.6)
        cm = model["metrics"]["confusion_matrix"]
        text(slide, x, 1.6, cw, 0.4, [name], size=18, bold=True, color=INK)
        rows = [["эталон \\ ответ", *classes]] + [[c, *(num(v) for v in row)] for c, row in zip(classes, cm)]
        table(slide, x, 2.1, cw, rows, [cw * 0.31, cw * 0.23, cw * 0.23, cw * 0.23], row_h=0.6, size=15,
              bold_first_col=True, highlight_rows=(2,), name=f"Матрица {name}")
    text(slide, MX, 4.75, CW, 1.6, [
        "Строки — эталон, столбцы — ответ модели; строка neutral выделена.",
        "Чувствительность к 7 группам близких повторов в test: macro-F1 меняется на +0,00006 у baseline "
        "и на +0,00013 у RuBERT.",
        "Усечение RuBERT до 64 токенов затронуло 3 152 записи (33,8 %); влияние на качество не измерялось.",
    ], size=14, bullets=True, bullet_color=INK, space_after=7)
    footer(slide, n, total, "Источник: reports/evaluation/final_metrics.json, confusion_matrix_*.csv (T13)")
    return "Резервный слайд для вопросов о распределении ошибок."


def s_backup_resources(slide, f, n, total):
    white_background(slide)
    set_title(slide, "Резерв: время, ресурсы и воспроизводимость")
    bt, rt = f["base"]["timing"], f["rubert"]["timing"]
    asr = f["voice"]["timing"]["asr_seconds"]
    rows = [["Измерение", "Значение"],
            ["Классификация test, мс/текст", f"{BASELINE_NAME}: {dec(bt['ms_per_text'], 3)} · {RUBERT_NAME}: "
                                             f"{dec(rt['ms_per_text'], 3)}"],
            ["Whisper на запись 8–16 с", f"медиана {dec(asr['median'], 3)} с, p95 {dec(asr['p95'], 3)} с"],
            ["Память baseline + Whisper", "пиковый RSS 898,3 MiB при загрузке"],
            ["Машина измерений", "Apple M1, 16 GiB, CPU, macOS 26.5.2, Python 3.13.7"]]
    table(slide, MX, 1.6, 7.6, rows, [3.0, 4.6], row_h=0.55, size=14, bold_first_col=True, name="Время и ресурсы")
    text(slide, MX, 4.55, 7.6, 1.6, [
        "Время — один прогон под фоновой нагрузкой, ориентировочно; задержка Telegram не измерялась.",
        "GPU не нужен для бота, T13 и T14.",
    ], size=14, bullets=True, bullet_color=INK, space_after=7)
    x = MX + 8.0
    w = SLIDE_W - MX - x
    card(slide, x, 1.6, w, 5.1)
    text(slide, x + 0.3, 1.75, w - 0.6, 0.4, ["Как проверить результаты"], size=17, bold=True, color=INK)
    text(slide, x + 0.3, 2.25, w - 0.6, 4.3, [
        "Фиксация T12: reports/freeze.json, протокол t12/1",
        "T13: reports/evaluation/ — метрики, предсказания, суммы",
        "T14: reports/voice_eval/ — агрегаты без предсказаний",
        "T15: reports/error_analysis/",
        "Сверка чисел отчёта: reports/final/preparation_checks.json",
    ], size=14, bullets=True, bullet_color=INK, space_after=8)
    footer(slide, n, total, "Источник: final_metrics.json (T13), voice_eval/metrics.json (T14), reports/freeze.md (T12)")
    return "Резервный слайд для вопросов о скорости, ресурсах и воспроизводимости."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="путь к .pptx")
    args = parser.parse_args(argv)
    path = build(load_facts(), args.out)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
