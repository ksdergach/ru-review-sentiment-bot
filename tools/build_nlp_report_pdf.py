"""Render the short NLP report from its Markdown source.

Optional document dependency: reportlab (not needed by the bot).
Usage: python tools/build_nlp_report_pdf.py --font-dir /path/to/DejaVu/fonts
The directory must contain DejaVuSans.ttf and DejaVuSans-Bold.ttf.
"""
from __future__ import annotations
import argparse
import re
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]

def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<link href="\2" color="#156677"><u>\1</u></link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font color="#40556A">\1</font>', text)

def build(font_dir: Path, output: Path) -> None:
    for name, file in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    styles = {
        'p': ParagraphStyle('p',fontName='Body',fontSize=9.2,leading=13.1,spaceAfter=7,textColor=colors.HexColor('#223343')),
        'h1': ParagraphStyle('h1',fontName='BodyBold',fontSize=21,leading=26,spaceAfter=14,textColor=colors.HexColor('#123F50')),
        'h2': ParagraphStyle('h2',fontName='BodyBold',fontSize=12.6,leading=17,spaceBefore=8,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#123F50')),
        'cell': ParagraphStyle('cell',fontName='Body',fontSize=8.3,leading=11),
        'code': ParagraphStyle('code',fontName='Body',fontSize=8.6,leading=13,backColor=colors.HexColor('#EDF3F6'),borderPadding=7,spaceAfter=12),
    }
    lines=(ROOT/'reports/nlp/report.md').read_text(encoding='utf-8').splitlines()
    story=[]; i=0
    while i < len(lines):
        line=lines[i];i+=1
        if not line.strip(): continue
        if line=='<!-- pagebreak -->': story.append(PageBreak());continue
        if line.startswith('```'):
            block=[]
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(escape(lines[i]));i+=1
            i+=1;story.append(Paragraph('<br/>'.join(block),styles['code']));continue
        if line.startswith('|'):
            raw=[line]
            while i<len(lines) and lines[i].startswith('|'):raw.append(lines[i]);i+=1
            rows=[[c.strip() for c in r.strip('|').split('|')] for r in raw if not re.fullmatch(r'[| :\-]+',r)]
            cells=[[Paragraph(inline('**'+c+'**' if j==0 else c),styles['cell']) for c in row] for j,row in enumerate(rows)]
            available=A4[0]-88
            widths=([90]+[(available-90)/(len(rows[0])-1)]*(len(rows[0])-1))
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E2EDF0')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8AA5AE')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F9')])]))
            story.extend([table,Spacer(1,9)]);continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['h1']));continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['h2']));continue
        if line.startswith('- '):
            story.append(Paragraph(inline(line[2:]),styles['p'],bulletText='•'));continue
        story.append(Paragraph(inline(line),styles['p']))
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas, doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#BACBD1'));canvas.line(44,38,A4[0]-44,38)
        canvas.setFont('Body',7.5);canvas.setFillColor(colors.HexColor('#536D79'))
        canvas.drawString(44,25,'NLP · Киноотзывы · nlp-v1.1 · 09.10.2026')
        canvas.drawRightString(A4[0]-44,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=37,bottomMargin=51,title='Анализ тональности киноотзывов - отчёт NLP',author='Команда ru-review-sentiment-bot',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/nlp-report.pdf')
    args=parser.parse_args();build(args.font_dir,args.output)
