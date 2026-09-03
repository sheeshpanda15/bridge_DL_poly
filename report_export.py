from __future__ import annotations

import shutil
import subprocess
from html import escape
from io import BytesIO
from pathlib import Path


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _iter_docx_blocks(doc):
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield DocxTable(child, doc)


def _paragraph_images(paragraph):
    from docx.oxml.ns import qn

    images = []
    for run in paragraph.runs:
        for blip in run._element.xpath(".//a:blip"):
            rel_id = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
            if rel_id and rel_id in paragraph.part.related_parts:
                images.append(paragraph.part.related_parts[rel_id].blob)
    return images


def _simple_docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    from docx import Document
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.pdfmetrics import registerFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    doc = Document(docx_path)
    full_text = "\n".join(p.text for p in doc.paragraphs)
    if _contains_cjk(full_text):
        registerFont(UnicodeCIDFont("STSong-Light"))
        body_font = "STSong-Light"
    else:
        body_font = "Helvetica"

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=10.5,
        leading=13.5,
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "ReportTitle",
        parent=body,
        fontSize=20,
        leading=24,
        spaceAfter=8,
        textColor=colors.HexColor("#0B2545"),
    )
    heading1 = ParagraphStyle(
        "ReportHeading1",
        parent=body,
        fontSize=15,
        leading=19,
        spaceBefore=14,
        spaceAfter=7,
        textColor=colors.HexColor("#2E74B5"),
    )
    heading2 = ParagraphStyle(
        "ReportHeading2",
        parent=body,
        fontSize=12.5,
        leading=16,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor("#1F4D78"),
    )
    caption = ParagraphStyle(
        "ReportCaption",
        parent=body,
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=body,
        fontSize=8.3,
        leading=10.5,
        spaceAfter=0,
    )

    story = []
    max_width = 6.5 * inch
    for block in _iter_docx_blocks(doc):
        if hasattr(block, "runs"):
            text = block.text.strip()
            for image_blob in _paragraph_images(block):
                image_data = BytesIO(image_blob)
                with PILImage.open(BytesIO(image_blob)) as pil_image:
                    width_px, height_px = pil_image.size
                width = min(max_width, 6.4 * inch)
                height = width * height_px / width_px
                story.append(Image(image_data, width=width, height=height))
                story.append(Spacer(1, 6))
            if not text:
                continue
            style_name = getattr(block.style, "name", "")
            if style_name == "Title":
                style = title
            elif style_name.startswith("Heading 1"):
                style = heading1
            elif style_name.startswith("Heading"):
                style = heading2
            elif text.startswith(("Figure ", "图 ")):
                style = caption
            else:
                style = body
            story.append(Paragraph(escape(text).replace("\n", "<br/>"), style))
        else:
            rows = []
            for row in block.rows:
                rows.append(
                    [
                        Paragraph(escape(cell.text.strip()).replace("\n", "<br/>"), cell_style)
                        for cell in row.cells
                    ]
                )
            if not rows:
                continue
            col_count = max(len(row) for row in rows)
            col_width = max_width / col_count
            table = Table(rows, colWidths=[col_width] * col_count, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), body_font),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                        ("LEADING", (0, 0), (-1, -1), 10.5),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F4D78")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3DF")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

    SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    ).build(story)


def export_docx_to_pdf(docx_path: Path) -> Path:
    """Export a DOCX to PDF using LibreOffice when available, otherwise Word COM."""
    docx_path = Path(docx_path).resolve()
    pdf_path = docx_path.with_suffix(".pdf")

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(pdf_path.parent),
                str(docx_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        command = r"""
& {
    param([string]$DocxPath, [string]$PdfPath)
    $ErrorActionPreference = 'Stop'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    try {
        $doc = $word.Documents.Open($DocxPath, $false, $true)
        try {
            $doc.ExportAsFixedFormat($PdfPath, 17)
        }
        finally {
            $doc.Close($false)
        }
    }
    finally {
        $word.Quit()
    }
}
"""
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                    "-DocxPath",
                    str(docx_path),
                    "-PdfPath",
                    str(pdf_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            _simple_docx_to_pdf(docx_path, pdf_path)

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"PDF export failed: {pdf_path}")
    return pdf_path
