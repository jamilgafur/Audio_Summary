# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
from pathlib import Path

import pymupdf  # installed as `pymupdf`, imported as `pymupdf`; fixture generation uses it too
import pytest

from extract import ScannedPDFError, extract

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_extract_real_paper():
    paper = extract(FIXTURE)
    assert len(paper["title"]) > 5
    total = sum(len(s["text"]) for s in paper["sections"])
    assert total > 20_000, f"suspiciously little text: {total} chars"
    assert len(paper["sections"]) >= 2
    # references section must be cut: no run of bibliography-looking lines at the end
    tail = paper["sections"][-1]["text"][-2000:]
    assert "https://doi.org" not in tail.lower()


def _make_pdf(path, lines):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), "\n".join(lines), fontsize=11)
    doc.save(path)


PROSE = (
    "We study how readers listen to papers aloud and why clear narration "
    "helps the comprehension of complex technical material. " * 6
)


def test_headings_split_into_sections(tmp_path):
    """Papers without a PDF outline must still be split on section headings."""
    p = tmp_path / "structured.pdf"
    _make_pdf(p, ["Introduction", PROSE, "Methods", PROSE, "Conclusion", PROSE])
    paper = extract(p)
    titles = [s["title"].lower() for s in paper["sections"]]
    assert len(paper["sections"]) >= 3, titles
    assert any("introduction" in t for t in titles)
    assert any("conclusion" in t for t in titles)


def test_table_rows_stripped(tmp_path):
    """Digit-heavy table rows must not survive into the narration text."""
    table_row = "12.34 56.78 90.12 34.56 78.90 12.34 45.67"
    p = tmp_path / "tabular.pdf"
    _make_pdf(p, [PROSE, table_row, table_row, table_row, PROSE])
    paper = extract(p)
    whole = "\n".join(s["text"] for s in paper["sections"])
    assert "56.78" not in whole and "45.67" not in whole


def test_scanned_pdf_rejected(tmp_path):
    src = pymupdf.open(FIXTURE)
    pix = src[0].get_pixmap(dpi=50)  # rasterize: pixels only, no text layer
    doc = pymupdf.open()
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    p = tmp_path / "scanned.pdf"
    doc.save(p)
    with pytest.raises(ScannedPDFError):
        extract(p)


# ---- plain-text sources -------------------------------------------------


def test_txt_extracts_first_line_as_title(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("How Readers Hear Papers\n\n" + PROSE)
    paper = extract(p)
    assert paper["title"] == "How Readers Hear Papers"
    assert paper["sections"][0]["title"] == "full text"
    assert PROSE[:40] in paper["sections"][0]["text"]


def test_tex_strips_commands_and_splits_sections(tmp_path):
    body = PROSE.replace("\n", " ")
    p = tmp_path / "paper.tex"
    p.write_text(
        "\\documentclass{article}\n\\title{A TeX Study\\\\With More}\n"
        "\\author{Ada Lovelace \\and Alan Turing}\n\\begin{document}\n\\maketitle\n"
        f"\\section{{Introduction}}\n{body} \\cite{{doe2024}} $E=mc^2$ see Figure~\\ref{{fig:x}}.\n"
        "% a comment that must never be read\n"
        f"\\section{{Methods}}\n{body} \\textbf{{important}} finding.\n"
        "\\section{References}\n\\bibliography{refs}\n\\end{document}\n"
    )
    paper = extract(p)
    assert paper["title"] == "A TeX Study"  # \\ line-break cut
    assert "Ada Lovelace" in paper["authors"]
    titles = [s["title"] for s in paper["sections"]]
    assert titles == ["Introduction", "Methods"], titles  # references dropped by _finish
    whole = " ".join(s["text"] for s in paper["sections"])
    for junk in ("\\cite", "doe2024", "E=mc", "\\ref", "fig:x", "textbf", "a comment"):
        assert junk not in whole
    assert "important" in whole


def test_docx_extracts_paragraph_text(tmp_path):
    import zipfile

    p = tmp_path / "notes.docx"
    paras = "".join(
        f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in ["A Word Document", PROSE[:360], PROSE[360:]]
    )
    xml = (
        '<?xml version="1.0"?><w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paras}</w:body></w:document>"
    )
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", xml)
    paper = extract(p)
    assert paper["title"] == "A Word Document"
    assert PROSE[:40] in paper["sections"][0]["text"]


def test_short_txt_rejected(tmp_path):
    p = tmp_path / "tiny.txt"
    p.write_text("too short.")
    with pytest.raises(ScannedPDFError):
        extract(p)
