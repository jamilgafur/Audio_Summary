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
    "helps the comprehension of complex technical material. "
    * 6
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
