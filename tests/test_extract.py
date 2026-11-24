# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
import re
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
    # 2-column pages must read column-by-column, not merged line-by-line
    whole = "\n".join(s["text"] for s in paper["sections"])
    assert "In recent years, researchers have explored data-driven emulation" in whole
    assert "WALRUS employs a space-time factorized transformer architecture" in whole
    assert all(len(line) <= 300 for line in whole.splitlines())  # no cross-column hybrid lines (joins may reach ~250)
    assert not re.search(r"[a-z]-\n[a-z]", whole)  # line-wrap word splits must be joined


def _make_2col_pdf(path, left_blocks, right_blocks, header="A Two Column Study"):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    if header:  # spans both columns like a paper title
        page.insert_textbox(pymupdf.Rect(50, 30, 562, 60), header, fontsize=14, align=1)
    for x0, x1, blocks in ((50, 290, left_blocks), (322, 562, right_blocks)):
        y = 80
        for block in blocks:
            height = 30
            while page.insert_textbox(pymupdf.Rect(x0, y, x1, y + height), block, fontsize=10) < 0:
                height += 30  # insert_textbox is all-or-nothing: retry taller until it fits
            y += height + 10
        assert y < 792, "fixture blocks overflow the page"
    doc.save(path)


def test_two_column_pdf_reads_columns_in_order(tmp_path):
    """IEEE-style 2-column PDF: full left column, then right; no merged lines."""
    left = ["I. INTRODUCTION", "Alphamark " + PROSE, "II. CONCLUSION", "Alphatail " + PROSE]
    right = ["Omegamark " + PROSE, "Omegatail " + PROSE]
    p = tmp_path / "ieee.pdf"
    _make_2col_pdf(p, left, right)
    paper = extract(p)
    titles = [s["title"].lower() for s in paper["sections"]]
    assert any("introduction" in t for t in titles), titles  # roman numeral heading
    assert any("conclusion" in t for t in titles), titles
    whole = "\n".join(s["text"] for s in paper["sections"])
    for line in whole.splitlines():
        assert not ("Alpha" in line and "Omega" in line), f"merged column line: {line}"
    for marker in ("Alphamark", "Alphatail", "Omegamark"):
        assert marker in whole
    assert whole.index("Alphamark") < whole.index("Omegamark")  # left column before right


def _make_pdf(path, lines):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), "\n".join(lines), fontsize=11)
    doc.save(path)


def test_line_wrap_hyphens_joined(tmp_path):
    """Words split at a line wrap must join ("inter-\\nnational" -> "international");
    digit hyphens ("3-year") and non-wraps keep their hyphen."""
    p = tmp_path / "hyphens.pdf"
    _make_pdf(p, ["The inter-", "national team ran a 3-", "year study. " + PROSE, "WAL-", "RUS outperforms."])
    paper = extract(p)
    whole = "\n".join(s["text"] for s in paper["sections"])
    assert "international team ran a 3-" in whole
    assert "WALRUS outperforms" in whole
    assert not re.search(r"[a-z]-\n[a-z]", whole)


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


def test_tex_escaped_specials_read_as_words(tmp_path):
    body = PROSE.replace("\n", " ")
    p = tmp_path / "paper.tex"
    p.write_text(
        "\\begin{document}\n\\section{Introduction}\n"
        + body
        + " Throughput rose 50\\% faster \\& cheaper: hash \\#1, var 2\\_3, cost \\$5, brace \\{x\\}. % comment \\% here\n"
        "\\end{document}\n"
    )
    whole = " ".join(s["text"] for s in extract(p)["sections"])
    assert "50 percent faster and cheaper" in whole
    for word in (" hash 1", " underscore 3", " dollar 5", " open brace x close brace "):
        assert word in whole
    assert "\\%" not in whole and "comment" not in whole  # % line tail is dropped, escapes inside it too


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
