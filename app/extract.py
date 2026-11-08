# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
"""Extract clean, narration-friendly text from a paper: PDF, .txt, .tex, or .docx."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pymupdf

SRC_EXT = {".pdf", ".txt", ".tex", ".docx"}

# shortcut: references/bibliography cut and citation stripping are heuristics;
# upgrade to GROBID if users feed papers with exotic layouts.
# shortcut: references header can be glued to a column-neighbor line by PDF text
# extraction; accepts own-line or after a big space gap. Upgrade to GROBID if
# users hit layouts this misses.
_REF_HEADER = re.compile(r"(?:^\s*|\S\s{8,})(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
_CITATION_NUM = re.compile(r"\[[\d,\s\-–]+\]")  # [12], [3-5], [1, 2, 7]
_CITATION_AUTHOR = re.compile(r"\([A-Z][A-Za-z\-]+(?: et al\.| and [A-Z][A-Za-z\-]+)?,? \d{4}[a-z]?\)")
_FIGURE_REF = re.compile(r"\b(?:Fig(?:ure)?|Table|Equation|Eq\.)\s*\d+[\w.]*", re.IGNORECASE)
_PAGE_NOISE = re.compile(r"^\s*(\d+|page\s+\d+(\s+of\s+\d+)?)\s*$", re.IGNORECASE)
_MOSTLY_SYMBOLS = re.compile(r"^[^a-zA-Z]*$")

MIN_TEXT_CHARS = 500  # below this the PDF is almost certainly scanned images


class ScannedPDFError(RuntimeError):
    """Raised when a PDF has no usable text layer (scanned images)."""


def _is_heading(line: str) -> bool:
    words = line.split()
    return bool(line) and len(line) < 80 and len(words) <= 10 and not line.endswith(".")


def _clean_lines(text: str) -> str:
    kept = []
    prev_line = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            kept.append("")
            continue
        if _PAGE_NOISE.match(line):
            continue
        if _MOSTLY_SYMBOLS.match(line) and not _is_heading(line):
            continue  # equation / table-junk line
        if line == prev_line and _is_heading(line):
            continue  # repeated running header
        # drop lines that are >60% digits (table rows)
        digits = sum(c.isdigit() for c in line)
        if len(line) > 8 and digits / len(line) > 0.6:
            continue
        line = _FIGURE_REF.sub("", line)
        line = _CITATION_AUTHOR.sub("", line)
        line = _CITATION_NUM.sub("", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        kept.append(line)
        prev_line = line
    return "\n".join(kept)


def extract(path: str | Path) -> dict:
    """Return {title, authors, sections: [{title, text}]} from any supported source."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix not in SRC_EXT:
        raise ValueError(f"unsupported file type: {suffix}")
    if suffix == ".txt":
        paper = _extract_txt(path)
    elif suffix == ".tex":
        paper = _extract_tex(path)
    else:
        paper = _extract_docx(path)
    paper["sections"] = _finish(paper["sections"])
    if len("\n".join(s["text"] for s in paper["sections"])) < MIN_TEXT_CHARS:
        raise ScannedPDFError("Not enough readable text in this file.")
    return paper


def _extract_pdf(path: Path) -> dict:
    doc = pymupdf.open(str(path))
    try:
        pages = [page.get_text(sort=True) for page in doc]
        title = (doc.metadata or {}).get("title") or Path(path).stem.replace("_", " ")
        authors = (doc.metadata or {}).get("author", "").strip()

        # Prefer the PDF's own outline when present; else headings heuristics.
        sections = _sections_from_toc(doc, pages) or _sections_from_headings(pages)
        sections = _finish(sections)
    finally:
        doc.close()

    whole = "\n".join(s["text"] for s in sections)
    if len(whole) < MIN_TEXT_CHARS:
        raw = _clean_lines("\n".join(pages))
        if len(raw) >= MIN_TEXT_CHARS:  # heuristics failed; fall back to full text
            sections = [{"title": "full text", "text": raw}]
            whole = raw
        else:
            raise ScannedPDFError("This PDF has no text layer (scanned images). OCR is not supported.")
    return {"title": title.strip(), "authors": authors, "sections": sections}


def _sections_from_toc(doc, pages: list[str]) -> list[dict]:
    toc = doc.get_toc(simple=True)
    level1 = [(title, page - 1) for level, title, page in toc if level == 1]
    if len(level1) < 2:
        return []
    sections = []
    for i, (title, start) in enumerate(level1):
        end = level1[i + 1][1] if i + 1 < len(level1) else len(pages)
        text = "\n".join(pages[start:end])
        sections.append({"title": title, "text": text})
    return sections


def _sections_from_headings(pages: list[str]) -> list[dict]:
    numbered = re.compile(
        r"^\s*(\d+\.?\s+)?(abstract|introduction|background|related work|"
        r"method(?:s|ology)?|approach|experiments?|results?|evaluation|"
        r"discussion|conclusion[s]?|acknowledg?ements?)\s*$",
        re.IGNORECASE,
    )
    sections: list[dict] = []
    current: dict | None = None
    for page in pages:
        paragraph: list[str] = []
        for line in page.splitlines() + [""]:
            clean = line.strip()
            if clean and numbered.match(clean):
                if current is not None:
                    current["text"] += "\n".join(paragraph)
                    sections.append(current)
                current = {"title": clean, "text": ""}
                paragraph = []
                continue
            if current is None:
                current = {"title": "opening", "text": ""}
            paragraph.append(clean)
            if not clean:  # paragraph break
                current["text"] += "\n".join(paragraph) + "\n\n"
                paragraph = []
    if current is not None:
        current["text"] += "\n".join(paragraph)
        sections.append(current)
    return sections


def _finish(sections: list[dict]) -> list[dict]:
    """Shared post-processing: cut references/appendix on raw text, then clean lines."""
    trimmed = []
    for s in sections:
        if _REF_HEADER.match(s["title"].strip()):
            break  # references/bibliography section: drop it and everything after
        m = _REF_HEADER.search(s["text"])
        if m:
            trimmed.append({"title": s["title"], "text": s["text"][: m.start()]})
            break  # everything after references is appendix noise
        trimmed.append(s)
    cleaned = [{"title": s["title"], "text": _clean_lines(s["text"])} for s in trimmed]
    return [s for s in cleaned if len(s["text"].strip()) > 50]


# ---- plain-text sources -------------------------------------------------


def _opening_title(first_line: str) -> str:
    return first_line if first_line and len(first_line) < 120 and not first_line.endswith(".") else ""


def _extract_txt(path: Path) -> dict:
    text = path.read_text(errors="replace")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = _opening_title(first)
    body = text[len(first) :] if title else text
    return {
        "title": title or path.stem.replace("_", " "),
        "authors": "",
        "sections": [{"title": "full text", "text": body}],
    }


_DOCX_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx(path: Path) -> dict:
    """A .docx is a zip of XML; pull paragraph text out of word/document.xml."""
    try:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError) as e:
        raise ScannedPDFError(f"Unreadable .docx: {e}")
    paras = ["".join(t.text or "" for t in p.iter(_DOCX_W + "t")) for p in root.iter(_DOCX_W + "p")]
    title = _opening_title(next((p.strip() for p in paras if p.strip()), ""))
    return {
        "title": title or path.stem.replace("_", " "),
        "authors": "",
        "sections": [{"title": "full text", "text": "\n".join(paras)}],
    }


# shortcut: .tex stripping is a heuristic — no macro expansion, no \input/\include
# following. A paper split across files only narrates the main file.
_BRACED = r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"  # one level of nesting


def _tex_arg(cmd: str, text: str) -> str:
    m = re.search(r"\\" + cmd + _BRACED, text)
    return m.group(1) if m else ""


def _tex_strip(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)  # comments
    t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", t, flags=re.S)  # math
    t = re.sub(
        r"\\begin\{(?:equation|align|figure|table|tabular)[^}]*\}.*?\\end\{(?:equation|align|figure|table|tabular)[^}]*\}",
        " ",
        t,
        flags=re.S,
    )  # floats & math envs
    t = re.sub(
        r"\\(?:cite[a-z]*|ref|pageref|label|footnote|url|href|bibliography(?:style)?)"
        rf"(?:\[[^\]]*\])?{_BRACED}(?:{_BRACED})?",
        " ",
        t,
    )  # refs/citations
    t = re.sub(r"\\(?:sub)*section\*?" + _BRACED, r"\1. ", t)  # keep headings as text
    t = re.sub(r"\\(?:text|emph|textbf|textit|texttt|textrm|underline|title)" + _BRACED, r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", t)  # remaining commands
    return re.sub(r"[{}$&]+", " ", t)


def _extract_tex(path: Path) -> dict:
    raw = path.read_text(errors="replace")
    title = re.sub(r"\\\\.*", "", _tex_arg("title", raw)).strip()  # line-break \\ in \title{...}
    authors = re.sub(r"\\and", ", ", _tex_arg("author", raw)).strip()
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", raw, flags=re.S)
    body = m.group(1) if m else raw
    chunks = re.split(r"\\section\*?\{([^}]*)\}", body)
    if len(chunks) < 3:
        sections = [{"title": "full text", "text": _tex_strip(body)}]
    else:
        sections = []
        if len(_tex_strip(chunks[0])) > 200:
            sections.append({"title": "opening", "text": _tex_strip(chunks[0])})
        for i in range(1, len(chunks) - 1, 2):
            sections.append({"title": _tex_strip(chunks[i]), "text": _tex_strip(chunks[i + 1])})
    return {"title": title, "authors": authors, "sections": sections}
