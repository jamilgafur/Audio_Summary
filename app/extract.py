"""Extract clean, narration-friendly text from a research-paper PDF."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

# shortcut: references/bibliography cut and citation stripping are heuristics;
# upgrade to GROBID if users feed papers with exotic layouts.
# shortcut: references header can be glued to a column-neighbor line by PDF text
# extraction; accepts own-line or after a big space gap. Upgrade to GROBID if
# users hit layouts this misses.
_REF_HEADER = re.compile(r"(?:^\s*|\S\s{8,})(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
_CITATION_NUM = re.compile(r"\[[\d,\s\-–]+\]")          # [12], [3-5], [1, 2, 7]
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
    """Return {title, authors, sections: [{title, text}]} from a PDF."""
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
            raise ScannedPDFError(
                "This PDF has no text layer (scanned images). OCR is not supported."
            )
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
    numbered = re.compile(r"^\s*(\d+\.?\s+)?(abstract|introduction|background|related work|"
                          r"method(?:s|ology)?|approach|experiments?|results?|evaluation|"
                          r"discussion|conclusion[s]?|acknowledg?ements?)\s*$", re.IGNORECASE)
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
