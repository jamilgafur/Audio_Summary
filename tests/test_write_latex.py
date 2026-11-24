# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
"""The Write tab speaks LaTeX through latexToSpeech() in index.html — every read path
(listen back, read-aloud, send to library) runs the draft through it — and clicks in the
read view map back to the raw draft via rawOffsetOf(). Exercise the real function sources
with node, the same runtime CI syntax-checks the page with."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).parent.parent / "app" / "static" / "index.html"

# (draft, what gets spoken) — the rules the Write tab promises
CASES = [
    ("We gained 50\\% speed.", "We gained 50 percent speed."),  # \% reads as the word
    ("Keep this % but drop this comment", "Keep this"),  # bare % comments out the line
    ("See \\cite{doe2024} and \\ref{fig:x}.", "See and."),  # citations/refs skipped
    ("\\textbf{bold} text", "bold text"),  # formatting unwraps to its text
    ("intro \\begin{equation}x=y\\end{equation} outro", "intro outro"),  # math env dropped
    ("value $x^2$ is", "value is"),  # inline math dropped
    ("Smith \\& Jones cost \\$5", "Smith and Jones cost dollar 5"),  # escaped specials speak
    ("\\section{Methods}\nBody text.", "Methods.\n\nBody text."),  # headings: own line + a beat
    ("End of text \\cite{a}\n\\section{Methods}\nBody.", "End of text\n\nMethods.\n\nBody."),  # no gluing
    ("\\section[short]{Long Title}\nBody.", "Long Title.\n\nBody."),  # optional [..] arg
    ("\\subsection{Details}\nBody.", "Details.\n\nBody."),  # subsections too
    (
        "\\begin{itemize}\n\\item First item.\n\\item Second item.\n\\end{itemize}",
        "First item.\n\nSecond item.",
    ),  # items: own paragraph
    (
        "\\begin{enumerate}\n\\item Alpha.\n\\item Beta.\n\\end{enumerate}",
        "1. Alpha.\n\n2. Beta.",
    ),  # enumerated speaks numbers
    ("\\begin{itemize}\n\\item[Note] Text here.\n\\end{itemize}", "Note. Text here."),  # custom label speaks
    ("Figure~\\ref{fig:x}.", "Figure."),  # ~ is a space
]

# (raw draft, spoken sentence, expected spot in the draft) — clicking the read view must
# land the editor at the sentence's source, even where speech and source differ
OFFSET_CASES = [
    ("Intro line.\nWe gained 50\\% speed.", "We gained 50 percent speed.", 12),  # \% spoken as the word
    ("See \\cite{a} results here.", "See results here.", 0),  # dropped cite: still the sentence start
    ("Alpha beta gamma.", "percent beta gamma.", 6),  # transformed lead word: fall back inside
    ("Nothing alike here.", "Completely different words.", -1),  # no match: no jump
]


def _fn(name: str) -> str:
    m = re.search(r"^function " + re.escape(name) + r".*?^}", INDEX.read_text(), re.M | re.S)
    assert m, f"{name} not found in index.html"
    return m.group(0)


def _run_node(js: str):
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_write_tab_latex_reads_as_speech():
    _run_node(
        _fn("latexToSpeech") + "\nconst cases = " + json.dumps(CASES) + ";\nfor (const [inp, want] of cases) {"
        " const got = latexToSpeech(inp);"
        " if (got !== want) {"
        " console.error(JSON.stringify(inp), '->', JSON.stringify(got), 'want', JSON.stringify(want));"
        " process.exit(1); } }\n"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_read_view_click_maps_to_draft_spot():
    _run_node(
        _fn("rawOffsetOf")
        + "\nconst cases = "
        + json.dumps(OFFSET_CASES)
        + ";\nfor (const [text, sent, want] of cases) {"
        " const got = rawOffsetOf(text, sent);"
        " if (got !== want) {"
        " console.error(JSON.stringify(sent), '->', got, 'want', want);"
        " process.exit(1); } }\n"
    )
