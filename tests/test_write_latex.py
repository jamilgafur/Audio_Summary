# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
"""The Write tab speaks LaTeX through latexToSpeech() in index.html — every read path
(listen back, read-aloud, send to library) runs the draft through it. Exercise the real
function source with node, the same runtime CI syntax-checks the page with."""

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
    ("\\section{Methods}\nBody text.", "Methods.\nBody text."),  # headings stay as text
    ("Figure~\\ref{fig:x}.", "Figure."),  # ~ is a space
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_write_tab_latex_reads_as_speech():
    src = INDEX.read_text()
    m = re.search(r"^function latexToSpeech.*?^}", src, re.M | re.S)
    assert m, "latexToSpeech not found in index.html"
    js = (
        m.group(0) + "\nconst cases = " + json.dumps(CASES) + ";\nfor (const [inp, want] of cases) {"
        " const got = latexToSpeech(inp);"
        " if (got !== want) {"
        " console.error(JSON.stringify(inp), '->', JSON.stringify(got), 'want', JSON.stringify(want));"
        " process.exit(1); } }\n"
    )
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
