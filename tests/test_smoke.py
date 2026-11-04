"""End-to-end smoke test: PDF -> (mocked LLM) -> real Kokoro TTS -> real MP3."""
import json
import os

os.environ["NARRATE_MOCK"] = "1"  # must precede app imports

import pymupdf
import pytest

from main import Pipeline

TEXT = (
    "We study the problem of reading research papers aloud. "
    "Our method combines text extraction, a language model, and neural speech synthesis. "
    "Experiments show that listeners prefer narration over verbatim equation reading. "
) * 15  # > MIN_TEXT_CHARS so it doesn't trip the scanned-PDF guard


@pytest.fixture(scope="module")
def mp3(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    pdf = tmp / "tinypaper.pdf"
    doc = pymupdf.open()
    for i in range(0, len(TEXT), 1800):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), TEXT[i : i + 1800], fontsize=11)
    doc.save(pdf)

    pipe = Pipeline(tmp / "in", tmp / "out")
    pdf.rename(pipe.input_dir / pdf.name)
    pipe.process(pipe.input_dir / "tinypaper.pdf")
    return pipe, tmp


def test_mp3_produced(mp3):
    pipe, _ = mp3
    out = pipe.mp3("tinypaper")
    assert out.exists() and out.stat().st_size > 10_000
    head = out.read_bytes()[:3]
    is_mp3 = head == b"ID3" or (head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
    assert is_mp3, f"not an MP3 (header {head!r})"


def test_progression_and_duration(mp3):
    pipe, _ = mp3
    st = json.loads(pipe.status_path("tinypaper").read_text())
    assert st["stage"] == "done"
    assert st["progress"] == 100
    assert st["duration"] > 5, "audio is far too short to be the script"


def test_script_cached(mp3):
    pipe, _ = mp3
    assert "mock narration" in pipe.script("tinypaper").read_text()
