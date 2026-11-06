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


def _make_pdf(path):
    doc = pymupdf.open()
    for i in range(0, len(TEXT), 1800):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 792), TEXT[i : i + 1800], fontsize=11)
    doc.save(path)


@pytest.fixture(scope="module")
def mp3(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    pdf = tmp / "tinypaper.pdf"
    _make_pdf(pdf)

    pipe = Pipeline(tmp / "in", tmp / "out")
    pdf.rename(pipe.input_dir / pdf.name)
    pipe.process(pipe.input_dir / "tinypaper.pdf")
    return pipe, tmp


def test_special_chars_in_filename_are_consumed(tmp_path_factory):
    """A hand-dropped "My Paper (draft).pdf" must be fully consumed at done-time:
    no leftover in input/, artifacts under the title slug."""
    tmp = tmp_path_factory.mktemp("special")
    pdf = tmp / "My Paper (draft).pdf"
    _make_pdf(pdf)

    pipe = Pipeline(tmp / "in", tmp / "out")
    pdf.rename(pipe.input_dir / pdf.name)
    pipe.process(pipe.input_dir / "My Paper (draft).pdf")

    assert list(pipe.input_dir.iterdir()) == []  # real path moved, not a rebuilt name
    assert (pipe.output_dir / "my-paper-draft.pdf").exists()
    assert json.loads(pipe.status_path("my-paper-draft").read_text())["stage"] == "done"


def test_verbatim_mode_skips_the_llm(tmp_path_factory):
    """Verbatim mode: extracted words straight to TTS — the script must be the paper's
    exact text, not the (mocked) LLM narration."""
    tmp = tmp_path_factory.mktemp("verbatim")
    pdf = tmp / "audit.pdf"
    _make_pdf(pdf)

    pipe = Pipeline(tmp / "in", tmp / "out")
    pdf.rename(pipe.input_dir / pdf.name)
    pipe.set_status("audit", "queued", mode="verbatim")
    pipe.process(pipe.input_dir / "audit.pdf")

    script = pipe.script("audit").read_text()
    assert "We study the problem" in script and "Read verbatim" in script
    assert "mock narration" not in script  # proof the LLM path was never entered
    assert pipe.mp3("audit").stat().st_size > 10_000
    assert json.loads(pipe.status_path("audit").read_text())["stage"] == "done"


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


def test_user_rename_survives_reprocess(mp3):
    """A display-title edit must not be clobbered (or turn into a file rename) on re-run."""
    pipe, _ = mp3
    st = json.loads(pipe.status_path("tinypaper").read_text())
    st["title"] = "My Custom Title"
    pipe.status_path("tinypaper").write_text(json.dumps(st))
    # a done job's PDF has moved out of input/ into output/ by now
    pipe.process(pipe.output_dir / "tinypaper.pdf")
    st = json.loads(pipe.status_path("tinypaper").read_text())
    assert st["title"] == "My Custom Title"
    assert pipe.mp3("tinypaper").exists()  # rename is display-only; slug untouched
