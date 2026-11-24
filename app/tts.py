# BUSL-1.1 — see LICENSE (change date 2030-11-07 -> Apache-2.0)
"""Narration script -> MP3 via Kokoro TTS (CPU, onnxruntime) + lameenc."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.request
from pathlib import Path

import lameenc
import numpy as np

_SAMPLE_RATE = 24000
_MODEL_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
_MODEL_URLS = {
    "kokoro-v1.0.onnx": _MODEL_BASE + "kokoro-v1.0.onnx",
    "kokoro-v1.0.fp16.onnx": _MODEL_BASE + "kokoro-v1.0.fp16.onnx",
}
_VOICES_URL = _MODEL_BASE + "voices-v1.0.bin"


def _model_name() -> str:
    """KOKORO_MODEL picks the onnx; fp16 halves RAM/bandwidth (bench before switching)."""
    name = os.getenv("KOKORO_MODEL", "kokoro-v1.0.onnx")
    if name not in _MODEL_URLS:
        raise ValueError(f"unknown KOKORO_MODEL: {name!r} (choices: {', '.join(_MODEL_URLS)})")
    return name


_KOKORO = None


def _models_dir() -> Path:
    d = Path(os.getenv("MODELS_DIR", "./models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_files() -> tuple[Path, Path]:
    paths = {}
    models = _models_dir()
    for name, url in [(_model_name(), _MODEL_URLS[_model_name()]), ("voices-v1.0.bin", _VOICES_URL)]:
        dest = models / name
        if not dest.exists():
            print(f"downloading {name} ...", flush=True)
            tmp = dest.with_suffix(".tmp")  # partial download must not pose as the model
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)  # urlretrieve has no timeout: a stall would hang forever
            os.replace(tmp, dest)
        paths[name] = dest
    return paths[_model_name()], paths["voices-v1.0.bin"]


def _load():
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro

        model, voices = _ensure_files()
        _KOKORO = Kokoro(str(model), str(voices))
    return _KOKORO


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, max_chars: int = 1200) -> list[str]:
    """Split script into TTS-sized pieces at paragraph, then sentence, boundaries.
    A sentence longer than the limit is hard-split — Kokoro must never get one huge ball."""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        current = ""
        for sentence in _SENTENCE.split(para):
            while len(sentence) > max_chars:
                pieces.append(current or sentence[:max_chars])
                current, sentence = "", sentence[max_chars:]
            if len(current) + len(sentence) > max_chars and current:
                pieces.append(current)
                current = ""
            current = (current + " " + sentence).strip() if current else sentence
        if current:
            pieces.append(current)
    return pieces


# kokoro-onnx crashes at >=510 phonemes: it truncates to MAX_PHONEME_LENGTH=510 and
# then indexes voice[len(tokens)] into a 510-row array. Its own phoneme batcher can
# overshoot (>540 observed), so cap total phonemes per create() ourselves, with margin.
_MAX_PHONEMES = 480


def _render(kokoro, piece: str, voice: str, speed: float):
    """One piece of audio, disk-cached by (voice, speed, text): a re-render after a
    small edit only synthesizes the pieces that changed — unchanged sentences replay
    from cache instantly.
    shortcut: the cache grows unboundedly; wipe <OUTPUT_DIR>/samples/pieces occasionally."""
    f = (
        Path(os.getenv("OUTPUT_DIR", "./output"))
        / "samples"
        / "pieces"
        / (hashlib.sha1(f"{voice}\n{speed}\n{_model_name()}\n{piece}".encode()).hexdigest()[:16] + ".npy")
    )
    if f.exists():
        return np.load(f)
    samples = _render_safe(kokoro, piece, voice, speed)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".npy.tmp")  # crash mid-save must not leave a "valid" cache entry
    with open(tmp, "wb") as w:
        np.save(w, samples)
    os.replace(tmp, f)
    return samples


def _render_safe(kokoro, text: str, voice: str, speed: float):
    """Synthesize under the phoneme cap, splitting at a word boundary when over."""
    if len(kokoro.tokenizer.phonemize(text, "en-us")) <= _MAX_PHONEMES:
        samples, _sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        return samples
    mid = len(text) // 2
    cut = text.rfind(" ", 0, mid)
    if cut <= 0:
        cut = text.find(" ", mid)
    if 0 < cut < len(text) - 1:
        left, right = text[:cut], text[cut + 1 :]  # word boundary: drop the space
    else:
        left, right = text[:mid], text[mid:]  # pathological space-free run: hard split
    return np.concatenate([_render_safe(kokoro, left, voice, speed), _render_safe(kokoro, right, voice, speed)])


def synthesize(
    script: str, out_path: str | Path, progress=None, voice: str | None = None, marks_path: str | Path | None = None
) -> float:
    """Render the script to an MP3 at out_path. Returns audio duration in seconds.
    When marks_path is given, also writes sentence→time marks for the follow-along
    view: exact per-piece durations from the rendered samples, char-proportional
    across the sentences inside each piece (re-syncs every piece, so no drift)."""
    kokoro = _load()
    voice = voice or os.getenv("VOICE", "af_heart")
    speed = float(os.getenv("SPEED", "1.1"))

    pieces = _pieces(script)
    chunks = []
    raw_marks = []  # (script offset, duration) per piece
    t = 0.0
    cursor = 0
    for i, piece in enumerate(pieces):
        samples = _render(kokoro, piece, voice, speed)
        chunks.append(samples)
        dur = len(samples) / _SAMPLE_RATE
        if marks_path is not None:
            start = script.find(piece, cursor)
            if start < 0:  # identical pieces confuse find-from-cursor; fall back
                start = script.find(piece)
            raw_marks.append((start, dur))
            cursor = start + len(piece)
        t += dur
        if progress:
            progress(i + 1, len(pieces))

    if marks_path is not None:
        _write_marks(script, pieces, raw_marks, marks_path)

    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    enc = lameenc.Encoder()
    enc.set_bit_rate(96)
    enc.set_in_sample_rate(_SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(2)
    mp3 = enc.encode(pcm16.tobytes()) + enc.flush()
    out = Path(out_path)
    tmp = out.with_suffix(".tmp")  # crash mid-write must not leave a "finished" MP3 behind
    tmp.write_bytes(mp3)
    os.replace(tmp, out)
    return len(pcm16) / _SAMPLE_RATE  # speed is already baked into the samples


def _write_marks(script: str, pieces: list[str], raw_marks: list[tuple[int, float]], marks_path: str | Path):
    """Sentence-level {s, e, t} marks: piece timing is exact from the samples; the
    split inside a piece assumes uniform speaking rate (Kokoro's is close)."""
    import json

    marks = []
    for piece, (start, dur) in zip(pieces, raw_marks):
        if start < 0:
            continue
        local = 0
        span = len(piece) or 1
        for sent in _SENTENCE.split(piece):
            off = piece.find(sent, local)
            if off < 0:
                off = local
            sd = dur * len(sent) / span
            marks.append({"s": start + off, "e": start + off + len(sent), "t": 0.0, "d": round(sd, 3)})
            local = off + len(sent)
    # cumulative times from durations (piece marks carry exact totals)
    t = 0.0
    for m in marks:
        m["t"] = round(t, 3)
        t += m.pop("d")
    out = Path(marks_path)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"marks": marks, "duration": round(t, 3)}))
    os.replace(tmp, out)
