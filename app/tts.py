"""Narration script -> MP3 via Kokoro TTS (CPU, onnxruntime) + lameenc."""
from __future__ import annotations

import os
import re
import shutil
import urllib.request
from pathlib import Path

import lameenc
import numpy as np

_SAMPLE_RATE = 24000
_MODEL_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
# shortcut: fp32 model (~325MB). Swap in kokoro-v1.0.fp16.onnx if RAM is ever tight.

_KOKORO = None


def _models_dir() -> Path:
    d = Path(os.getenv("MODELS_DIR", "./models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_files() -> tuple[Path, Path]:
    paths = {}
    models = _models_dir()
    for name, url in _MODEL_FILES.items():
        dest = models / name
        if not dest.exists():
            print(f"downloading {name} ...", flush=True)
            tmp = dest.with_suffix(".tmp")  # partial download must not pose as the model
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)  # urlretrieve has no timeout: a stall would hang forever
            os.replace(tmp, dest)
        paths[name] = dest
    return paths["kokoro-v1.0.onnx"], paths["voices-v1.0.bin"]


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


def synthesize(script: str, out_path: str | Path, progress=None, voice: str | None = None) -> float:
    """Render the script to an MP3 at out_path. Returns audio duration in seconds."""
    kokoro = _load()
    voice = voice or os.getenv("VOICE", "af_heart")
    speed = float(os.getenv("SPEED", "1.1"))

    pieces = _pieces(script)
    chunks = []
    for i, piece in enumerate(pieces):
        samples, _sr = kokoro.create(piece, voice=voice, speed=speed, lang="en-us")
        chunks.append(samples)
        if progress:
            progress(i + 1, len(pieces))

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
