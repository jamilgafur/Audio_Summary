"""FastAPI app: folder watcher + pipeline worker + tiny web UI."""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from extract import ScannedPDFError, extract
from narrate import narrate
from tts import synthesize

INPUT_DIR = Path(os.getenv("INPUT_DIR", "./input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

_SAFE_STEM = re.compile(r"[^\w\-. ]+")

_STAGE_PCT = {"queued": 0, "extracting": 5, "narrating": 10, "rendering audio": 75, "done": 100}


def _safe(name: str) -> str:
    stem = _SAFE_STEM.sub("", Path(name).stem).strip()
    if not stem:
        raise HTTPException(400, "invalid filename")
    return stem


class Pipeline:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.q: queue.Queue[Path] = queue.Queue()
        for d in (input_dir, output_dir):
            d.mkdir(parents=True, exist_ok=True)

    # paths -----------------------------------------------------------
    def mp3(self, stem: str) -> Path:
        return self.output_dir / f"{stem}.mp3"

    def script(self, stem: str) -> Path:
        return self.output_dir / f"{stem}.script.txt"

    def status_path(self, stem: str) -> Path:
        return self.output_dir / f"{stem}.status.json"

    # status ----------------------------------------------------------
    def set_status(self, stem: str, stage: str, **extra):
        data = {
            "stage": stage,
            "progress": _STAGE_PCT.get(stage, 0),
            "updated": time.time(),
            **extra,
        }
        self.status_path(stem).write_text(json.dumps(data))

    # full pipeline ---------------------------------------------------
    def process(self, pdf: Path):
        stem = _safe(pdf.name)
        try:
            self.set_status(stem, "extracting")
            paper = extract(pdf)

            script_path = self.script(stem)
            if script_path.exists():
                script = script_path.read_text()  # resume: LLM step already paid for
                self.set_status(stem, "narrating", detail="cached script")
            else:
                self.set_status(stem, "narrating", detail="starting")
                def on_narrate(i, n):
                    pct = 10 + round(65 * i / n)
                    self.set_status(stem, "narrating", progress=pct, detail=f"part {i}/{n}")
                script = narrate(paper, progress=on_narrate)
                script_path.write_text(script)

            def on_tts(i, n):
                pct = 75 + round(24 * i / n)
                self.set_status(stem, "rendering audio", progress=pct, detail=f"piece {i}/{n}")
            duration = synthesize(script, self.mp3(stem), progress=on_tts)
            self.set_status(stem, "done", duration=round(duration))
        except ScannedPDFError as e:
            self.set_status(stem, "failed", error=str(e))
        except Exception as e:  # keep the worker alive; surface the failure
            traceback.print_exc()
            self.set_status(stem, "failed", error=str(e))

    # threads ---------------------------------------------------------
    def worker(self):
        while True:
            self.process(self.q.get())

    def watcher(self):
        while True:
            try:
                self.scan()
            except Exception:
                traceback.print_exc()
            time.sleep(5)

    def scan(self):
        for pdf in sorted(self.input_dir.glob("*.pdf")):
            stem = _safe(pdf.name)
            if self.mp3(stem).exists() or self.status_path(stem).exists():
                continue
            self.set_status(stem, "queued")
            self.q.put(pdf)

    def jobs(self) -> list[dict]:
        seen = {}
        for f in sorted(self.output_dir.glob("*.status.json"), key=os.path.getmtime, reverse=True):
            stem = f.stem[: -len(".status")]
            try:
                seen[stem] = {"name": stem, **json.loads(f.read_text())}
            except json.JSONDecodeError:
                continue
        for mp3 in self.output_dir.glob("*.mp3"):
            stem = mp3.stem
            if stem not in seen:
                seen[stem] = {"name": stem, "stage": "done", "progress": 100}
        for pdf in self.input_dir.glob("*.pdf"):
            stem = pdf.stem
            seen.setdefault(stem, {"name": stem, "stage": "queued", "progress": 0})
        return list(seen.values())


pipeline = Pipeline(INPUT_DIR, OUTPUT_DIR)


@asynccontextmanager
async def lifespan(_app):
    for target in (pipeline.worker, pipeline.watcher):
        threading.Thread(target=target, daemon=True).start()
    pipeline.scan()
    yield


app = FastAPI(lifespan=lifespan)
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (_STATIC / "index.html").read_text()


@app.get("/api/jobs")
def jobs():
    return pipeline.jobs()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: PDFs only")
        dest = pipeline.input_dir / f"{_safe(f.filename)}.pdf"
        tmp = dest.with_suffix(".tmp")  # atomic: watcher only globs *.pdf
        tmp.write_bytes(await f.read())
        os.replace(tmp, dest)
        saved.append(dest.name)
    pipeline.scan()
    return {"saved": saved}


@app.get("/api/audio/{name}")
def audio(name: str):
    mp3 = pipeline.mp3(_safe(name))
    if not mp3.exists():
        raise HTTPException(404, "not found")
    return FileResponse(mp3, media_type="audio/mpeg")


@app.get("/api/script/{name}", response_class=PlainTextResponse)
def script(name: str):
    s = pipeline.script(_safe(name))
    if not s.exists():
        raise HTTPException(404, "not found")
    return s.read_text()


@app.post("/api/delete/{name}")
def delete(name: str):
    stem = _safe(name)
    removed = 0
    for p in (
        pipeline.input_dir / f"{stem}.pdf",
        pipeline.mp3(stem),
        pipeline.script(stem),
        pipeline.status_path(stem),
    ):
        if p.exists():
            p.unlink()
            removed += 1
    return {"removed": removed}
