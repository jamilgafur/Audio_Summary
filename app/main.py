"""FastAPI app: folder watcher + pipeline worker + tiny web UI."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
from contextlib import asynccontextmanager
from email.utils import formatdate
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extract import ScannedPDFError, extract
from narrate import narrate, verbatim
from tts import synthesize

INPUT_DIR = Path(os.getenv("INPUT_DIR", "./input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

_SAFE_STEM = re.compile(r"[^\w\-. ]+")
_SLUG_BAD = re.compile(r"[^a-z0-9]+")
_VOICE = re.compile(r"^[ab][fm]_[a-z]+$")  # kokoro voice names: af_heart, bm_george, ...
_MAX_UPLOAD = 50 * 1024 * 1024  # LAN-open endpoint; cap memory + disk abuse


def _slug(title: str) -> str:
    """'Walrus: A Foundation Model!' -> 'walrus-a-foundation-model' (ascii; accents dropped)."""
    return _SLUG_BAD.sub("-", (title or "").lower()).strip("-")[:60].strip("-")


def _stem_of(status_file: Path) -> str:
    """'walrus.status.json' -> 'walrus'."""
    return status_file.stem[: -len(".status")]

_STAGE_PCT = {"queued": 0, "extracting": 5, "narrating": 10, "rendering audio": 75, "done": 100}
_ACTIVE_STAGES = {"queued", "extracting", "narrating", "rendering audio"}


def _safe(name: str) -> str:
    stem = _SAFE_STEM.sub("", Path(name).stem).strip()
    if not stem:
        raise HTTPException(400, "invalid filename")
    return stem


def _voice_or_400(voice: str) -> str:
    if not _VOICE.match(voice):
        raise HTTPException(400, f"unknown voice: {voice!r}")
    return voice


def _mode_or_400(mode: str) -> str:
    if mode not in ("summary", "verbatim"):
        raise HTTPException(400, f"unknown mode: {mode!r}")
    return mode


def _busy_or_409(stem: str):
    """Archive/delete/rerender while the worker owns the job half-moves its files."""
    stage = pipeline.status_data(stem).get("stage")
    if stage in _ACTIVE_STAGES:
        raise HTTPException(409, f"job is {stage}; wait for it to finish first")


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

    def find_pdf(self, stem: str) -> Path | None:
        """A job's source PDF: input/ while it runs, output/ once done."""
        for d in (self.input_dir, self.output_dir):
            p = d / f"{stem}.pdf"
            if p.exists():
                return p
        return None

    # status ----------------------------------------------------------
    def set_status(self, stem: str, stage: str, **extra):
        """Merge into status.json: title/authors/tags/voice survive stage rewrites."""
        data = self.status_data(stem)
        if data.get("stage") != stage:
            data.pop("error", None)
            data.pop("detail", None)  # per-stage leftovers don't carry over
        data.update({"stage": stage, "progress": _STAGE_PCT.get(stage, 0), "updated": time.time(), **extra})
        if stage in ("extracting", "narrating", "rendering audio"):
            data.setdefault("started", data["updated"])  # UI ETA basis
        p = self.status_path(stem)
        tmp = p.with_suffix(".tmp")  # atomic: a crash mid-write never leaves a truncated status
        tmp.write_text(json.dumps(data))
        os.replace(tmp, p)

    def status_data(self, stem: str) -> dict:
        try:
            return json.loads(self.status_path(stem).read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def find_status(self, stem: str) -> Path | None:
        """Status file for a live or archived job, wherever it sits."""
        p = self.status_path(stem)
        if p.exists():
            return p
        p = self.archive_dir() / f"{stem}.status.json"
        return p if p.exists() else None

    def free_slug(self, slug: str) -> str:
        """Title slugs collide across papers — pick {slug}-2, -3, ... instead."""
        base, n = slug, 2
        while self.mp3(slug).exists() or self.status_path(slug).exists():
            slug = f"{base}-{n}"
            n += 1
        return slug

    # crash recovery --------------------------------------------------
    def recover_orphans(self):
        """A worker crash (e.g. llama.cpp CUDA abort) leaves a status stuck in an
        active stage with nobody processing it — the queue is in-memory. Requeue on
        startup, but cap retries so a persistent crash ends in 'failed', not a loop."""
        for f in sorted(self.output_dir.glob("*.status.json")):
            stem = _stem_of(f)
            data = self.status_data(stem)
            stage = data.get("stage")
            if stage not in _ACTIVE_STAGES:
                continue
            pdf = self.find_pdf(stem)
            if pdf is None:
                self.set_status(stem, "failed", error="source PDF missing")
            elif stage == "queued":
                self.q.put(pdf)  # only waiting in line — not a crash, spends no attempt
            elif data.get("attempts", 0) + 1 > 2:
                self.set_status(stem, "failed", error="interrupted twice; see the logs, then retry manually")
            else:
                self.set_status(stem, "queued", attempts=data.get("attempts", 0) + 1)
                self.q.put(pdf)

    # full pipeline ---------------------------------------------------
    def process(self, pdf: Path):
        stem = _safe(pdf.name)
        try:
            self.set_status(stem, "extracting")
            paper = extract(pdf)
            paper_title = paper["title"] or ""
            # A user rename (POST /api/meta) wins as the display title, but the
            # on-disk slug still derives from the PDF title — rename is display-only.
            title = self.status_data(stem).get("title") or paper_title
            authors = paper.get("authors") or ""

            mode = self.status_data(stem).get("mode") or "summary"
            script_path = self.script(stem)
            if script_path.exists():
                script = script_path.read_text()  # resume: LLM step already paid for
                self.set_status(stem, "narrating", detail="cached script", title=title, authors=authors)
            elif mode == "verbatim":
                # exact text, no LLM: cheap, crash-free, for auditing your own paper
                self.set_status(stem, "narrating", detail="verbatim (no LLM)", title=title, authors=authors)
                script = verbatim(paper)
                script_path.write_text(script)
            else:
                self.set_status(stem, "narrating", detail="starting", title=title, authors=authors)
                backend_note = ""
                def on_backend(mode):
                    nonlocal backend_note
                    backend_note = " · CPU fallback (VRAM busy)" if mode == "cpu-fallback" else ""
                def on_narrate(i, n):
                    pct = 10 + round(65 * i / n)
                    self.set_status(stem, "narrating", progress=pct, detail=f"part {i}/{n}{backend_note}", title=title)
                script = narrate(paper, progress=on_narrate, on_backend=on_backend)
                script_path.write_text(script)
                # the narration child process has exited by now; its VRAM is free for other services

            def on_tts(i, n):
                pct = 75 + round(24 * i / n)
                self.set_status(stem, "rendering audio", progress=pct, detail=f"piece {i}/{n}", title=title)
            voice = self.status_data(stem).get("voice")  # set at upload or re-render time
            duration = synthesize(script, self.mp3(stem), progress=on_tts, voice=voice)

            # Final artifacts are title-named and self-contained in output/:
            # walrus-a-foundation-model.{pdf,mp3,script.txt,status.json}
            slug = _slug(paper_title) or stem
            if slug != stem:
                slug = self.free_slug(slug)  # free => no mp3/status there to stomp
                self.mp3(stem).replace(self.mp3(slug))
                if script_path.exists():
                    script_path.replace(self.script(slug))
                self.status_path(stem).replace(self.status_path(slug))  # keep tags/voice/started
                stem = slug
            if pdf.parent == self.input_dir and pdf.exists():
                # done jobs keep their PDF in output/, under the safe stem — use the
                # real path, not a rebuilt one, or special chars in the name strand it
                # input/ and output/ are separate mounts: rename() would hit EXDEV
                shutil.move(pdf, self.output_dir / f"{stem}.pdf")
            self.set_status(stem, "done", duration=round(duration), title=title)
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

    def archive_dir(self) -> Path:
        return self.output_dir / "archive"

    def jobs(self) -> list[dict]:
        seen = {}
        for f in sorted(self.output_dir.glob("*.status.json"), key=os.path.getmtime, reverse=True):
            stem = _stem_of(f)
            try:
                seen[stem] = {"name": stem, **json.loads(f.read_text())}
            except json.JSONDecodeError:
                continue
        for mp3 in self.output_dir.glob("*.mp3"):
            stem = mp3.stem
            if stem not in seen:
                seen[stem] = {"name": stem, "stage": "done", "progress": 100}
        arc = self.archive_dir()
        if arc.exists():
            for f in arc.glob("*.status.json"):
                stem = _stem_of(f)
                try:
                    # a live job of the same stem (e.g. re-uploaded) must win, or
                    # unarchive would stomp its files with the archived copies
                    seen.setdefault(stem, {"name": stem, **json.loads(f.read_text()), "archived": True})
                except json.JSONDecodeError:
                    continue
            for mp3 in arc.glob("*.mp3"):
                seen.setdefault(
                    mp3.stem, {"name": mp3.stem, "stage": "done", "progress": 100, "archived": True}
                )
        for pdf in self.input_dir.glob("*.pdf"):
            try:
                stem = _safe(pdf.name)  # match scan()'s stem, or special chars ghost a second card
            except HTTPException:
                continue  # unprocessable name ("!!!.pdf") — the watcher rejects it too
            seen.setdefault(stem, {"name": stem, "stage": "queued", "progress": 0})
        return list(seen.values())


pipeline = Pipeline(INPUT_DIR, OUTPUT_DIR)


@asynccontextmanager
async def lifespan(_app):
    for target in (pipeline.worker, pipeline.watcher):
        threading.Thread(target=target, daemon=True).start()
    pipeline.recover_orphans()
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
async def upload(files: list[UploadFile] = File(...), voice: str = Form(""), mode: str = Form("")):
    if voice:
        voice = _voice_or_400(voice)
    if mode:
        mode = _mode_or_400(mode)
    saved = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: PDFs only")
        data = await f.read(_MAX_UPLOAD + 1)  # bounded read
        if len(data) > _MAX_UPLOAD:
            raise HTTPException(413, f"{f.filename}: over the 50 MB limit")
        dest = pipeline.input_dir / f"{_safe(f.filename)}.pdf"
        tmp = dest.with_suffix(".tmp")  # atomic: watcher only globs *.pdf
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        saved.append(dest.name)
    pipeline.scan()
    if voice or mode:
        for name in saved:
            stem = Path(name).stem
            d = pipeline.status_data(stem)
            # annotate only jobs the worker will actually touch: `d and` guards
            # mp3-only legacy jobs (no status) from getting a phantom "queued"
            if d and d.get("stage") not in ("done", "failed"):
                extra = {}
                if voice:
                    extra["voice"] = voice
                # mode switch only while queued: later stages already run under the old mode
                if mode and mode != d.get("mode", "summary") and d.get("stage") == "queued":
                    pipeline.script(stem).unlink(missing_ok=True)
                    extra["mode"] = mode
                if extra:
                    pipeline.set_status(stem, d["stage"], **extra)
    return {"saved": saved}


@app.get("/api/audio/{name}")
def audio(name: str, dl: bool = False):
    stem = _safe(name)
    mp3 = pipeline.mp3(stem)
    if not mp3.exists():
        raise HTTPException(404, "not found")
    if not dl:
        return FileResponse(mp3, media_type="audio/mpeg")
    title = stem
    st = pipeline.status_path(stem)
    if st.exists():
        try:
            title = json.loads(st.read_text()).get("title") or stem
        except json.JSONDecodeError:
            pass
    return FileResponse(mp3, media_type="audio/mpeg", filename=f"{_slug(title) or stem}.mp3")


@app.get("/feed.xml")
def feed(request: Request):
    """RSS 2.0 podcast feed of finished MP3s — subscribe from any podcast app."""
    base = str(request.base_url).rstrip("/")
    meta = {}
    for f in pipeline.output_dir.glob("*.status.json"):
        try:
            meta[_stem_of(f)] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    items = []
    for mp3 in sorted(pipeline.output_dir.glob("*.mp3"), key=os.path.getmtime, reverse=True):
        st = mp3.stat()
        d = meta.get(mp3.stem, {})
        title = escape(d.get("title") or mp3.stem)
        author = escape(d.get("authors") or "Paper Reader")
        dur = int(d.get("duration") or 0)
        items.append(
            f"<item><title>{title}</title><guid>{escape(mp3.stem)}</guid>"
            f"<pubDate>{formatdate(st.st_mtime)}</pubDate>"
            f"<itunes:author>{author}</itunes:author>"
            + (f"<itunes:duration>{dur}</itunes:duration>" if dur else "")
            + f'<enclosure url="{base}/api/audio/{mp3.stem}" length="{st.st_size}" type="audio/mpeg"/>'
            f"</item>"
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'
        f"<title>Paper Reader</title><link>{base}/</link>"
        "<description>Narrated papers</description>" + "".join(items) + "</channel></rss>"
    )
    return Response(content=xml, media_type="application/rss+xml")


@app.get("/api/pdf/{name}")
def pdf(name: str):
    """Source PDF for the in-page viewer. Done jobs keep it in output/; also check
    archive/ and input/ so failed or archived jobs are still readable."""
    stem = _safe(name)
    for d in (pipeline.output_dir, pipeline.archive_dir(), pipeline.input_dir):
        p = d / f"{stem}.pdf"
        if p.exists():
            return FileResponse(p, media_type="application/pdf")  # inline, not attachment
    raise HTTPException(404, "not found")


@app.get("/api/script/{name}", response_class=PlainTextResponse)
def script(name: str):
    s = pipeline.script(_safe(name))
    if not s.exists():
        raise HTTPException(404, "not found")
    return s.read_text()


@app.post("/api/retry/{name}")
def retry(name: str):
    """Clear a failed job's status so the watcher re-queues its PDF."""
    stem = _safe(name)
    st = pipeline.status_path(stem)
    cleared = False
    if st.exists():
        try:
            failed = json.loads(st.read_text()).get("stage") == "failed"
        except json.JSONDecodeError:
            failed = True
        if failed:
            st.unlink()
            cleared = True
    pdf = pipeline.find_pdf(stem)
    if cleared and pdf is not None:
        # queue directly: scan() would skip it if a partial mp3 exists
        pipeline.set_status(stem, "queued")
        pipeline.q.put(pdf)
    return {"requeued": cleared}


class Meta(BaseModel):
    tags: list[str] | None = None
    title: str | None = None


@app.post("/api/meta/{name}")
def meta(name: str, m: Meta):
    """Edit a job's tags / display title. Works on archived jobs too."""
    stem = _safe(name)
    st = pipeline.find_status(stem)
    if st is None:
        raise HTTPException(404, "not found")
    if st.parent != pipeline.archive_dir():
        _busy_or_409(stem)  # the worker rewrites this file every few seconds — edits would be lost
    try:
        data = json.loads(st.read_text())
    except json.JSONDecodeError:
        data = {}
    if m.tags is not None:
        data["tags"] = [t.strip() for t in m.tags if t.strip()][:10]
    if m.title is not None:
        if not m.title.strip():
            raise HTTPException(400, "empty title")
        data["title"] = m.title.strip()
    st.write_text(json.dumps(data))
    return data


@app.post("/api/rerender/{name}")
def rerender(name: str, voice: str = "", mode: str = ""):
    """Re-run TTS from the cached narration script — unless the mode changes, in
    which case the script is regenerated (verbatim needs no LLM; summary does)."""
    stem = _safe(name)
    _busy_or_409(stem)
    mp3 = pipeline.mp3(stem)
    if not mp3.exists():
        raise HTTPException(409, "only finished jobs can be re-rendered")
    pdf = pipeline.find_pdf(stem)
    if pdf is None:
        raise HTTPException(404, "source PDF missing")
    keep = {k: v for k, v in pipeline.status_data(stem).items() if k in ("tags", "title", "authors", "voice", "mode")}
    if voice:
        keep["voice"] = _voice_or_400(voice)
    if mode:
        mode = _mode_or_400(mode)
        if mode != keep.get("mode", "summary"):  # cached script belongs to the old mode — regenerate
            keep["mode"] = mode
            pipeline.script(stem).unlink(missing_ok=True)
    pipeline.status_path(stem).unlink(missing_ok=True)  # clean slate: started/duration restamp
    pipeline.set_status(stem, "queued", **keep)
    mp3.unlink()  # last: a crash leaves "queued + stale mp3" — the rerun overwrites it
    pipeline.q.put(pdf)
    return {"requeued": True}


@app.post("/api/archive/{name}")
def archive(name: str, undo: bool = False):
    """Move all of a job's files into/out of output/archive/."""
    stem = _safe(name)
    _busy_or_409(stem)
    arc = pipeline.archive_dir()
    moved = 0
    if undo:
        for p in sorted(arc.glob(f"{stem}.*")):
            p.replace(pipeline.output_dir / p.name)
            moved += 1
    else:
        arc.mkdir(exist_ok=True)
        candidates = (
            pipeline.input_dir / f"{stem}.pdf",
            pipeline.input_dir / f"{stem}.tmp",  # orphaned half-upload
            pipeline.output_dir / f"{stem}.pdf",
            pipeline.mp3(stem),
            pipeline.script(stem),
            pipeline.status_path(stem),
            pipeline.output_dir / f"{stem}.tmp",  # orphaned half-written mp3
            pipeline.output_dir / f"{stem}.status.tmp",  # orphaned half-written status
        )
        for p in candidates:
            if p.exists():
                shutil.move(p, arc / p.name)  # pdf may come from the input/ mount
                moved += 1
    return {"moved": moved}


@app.post("/api/delete/{name}")
def delete(name: str):
    stem = _safe(name)
    _busy_or_409(stem)
    removed = 0
    arc = pipeline.archive_dir()
    for p in (
        pipeline.input_dir / f"{stem}.pdf",
        pipeline.input_dir / f"{stem}.tmp",
        pipeline.output_dir / f"{stem}.pdf",
        pipeline.mp3(stem),
        pipeline.script(stem),
        pipeline.status_path(stem),
        pipeline.output_dir / f"{stem}.tmp",
        pipeline.output_dir / f"{stem}.status.tmp",
        arc / f"{stem}.pdf",
        arc / f"{stem}.tmp",
        arc / f"{stem}.mp3",
        arc / f"{stem}.script.txt",
        arc / f"{stem}.status.json",
    ):
        if p.exists():
            p.unlink()
            removed += 1
    return {"removed": removed}
