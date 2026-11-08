"""Web layer: UI serving + upload validation. No LLM (mocked), no worker threads."""

import json
import os
import tempfile
from pathlib import Path

os.environ["NARRATE_MOCK"] = "1"  # must precede app imports
_td = tempfile.mkdtemp()
os.environ["INPUT_DIR"] = os.path.join(_td, "in")
os.environ["OUTPUT_DIR"] = os.path.join(_td, "out")

from fastapi.testclient import TestClient  # noqa: E402  (env above must land first)

from main import app  # noqa: E402

client = TestClient(app)  # no `with`: lifespan threads stay off


def test_active_job_operations_return_409():
    """Archive/delete/rerender/meta would interfere with files the worker rewrites."""
    import main

    sp = main.pipeline.status_path("busyjob")
    sp.write_text('{"stage":"narrating"}')
    try:
        for ep in ("/api/delete/busyjob", "/api/archive/busyjob", "/api/rerender/busyjob"):
            r = client.post(ep)
            assert r.status_code == 409, ep
        assert client.post("/api/meta/busyjob", json={"tags": ["x"]}).status_code == 409
    finally:
        sp.unlink()


def test_tts_splitter_respects_max_chars():
    """One sentence longer than the limit must be hard-split, never fed whole to Kokoro."""
    from tts import _pieces

    assert _pieces("") == []
    assert _pieces("short one. short two.") == ["short one. short two."]
    long_para = "word " * 500 + "."  # single ~2500-char sentence, no breaks
    pieces = _pieces(long_para, max_chars=1200)
    assert pieces and all(len(p) <= 1200 for p in pieces)
    assert "".join(pieces).replace(" ", "") == long_para.replace(" ", "")  # no text lost
    two_paras = "first paragraph. " * 100 + "\n\n" + "second paragraph. " * 100
    pieces = _pieces(two_paras, max_chars=200)
    assert all(len(p) <= 200 for p in pieces)


def test_llm_gpu_oom_falls_back_to_cpu(monkeypatch):
    """VRAM taken by a resident service -> narrate on CPU instead of failing the job."""
    import huggingface_hub
    import pytest

    import narrate

    try:
        import llama_cpp
    except Exception as e:  # CUDA-built libllama won't even load without a driver
        pytest.skip(f"llama_cpp not loadable here: {e}")

    layers_tried = []

    class FakeLlama:
        def __init__(self, model_path=None, n_ctx=None, n_gpu_layers=None, verbose=None):
            layers_tried.append(n_gpu_layers)
            if n_gpu_layers != 0:
                raise RuntimeError("Vulkan: No/-1 devices found")  # what a contested box actually raises

        def create_chat_completion(self, messages=None, temperature=None, max_tokens=None):
            return {"choices": [{"message": {"content": "spoken words."}}]}

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/fake.gguf")
    monkeypatch.setattr(llama_cpp, "Llama", FakeLlama)
    monkeypatch.setenv("N_GPU_LAYERS", "12")
    events = []
    script = narrate._generate(
        {"title": "T", "authors": "", "sections": [{"title": "s", "text": "word " * 30}]},
        lambda *a: events.append(a),
        force_cpu=False,
    )
    assert layers_tried == [12, 0]
    assert ("backend", "cpu-fallback") in events
    assert "spoken words." in script


def _abort_entry(conn):
    """Spawn-target for the crash test: dies by SIGABRT like llama.cpp's ggml_abort."""
    import os

    os.abort()


def test_child_main_packages_events_for_the_pipe(monkeypatch):
    """Regression: conn.send takes one payload; events must go over the wire as tuples."""
    import multiprocessing as mp

    import narrate

    def fake_generate(paper, emit, force_cpu):
        emit("backend", "gpu")
        emit("progress", 1, 2)
        return "the script"

    monkeypatch.setattr(narrate, "_generate", fake_generate)
    parent_conn, child_conn = mp.Pipe()
    narrate._child_main(child_conn, False, {"title": "T", "authors": "", "sections": []})
    assert parent_conn.recv() == ("backend", "gpu")
    assert parent_conn.recv() == ("progress", 1, 2)
    assert parent_conn.recv() == ("done", "the script")


def test_child_abort_surfaces_as_crash_error_not_app_death():
    """A hard abort inside the narration child must raise, not kill the app."""
    import narrate
    import pytest

    with pytest.raises(narrate.ChildCrashedError):
        narrate._run_child(_abort_entry, ())


def test_narrate_retries_once_on_cpu_after_gpu_crash(monkeypatch):
    import narrate

    calls = []

    def fake_run(target, args, progress=None, on_backend=None):
        calls.append(args[0])
        if not args[0]:
            raise narrate.ChildCrashedError(-6, "gpu")
        return "script", "cpu"

    monkeypatch.setattr(narrate, "_run_child", fake_run)
    monkeypatch.delenv("NARRATE_MOCK", raising=False)  # reach the real narrate() path
    assert narrate.narrate({"title": "T", "authors": "", "sections": []}) == "script"
    assert calls == [False, True]


def test_narrate_no_retry_when_crash_was_already_on_cpu(monkeypatch):
    import narrate
    import pytest

    calls = []

    def fake_run(target, args, progress=None, on_backend=None):
        calls.append(args[0])
        raise narrate.ChildCrashedError(-6, "cpu-fallback")

    monkeypatch.setattr(narrate, "_run_child", fake_run)
    monkeypatch.delenv("NARRATE_MOCK", raising=False)
    with pytest.raises(narrate.ChildCrashedError):
        narrate.narrate({"title": "T", "authors": "", "sections": []})
    assert calls == [False]


def test_ui_served_without_tables():
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="drop"' in r.text
    assert "<table" not in r.text.lower()  # readability: card layout only
    assert 'data-tab="feeds"' in r.text  # in-app tabs: Library / Write / Voices / Feeds
    assert 'data-tab="write"' in r.text  # draft scratchpad with instant listen
    assert 'id="wvoice"' in r.text  # Write tab has its own voice control
    assert 'id="wpane"' in r.text and 'id="wplay"' in r.text  # read-aloud transport + highlight pane
    assert 'id="wsplit"' in r.text  # side-by-side read-view | editor
    assert 'class="voicesel"' in r.text  # per-paper voice in the card menu
    assert 'id="vpdf"' not in r.text  # the reader lives in its own tab now


def test_view_page_served():
    r = client.get("/view")
    assert r.status_code == 200
    assert 'id="vpdf"' in r.text and 'id="vedit"' in r.text
    assert 'id="vvoice"' in r.text  # reader voice drives preview and save & re-render
    assert "<table" not in r.text.lower()


def test_serve_source_inline_for_viewer():
    """The read & listen viewer needs the source inline (not as a download)."""
    import main

    out = Path(main.pipeline.output_dir)
    (out / "readable.pdf").write_bytes(b"%PDF-1.4 fake")
    (out / "readable-txt.txt").write_text("plain notes")
    try:
        r = client.get("/api/source/readable")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert "attachment" not in r.headers.get("content-disposition", "")
        assert r.content == b"%PDF-1.4 fake"
        r = client.get("/api/source/readable-txt")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert client.get("/api/source/unknown-job").status_code == 404
    finally:
        (out / "readable.pdf").unlink(missing_ok=True)
        (out / "readable-txt.txt").unlink(missing_ok=True)


def test_pico_css_is_local():
    r = client.get("/static/pico.min.css")
    assert r.status_code == 200 and len(r.content) > 10_000


def test_verbatim_script_is_exact_text_no_llm():
    import narrate

    paper = {
        "title": "My Paper",
        "authors": "A N Author",
        "sections": [
            {"title": "Methods", "text": "We did the thing exactly."},
            {"title": "full text", "text": "fallback body."},
            {"title": "", "text": "   "},
        ],
    }
    script = narrate.verbatim(paper)
    assert script.startswith("You're listening to My Paper, by A N Author. Read verbatim.")
    assert "Methods.\nWe did the thing exactly." in script
    assert "fallback body." in script and "full text" not in script  # synthetic heading not spoken
    assert script.endswith("fallback body.")  # whitespace-only section skipped


def test_upload_rejects_non_pdf():
    r = client.post("/api/upload", files={"files": ("video.mp4", b"hello")})
    assert r.status_code == 400
    assert "supported types" in r.json()["detail"]


def test_upload_pdf_registers_job():
    r = client.post("/api/upload", files={"files": ("mypaper.pdf", b"%PDF-1.4 fake")})
    assert r.status_code == 200
    assert r.json()["saved"] == ["mypaper.pdf"]
    names = [j["name"] for j in client.get("/api/jobs").json()]
    assert "mypaper" in names


def test_upload_rejects_oversize(monkeypatch):
    import main

    monkeypatch.setattr(main, "_MAX_UPLOAD", 10)
    r = client.post("/api/upload", files={"files": ("big.pdf", b"%PDF-1.4 way over ten bytes")})
    assert r.status_code == 413


def test_feed_lists_mp3_as_podcast_episode():
    import main  # test_smoke may import main first; use the live pipeline's dir

    mp3 = Path(main.pipeline.output_dir) / "some-paper.mp3"
    mp3.write_bytes(b"fake-mp3-bytes")
    try:
        r = client.get("/feed.xml")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/rss+xml")
        assert "<enclosure" in r.text and "some-paper" in r.text
        assert r.text.count("<item>") == 1
    finally:
        mp3.unlink()


def test_feed_shows_paper_title():
    import main

    out = Path(main.pipeline.output_dir)
    (out / "titled-one.mp3").write_bytes(b"mp3")
    (out / "titled-one.status.json").write_text(
        '{"stage":"done","title":"Walrus: A Cross-Domain Foundation Model","authors":"Michael McCabe","duration":905}'
    )
    try:
        r = client.get("/feed.xml")
        assert "Walrus: A Cross-Domain Foundation Model" in r.text
        assert "xmlns:itunes" in r.text
        assert "<itunes:author>Michael McCabe</itunes:author>" in r.text
        assert "<itunes:duration>905</itunes:duration>" in r.text
    finally:
        (out / "titled-one.mp3").unlink()
        (out / "titled-one.status.json").unlink()


def test_slug():
    import main

    assert main._slug("Walrus: A Cross-Domain Foundation Model (v2)!") == "walrus-a-cross-domain-foundation-model-v2"
    assert main._slug("") == ""


def test_recover_orphans_requeues_stuck_job_then_gives_up():
    """A crash mid-job (e.g. CUDA abort) must not freeze the UI at 'narrating' forever."""
    import main

    p = main.pipeline
    (Path(p.input_dir) / "stuck.pdf").write_bytes(b"%PDF-1.4 fake")
    sp = p.status_path("stuck")
    sp.write_text('{"stage":"narrating","title":"T","tags":["x"]}')
    try:
        p.recover_orphans()
        data = json.loads(sp.read_text())
        assert data["stage"] == "queued" and data["attempts"] == 1
        assert data["title"] == "T" and data["tags"] == ["x"]  # metadata survives
        assert any(pdf.name == "stuck.pdf" for pdf in list(p.q.queue))
        # second crash mid-narration: attempts -> 2; third: give up, mark failed
        for final in (2, "failed"):
            data = json.loads(sp.read_text())
            data["stage"] = "narrating"  # worker was mid-job when it died again
            sp.write_text(json.dumps(data))
            p.recover_orphans()
            data = json.loads(sp.read_text())
            assert (data["attempts"] if final == 2 else data["stage"]) == final
    finally:
        sp.unlink(missing_ok=True)
        (Path(p.input_dir) / "stuck.pdf").unlink(missing_ok=True)
        while not p.q.empty():
            p.q.get_nowait()


def test_recover_orphans_queued_job_spends_no_attempt():
    """A job merely waiting in line is requeued without burning one of its 2 retries."""
    import main

    p = main.pipeline
    (Path(p.input_dir) / "waiting.pdf").write_bytes(b"%PDF-1.4 fake")
    sp = p.status_path("waiting")
    sp.write_text('{"stage":"queued"}')
    try:
        p.recover_orphans()
        data = json.loads(sp.read_text())
        assert data["stage"] == "queued" and "attempts" not in data
        assert any(pdf.name == "waiting.pdf" for pdf in list(p.q.queue))
    finally:
        sp.unlink(missing_ok=True)
        (Path(p.input_dir) / "waiting.pdf").unlink(missing_ok=True)
        while not p.q.empty():
            p.q.get_nowait()


def test_recover_orphans_leaves_done_jobs_alone():
    import main

    p = main.pipeline
    sp = p.status_path("donejob")
    sp.write_text('{"stage":"done","progress":100}')
    try:
        p.recover_orphans()
        assert json.loads(sp.read_text())["stage"] == "done"
        assert all(pdf.stem != "donejob" for pdf in list(p.q.queue))
    finally:
        sp.unlink(missing_ok=True)


def test_retry_requeues_failed_job():
    import main

    sp = main.pipeline.status_path("zombiejob")
    sp.write_text('{"stage":"failed","error":"boom"}')
    r = client.post("/api/retry/zombiejob")
    assert r.json()["requeued"] is True
    assert not sp.exists()


def test_meta_sets_tags_and_title():
    import main

    sp = main.pipeline.status_path("metatest")
    sp.write_text('{"stage":"done"}')
    try:
        r = client.post("/api/meta/metatest", json={"tags": [" rl ", "", "survey"], "title": "A Nice Paper"})
        assert r.status_code == 200
        data = json.loads(sp.read_text())
        assert data["tags"] == ["rl", "survey"]
        assert data["title"] == "A Nice Paper"
    finally:
        sp.unlink()


def test_meta_404_on_unknown_job():
    assert client.post("/api/meta/nope-stem", json={"title": "x"}).status_code == 404


def test_meta_edits_archived_job():
    import main

    arc = main.pipeline.archive_dir()
    arc.mkdir(exist_ok=True)
    sp = arc / "archmeta.status.json"
    sp.touch()
    sp.write_text('{"stage":"done"}')
    try:
        r = client.post("/api/meta/archmeta", json={"tags": ["old"]})
        assert r.status_code == 200
        assert json.loads(sp.read_text())["tags"] == ["old"]
    finally:
        sp.unlink()


def test_free_slug_avoids_collisions():
    import main

    p = main.pipeline
    dupe = Path(p.output_dir) / "dupe.mp3"
    dupe.write_bytes(b"x")
    try:
        assert p.free_slug("dupe") == "dupe-2"
        assert p.free_slug("fresh") == "fresh"
    finally:
        dupe.unlink()


def test_voice_validation():
    import pytest
    from fastapi import HTTPException

    import main

    assert main._voice_or_400("af_heart") == "af_heart"
    with pytest.raises(HTTPException):
        main._voice_or_400("not a voice")


def test_upload_rejects_unknown_mode():
    r = client.post("/api/upload", files={"files": ("m.pdf", b"%PDF-1.4 fake")}, data={"mode": "skim"})
    assert r.status_code == 400
    assert "unknown mode" in r.json()["detail"]


def test_rerender_mode_switch_regenerates_script():
    """Verbatim needs a different script than summary — a mode switch must not reuse the cache."""
    import main

    out = Path(main.pipeline.output_dir)
    (out / "mm.mp3").write_bytes(b"m")
    (out / "mm.pdf").write_bytes(b"p")
    (out / "mm.script.txt").write_text("summary script")
    (out / "mm.status.json").write_text('{"stage":"done","mode":"summary"}')
    try:
        r = client.post("/api/rerender/mm", params={"mode": "verbatim"})
        assert r.status_code == 200 and r.json()["requeued"] is True
        data = json.loads((out / "mm.status.json").read_text())
        assert data["mode"] == "verbatim" and data["stage"] == "queued"
        assert not (out / "mm.script.txt").exists()  # old mode's script discarded
        assert not (out / "mm.mp3").exists()
    finally:
        while not main.pipeline.q.empty():
            main.pipeline.q.get_nowait()
        for suf in (".mp3", ".pdf", ".script.txt", ".status.json"):
            (out / ("mm" + suf)).unlink(missing_ok=True)


def test_rerender_requeues_and_keeps_meta():
    import main

    out = Path(main.pipeline.output_dir)
    (out / "rr.mp3").write_bytes(b"m")
    (out / "rr.pdf").write_bytes(b"p")
    (out / "rr.status.json").write_text('{"stage":"done","title":"T","tags":["a"],"voice":"af_heart","started":1}')
    try:
        r = client.post("/api/rerender/rr", params={"voice": "am_adam"})
        assert r.status_code == 200 and r.json()["requeued"] is True
        assert not (out / "rr.mp3").exists()
        data = json.loads((out / "rr.status.json").read_text())
        assert data["stage"] == "queued" and data["tags"] == ["a"] and data["voice"] == "am_adam"
        assert "started" not in data  # restamps when processing resumes
        queued = []
        while not main.pipeline.q.empty():  # shared queue may hold earlier tests' pdfs
            queued.append(main.pipeline.q.get_nowait().name)
        assert "rr.pdf" in queued
    finally:
        for suf in (".mp3", ".pdf", ".script.txt", ".status.json"):
            (out / ("rr" + suf)).unlink(missing_ok=True)


def test_upload_voice_ignores_mp3_only_legacy_job():
    """A finished job with no status.json must not get a phantom 'queued' card."""
    import main

    out = Path(main.pipeline.output_dir)
    inp = Path(main.pipeline.input_dir)
    (out / "legacy.mp3").write_bytes(b"x")
    (inp / "legacy.pdf").write_bytes(b"%PDF-1.4 fake")
    try:
        r = client.post("/api/upload", files={"files": ("legacy.pdf", b"%PDF-1.4 fake")}, data={"voice": "af_heart"})
        assert r.status_code == 200
        assert not (out / "legacy.status.json").exists()
    finally:
        for p in (out / "legacy.mp3", out / "legacy.status.json", inp / "legacy.pdf"):
            p.unlink(missing_ok=True)


def test_live_job_wins_over_archived_same_stem():
    """Archive pass must not overwrite the live card — unarchive would stomp live files."""
    import main

    p = main.pipeline
    out, arc = Path(p.output_dir), p.archive_dir()
    arc.mkdir(exist_ok=True)
    (out / "clash.status.json").write_text('{"stage":"narrating","title":"new"}')
    (arc / "clash.status.json").write_text('{"stage":"done","title":"old"}')
    try:
        jobs = [j for j in client.get("/api/jobs").json() if j["name"] == "clash"]
        assert len(jobs) == 1
        assert jobs[0]["title"] == "new" and not jobs[0].get("archived")
    finally:
        (out / "clash.status.json").unlink(missing_ok=True)
        (arc / "clash.status.json").unlink(missing_ok=True)


def test_special_char_input_pdf_shows_one_card_under_safe_stem():
    import main

    p = main.pipeline
    (Path(p.input_dir) / "Café (draft).pdf").write_bytes(b"%PDF-1.4 fake")
    try:
        names = [j["name"] for j in client.get("/api/jobs").json()]
        assert "Café draft" in names  # _safe()'s stem (é is \w, parens stripped), same as scan()
        assert "Café (draft)" not in names  # no ghost card under the raw name
    finally:
        (Path(p.input_dir) / "Café (draft).pdf").unlink(missing_ok=True)


def test_archive_roundtrip():
    import main

    out = Path(main.pipeline.output_dir)
    for suffix, data in ((".mp3", b"mp"), (".script.txt", b"s"), (".status.json", b'{"stage":"done"}'), (".pdf", b"p")):
        (out / ("arctest" + suffix)).write_bytes(data)
    try:
        assert client.post("/api/archive/arctest").json()["moved"] == 4
        arc = out / "archive"
        assert (arc / "arctest.mp3").exists() and not (out / "arctest.mp3").exists()
        jobs = client.get("/api/jobs").json()
        assert any(j["name"] == "arctest" and j.get("archived") for j in jobs)
        assert client.post("/api/archive/arctest?undo=1").json()["moved"] == 4
        assert (out / "arctest.mp3").exists()
    finally:
        for suffix in (".mp3", ".script.txt", ".status.json", ".pdf"):
            (out / ("arctest" + suffix)).unlink(missing_ok=True)


def test_find_source_matches_handdropped_raw_name():
    """A hand-dropped "Café (draft).pdf" must be findable under its safe stem."""
    import main

    raw = Path(main.pipeline.input_dir) / "Café (draft).pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    try:
        assert main.pipeline.find_source("Café draft") == raw
    finally:
        raw.unlink(missing_ok=True)


def test_delete_removes_handdropped_raw_named_pdf():
    """Deleting a failed special-char job must not leave the PDF for the watcher
    to re-queue — that resurrected the job forever."""
    import main

    p = main.pipeline
    raw = Path(p.input_dir) / "My Paper (draft).pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    sp = p.status_path("My Paper draft")
    sp.write_text('{"stage":"failed","error":"scanned"}')
    try:
        r = client.post("/api/delete/My Paper draft")
        assert r.status_code == 200
        assert not raw.exists()
        assert list(Path(p.input_dir).glob("My Paper*")) == []
    finally:
        raw.unlink(missing_ok=True)
        sp.unlink(missing_ok=True)


def test_serve_source_finds_handdropped_raw_name():
    import main

    raw = Path(main.pipeline.input_dir) / "My Paper (draft).pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    try:
        r = client.get("/api/source/My Paper draft")
        assert r.status_code == 200 and r.content == b"%PDF-1.4 fake"
    finally:
        raw.unlink(missing_ok=True)


def test_upload_accepts_txt_and_rejects_unknown_extension():
    r = client.post("/api/upload", files={"files": ("notes.txt", b"just some notes " * 40)})
    assert r.status_code == 200
    assert r.json()["saved"] == ["notes.txt"]
    r = client.post("/api/upload", files={"files": ("slides.pptx", b"nope")})
    assert r.status_code == 400
    assert "supported types" in r.json()["detail"]
    import main

    (Path(main.pipeline.input_dir) / "notes.txt").unlink(missing_ok=True)
    main.pipeline.status_path("notes").unlink(missing_ok=True)
    while not main.pipeline.q.empty():
        main.pipeline.q.get_nowait()


def test_script_save_backs_up_and_validates():
    import main

    s = main.pipeline.script("editable")
    s.write_text("original script")
    try:
        r = client.post("/api/script/editable", content="edited words")
        assert r.status_code == 200 and r.json()["saved"] == len("edited words")
        assert s.read_text() == "edited words"
        assert s.with_suffix(".bak").read_text() == "original script"
        r = client.post("/api/script/editable", content="   ")
        assert r.status_code == 400
        assert s.read_text() == "edited words"  # untouched
        # second save: .bak moves to the immediately-previous version
        client.post("/api/script/editable", content="v3")
        assert s.with_suffix(".bak").read_text() == "edited words"
        assert client.post("/api/script/ghost-job", content="x").status_code == 404
    finally:
        s.unlink(missing_ok=True)
        s.with_suffix(".bak").unlink(missing_ok=True)


def test_script_save_409_while_job_active():
    import main

    s = main.pipeline.script("busyedit")
    s.write_text("words")
    main.pipeline.status_path("busyedit").write_text('{"stage":"narrating"}')
    try:
        assert client.post("/api/script/busyedit", content="new").status_code == 409
        assert s.read_text() == "words"
    finally:
        s.unlink(missing_ok=True)
        main.pipeline.status_path("busyedit").unlink(missing_ok=True)


def test_preview_synthesizes_snippet(monkeypatch):
    import main

    written = []

    def fake_synth(script, out_path, progress=None, voice=None):
        written.append((len(script), voice))
        Path(out_path).write_bytes(b"preview-mp3")

    monkeypatch.setattr(main, "synthesize", fake_synth)
    try:
        r = client.post("/api/preview", json={"text": "read this bit", "voice": "bm_lewis"})
        assert r.status_code == 200 and r.content == b"preview-mp3"
        assert written == [(len("read this bit"), "bm_lewis")]
        r = client.post("/api/preview", json={"text": "read this bit", "voice": "bm_lewis"})
        assert len(written) == 1  # hash cache hit
        assert client.post("/api/preview", json={"text": " "}).status_code == 400
        assert client.post("/api/preview", json={"text": "x", "voice": "nah"}).status_code == 400
        r = client.post("/api/preview", json={"text": "y" * 1600})
        assert r.status_code == 413
    finally:
        import hashlib

        key = hashlib.sha1("bm_lewis\nread this bit".encode()).hexdigest()[:16]
        (Path(main.pipeline.output_dir) / "samples" / f"preview-{key}.mp3").unlink(missing_ok=True)


def test_preview_get_shares_cache_and_immutable_headers(monkeypatch):
    """GET /api/preview: the read-aloud engine's path — browser-cacheable (immutable)
    and sharing the hash cache with POST."""
    import main

    written = []

    def fake_synth(script, out_path, progress=None, voice=None):
        written.append(script)
        Path(out_path).write_bytes(b"get-preview")

    monkeypatch.setattr(main, "synthesize", fake_synth)
    try:
        r = client.get("/api/preview", params={"text": "same words", "voice": "bm_lewis"})
        assert r.status_code == 200 and r.content == b"get-preview"
        assert "immutable" in r.headers.get("cache-control", "")
        client.get("/api/preview", params={"text": "same words", "voice": "bm_lewis"})
        assert len(written) == 1  # hash cache hit
        assert client.post("/api/preview", json={"text": "same words", "voice": "bm_lewis"}).status_code == 200
        assert len(written) == 1  # GET/POST share the cache
        assert client.get("/api/preview", params={"text": " "}).status_code == 400
    finally:
        import hashlib

        key = hashlib.sha1("bm_lewis\nsame words".encode()).hexdigest()[:16]
        (Path(main.pipeline.output_dir) / "samples" / f"preview-{key}.mp3").unlink(missing_ok=True)


def test_follow_marks_404_until_rendered():
    assert client.get("/api/follow/nothing-here").status_code == 404
    import main

    f = main.pipeline.follow("marked")
    f.write_text('{"marks":[],"duration":0}')
    try:
        r = client.get("/api/follow/marked")
        assert r.status_code == 200 and r.json()["duration"] == 0
    finally:
        f.unlink(missing_ok=True)


def test_tts_marks_align_with_script(monkeypatch, tmp_path):
    """Follow-along marks: sentence offsets must slice the script, and cumulative
    durations must add up to the returned audio length."""
    import numpy as np

    import tts

    class FakeKokoro:
        def create(self, text, voice=None, speed=None, lang=None):
            return np.zeros(len(text) * 10, dtype=np.float32), 24000

    monkeypatch.setattr(tts, "_load", lambda: FakeKokoro())
    script = "First sentence here. Second one is a bit longer than the first! Third? Yes, a third."
    out = tmp_path / "a.mp3"
    marks_path = tmp_path / "a.follow.json"
    duration = tts.synthesize(script, out, marks_path=marks_path)
    data = json.loads(marks_path.read_text())
    assert data["marks"], "no marks written"
    assert data["marks"][0]["t"] == 0
    for m in data["marks"]:
        assert script[m["s"] : m["e"]].strip(), "mark slice lands outside its sentence"
    assert abs(data["duration"] - duration) < 1.0
    assert out.exists()


def test_tts_piece_cache_skips_unchanged_pieces(monkeypatch, tmp_path):
    """The edit→re-render loop: only the piece containing the edit re-renders;
    unchanged pieces replay from the on-disk cache."""
    import numpy as np

    import tts

    calls = []

    class FakeKokoro:
        def create(self, text, voice=None, speed=None, lang=None):
            calls.append(text)
            return np.zeros(max(1, len(text) * 10), dtype=np.float32), 24000

    monkeypatch.setattr(tts, "_load", lambda: FakeKokoro())
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    sents = [f"Sentence number {i:02d} has enough words in it to sound natural." for i in range(25)]
    tts.synthesize(" ".join(sents), tmp_path / "a.mp3")
    assert len(calls) >= 2, "script should span multiple TTS pieces"
    calls.clear()
    sents[18] = "Sentence number 18 was reworded to flow much better."
    tts.synthesize(" ".join(sents), tmp_path / "b.mp3")
    assert len(calls) == 1 and "reworded" in calls[0], f"needlessly re-rendered: {calls}"


def test_recover_orphans_finds_handdropped_raw_named_pdf():
    """A crash mid-job on a raw-named PDF must re-queue, not claim the PDF is missing."""
    import main

    p = main.pipeline
    raw = Path(p.input_dir) / "My Paper (draft).pdf"
    raw.write_bytes(b"%PDF-1.4 fake")
    sp = p.status_path("My Paper draft")
    sp.write_text('{"stage":"narrating"}')
    try:
        p.recover_orphans()
        data = json.loads(sp.read_text())
        assert data["stage"] == "queued" and data.get("error") is None
        assert any(pdf.name == "My Paper (draft).pdf" for pdf in list(p.q.queue))
    finally:
        raw.unlink(missing_ok=True)
        sp.unlink(missing_ok=True)
        while not p.q.empty():
            p.q.get_nowait()


def test_unarchive_refuses_to_stomp_live_job():
    """Unarchiving onto a live same-stem job would silently replace its files."""
    import main

    p = main.pipeline
    out, arc = Path(p.output_dir), p.archive_dir()
    arc.mkdir(exist_ok=True)
    (out / "stomp.mp3").write_bytes(b"new")
    (arc / "stomp.mp3").write_bytes(b"old")
    try:
        r = client.post("/api/archive/stomp?undo=1")
        assert r.status_code == 409
        assert (out / "stomp.mp3").read_bytes() == b"new"  # untouched
        assert (arc / "stomp.mp3").read_bytes() == b"old"  # still archived
    finally:
        (out / "stomp.mp3").unlink(missing_ok=True)
        (arc / "stomp.mp3").unlink(missing_ok=True)


def test_upload_url_fetches_and_queues(monkeypatch):
    import main

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"%PDF-1.4 from the web"

    seen = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: (seen.append(req.full_url), FakeResp())[1])
    try:
        r = client.post(
            "/api/upload-url", json={"url": "https://example.com/papers/cool-paper.pdf", "voice": "bf_emma"}
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"saved": "cool-paper.pdf", "queued": True}
        data = json.loads(main.pipeline.status_path("cool-paper").read_text())
        assert data["stage"] == "queued" and data["voice"] == "bf_emma"
        assert (Path(main.pipeline.input_dir) / "cool-paper.pdf").read_bytes() == b"%PDF-1.4 from the web"
    finally:
        while not main.pipeline.q.empty():
            main.pipeline.q.get_nowait()
        (Path(main.pipeline.input_dir) / "cool-paper.pdf").unlink(missing_ok=True)
        main.pipeline.status_path("cool-paper").unlink(missing_ok=True)


def test_upload_url_rewrites_arxiv_abs_page(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"%PDF-1.4 arxiv"

    seen = []
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: (seen.append(req.full_url), FakeResp())[1])
    try:
        r = client.post("/api/upload-url", json={"url": "https://arxiv.org/abs/2301.12345v2"})
        assert r.status_code == 200, r.text
        assert seen == ["https://arxiv.org/pdf/2301.12345v2"]
        assert r.json()["saved"] == "2301.12345v2.pdf"
    finally:
        import main

        while not main.pipeline.q.empty():
            main.pipeline.q.get_nowait()
        (Path(main.pipeline.input_dir) / "2301.12345v2.pdf").unlink(missing_ok=True)
        main.pipeline.status_path("2301.12345v2").unlink(missing_ok=True)


def test_upload_url_rejects_non_pdf_response(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"<html>paywall</html>"

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: FakeResp())
    r = client.post("/api/upload-url", json={"url": "https://example.com/landing"})
    assert r.status_code == 400
    assert "did not return a PDF" in r.json()["detail"]


def test_voice_sample_synthesizes_once_then_serves_cache(monkeypatch):
    import main

    calls = []

    def fake_synth(script, out_path, progress=None, voice=None):
        calls.append(voice)
        Path(out_path).write_bytes(b"fake-sample-mp3")

    monkeypatch.setattr(main, "synthesize", fake_synth)
    try:
        r = client.get("/api/voice-sample/af_heart")
        assert r.status_code == 200 and r.content == b"fake-sample-mp3"
        r = client.get("/api/voice-sample/af_heart")
        assert r.status_code == 200
        assert calls == ["af_heart"]  # second request hit the cache
        assert client.get("/api/voice-sample/not_a_voice").status_code == 400
    finally:
        (Path(main.pipeline.output_dir) / "samples" / "af_heart.mp3").unlink(missing_ok=True)


def test_feed_filters_by_tag():
    import main

    out = Path(main.pipeline.output_dir)
    (out / "tagged-a.mp3").write_bytes(b"m")
    (out / "tagged-a.status.json").write_text('{"title":"A","tags":["llm"]}')
    (out / "tagged-b.mp3").write_bytes(b"m")
    (out / "tagged-b.status.json").write_text('{"title":"B","tags":["vision"]}')
    try:
        r = client.get("/feed.xml", params={"tag": "llm"})
        assert "tagged-a" in r.text and "tagged-b" not in r.text
        assert client.get("/feed.xml").text.count("<item>") >= 2  # no tag: everything
    finally:
        for n in ("tagged-a", "tagged-b"):
            (out / f"{n}.mp3").unlink(missing_ok=True)
            (out / f"{n}.status.json").unlink(missing_ok=True)


def test_upload_enqueues_with_voice_and_mode_already_set():
    """The worker reads mode the instant it picks the job up: voice/mode must be
    in the status file at queue time, not annotated afterwards — else an empty
    queue races the annotation and runs the wrong (expensive) mode."""
    import main

    p = main.pipeline
    try:
        r = client.post(
            "/api/upload",
            files={"files": ("atomic.pdf", b"%PDF-1.4 fake")},
            data={"voice": "am_adam", "mode": "verbatim"},
        )
        assert r.status_code == 200
        data = json.loads(p.status_path("atomic").read_text())
        assert data["stage"] == "queued"
        assert data["voice"] == "am_adam" and data["mode"] == "verbatim"
        assert any(pdf.name == "atomic.pdf" for pdf in list(p.q.queue))
    finally:
        while not p.q.empty():
            p.q.get_nowait()
        (Path(p.input_dir) / "atomic.pdf").unlink(missing_ok=True)
        p.status_path("atomic").unlink(missing_ok=True)
