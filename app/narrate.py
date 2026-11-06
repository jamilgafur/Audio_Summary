"""Turn extracted paper text into a spoken narration script via a local GGUF LLM.

The LLM runs in a spawned child process: llama.cpp hard-aborts mid-job (ggml_abort
on CUDA OOM kills the process — not a catchable exception), and that must not take
the web app down with it. A crash surfaces as ChildCrashedError, and the caller
retries once on CPU. The child's exit also frees the VRAM deterministically.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

MAX_CHUNK_WORDS = 1200  # ~1600 tokens; fits an 8k context with room for the reply

_SYSTEM = (
    "You rewrite research-paper excerpts as clear spoken narration for an expert audience. "
    "Explain the ideas naturally, as a good colleague summarizing the paper aloud. "
    "Never mention equations, figure numbers, table numbers, or citations. "
    "Output plain prose paragraphs only — no markdown, no headers, no bullet points."
)


class ChildCrashedError(RuntimeError):
    """The narration child died without returning a script (usually ggml_abort on CUDA OOM)."""

    def __init__(self, exitcode: int | None, backend: str | None):
        super().__init__(f"narration crashed (child exit code {exitcode}, backend={backend})")
        self.exitcode = exitcode
        self.backend = backend  # backend the child reported before dying; None = died at/before load


def _chunk_words(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + MAX_CHUNK_WORDS]) for i in range(0, len(words), MAX_CHUNK_WORDS)] or [""]


def verbatim(paper: dict) -> str:
    """No-LLM mode: the extracted text read exactly as written — for auditing your
    own paper aloud. Section headings are spoken (they're part of the paper)."""
    title, authors = paper["title"], paper["authors"] or "unknown authors"
    parts = [f"You're listening to {title}, by {authors}. Read verbatim.\n\n"]
    for s in paper["sections"]:
        text = s["text"].strip()
        if not text:
            continue
        heading = (s.get("title") or "").strip()
        parts.append((heading + ".\n" if heading and heading != "full text" else "") + text)
    return "\n\n".join(parts)


def _load_llm(force_cpu: bool):
    """Child-side model load. Returns (llm, backend) with backend in gpu|cpu|cpu-fallback."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    repo = os.getenv("GGUF_REPO", "bartowski/Qwen2.5-7B-Instruct-GGUF")
    fname = os.getenv("GGUF_FILE", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    models_dir = os.getenv("MODELS_DIR", "./models")
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(repo, fname, local_dir=models_dir)
    layers = 0 if force_cpu else int(os.getenv("N_GPU_LAYERS", "-1"))  # -1: all layers
    try:
        llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=layers, verbose=False)
        return llm, "gpu" if layers else "cpu"
    except Exception:
        if not layers:  # already CPU; nothing to fall back to — fail the job
            raise
        # VRAM was taken by a resident service (ollama/docling) — narrate slow rather than never
        print("llama.cpp GPU context failed; retrying on CPU (n_gpu_layers=0)", flush=True)
        llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=0, verbose=False)
        return llm, "cpu-fallback"


def _generate(paper: dict, emit, force_cpu: bool) -> str:
    """The whole narration. emit(kind, *args) reports backend/progress events."""
    title, authors = paper["title"], paper["authors"] or "unknown authors"
    intro = f"You're listening to {title}, by {authors}.\n\n"

    llm, backend = _load_llm(force_cpu)
    emit("backend", backend)
    chunks = [c for s in paper["sections"] for c in _chunk_words(s["text"])]
    parts = [intro]
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"This is part {i + 1} of {len(chunks)} of the paper \"{title}\". "
                    "Rewrite it as spoken narration in 2-4 paragraphs.\n\n" + chunk
                ),
            },
        ]
        reply = llm.create_chat_completion(messages=messages, temperature=0.5, max_tokens=1400)
        parts.append(reply["choices"][0]["message"]["content"].strip())
        emit("progress", i + 1, len(chunks))
    return "\n\n".join(parts)


def _child_main(conn, force_cpu: bool, paper: dict):
    """Runs in the child. Any exception becomes an 'error' event; a hard abort just kills it."""
    try:
        conn.send(("done", _generate(paper, lambda *a: conn.send(a), force_cpu)))  # conn.send takes ONE payload
    except BaseException as e:  # payload must survive pickling: send the repr, not the exception
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass  # child dies anyway; parent sees a crash


def _run_child(target, args: tuple, progress=None, on_backend=None) -> tuple[str, str | None]:
    """Run target(conn, *args) in a spawned child, streaming its events to our callbacks.

    Returns (script, backend). Raises RuntimeError on in-child exceptions, and
    ChildCrashedError when the child dies without a script — the app keeps running either way.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=target, args=(child_conn, *args), daemon=True)
    proc.start()
    child_conn.close()  # parent holds the read end only
    script = backend = None
    try:
        while True:
            if parent_conn.poll(0.5):
                try:
                    kind, *rest = parent_conn.recv()
                except (EOFError, OSError):  # pipe torn down by a dying child
                    break
                if kind == "progress" and progress:
                    progress(*rest)
                elif kind == "backend":
                    backend = rest[0]
                    if on_backend:
                        on_backend(*rest)
                elif kind == "done":
                    script = rest[0]
                elif kind == "error":
                    raise RuntimeError(f"narration failed: {rest[0]}")
            elif not proc.is_alive():
                break
    finally:
        proc.join()
        exitcode = proc.exitcode  # read before close; closed process objects raise
        proc.close()
        parent_conn.close()
    if script is not None:
        return script, backend
    if exitcode == 0:  # exited cleanly but breached the contract — a bug, not a retry case
        raise RuntimeError("narration child exited without returning a script")
    raise ChildCrashedError(exitcode, backend)


def narrate(paper: dict, progress=None, on_backend=None) -> str:
    """paper: dict from extract.extract(). Returns the full narration script."""
    if os.getenv("NARRATE_MOCK"):  # test path: no model download, instant
        title, authors = paper["title"], paper["authors"] or "unknown authors"
        return f"You're listening to {title}, by {authors}.\n\n" + "This is a mock narration used by the test suite. " * 30

    try:
        return _run_child(_child_main, (False, paper), progress, on_backend)[0]
    except ChildCrashedError as e:
        if e.backend in ("cpu", "cpu-fallback"):
            raise  # already on CPU and still died — retrying gains nothing
        print(f"{e}; retrying once on CPU", flush=True)
        return _run_child(_child_main, (True, paper), progress, on_backend)[0]
