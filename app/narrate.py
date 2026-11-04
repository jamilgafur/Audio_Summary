"""Turn extracted paper text into a spoken narration script via a local GGUF LLM."""
from __future__ import annotations

import os
from pathlib import Path

MAX_CHUNK_WORDS = 1200  # ~1600 tokens; fits an 8k context with room for the reply

_SYSTEM = (
    "You rewrite research-paper excerpts as clear spoken narration for an expert audience. "
    "Explain the ideas naturally, as a good colleague summarizing the paper aloud. "
    "Never mention equations, figure numbers, table numbers, or citations. "
    "Output plain prose paragraphs only — no markdown, no headers, no bullet points."
)

_LLM = None  # cached across papers; loading the model is the expensive part


def _chunk_words(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + MAX_CHUNK_WORDS]) for i in range(0, len(words), MAX_CHUNK_WORDS)] or [""]


def _load_llm():
    global _LLM
    if _LLM is None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        repo = os.getenv("GGUF_REPO", "bartowski/Qwen2.5-7B-Instruct-GGUF")
        fname = os.getenv("GGUF_FILE", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        models_dir = os.getenv("MODELS_DIR", "./models")
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(repo, fname, local_dir=models_dir)
        _LLM = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "-1")),  # -1: all layers; 0: CPU only
            verbose=False,
        )
    return _LLM


def narrate(paper: dict, progress=None) -> str:
    """paper: dict from extract.extract(). Returns the full narration script."""
    title, authors = paper["title"], paper["authors"] or "unknown authors"
    intro = f"You're listening to {title}, by {authors}.\n\n"

    if os.getenv("NARRATE_MOCK"):  # test path: no model download, instant
        return intro + "This is a mock narration used by the test suite. " * 30

    llm = _load_llm()
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
        if progress:
            progress(i + 1, len(chunks))
    return "\n\n".join(parts)
