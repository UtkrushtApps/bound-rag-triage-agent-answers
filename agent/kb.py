"""Knowledge-base loader and retrieval utilities.

The KB is stored as a flat JSONL of chunks under fixtures/kb_chunks.jsonl.
Each chunk has: {id, section, text}.

Public functions:
  - load_corpus() -> dict  : loads chunks + builds fastembed embeddings (cached).
  - dump_all()    -> list  : returns ALL chunk texts (retained for compatibility).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

_CHUNKS_PATH = Path("fixtures/kb_chunks.jsonl")
_CACHE_PATH = Path(".cache/kb_corpus.pkl")
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding row for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _write_cache(corpus: dict[str, Any]) -> None:
    """Persist the prepared corpus to the local cache path."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("wb") as handle:
        pickle.dump(corpus, handle)


def _build_corpus() -> dict[str, Any]:
    """Build the corpus and embeddings from the JSONL fixture."""
    from fastembed import TextEmbedding  # noqa: PLC0415

    chunks_raw = [json.loads(line) for line in _CHUNKS_PATH.read_text().splitlines() if line.strip()]
    texts = [chunk["text"] for chunk in chunks_raw]

    model = TextEmbedding(model_name=_MODEL_NAME)
    embeddings = np.asarray(list(model.embed(texts)), dtype=np.float32)
    normalized_embeddings = _normalize_rows(embeddings)

    corpus = {
        "chunks": chunks_raw,
        "texts": texts,
        "embeddings": embeddings,
        "normalized_embeddings": normalized_embeddings,
        "model_name": _MODEL_NAME,
    }
    _write_cache(corpus)
    return corpus


def load_corpus() -> dict[str, Any]:
    """Load KB chunks and their fastembed embeddings. Caches to .cache/."""
    if _CACHE_PATH.exists():
        with _CACHE_PATH.open("rb") as handle:
            corpus = pickle.load(handle)

        # Backfill derived fields if an older cache format is present.
        embeddings = np.asarray(corpus["embeddings"], dtype=np.float32)
        corpus["embeddings"] = embeddings
        if "normalized_embeddings" not in corpus:
            corpus["normalized_embeddings"] = _normalize_rows(embeddings)
            _write_cache(corpus)
        return corpus

    return _build_corpus()


def dump_all() -> list[str]:
    """Return ALL KB chunk texts — retained for compatibility/testing."""
    corpus = load_corpus()
    return corpus["texts"]
