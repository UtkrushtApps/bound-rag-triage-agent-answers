"""Session policy: cost ceiling enforcement and relevance-keyed KB recall."""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np

from agent import kb

logger = logging.getLogger(__name__)

BUDGET_CEILING: float = 0.08
TOP_K_CHUNKS: int = 4

# Conservative preflight budget estimate.
PROMPT_RATE_PER_1K: float = 0.00025
COMPLETION_RATE_PER_1K: float = 0.00125
ESTIMATED_COMPLETION_TOKENS: int = 384
MIN_PRECALL_COST: float = 0.006


class BudgetExceededError(Exception):
    """Raised by enforce_budget when the session cost ceiling would be breached."""

    def __init__(self, session_cost: float, projected: float, ceiling: float) -> None:
        self.session_cost = session_cost
        self.projected = projected
        self.ceiling = ceiling
        super().__init__(
            f"budget ceiling: session={session_cost:.4f} projected={projected:.4f} ceiling={ceiling:.4f}"
        )


@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    """Reuse the query embedding model across recall calls."""
    from fastembed import TextEmbedding  # noqa: PLC0415

    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a single embedding vector."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _estimate_text_tokens(text: str) -> int:
    """Fast, model-agnostic token estimate using character length."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _estimate_next_call_cost(chunks: list[str], history: list[dict[str, Any]]) -> float:
    """Conservatively estimate the cost of the next model call.

    The graph already computes actual spend after the call. This policy guard is a
    preflight safety check, so it intentionally reserves a minimum amount for the
    upcoming completion to avoid crossing the hard session ceiling.
    """
    system_overhead_tokens = 300
    query_reserve_tokens = 64
    history_overhead_tokens = len(history) * 8

    prompt_tokens = system_overhead_tokens + query_reserve_tokens + history_overhead_tokens
    prompt_tokens += sum(_estimate_text_tokens(chunk) for chunk in chunks)
    prompt_tokens += sum(_estimate_text_tokens(str(message.get("content", ""))) for message in history)

    estimated_cost = (
        prompt_tokens * PROMPT_RATE_PER_1K
        + ESTIMATED_COMPLETION_TOKENS * COMPLETION_RATE_PER_1K
    ) / 1000

    return max(float(estimated_cost), MIN_PRECALL_COST)


def enforce_budget(
    session_cost: float,
    chunks: list[str],
    history: list[dict[str, Any]],
) -> None:
    """Raise BudgetExceededError if the next model call would breach the ceiling."""
    next_call_cost = _estimate_next_call_cost(chunks=chunks, history=history)
    projected = float(session_cost + next_call_cost)

    logger.info(
        "[budget] session_cost=%.5f projected=%.5f ceiling=%.5f chunks=%d history_messages=%d",
        session_cost,
        projected,
        BUDGET_CEILING,
        len(chunks),
        len(history),
    )

    if session_cost >= BUDGET_CEILING or projected >= BUDGET_CEILING:
        raise BudgetExceededError(
            session_cost=float(session_cost),
            projected=projected,
            ceiling=float(BUDGET_CEILING),
        )


def recall(query: str, top_k: int = TOP_K_CHUNKS) -> list[str]:
    """Return the top_k KB chunk texts most semantically relevant to query."""
    query = query.strip()
    if not query:
        return []

    limit = max(0, min(int(top_k), TOP_K_CHUNKS))
    if limit == 0:
        return []

    corpus = kb.load_corpus()
    texts: list[str] = corpus["texts"]
    normalized_embeddings = np.asarray(corpus["normalized_embeddings"], dtype=np.float32)

    query_embedding = np.asarray(list(_embedding_model().embed([query]))[0], dtype=np.float32)
    query_embedding = _normalize(query_embedding).astype(np.float32)

    scores = normalized_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:limit]
    results = [texts[int(idx)] for idx in ranked_indices]

    logger.info(
        "[recall] query=%r top_k=%d returned=%d",
        query,
        top_k,
        len(results),
    )
    return results
