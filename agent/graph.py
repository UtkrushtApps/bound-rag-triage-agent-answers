"""LangGraph graph definition for the triage agent.

The graph has three nodes:
  - retrieve: pulls KB context for the current query
  - reason:   calls the model with context + conversation history
  - respond:  formats and validates the final output

This implementation fixes three production issues:
  1. Retrieval is relevance-keyed and bounded to top-k KB chunks.
  2. Every model call is budget-guarded before execution.
  3. The retrieve/reason loop is capped at MAX_STEPS and degrades to escalation.
"""
from __future__ import annotations

import logging
import os
from typing import Any, NotRequired, TypedDict

import litellm
from langgraph.graph import END, StateGraph

from agent import policy, prompts

logger = logging.getLogger(__name__)

MODEL = os.environ.get("TRIAGE_MODEL", "anthropic/claude-3-haiku-20240307")
MAX_STEPS = 5


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class TriageState(TypedDict):
    ticket: dict[str, Any]
    messages: list[dict[str, Any]]
    chunks: list[str]
    steps: int
    session_cost: float
    output: dict[str, Any]
    escalation_reason: NotRequired[str]
    budget_error: NotRequired[dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_text(value: Any) -> str:
    """Convert LLM content payloads into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _usage_value(usage: Any, key: str) -> int:
    """Read token counts from either a dict-like or object-like usage payload."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _last_assistant_message(messages: list[dict[str, Any]]) -> str:
    """Return the most recent assistant message content, if any."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _coerce_text(message.get("content"))
    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve_node(state: TriageState) -> TriageState:
    """Retrieve only the most relevant KB chunks for the current query."""
    query = state["ticket"]["query"]
    chunks = policy.recall(query=query, top_k=policy.TOP_K_CHUNKS)

    logger.info(
        "[retrieve] query=%r chunks_retrieved=%d top_k=%d",
        query,
        len(chunks),
        policy.TOP_K_CHUNKS,
    )

    return {
        **state,
        "chunks": chunks,
    }


def reason_node(state: TriageState) -> TriageState:
    """Call the model with the current context.

    BudgetExceededError is converted into explicit escalation state so the graph
    can terminate cleanly without making the model call.
    """
    query = state["ticket"]["query"]
    chunks = state["chunks"]
    history = state["messages"]
    session_cost = state["session_cost"]
    steps = state["steps"]

    logger.info(
        "[reason] step=%d query=%r session_cost=%.5f chunks=%d history_messages=%d",
        steps,
        query,
        session_cost,
        len(chunks),
        len(history),
    )

    try:
        policy.enforce_budget(session_cost=session_cost, chunks=chunks, history=history)
    except policy.BudgetExceededError as exc:
        logger.warning(
            "[reason] budget_blocked query=%r session_cost=%.5f projected=%.5f ceiling=%.5f",
            query,
            exc.session_cost,
            exc.projected,
            exc.ceiling,
        )
        return {
            **state,
            "escalation_reason": "budget_ceiling",
            "budget_error": {
                "session_cost": float(exc.session_cost),
                "projected": float(exc.projected),
                "ceiling": float(exc.ceiling),
            },
        }

    system_prompt = prompts.build_system(chunks)
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": query}
    ]

    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        max_tokens=512,
    )

    content = _coerce_text(resp.choices[0].message.content)
    usage = getattr(resp, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")

    # Rough actual cost estimate: haiku ~$0.00025/1k prompt, $0.00125/1k completion
    step_cost = (prompt_tokens * 0.00025 + completion_tokens * 0.00125) / 1000
    new_cost = session_cost + step_cost

    logger.info(
        "[reason] query=%r prompt_tokens=%d completion_tokens=%d step_cost=%.5f session_cost=%.5f",
        query,
        prompt_tokens,
        completion_tokens,
        step_cost,
        new_cost,
    )

    new_messages = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": content},
    ]

    next_state: TriageState = {
        **state,
        "messages": new_messages,
        "steps": steps + 1,
        "session_cost": new_cost,
    }
    next_state.pop("escalation_reason", None)
    next_state.pop("budget_error", None)
    return next_state


def respond_node(state: TriageState) -> TriageState:
    """Format the final triage output from the last assistant message."""
    last_msg = _last_assistant_message(state["messages"])
    output = prompts.parse_response(last_msg, ticket_id=state["ticket"]["id"])
    logger.info("[respond] ticket=%s disposition=%s", state["ticket"]["id"], output.get("disposition"))
    return {**state, "output": output}


def escalate_node(state: TriageState) -> TriageState:
    """Produce a safe escalation response for budget or loop-limit exhaustion."""
    reason = state.get("escalation_reason")
    if reason not in {"budget_ceiling", "loop_cap"}:
        if state["session_cost"] >= policy.BUDGET_CEILING:
            reason = "budget_ceiling"
        elif state["steps"] >= MAX_STEPS:
            reason = "loop_cap"
        else:
            reason = "loop_cap"

    logger.warning(
        "[escalate] ticket=%s reason=%s steps=%d cost=%.5f",
        state["ticket"]["id"],
        reason,
        state["steps"],
        state["session_cost"],
    )

    if reason == "budget_ceiling":
        summary = "Automatic escalation: session budget ceiling reached before the next model call. A human specialist will follow up."
    else:
        summary = "Automatic escalation: reasoning step limit reached. A human specialist will follow up."

    output = {
        "ticket_id": state["ticket"]["id"],
        "disposition": "escalate",
        "reason": reason,
        "summary": summary,
        "steps": state["steps"],
        "session_cost": float(round(state["session_cost"], 5)),
    }
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def should_continue(state: TriageState) -> str:
    """Decide whether to loop, respond, or escalate.

    Boundary behavior is intentional:
      - steps >= MAX_STEPS  => escalate
      - cost  >= ceiling    => escalate
      - FINAL_ANSWER        => respond
      - otherwise           => retrieve
    """
    if state.get("escalation_reason") == "budget_ceiling":
        return "escalate"

    if state["steps"] >= MAX_STEPS:
        return "escalate"

    if state["session_cost"] >= policy.BUDGET_CEILING:
        return "escalate"

    last_msg = _last_assistant_message(state["messages"])
    if "FINAL_ANSWER" in last_msg:
        return "respond"

    return "retrieve"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and return the (uncompiled) triage graph."""
    g = StateGraph(TriageState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("reason", reason_node)
    g.add_node("respond", respond_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "reason")
    g.add_conditional_edges(
        "reason",
        should_continue,
        {
            "retrieve": "retrieve",
            "respond": "respond",
            "escalate": "escalate",
        },
    )
    g.add_edge("respond", END)
    g.add_edge("escalate", END)

    return g
