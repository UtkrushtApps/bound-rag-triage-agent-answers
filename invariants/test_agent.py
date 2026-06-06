"""Candidate-facing invariant tests.

Run these yourself after filling the stubs:
  pytest invariants/ -v

These tests do NOT run during the readiness check (run.sh) and are NOT
a generation-time gate. They are feedback for the candidate.

All tests except the live-model integration test are LLM-free and deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path("fixtures/tickets.jsonl")


def load_tickets() -> list[dict]:
    return [json.loads(l) for l in FIXTURES.read_text().splitlines() if l.strip()]


def make_state(
    ticket: dict,
    steps: int = 0,
    session_cost: float = 0.0,
    messages: list | None = None,
) -> dict:
    return {
        "ticket": ticket,
        "messages": messages or [],
        "chunks": [],
        "steps": steps,
        "session_cost": session_cost,
        "output": {},
    }


# ---------------------------------------------------------------------------
# 1. Loop cap — should_continue must route to escalate at MAX_STEPS
# ---------------------------------------------------------------------------


def test_loop_cap_routes_to_escalate():
    """should_continue must return 'escalate' when steps >= MAX_STEPS.

    This test is LLM-free. It directly calls the router with a state at
    the step cap and asserts the edge returned is 'escalate'.
    """
    from agent.graph import MAX_STEPS, should_continue

    ticket = load_tickets()[0]
    state_at_cap = make_state(
        ticket,
        steps=MAX_STEPS,
        session_cost=0.01,
        messages=[{"role": "assistant", "content": "Still reasoning, no answer yet."}],
    )
    result = should_continue(state_at_cap)
    assert result == "escalate", (
        f"should_continue returned {result!r} when steps={MAX_STEPS} >= MAX_STEPS={MAX_STEPS}. "
        "Expected 'escalate'. Fix the missing loop-cap guard in agent/graph.py should_continue()."
    )


def test_loop_cap_one_below_does_not_escalate():
    """should_continue must NOT escalate when steps == MAX_STEPS - 1 (one step remaining)."""
    from agent.graph import MAX_STEPS, should_continue

    ticket = load_tickets()[0]
    state_below_cap = make_state(
        ticket,
        steps=MAX_STEPS - 1,
        session_cost=0.01,
        messages=[{"role": "assistant", "content": "Still thinking."}],
    )
    result = should_continue(state_below_cap)
    assert result != "escalate", (
        f"should_continue escalated at steps={MAX_STEPS - 1} but cap is {MAX_STEPS}. "
        "The loop cap is firing one step too early."
    )


def test_loop_cap_beyond_max_also_escalates():
    """should_continue must escalate when steps > MAX_STEPS (e.g. a state that overshot)."""
    from agent.graph import MAX_STEPS, should_continue

    ticket = load_tickets()[0]
    state_over = make_state(
        ticket,
        steps=MAX_STEPS + 2,
        session_cost=0.01,
        messages=[{"role": "assistant", "content": "Still going."}],
    )
    result = should_continue(state_over)
    assert result == "escalate", (
        f"should_continue returned {result!r} when steps={MAX_STEPS + 2} > MAX_STEPS={MAX_STEPS}."
    )


# ---------------------------------------------------------------------------
# 2. Cost ceiling — enforce_budget contract (LLM-free, deterministic)
# ---------------------------------------------------------------------------


def test_budget_ceiling_enforced_near_limit():
    """enforce_budget must raise BudgetExceededError when projected cost would exceed $0.08.

    Uses a session already at $0.075 with 4 chunks and 3 history turns — the
    projected next call will push over the $0.08 ceiling.
    """
    from agent.policy import BUDGET_CEILING, BudgetExceededError, enforce_budget

    near_ceiling = BUDGET_CEILING - 0.005  # $0.075
    chunks = ["context chunk A", "context chunk B", "context chunk C", "context chunk D"]
    history = [
        {"role": "user", "content": "what is wrong with the autoclave?"},
        {"role": "assistant", "content": "Let me check the seals."},
        {"role": "user", "content": "still leaking"},
        {"role": "assistant", "content": "Try the hinge bolts."},
        {"role": "user", "content": "checked, still leaking"},
        {"role": "assistant", "content": "Escalating to engineering."},
    ]

    with pytest.raises(BudgetExceededError) as exc_info:
        enforce_budget(session_cost=near_ceiling, chunks=chunks, history=history)

    err = exc_info.value
    assert err.session_cost == near_ceiling, (
        f"BudgetExceededError.session_cost should be {near_ceiling}, got {err.session_cost}"
    )
    assert err.ceiling == BUDGET_CEILING, (
        f"BudgetExceededError.ceiling should be {BUDGET_CEILING}, got {err.ceiling}"
    )
    assert err.projected > 0, "BudgetExceededError.projected must be a positive cost estimate"


def test_budget_allows_fresh_session():
    """enforce_budget must NOT raise when session cost is zero."""
    from agent.policy import BudgetExceededError, enforce_budget

    try:
        enforce_budget(session_cost=0.0, chunks=["short context"], history=[])
    except BudgetExceededError:
        pytest.fail(
            "enforce_budget raised BudgetExceededError on a fresh session (session_cost=0.0). "
            "The budget guard is firing too eagerly."
        )
    except NotImplementedError:
        pytest.fail("enforce_budget is not implemented yet.")


def test_budget_ceiling_exactly_at_limit_is_blocked():
    """A session already AT the ceiling must be blocked even for a tiny projected cost."""
    from agent.policy import BUDGET_CEILING, BudgetExceededError, enforce_budget

    with pytest.raises(BudgetExceededError):
        enforce_budget(
            session_cost=BUDGET_CEILING,
            chunks=["one chunk"],
            history=[],
        )


def test_budget_error_attributes_are_populated():
    """BudgetExceededError must carry session_cost, projected, and ceiling attributes."""
    from agent.policy import BUDGET_CEILING, BudgetExceededError, enforce_budget

    try:
        enforce_budget(
            session_cost=BUDGET_CEILING,
            chunks=["chunk"],
            history=[],
        )
    except BudgetExceededError as exc:
        assert hasattr(exc, "session_cost"), "BudgetExceededError missing .session_cost"
        assert hasattr(exc, "projected"), "BudgetExceededError missing .projected"
        assert hasattr(exc, "ceiling"), "BudgetExceededError missing .ceiling"
        assert isinstance(exc.projected, float) and exc.projected > 0, (
            f".projected must be a positive float, got {exc.projected!r}"
        )
    except NotImplementedError:
        pytest.fail("enforce_budget is not implemented yet.")


# ---------------------------------------------------------------------------
# 3. Recall — relevance-keyed retrieval (LLM-free, uses local fastembed)
# ---------------------------------------------------------------------------


def test_recall_returns_at_most_top_k():
    """recall() must return at most TOP_K_CHUNKS texts, never the full corpus."""
    from agent.policy import TOP_K_CHUNKS, recall

    total_chunks = sum(
        1 for l in Path("fixtures/kb_chunks.jsonl").read_text().splitlines() if l.strip()
    )

    result = recall("autoclave door seal leak Tuttnauer", top_k=TOP_K_CHUNKS)

    assert isinstance(result, list), f"recall() must return a list, got {type(result)}"
    assert len(result) <= TOP_K_CHUNKS, (
        f"recall() returned {len(result)} chunks but top_k={TOP_K_CHUNKS}. "
        f"It must not dump the full corpus ({total_chunks} chunks)."
    )
    assert len(result) < total_chunks, (
        f"recall() returned all {total_chunks} chunks — the relevance filter is not working."
    )


def test_recall_top_result_is_relevant_for_autoclave_query():
    """Top recall result for an autoclave query must contain autoclave-related content."""
    from agent.policy import recall

    results = recall("autoclave door seal leak steam", top_k=3)
    assert results, "recall() returned an empty list"
    combined = " ".join(results).lower()
    assert any(word in combined for word in ["autoclave", "seal", "gasket", "steam", "door"]), (
        f"Top recall results for an autoclave query contain no autoclave-related text. "
        f"Got: {results[:1]}"
    )


def test_recall_top_result_is_relevant_for_infusion_pump_query():
    """Top recall result for an infusion pump query must reference pumps or occlusion."""
    from agent.policy import recall

    results = recall("infusion pump P-500 false occlusion alarm clear line", top_k=3)
    assert results, "recall() returned an empty list for infusion pump query"
    combined = " ".join(results).lower()
    assert any(word in combined for word in ["infusion", "pump", "occlusion", "p-500", "platen"]), (
        f"Top recall results for an infusion pump query missing expected terms. Got: {results[:1]}"
    )


def test_recall_discriminating_pm_fixture():
    """PM scheduling query must retrieve PM chunks, NOT only clinical-repair chunks.

    This is the discriminating fixture: a purely administrative query that shares
    no surface vocabulary with the equipment-fault repair chunks. Keyword or TF-IDF
    approaches fail here; semantic embedding recall succeeds.
    """
    from agent.policy import recall

    results = recall(
        "preventive maintenance schedule for infusion pumps and patient monitors", top_k=4
    )
    assert results, "recall() returned an empty list for PM scheduling query"
    combined = " ".join(results).lower()

    assert any(
        word in combined for word in ["preventive", "pm", "interval", "campaign", "cmms", "maintenance"]
    ), (
        f"recall() missed PM scheduling chunks for an explicit PM query. "
        f"Semantic similarity is not distinguishing administrative from clinical content. "
        f"Got: {results}"
    )

    repair_only = all(all(w in r.lower() for w in ["part:"]) for r in results)
    assert not repair_only, (
        "recall() returned only repair/fault chunks for a PM scheduling query. "
        "The embedding model is not surfacing the PM scheduling section."
    )


def test_recall_returns_strings():
    """recall() must return a list of strings (chunk texts), not chunk dicts."""
    from agent.policy import recall

    results = recall("autoclave", top_k=2)
    for item in results:
        assert isinstance(item, str), (
            f"recall() must return list[str], but got an item of type {type(item)}: {item!r}"
        )


# ---------------------------------------------------------------------------
# 4. Escalation output shape (LLM-free)
# ---------------------------------------------------------------------------


def test_escalate_node_produces_valid_output_on_budget_hit():
    """escalate_node must return a well-formed output dict when the budget ceiling is hit."""
    from agent.graph import escalate_node
    from agent.policy import BUDGET_CEILING

    ticket = load_tickets()[0]
    state = make_state(ticket, steps=3, session_cost=BUDGET_CEILING + 0.01)
    result = escalate_node(state)
    out = result["output"]

    assert out.get("disposition") == "escalate", (
        f"Expected disposition='escalate', got: {out.get('disposition')!r}"
    )
    assert "summary" in out and out["summary"], "escalation output must include a non-empty summary"
    assert out.get("ticket_id") == ticket["id"], (
        f"ticket_id mismatch: expected {ticket['id']!r}, got {out.get('ticket_id')!r}"
    )
    assert isinstance(out.get("session_cost"), float), "session_cost must be a float"
    assert out.get("reason") == "budget_ceiling", (
        f"Expected reason='budget_ceiling' when cost exceeds ceiling, got {out.get('reason')!r}"
    )


def test_escalate_node_produces_valid_output_on_loop_cap():
    """escalate_node must return disposition='escalate' with reason='loop_cap' at step limit."""
    from agent.graph import MAX_STEPS, escalate_node

    ticket = load_tickets()[3]
    state = make_state(ticket, steps=MAX_STEPS, session_cost=0.005)
    result = escalate_node(state)
    out = result["output"]

    assert out.get("disposition") == "escalate"
    assert out.get("reason") == "loop_cap", (
        f"Expected reason='loop_cap' at step cap with low cost, got {out.get('reason')!r}"
    )
    assert "summary" in out and out["summary"]
    assert out.get("ticket_id") == ticket["id"]
