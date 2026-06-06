"""Entry point: CLI runner + selfcheck probe."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _configure_logging() -> None:
    """Configure process-wide logging once for CLI runs."""
    logging.basicConfig(
        level=os.environ.get("AGENT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _ping_model() -> None:
    """One-token ping to confirm the model endpoint is reachable. Key-gated."""
    import litellm  # noqa: PLC0415

    resp = litellm.completion(
        model=os.environ.get("TRIAGE_MODEL", "anthropic/claude-3-haiku-20240307"),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    name = resp.model or "(unknown)"
    print(f"note: model ping ok — {name}")


def selfcheck() -> None:
    """LLM-free readiness probe.

    Checks:
      1. All agent modules import cleanly (stubs raise NotImplementedError only on CALL).
      2. Fixture files exist and have the expected shape.
      3. KB chunk file is present and non-empty (does NOT build embeddings — that is the
         candidate's concern and would require a model download at the gate).
      4. Graph topology compiles (no node execution).
      5. Policy stubs are present (does NOT call them).
      6. Key-gated model ping (skipped when no key is present).
    """
    print("[selfcheck] importing agent modules...")

    # 1. Imports — raises immediately if any module has a syntax error or bad import
    import importlib  # noqa: PLC0415

    for mod_name in ("agent.kb", "agent.policy", "agent.prompts", "agent.graph"):
        try:
            importlib.import_module(mod_name)
        except NotImplementedError:
            # A stub that raises NotImplementedError at module level is a scaffold bug;
            # stubs should only raise when their function is CALLED.
            print(
                f"[selfcheck] FAIL: {mod_name} raised NotImplementedError at import time",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            print(f"[selfcheck] FAIL: could not import {mod_name}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"[selfcheck] ok: import {mod_name}")

    import agent.graph as _graph  # noqa: PLC0415
    import agent.kb as _kb  # noqa: PLC0415
    import agent.policy as _policy  # noqa: PLC0415

    # 2. Fixture file: tickets
    fixture_path = Path("fixtures/tickets.jsonl")
    if not fixture_path.exists():
        print(f"[selfcheck] FAIL: missing fixture {fixture_path}", file=sys.stderr)
        sys.exit(1)
    tickets = [json.loads(l) for l in fixture_path.read_text().splitlines() if l.strip()]
    if len(tickets) < 8:
        print(f"[selfcheck] FAIL: expected >= 8 fixture tickets, got {len(tickets)}", file=sys.stderr)
        sys.exit(1)
    for t in tickets:
        if "id" not in t or "query" not in t:
            print(f"[selfcheck] FAIL: ticket missing id/query: {t}", file=sys.stderr)
            sys.exit(1)
    print(f"[selfcheck] ok: tickets fixture ({len(tickets)} rows)")

    # 3. KB chunks file: exists and is non-empty — do NOT build embeddings here
    kb_path = Path("fixtures/kb_chunks.jsonl")
    if not kb_path.exists():
        print(f"[selfcheck] FAIL: missing KB fixture {kb_path}", file=sys.stderr)
        sys.exit(1)
    kb_lines = [l for l in kb_path.read_text().splitlines() if l.strip()]
    if not kb_lines:
        print(f"[selfcheck] FAIL: KB fixture is empty", file=sys.stderr)
        sys.exit(1)
    # Verify rows are parseable and have the expected keys
    for raw in kb_lines:
        row = json.loads(raw)
        if "id" not in row or "text" not in row:
            print(f"[selfcheck] FAIL: KB chunk missing id/text: {raw[:80]}", file=sys.stderr)
            sys.exit(1)
    print(f"[selfcheck] ok: KB chunks fixture ({len(kb_lines)} chunks, embeddings built on first agent run)")

    # 4. Graph topology compiles — does NOT execute any node
    try:
        g = _graph.build_graph()
        _ = g.compile()
        print("[selfcheck] ok: graph compiled")
    except Exception as exc:  # noqa: BLE001
        print(f"[selfcheck] FAIL: graph.compile() raised: {exc}", file=sys.stderr)
        sys.exit(1)

    # 5. Policy stubs are present (do NOT call them)
    for attr in ("enforce_budget", "recall", "BUDGET_CEILING", "TOP_K_CHUNKS", "BudgetExceededError"):
        if not hasattr(_policy, attr):
            print(f"[selfcheck] FAIL: agent.policy missing attribute '{attr}'", file=sys.stderr)
            sys.exit(1)
    print("[selfcheck] ok: policy attributes present")

    # 6. Key-gated model ping — skipped at the deploy gate (no key present)
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            _ping_model()
        except Exception as exc:  # noqa: BLE001
            print(f"[selfcheck] FAIL: model ping: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("note: no ANTHROPIC_API_KEY/OPENAI_API_KEY found — skipping model ping")

    print("[selfcheck] all checks passed")


def run_tickets(path: str) -> None:
    """Process tickets from a JSONL file and print structured results."""
    from agent.graph import build_graph  # noqa: PLC0415

    tickets = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    app = build_graph().compile()

    for ticket in tickets:
        print(f"\n--- ticket {ticket['id']} ---")
        result = app.invoke(
            {
                "ticket": ticket,
                "steps": 0,
                "session_cost": 0.0,
                "messages": [],
                "chunks": [],
                "output": {},
            }
        )
        print(json.dumps(result.get("output", {}), indent=2))


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("--selfcheck", action="store_true", help="run readiness probe and exit")
    parser.add_argument("--ticket", metavar="FILE", help="process tickets from JSONL file")
    args = parser.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit(0)

    if args.ticket:
        run_tickets(args.ticket)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
