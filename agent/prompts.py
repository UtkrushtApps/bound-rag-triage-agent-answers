"""System prompt builder and response parser for the triage agent."""
from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_TEMPLATE = """You are a support triage assistant for medical-equipment field technicians.
Your job is to analyze the technician's query and the provided knowledge-base context,
then produce a structured triage decision.

Knowledge base context:
---
{context}
---

Respond with a JSON object wrapped in a FINAL_ANSWER block:

FINAL_ANSWER
{{"disposition": "resolve" | "parts_request" | "escalate", "summary": "<one sentence>", "detail": "<optional steps or part numbers>"}}
END_ANSWER

If you need to reason before answering, write your reasoning first, then the FINAL_ANSWER block.
Do not make up part numbers or procedures not present in the context.
If the context is insufficient, disposition must be "escalate".
"""


def build_system(chunks: list[str]) -> str:
    """Render the system prompt with the provided KB chunks as context."""
    context = "\n\n".join(chunks) if chunks else "(no context available)"
    return SYSTEM_TEMPLATE.format(context=context)


def parse_response(text: str, ticket_id: str) -> dict[str, Any]:
    """Extract structured triage output from the model's reply."""
    pattern = r"FINAL_ANSWER\s*\n(.*?)\nEND_ANSWER"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {
            "ticket_id": ticket_id,
            "disposition": "escalate",
            "summary": "Could not parse model response — escalating.",
            "detail": text[:300],
            "parse_error": True,
        }

    try:
        payload = json.loads(match.group(1).strip())
        payload["ticket_id"] = ticket_id
        assert payload.get("disposition") in {"resolve", "parts_request", "escalate"}
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "ticket_id": ticket_id,
            "disposition": "escalate",
            "summary": f"Malformed model response ({exc}) — escalating.",
            "parse_error": True,
        }
