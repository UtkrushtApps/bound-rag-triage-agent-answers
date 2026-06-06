# Solution Steps

1. Implement semantic recall in `agent/policy.py` by loading the KB corpus from `agent.kb.load_corpus()`, embedding the incoming query with `fastembed`, computing cosine similarity against cached KB embeddings, ranking scores descending, and returning only the top `TOP_K_CHUNKS` chunk texts as `list[str]`.

2. Upgrade `agent/kb.py` so corpus loading remains cached but also stores normalized embeddings for cosine similarity; if an older cache exists, backfill the normalized form before returning it.

3. Implement a conservative preflight budget estimator in `agent/policy.py` that approximates the next model-call cost from chunk/history size, then raise `BudgetExceededError(session_cost, projected, ceiling)` whenever the current or projected session spend reaches/exceeds `BUDGET_CEILING`.

4. Fix `agent/graph.py` retrieval so `retrieve_node()` calls `policy.recall()` instead of dumping the entire corpus, and add structured logging for the query and number of chunks returned.

5. Update `reason_node()` to call `policy.enforce_budget()` before every `litellm.completion()` call, log the budget check, and on `BudgetExceededError` convert the state into an explicit escalation path without making the model call.

6. Keep actual token/cost accounting in `reason_node()` after successful completions: read usage tokens from the response, compute step cost, increment `steps`, accumulate `session_cost`, and log query, token counts, step cost, and new total cost.

7. Fix `should_continue()` so it routes to `"escalate"` when `steps >= MAX_STEPS` and when the session is at/over the budget ceiling, with those guards checked before the `FINAL_ANSWER` condition so boundary behavior matches the invariants.

8. Update `escalate_node()` to produce a structured escalation payload with `disposition="escalate"`, the correct `reason` (`budget_ceiling` or `loop_cap`), the ticket id, summary, step count, and session cost.

9. Preserve clean exception behavior by catching only `BudgetExceededError` for graceful degradation, avoiding broad exception swallowing, and letting unrelated failures propagate normally.

10. Optionally configure CLI logging in `agent/__main__.py` so the structured per-step logs emitted by the graph are visible when running tickets or selfcheck, then verify with `./run.sh` and `pytest invariants/ -v`.

