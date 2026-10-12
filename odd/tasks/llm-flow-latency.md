# LLM Flow Latency Instrumentation

## Objective
Add correlated, non-sensitive elapsed-time logs for each meaningful `/agent/chat` processing phase, from HTTP request arrival through response completion, so slow model calls can be distinguished from lookup, retrieval, memory, and application overhead.

## Problem and rationale
The current flow performs serial classification, optional retrieval, generation, and persistence. Existing Langfuse spans may help when enabled, but application logs do not provide one consistently correlated phase breakdown. Timings should make the active model and individual model-call costs visible without logging prompts, user messages, responses, credentials, or VKs.

## Scope
- Add request correlation and total HTTP elapsed time for `/agent/chat` without changing the response schema.
- Add stable stage timing events for store resolution, intent classification, memory read/write, context retrieval, model discovery/selection, LLM completion attempts (including fallback), answer post-processing, and response handling.
- Use a monotonic clock; emit elapsed milliseconds, request correlation, stage/outcome, and safe diagnostic metadata such as model name, cache outcome, retry attempt, and result count.
- Add tests and short operator documentation describing the log fields and how to compare stages.

## Constraints and non-goals
- Preserve the existing uncommitted `.gitignore` change; do not include or modify it.
- No secrets or user content in new timing logs; do not change API response fields or model-selection policy.
- No external provider calls in tests.
- TDD mode: strict, explicitly selected by the user. Runner: `./.venv/bin/pytest`.
- Work on branch `feature/test`. Do not commit unless the user explicitly authorizes a commit.

## Tasks
- [x] T1 — Add failing tests for correlated request and phase timing, safe fields, and unchanged response contract.
- [x] T2 — Implement reusable timing instrumentation and wire the full request/model/retrieval/memory path.
- [x] T3 — Run focused and relevant regression tests, document how to read timing events, and record verification evidence.

## Acceptance criteria
- Every `/agent/chat` call produces a correlated request-total timing event for success and handled failure.
- Logs distinguish classifier and answer model selection from each provider completion attempt, and separately expose retrieval, memory, and post-processing latency.
- Timings use a monotonic clock and safe metadata; no request/prompt/response content, VK, API key, or secret is emitted by the new events.
- The response contract and model behavior remain unchanged.
- New tests fail before implementation (RED), pass after implementation (GREEN), and are rerun after refactoring (REFACTOR).

## Progress and evidence
- Branch `feature/test` created from `kevin`.
- Pre-existing `.gitignore` modification (`.atl/` ignore rule) was present before this task and remains untouched.
- TDD mode resolved as strict by user; runner is `./.venv/bin/pytest`.
- T1 completed: tests first failed at timing assertions (not collection/setup), then passed with the implementation.
- T2 completed: correlated monotonic timing events cover request total through final ASGI body send, store resolution, classification, memory reads/writes, context retrieval, embedding, Qdrant queries, model discovery/selection, each completion attempt including fallback, post-processing, and response handling. The outer middleware returns `X-Request-ID`; `request_total` records status/outcome. Timing logs use an allowlist and omit conversational data and credentials. README includes operator guidance.
- T3 completed: final independent command `./.venv/bin/pytest -q` — 142 passed, 1 existing Starlette/httpx deprecation warning. Targeted ASGI timing/integration tests also passed (12 passed). The classifier-specific timing test exercises mocked model selection and completion and checks correlated safe fields.
- User authorized local installation of `asyncpg==0.32.0` and `cachetools==7.2.1` into `.venv`; `pyproject.toml` and `uv.lock` are unchanged.
- Native risk assess was attempted but the review facade rejected the supplied `intendedUntracked` field as unsupported for ASSESS; no native review outcome or receipt is claimed.
- Feature implementation commit: `4ec00f7` (`feat(observability): instrument chat flow latency`), created on `feature/test` after explicit user authorization.
- Authored diff forecast is 704 lines, above the 400-line review budget. ODD `ask-on-risk` resolved to `feature-branch-chain`; no PR/tracker creation is authorized by this request. Logical future slices: (1) request correlation and ASGI end-to-end timing; (2) per-stage model, retrieval, and memory diagnostics with operator docs.

## Next step
Commit the verified feature without including the pre-existing `.gitignore` change, then provide the command to publish the new branch with upstream tracking.