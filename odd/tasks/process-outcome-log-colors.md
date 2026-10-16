# Colored Process Outcome Logs

## Objective
Make the existing `/agent/chat` stage outcomes immediately distinguishable in terminal logs: successful stages green and failed stages red, while retaining machine-readable timing fields.

## Problem and rationale
Timing events already include `outcome=success|failure`, but all are emitted as uncolored INFO lines. Operators can inspect the outcome text, yet cannot scan for successful versus failing processing stages quickly. Color should be a presentation-only layer, not alter event content or the API.

## Scope
- Format timing events with a green success outcome and red failure outcome when output is an interactive terminal.
- Preserve plain log messages, structured fields, and non-terminal output; honor `NO_COLOR`.
- Keep standard warning/error severity coloring sensible without making unrelated INFO logs look like successful process events.
- Add tests for success/failure colors and unchanged event messages, and update the operator guidance.

## Constraints and non-goals
- Preserve the existing staged `.gitignore` change; do not modify or include it.
- A failed nested attempt remains failure-colored even if a later fallback succeeds; the outer request can still correctly be success-colored.
- Do not change timing outcome semantics, API response shape, or process behavior.
- Strict TDD inherited from the existing project feature record; runner: `./.venv/bin/pytest`.
- Work on branch `feature/test`. Do not commit without explicit user authorization.

## Tasks
- [x] T1 — Add outcome-color regression tests, implement terminal-only log coloring, and document operator behavior.

## Acceptance criteria
- `timing_event` lines with `outcome=success` are green in an interactive terminal.
- `timing_event` lines with `outcome=failure` are red in an interactive terminal.
- Non-terminal output and `NO_COLOR` output contain no ANSI color sequences.
- Formatted events retain their existing text and structured fields; a recovered failed stage remains individually visible.
- Focused and full regression tests pass without external provider calls.

## Progress and evidence
- Exploration confirmed that `app/observability.py` already emits per-stage `outcome=success|failure`; exceptions escaping a timed stage are marked failure, and HTTP status >=400 marks `request_total` as failure.
- `app/observability.py` now provides `TerminalColorFormatter`: matching terminal timing events are green/red according to their outcome; non-TTY output and `NO_COLOR` remain plain. Warning/error severity logs are yellow/red; unrelated INFO messages remain uncolored.
- `app/main.py` configures the application handler with the formatter. Event text and structured fields remain unchanged; README explains terminal and plain-output behavior.
- Failed nested model attempts remain individually visible as failures even when a fallback and outer request succeed.
- Strict-TDD evidence: three new tests failed before implementation because the formatter did not exist; the new tests passed after implementation. Writer reported triangulation/refactor coverage for non-TTY, `NO_COLOR`, unrelated INFO and severity colors.
- Verification: `./.venv/bin/pytest -q tests/test_timing.py` passed (8 tests) in both writer and independent verification; `./.venv/bin/pytest -q` passed (146 tests). Both runs reported one existing Starlette/httpx deprecation warning.
- Native review assessment was unavailable: the controller rejected the ASSESS input (`intendedUntracked` unsupported). INSPECT did not produce a START route after validation errors; no lineage, receipt, or review approval was created. Because risk assessment was unavailable, an independent read-only verifier was run and passed the focused test command.
- Existing staged `.gitignore` change remains untouched. No files were staged or committed; committing requires explicit user authorization.

## Next step
Implementation and checks are complete. If a native review is needed, resolve the controller's inspect/assess input issue first; delivery actions remain a separate user decision.
