"""Fase 7 — Observabilidad y trazabilidad (Langfuse v4).

Helpers centralizados para trazar `run_agent` y sus herramientas. Diseñados
para ser **no-op sin coste** cuando Langfuse no está configurado (sin keys en
`.env` o `LANGFUSE_ENABLED=false`): en ese caso el agente y los endpoints
siguen funcionando igual que antes, solo sin traces. Los eventos de timing
HTTP usan metadatos explícitos y allowlisted, nunca contenido conversacional.

API usada (Langfuse Python SDK >= 4.15, modelo observations-first):
- `observe(name=..., as_type=...)` — decorador que crea un span/tool/generation
  y captura entrada, salida, tiempo y errores automáticamente.
- `get_client()` — cliente configurado por env vars; se deshabilita solo si no
  hay public_key.
- `propagate_attributes(...)` — atributos correlacionados (user_id, session_id,
  metadata, tags, trace_name) propagados a toda observación anidada.
"""

import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langfuse import propagate_attributes

logger = logging.getLogger(__name__)
_REQUEST_ID: ContextVar[str | None] = ContextVar("timing_request_id", default=None)
_SAFE_FIELDS = {
    "model_role", "model", "attempt", "cache_outcome", "result_count", "operation",
    "status_code",
}
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._/-]")
_TIMING_OUTCOME = re.compile(r"(?:^|\s)outcome=(success|failure)(?:\s|$)")


class TerminalColorFormatter(logging.Formatter):
    """Add terminal-only color without changing the underlying log record."""

    _COLORS = {
        "success": "\x1b[32m",
        "failure": "\x1b[31m",
        "warning": "\x1b[33m",
        "error": "\x1b[31m",
    }
    _RESET = "\x1b[0m"

    def __init__(self, fmt: str, *, stream=None):
        super().__init__(fmt)
        self.stream = stream if stream is not None else sys.stderr

    def _terminal_colors_enabled(self) -> bool:
        is_tty = getattr(self.stream, "isatty", None)
        return (
            "NO_COLOR" not in os.environ
            and callable(is_tty)
            and is_tty()
        )

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self._terminal_colors_enabled():
            return rendered

        message = record.getMessage()
        color = None
        if message.startswith("timing_event "):
            match = _TIMING_OUTCOME.search(message)
            if match:
                color = self._COLORS[match.group(1)]
        elif record.levelno >= logging.ERROR:
            color = self._COLORS["error"]
        elif record.levelno >= logging.WARNING:
            color = self._COLORS["warning"]

        return f"{color}{rendered}{self._RESET}" if color else rendered


def _safe_label(value: object) -> str:
    return _SAFE_LABEL.sub("_", str(value))[:100] or "unknown"


def emit_timing(stage: str, started_ns: int, outcome: str, **fields: object) -> None:
    """Emit a structured timing line only inside a correlated HTTP request."""
    request_id = _REQUEST_ID.get()
    if request_id is None:
        return
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000
    safe_fields: list[str] = []
    for name, value in fields.items():
        if name not in _SAFE_FIELDS or value is None:
            continue
        if name in {"attempt", "result_count", "status_code"}:
            try:
                rendered = str(max(0, int(value)))
            except (TypeError, ValueError):
                continue
        else:
            rendered = _safe_label(value)
        safe_fields.append(f"{name}={rendered}")
    suffix = " " + " ".join(safe_fields) if safe_fields else ""
    logger.info(
        "timing_event request_id=%s stage=%s elapsed_ms=%.3f outcome=%s%s",
        request_id,
        _safe_label(stage),
        elapsed_ms,
        _safe_label(outcome),
        suffix,
    )


class RequestTimingMiddleware:
    """Correlate /agent/chat through the final ASGI response-body send."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/agent/chat":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        token = _REQUEST_ID.set(request_id)
        started_ns = time.monotonic_ns()
        status_code = None
        unhandled_exception = False

        async def send_with_correlation(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (name, value) for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}

            await send(message)

        try:
            await self.app(scope, receive, send_with_correlation)
        except BaseException:
            unhandled_exception = True
            raise
        finally:
            outcome = (
                "failure"
                if unhandled_exception or status_code is None or status_code >= 400
                else "success"
            )
            fields = {"status_code": status_code} if status_code is not None else {}
            emit_timing("request_total", started_ns, outcome, **fields)
            _REQUEST_ID.reset(token)


@contextmanager
def timed_stage(stage: str, **fields: object):
    """Time one stage without recording exception text or arbitrary values."""
    started_ns = time.monotonic_ns()
    outcome = "success"
    try:
        yield fields
    except BaseException:
        outcome = "failure"
        raise
    finally:
        emit_timing(stage, started_ns, outcome, **fields)


def trace_attributes(
    *,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Context manager que propaga atributos correlacionados a la traza actual.

    Import en callers con `@observe()`:

        with trace_attributes(
            trace_name=f"agent/{conversation_id}",
            user_id=str(user_id),
            session_id=conversation_id,
            metadata={"inbox_id": inbox_id, "canal": channel},
            tags=["agent"],
        ):
            ...
    """
    return propagate_attributes(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )


def usage_details_from_completion(completion: Any) -> dict[str, int]:
    """Extrae usage_details de una respuesta de Bifrost/OpenAI-compatible.

    Espera la forma estándar `usage.prompt_tokens` / `usage.completion_tokens`.
    `completion_tokens_details.reasoning_tokens` puede existir y se deja como
    referencia.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    details: dict[str, int] = {
        "input": getattr(usage, "prompt_tokens", 0) or 0,
        "output": getattr(usage, "completion_tokens", 0) or 0,
    }
    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None and getattr(ctd, "reasoning_tokens", None) is not None:
        details["reasoning_tokens"] = ctd.reasoning_tokens
    return details
