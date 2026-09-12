"""Fase 7 — Observabilidad y trazabilidad (Langfuse v4).

Helpers centralizados para trazar `run_agent` y sus herramientas. Diseñados
para ser **no-op sin coste** cuando Langfuse no está configurado (sin keys en
`.env` o `LANGFUSE_ENABLED=false`): en ese caso el agente y los endpoints
siguen funcionando igual que antes, solo sin traces.

API usada (Langfuse Python SDK >= 4.15, modelo observations-first):
- `observe(name=..., as_type=...)` — decorador que crea un span/tool/generation
  y captura entrada, salida, tiempo y errores automáticamente.
- `get_client()` — cliente configurado por env vars; se deshabilita solo si no
  hay public_key.
- `propagate_attributes(...)` — atributos correlacionados (user_id, session_id,
  metadata, tags, trace_name) propagados a toda observación anidada.
"""

import logging
from typing import Any

from langfuse import get_client, propagate_attributes

logger = logging.getLogger(__name__)

# True si el cliente Langfuse quedó realmente activo (se puede trazar).
# El cliente se deshabilita automáticamente si falta LANGFUSE_PUBLIC_KEY, pero
# exponemos este flag para decisones de bajo coste en el código (p.ej. no
# construir payloads de traces pesados si no se van a enviar).
Langfuse = get_client()


def is_enabled() -> bool:
    """Devuelve True si Langfuse está configurado correctamente.

    Un signo práctico: el public key fue suministrado. Si el cliente se
    deshabilitó, get_client() existe pero no emite nada; este helper evita
    construir input/output JSON innecesarios.
    """
    return bool(Langfuse._client and Langfuse._client.auth_public_key)


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


def usage_details_from_groq(completion: Any) -> dict[str, int]:
    """Extrae usage_details de una respuesta Groq.

    Para los modelos de razonamiento (gpt-oss) la mayor parte del coste está en
    los tokens de entrada/salida. Groq expone `usage.prompt_tokens` y
    `usage.completion_tokens`; `completion_tokens_details.reasoning_tokens`
    puede existir y se deja como referencia.
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
