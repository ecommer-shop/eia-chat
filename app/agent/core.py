"""Fase 3 — Núcleo del agente (loop a mano, sin framework).

Flujo determinista:

    resolve_store(inbox_id) → classify_intent(query) → [tienda no mapeada → escalate]
    → dispatch de herramientas por intención (search_catalogo/search_docs)
    → get_memory → answer (Groq) → save_message

Reglas:
- Tienda no mapeada (fail-closed) → herramienta terminal `escalate`, sin LLM.
- Intención solo CONVERSACIONAL → se salta la búsqueda (sin coste de embedding).
- Intenciones mixtas (CATALOGO + POLITICAS/INFO_GENERAL) → ambas herramientas.

`/chat` no se toca: este módulo es la base de la Fase 6 (`/agent/chat`).
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from langfuse import observe

from app.agent import tools
from app.intent_classifier import classify_intent
from app.memory import save_message
from app.observability import trace_attributes
from app.store_resolver import resolve_store

logger = logging.getLogger(__name__)

TRIVIAL_REPLY = "¡Hola! ¿En qué puedo ayudarte hoy?"

_ALPHA_RE = re.compile(r"[a-záéíóúüñ]", re.IGNORECASE)
_CATEGORIES_QUERY_RE = re.compile(r"categor[ií]as", re.IGNORECASE)


def _is_trivial(query: str) -> bool:
    return not _ALPHA_RE.search(query)


def _last_user_query(history: list[dict]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


@dataclass
class AgentResult:
    """Salida del agente.

    `escalado=True` indica que el agente delegó la conversación a un humano
    (gancho para la integración futura con Chatwoot).
    """

    answer: str
    intent_detected: str
    sources_used: int
    escalado: bool = False
    tools_used: list[str] = field(default_factory=list)
    conversation_id: str = ""


def _new_conversation_id() -> str:
    return f"agent-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _pick_tools(intents: list[str]) -> list[str]:
    """Selecciona las herramientas de búsqueda según las intenciones."""
    names: list[str] = []
    if "CATALOGO" in intents:
        names.append("search_catalogo")
    if any(i in ("POLITICAS", "INFO_GENERAL") for i in intents):
        names.append("search_docs")
    return names


def _is_categories_query(query: str) -> bool:
    return bool(_CATEGORIES_QUERY_RE.search(query))


async def _collect_context(
    query: str, store, intents: list[str]
) -> tuple[list[dict], list[str]]:
    context: list[dict] = []
    used: list[str] = []
    doc_intents = [i for i in intents if i in ("POLITICAS", "INFO_GENERAL")]

    catalog_name: str | None = None
    if "CATALOGO" in intents:
        catalog_name = "list_categorias" if _is_categories_query(query) else "search_catalogo"

    for name in ([catalog_name] if catalog_name else []) + (["search_docs"] if doc_intents else []):
        if name == "search_catalogo":
            items = await tools.search_catalogo(query, store)
        elif name == "search_docs":
            items = await tools.search_docs(query, store, doc_intents)
        elif name == "list_categorias":
            categories = await tools.list_categorias(store)
            if categories:
                items = [{
                    "score": 1.0,
                    "payload": {
                        "metadata": {"categories": categories, "name": "Catálogo de la tienda"},
                        "text": f"Categorías disponibles en la tienda: {', '.join(categories)}.",
                    },
                }]
            else:
                items = []
        else:
            continue
        context.extend(items)
        used.append(name)
    return context, used


@observe(as_type="agent")
async def run_agent(
    query: str,
    conversation_id: str,
    inbox_id: int,
    user_id: int | None = None,
    channel: str | None = None,
) -> AgentResult:
    """Punto de entrada lógico del agente.

    `channel` se acepta por paridad con `/chat`, pero la selección de prompt
    por canal sigue viniendo del `system_prompt` de la tienda resuelta por
    `inbox_id` (igual que en el pipeline actual).

    Trazabilidad: la función completa es un span `agent`; los atributos
    correlacionados (conversation/tienda/canal) se propagan a todas las
    observaciones anidadas (herramientas).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("El mensaje no puede estar vacío.")

    conversation_id = conversation_id or _new_conversation_id()
    store = resolve_store(inbox_id)

    trace_ctx = trace_attributes(
        trace_name=f"agent-chat",
        user_id=str(user_id) if user_id is not None else None,
        session_id=conversation_id,
        tags=["agent", store.store_name, store.channel_name],
        metadata={
            "inbox_id": inbox_id,
            "tienda": store.store_name,
            "canal": store.channel_name,
            "is_mapped": store.is_mapped,
            "conversation_id": conversation_id,
        },
    )

    with trace_ctx:
        intents = await classify_intent(query)
        intent_str = ", ".join(intents)
        logger.info(
            "Agente: '%s' | tienda=%s (%s) | intents=%s",
            query, store.store_name, store.channel_name, intent_str,
        )

        if not store.is_mapped:
            answer = await tools.escalate(store)
            result = AgentResult(
                answer=answer,
                intent_detected=intent_str,
                sources_used=0,
                escalado=True,
                tools_used=["escalate"],
                conversation_id=conversation_id,
            )
            save_message(conversation_id, "user", query)
            save_message(conversation_id, "assistant", answer)
            return result

        history = await tools.get_memory(conversation_id)
        tools_used: list[str] = ["get_memory"]

        if _is_trivial(query):
            return AgentResult(
                answer=TRIVIAL_REPLY,
                intent_detected="CONVERSACIONAL",
                sources_used=0,
                escalado=False,
                tools_used=tools_used,
                conversation_id=conversation_id,
            )

        if intents == ["CONVERSACIONAL"]:
            context_items: list[dict] = []
        else:
            context_items, search_used = await _collect_context(query, store, intents)
            tools_used.extend(search_used)

            if not context_items:
                prev_query = _last_user_query(history)
                if prev_query and prev_query != query:
                    rescued, rescue_used = await _collect_context(prev_query, store, intents)
                    if rescued:
                        context_items = rescued
                        tools_used.extend(rescue_used)

        answer = await tools.answer(query, store, context_items, intent_str, history)
        tools_used.append("answer")

        save_message(conversation_id, "user", query)
        save_message(conversation_id, "assistant", answer)

        return AgentResult(
            answer=answer,
            intent_detected=intent_str,
            sources_used=len(context_items),
            escalado=False,
            tools_used=tools_used,
            conversation_id=conversation_id,
        )