"""Fase 3 — Núcleo del agente (loop a mano, sin framework).

Flujo determinista:

    resolve_store(account_id, inbox_id) → [tienda mapeada sin VK Bifrost → error 503]
    → classify_intent_detailed(query, vk) → [tienda no mapeada → escalate]
    → dispatch de herramientas por intención (search_catalogo/search_docs)
    → get_memory → answer (Bifrost) → save_message

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
from app.exceptions import StoreWithoutBifrostKeyError
from app.intent_classifier import classify_intent_detailed
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


def _is_categories_query(query: str) -> bool:
    return bool(_CATEGORIES_QUERY_RE.search(query))


async def _collect_context(
    query: str,
    store,
    intents: list[str],
    search_query: str | None = None,
    account_id: int | None = None,
    inbox_id: int | None = None,
    chatwoot_conversation_id: int | None = None,
    order_code: str | None = None,
) -> tuple[list[dict], list[str]]:
    context: list[dict] = []
    used: list[str] = []
    doc_intents = [i for i in intents if i in ("POLITICAS", "INFO_GENERAL")]
    # La query extraída por el clasificador (`params["query"]`) es la que se
    # usa contra las herramientas de búsqueda; sin ella se cae al texto crudo.
    effective_query = search_query or query

    catalog_name: str | None = None
    # `BUSCAR` es búsqueda explícita de catálogo (siempre search_catalogo);
    # `CATALOGO` puede ser búsqueda o listado de categorías.
    if "BUSCAR" in intents:
        catalog_name = "search_catalogo"
    elif "CATALOGO" in intents:
        catalog_name = (
            "list_categorias" if _is_categories_query(effective_query) else "search_catalogo"
        )

    names: list[str] = []
    # Fase 1b: carrito y pedido antes que el catálogo (respuesta acotada).
    if "CARRITO" in intents:
        names.append("get_cart")
    if "PEDIDO" in intents:
        names.append("get_order")
    if catalog_name:
        names.append(catalog_name)
    if doc_intents:
        names.append("search_docs")

    for name in names:
        if name == "search_catalogo":
            items = await tools.search_catalogo(effective_query, store)
        elif name == "search_docs":
            items = await tools.search_docs(effective_query, store, doc_intents)
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
        elif name in ("get_cart", "get_order"):
            # Las tools de identidad devuelven texto plano ya listo para el
            # usuario (o un fallback cordial). Se inyecta como contexto.
            if name == "get_cart":
                text = await tools.get_cart(
                    store, account_id, inbox_id, chatwoot_conversation_id
                )
            else:
                text = await tools.get_order(
                    store, account_id, inbox_id, chatwoot_conversation_id, order_code
                )
            items = [{"score": 1.0, "payload": {"text": text}}] if text else []
        else:
            continue
        context.extend(items)
        used.append(name)
    return context, used


@observe(as_type="agent")
async def run_agent(
    query: str,
    conversation_id: str,
    account_id: int | None,
    inbox_id: int,
    user_id: int | None = None,
    channel: str | None = None,
    chatwoot_conversation_id: int | None = None,
) -> AgentResult:
    """Punto de entrada lógico del agente.

    `channel` se acepta por paridad con `/chat`, pero la selección de prompt
    por canal sigue viniendo del `system_prompt` de la tienda resuelta por
    `account_id` + `inbox_id` (igual que en el pipeline actual).

    `chatwoot_conversation_id` es el entero puro del `conversation.id` de
    Chatwoot (distinto del `conversation_id` sintético que usamos para la
    memoria) y es la única entrada de identidad de las tools `get_cart` /
    `get_order`. Default `None` para back-compat: sin él, esas tools
    degradan a un mensaje cordial y el resto del flujo no cambia.

    Trazabilidad: la función completa es un span `agent`; los atributos
    correlacionados (conversation/tienda/canal) se propagan a todas las
    observaciones anidadas (herramientas).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("El mensaje no puede estar vacío.")

    conversation_id = conversation_id or _new_conversation_id()
    store = await resolve_store(account_id, inbox_id)

    trace_ctx = trace_attributes(
        trace_name=f"agent-chat",
        user_id=str(user_id) if user_id is not None else None,
        session_id=conversation_id,
        tags=["agent", store.store_name, store.channel_name],
        metadata={
            "account_id": account_id,
            "inbox_id": inbox_id,
            "tienda": store.store_name,
            "canal": store.channel_name,
            "is_mapped": store.is_mapped,
            "conversation_id": conversation_id,
            "chatwoot_conversation_id": chatwoot_conversation_id,
        },
    )

    with trace_ctx:
        if store.is_mapped and not store.bifrost_vk:
            # Toda tienda activa debe tener una bifrost_key: si no la tiene es
            # un error de creación de claves (no se usa la VK de otra tienda).
            raise StoreWithoutBifrostKeyError(account_id, store.store_name)

        classification = await classify_intent_detailed(query, vk=store.bifrost_vk)
        intents = classification.intents
        search_query = classification.params.get("query")
        order_code = classification.params.get("code")
        intent_str = ", ".join(intents)
        logger.info(
            "Agente: '%s' | tienda=%s (%s) | intents=%s",
            query, store.store_name, store.channel_name, intent_str,
        )

        # `BUSCAR` es la señal de dispatch hacia el catálogo remoto; para el
        # grounding/renderer (que hoy solo entienden `CATALOGO`) se normaliza.
        # `intent_str` conserva la etiqueta original para trazabilidad.
        dispatch_intents = [
            "CATALOGO" if i == "BUSCAR" else i for i in intents
        ]

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
            context_items, search_used = await _collect_context(
                query,
                store,
                intents,
                search_query=search_query,
                account_id=account_id,
                inbox_id=inbox_id,
                chatwoot_conversation_id=chatwoot_conversation_id,
                order_code=order_code,
            )
            tools_used.extend(search_used)

            if not context_items:
                prev_query = _last_user_query(history)
                if prev_query and prev_query != query:
                    rescued, rescue_used = await _collect_context(
                        prev_query,
                        store,
                        intents,
                        search_query=search_query,
                        account_id=account_id,
                        inbox_id=inbox_id,
                        chatwoot_conversation_id=chatwoot_conversation_id,
                        order_code=order_code,
                    )
                    if rescued:
                        context_items = rescued
                        tools_used.extend(rescue_used)

        answer = await tools.answer(
            query, store, context_items,
            ", ".join(dispatch_intents), history,
        )
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