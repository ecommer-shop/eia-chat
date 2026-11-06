"""Fase 3 — Núcleo del agente (loop a mano, sin framework).

Flujo determinista:

    resolve_store(account_id, inbox_id) → [tienda mapeada sin VK Bifrost → error 503]
    → classify_intent_detailed(query, vk)
    → [pending_order + confirmación → place_order(confirm=True)]
    → [intención COMPRAR → place_order(confirm=False) + estado pendiente]
    → dispatch de herramientas por intención (search_catalogo/search_docs)
    → get_memory → answer (Bifrost) → save_message

Reglas:
- Tienda no mapeada (fail-closed) → herramienta terminal `escalate`, sin LLM.
- Intención solo CONVERSACIONAL → se salta la búsqueda (sin coste de embedding).
- Intenciones mixtas (CATALOGO + POLITICAS/INFO_GENERAL) → ambas herramientas.
- La compra (`COMPRAR`) es en dos turnos y nunca se cobra sin confirmación
  explícita: el turno 1 es solo preview con el total exacto.

`/chat` no se toca: este módulo es la base de la Fase 6 (`/agent/chat`).
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from langfuse import observe

from app.agent import tools
from app.config import settings
from app.exceptions import StoreWithoutBifrostKeyError
from app.intent_classifier import classify_intent_detailed, _is_confirmation
from app.memory import clear_pending_order, get_pending_order, save_message, save_pending_order
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


# --- Compra en dos turnos (place_order) -----------------------------------
# El estado pendiente NO va al historial de mensajes: vive en su propia clave
# de Redis (`app.memory.save_pending_order`) para no contaminar el prompt del
# LLM con estado interno. `run_agent` lo lee al principio del turno y decide si
# el mensaje es la confirmación que ejecuta el pago.

def _pending_payment_code(pending: dict, params: dict) -> str | None:
    """Método de pago para el turno 2: el que dijo el comprador, o el guardado."""
    mentioned = params.get("payment_method")
    if mentioned:
        return mentioned
    stored = pending.get("paymentMethodCode")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    return settings.ASSISTANT_PAYMENT_METHOD_CODE or None


def _pending_matches_identity(pending: dict, account_id, chatwoot_conversation_id) -> bool:
    """El pendiente solo aplica a la misma identidad que lo creó."""
    stored_account = pending.get("account_id")
    stored_conversation = pending.get("chatwoot_conversation_id")
    if stored_account is not None and account_id is not None:
        if int(stored_account) != int(account_id):
            return False
    if stored_conversation is not None and chatwoot_conversation_id is not None:
        if int(stored_conversation) != int(chatwoot_conversation_id):
            return False
    return True


def _pending_state(
    result,
    payment_method_code: str | None,
    account_id,
    inbox_id,
    chatwoot_conversation_id,
) -> dict:
    """Estado que se guarda tras un preview útil del turno 1."""
    return {
        "status": result.status,
        "paymentMethodCode": payment_method_code,
        "account_id": account_id,
        "inbox_id": inbox_id,
        "chatwoot_conversation_id": chatwoot_conversation_id,
        "order": result.order,
        "message": result.message,
    }


async def _run_pending_confirmation(
    query: str,
    store,
    pending: dict,
    params: dict,
    account_id,
    inbox_id,
    chatwoot_conversation_id,
    conversation_id: str,
    tools_used: list[str],
    history: list[dict],
) -> AgentResult | None:
    """Turno 2: el comprador confirma → se ejecuta el pago.

    Devuelve `None` cuando el mensaje no es una confirmación: el caller limpia
    el estado pendiente y sigue con el turno normal.
    """
    if not _is_confirmation(query):
        return None

    payment_code = _pending_payment_code(pending, params)
    order_code = (pending.get("order") or {}).get("code") if isinstance(
        pending.get("order"), dict
    ) else None

    logger.info(
        "Agente: confirmación de compra | conversation=%s | pedido=%s | pago=%s",
        conversation_id, order_code or "-", payment_code or "-",
    )

    result = await tools.place_order(
        store,
        account_id,
        inbox_id,
        chatwoot_conversation_id,
        confirm=True,
        payment_method_code=payment_code,
    )
    tools_used.append("place_order")

    answer = tools.format_place_order_result(result, payment_code)

    if result.ok and not result.is_confirmation_required:
        # Pago ejecutado (placed / awaiting_payment): el flujo termina acá.
        clear_pending_order(conversation_id)
    elif result.status == "confirmation_required":
        # El backend no ejecutó nada y devolvió un preview nuevo: se conserva
        # el pendiente actualizado para el turno siguiente.
        save_pending_order(
            conversation_id,
            _pending_state(
                result, payment_code, account_id, inbox_id, chatwoot_conversation_id
            ),
        )
    else:
        # Error accionable (pago rechazado, carrito desaparecido, red): se
        # limpia el pendiente para no reintentar un cobro con estado rancio.
        clear_pending_order(conversation_id)

    save_message(conversation_id, "user", query)
    save_message(conversation_id, "assistant", answer)

    return AgentResult(
        answer=answer,
        intent_detected="COMPRAR",
        sources_used=1,
        escalado=False,
        tools_used=tools_used,
        conversation_id=conversation_id,
    )


async def _run_place_order_preview(
    query: str,
    store,
    params: dict,
    account_id,
    inbox_id,
    chatwoot_conversation_id,
    conversation_id: str,
    tools_used: list[str],
    history: list[dict],
) -> AgentResult:
    """Turno 1: preview del pago (`confirm=False`) + pregunta de confirmación.

    Nunca cobra: el backend devuelve el total exacto y el agente pregunta. Si
    el comprador ya nombró el medio de pago en este turno, se guarda para el
    turno 2; si no, se usa el fallback de config y se le pregunta igual.
    """
    payment_code = params.get("payment_method") or settings.ASSISTANT_PAYMENT_METHOD_CODE

    result = await tools.place_order(
        store,
        account_id,
        inbox_id,
        chatwoot_conversation_id,
        confirm=False,
        payment_method_code=payment_code,
    )
    tools_used.append("place_order")

    answer = tools.format_place_order_preview(result, payment_code)

    if result.ok and result.order is not None:
        save_pending_order(
            conversation_id,
            _pending_state(
                result, payment_code, account_id, inbox_id, chatwoot_conversation_id
            ),
        )
    else:
        clear_pending_order(conversation_id)

    save_message(conversation_id, "user", query)
    save_message(conversation_id, "assistant", answer)

    return AgentResult(
        answer=answer,
        intent_detected="COMPRAR",
        sources_used=1 if result.order is not None else 0,
        escalado=False,
        tools_used=tools_used,
        conversation_id=conversation_id,
    )


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

        # Flujo de compra en dos turnos: si había un `pending_order` y el
        # mensaje confirma, se ejecuta el pago; si NO confirma, el pendiente se
        # descarta y el turno sigue por el camino normal.
        pending = get_pending_order(conversation_id)
        if pending and not _pending_matches_identity(
            pending, account_id, chatwoot_conversation_id
        ):
            # Pendiente de otra identidad (o sin identidad en este turno): se
            # descarta para no ejecutar un cobro con coordenadas equivocadas, y
            # no se abre un preview nuevo con esas mismas coordenadas.
            logger.info(
                "Agente: pending_order de otra identidad — se descarta (conv=%s)",
                conversation_id,
            )
            clear_pending_order(conversation_id)
            pending = None

        if pending:
            confirmed = await _run_pending_confirmation(
                query,
                store,
                pending,
                classification.params,
                account_id,
                inbox_id,
                chatwoot_conversation_id,
                conversation_id,
                tools_used,
                history,
            )
            if confirmed is not None:
                return confirmed
            clear_pending_order(conversation_id)
            # Un mensaje que no confirma deja de ser una compra en curso: cae
            # al dispatch normal de su propia intención.
            intents = [i for i in intents if i != "COMPRAR"] or ["CONVERSACIONAL"]

        if "COMPRAR" in intents:
            if chatwoot_conversation_id is None or account_id is None:
                # Sin identidad no hay compra posible: se responde cordial sin
                # salir a la red ni guardar estado pendiente.
                answer = tools.PLACE_ORDER_NOT_IDENTIFIED_MESSAGE
                tools_used.append("place_order")
                save_message(conversation_id, "user", query)
                save_message(conversation_id, "assistant", answer)
                return AgentResult(
                    answer=answer,
                    intent_detected=intent_str,
                    sources_used=0,
                    escalado=False,
                    tools_used=tools_used,
                    conversation_id=conversation_id,
                )
            # Preview siempre (`confirm=False`): se muestra el total exacto y
            # se pregunta. La ejecución ocurre recién en el turno siguiente.
            return await _run_place_order_preview(
                query,
                store,
                classification.params,
                account_id,
                inbox_id,
                chatwoot_conversation_id,
                conversation_id,
                tools_used,
                history,
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