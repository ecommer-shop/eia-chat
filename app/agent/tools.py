"""Fase 3 — Herramientas del agente.

Son wrappers delgados sobre los módulos existentes de eia-rag: no duplican
lógica de búsqueda, memoria ni generación. `answer` y `escalate` son
herramientas terminales: producen el texto final de la respuesta.

`TOOL_REGISTRY` centraliza el catálogo de herramientas; el núcleo del agente
(`core.py`) decide cuáles invocar según la intención detectada.
"""

import logging
from typing import Awaitable, Callable

from langfuse import get_client, observe

from app.clients import get_llm, invalidate_allowed_models, pick_model
from app.config import settings
from app.catalog_client import search_products as search_catalog_remote
from app.identity_client import get_cart as get_cart_remote
from app.identity_client import get_order as get_order_remote
from app.grounding import apply_grounding
from app.llm_generator import build_prompt
from app.memory import get_history
from app.models import StoreConfig
from app.observability import usage_details_from_completion
from app.renderer import annotate_products, render_product_footer
from app.retriever import search_context, list_categories

logger = logging.getLogger(__name__)

# Fallback cordial de escalada a humano (sin integración con Chatwoot aún).
# Mantener el término "configurada": el golden set (gs_025/gs_026) valida que
# la respuesta fail-closed lo incluya.
ESCALATION_MESSAGE = (
    "Aun no tengo informacion configurada para esta tienda. "
    "Un miembro del equipo te va a contactar pronto."
)


def _update_retriever_scores(result: list[dict]) -> None:
    """Adjunta los scores de los resultados al span retriever activo.

    Ayuda a diagnosticar alucinaciones: si el score es bajo, es más probable que
    la generación invente. Best-effort, nunca interrumpe el flujo.
    """
    try:
        if not result:
            get_client().update_current_span(metadata={"retrieved": 0})
        else:
            get_client().update_current_span(metadata={
                "retrieved": len(result),
                "scores": [round(item.get("score", 0), 4) for item in result],
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("No se pudo registrar scores de retriever: %s", e)


@observe(as_type="retriever")
async def search_catalogo(query: str, store: StoreConfig, limit: int = 10) -> list[dict]:
    """Busca productos del catálogo (intención CATALOGO/BUSCAR).

    Si el flag `ASSISTANT_CATALOG_ENABLED` está activo, consulta primero el
    endpoint de catálogo del backend (Fase 0.5). Si el endpoint no está
    disponible o no devuelve resultados, degrada al RAG (Qdrant) con la misma
    query. La ruta de degradación es transparente para el agente.
    """
    logger.info("Tool search_catalogo :: store=%s query='%s'", store.store_name, query)

    if settings.ASSISTANT_CATALOG_ENABLED:
        try:
            remote = await search_catalog_remote(query, limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.error("search_catalogo: fallo cliente catálogo: %s", e)
            remote = []
        if remote:
            _update_retriever_scores(remote)
            return remote
        logger.info(
            "search_catalogo: catálogo remoto sin resultados — degradando a RAG"
        )

    result = await search_context(query, store, ["CATALOGO"], limit=limit)
    _update_retriever_scores(result)
    return result


@observe(as_type="retriever")
async def search_docs(
    query: str, store: StoreConfig, intents: list[str], limit: int = 10
) -> list[dict]:
    """Busca documentos de POLITICAS/INFO_GENERAL."""
    logger.info("Tool search_docs :: store=%s intents=%s", store.store_name, intents)
    result = await search_context(query, store, list(intents), limit=limit)
    _update_retriever_scores(result)
    return result


@observe(as_type="tool")
async def list_categorias(store: StoreConfig) -> list[str]:
    """Lista las categorías reales del catálogo de la tienda (estructurado).

    No es una búsqueda semántica: recorre el payload para responder consultas
    de agregación ("¿qué categorías tienen?") sin inventar categorías.
    """
    logger.info("Tool list_categorias :: store=%s", store.store_name)
    return await list_categories(store)


@observe(as_type="tool")
async def get_memory(conversation_id: str) -> list[dict]:
    """Recupera el historial de la conversación (Redis/Valkey)."""
    return get_history(conversation_id)


# Fallbacks cordiales de las tools de identidad (Fase 1b). Nunca lanzan hacia
# el agente: sin identidad o sin configuración el turno sigue y el usuario
# recibe una respuesta útil en vez de un 500.
CART_NOT_IDENTIFIED_MESSAGE = (
    "No puedo ver tu carrito porque no estás identificado. "
    "Iniciá sesión en la tienda para que pueda ayudarte con eso."
)
CART_EMPTY_MESSAGE = (
    "No tenés un carrito activo en este momento. "
    "Cuando agregues productos a tu carrito, puedo contarte qué llevás y cuánto suma."
)
ORDER_NOT_IDENTIFIED_MESSAGE = (
    "No puedo ver tu pedido porque no estás identificado. "
    "Iniciá sesión en la tienda para que pueda ayudarte con eso."
)
ORDER_NEEDS_CODE_MESSAGE = (
    "Para consultar tu pedido necesito el número de pedido que te enviamos por correo "
    "al confirmar la compra. ¿Me lo pasás?"
)
ORDER_NOT_FOUND_MESSAGE = (
    "No encuentro un pedido con ese número en tu cuenta. "
    "Revisá el código y volvé a intentarlo, o iniciá sesión con la cuenta con la que compraste."
)

# Nombre legible del estado del pedido (Vendure `order.state`).
_ORDER_STATE_LABELS = {
    "AddingItems": "en armado (todavía es un carrito)",
    "ArrangingPayment": "pendiente de pago",
    "PaymentAuthorized": "pago autorizado",
    "PaymentSettled": "pago confirmado",
    "PartiallyShipped": "enviado parcialmente",
    "Shipped": "enviado",
    "PartiallyDelivered": "entregado parcialmente",
    "Delivered": "entregado",
    "Modifying": "en modificación",
    "ArrangingAdditionalPayment": "pendiente de un pago adicional",
    "Cancelled": "cancelado",
}


def _money(order: dict, amount_key: str, decimal_key: str) -> str:
    """Importe listo para el usuario: `$ 185.00` (moneda del pedido)."""
    decimal = order.get(decimal_key)
    currency = order.get("currencyCode") or ""
    if isinstance(decimal, str) and decimal.strip():
        return f"$ {decimal.strip()} {currency}".strip()
    amount = order.get(amount_key)
    if isinstance(amount, (int, float)):
        return f"$ {amount} {currency}".strip()
    return ""


def _format_cart(order: dict) -> str:
    """Resume el carrito: líneas, cantidades y total (texto plano)."""
    lines = [line for line in (order.get("lines") or []) if isinstance(line, dict)]
    if not lines:
        return "Tu carrito está vacío. ¿Querés que te ayude a encontrar algo para agregar?"

    parts: list[str] = []
    for line in lines:
        variant = line.get("productVariant") or {}
        name = variant.get("name") or "Producto"
        quantity = line.get("quantity")
        subtotal = line.get("linePriceWithTaxDecimal")
        segment = f"{name}"
        if quantity:
            segment += f" x{quantity}"
        if isinstance(subtotal, str) and subtotal.strip():
            segment += f" ({_money(line, 'linePriceWithTax', 'linePriceWithTaxDecimal')})"
        parts.append(segment)

    total_quantity = order.get("totalQuantity") or sum(
        (line.get("quantity") or 0) for line in lines
    )
    total = _money(order, "totalWithTax", "totalWithTaxDecimal")

    summary = f"Tu carrito tiene {len(lines)} producto{'s' if len(lines) != 1 else ''}: "
    summary += "; ".join(parts) + "."
    if total_quantity:
        summary += f" En total son {total_quantity} unidad{'es' if total_quantity != 1 else ''}"
        summary += f" por {total}." if total else "."
    elif total:
        summary += f" El total es {total}."
    return summary


def _format_order(order: dict) -> str:
    """Resume un pedido: código, estado, líneas y total (texto plano)."""
    code = order.get("code") or ""
    state = _ORDER_STATE_LABELS.get(order.get("state"), order.get("state") or "")

    header = "Tu pedido"
    if code:
        header += f" {code}"
    if state:
        header += f" está {state}"

    lines = [line for line in (order.get("lines") or []) if isinstance(line, dict)]
    detail_parts: list[str] = []
    for line in lines:
        variant = line.get("productVariant") or {}
        name = variant.get("name") or "Producto"
        quantity = line.get("quantity")
        detail_parts.append(f"{name} x{quantity}" if quantity else name)

    total = _money(order, "totalWithTax", "totalWithTaxDecimal")
    message = header + "."
    if detail_parts:
        message += " Incluye: " + "; ".join(detail_parts) + "."
    if total:
        message += f" Total: {total}."
    return message


@observe(as_type="tool")
async def get_cart(
    store: StoreConfig,
    account_id: int | None,
    inbox_id: int | None,
    chatwoot_conversation_id: int | None,
) -> str:
    """Carrito activo del comprador de esta conversación (fase 1b).

    Degrada con un mensaje cordial cuando falta la identidad (sin
    `chatwoot_conversation_id`, sin config del backend o 404 = anónimo/sin
    sesión). Nunca lanza: el turno sigue con la respuesta generada.
    """
    if chatwoot_conversation_id is None or account_id is None:
        logger.info("Tool get_cart :: sin identidad de conversación — degrada")
        return CART_NOT_IDENTIFIED_MESSAGE

    logger.info(
        "Tool get_cart :: store=%s account=%s inbox=%s conversation=%s",
        store.store_name, account_id, inbox_id, chatwoot_conversation_id,
    )
    try:
        order = await get_cart_remote(account_id, inbox_id, chatwoot_conversation_id)
    except Exception as e:  # noqa: BLE001
        logger.error("get_cart: fallo inesperado del cliente de identidad: %s", e)
        return CART_NOT_IDENTIFIED_MESSAGE

    if order is None:
        # No se distingue "anónimo" de "sin carrito": el cliente ve un mensaje
        # útil en ambos casos y no se filtran detalles internos.
        return CART_EMPTY_MESSAGE
    return _format_cart(order)


@observe(as_type="tool")
async def get_order(
    store: StoreConfig,
    account_id: int | None,
    inbox_id: int | None,
    chatwoot_conversation_id: int | None,
    code: str | None = None,
) -> str:
    """Un pedido del comprador, por `code` (fase 1b).

    Sin código no hay consulta posible (el backend exige `code`): se pide el
    número. Igual que `get_cart`, degrada con un mensaje cordial sin lanzar.
    """
    if chatwoot_conversation_id is None or account_id is None:
        logger.info("Tool get_order :: sin identidad de conversación — degrada")
        return ORDER_NOT_IDENTIFIED_MESSAGE

    if not code or not str(code).strip():
        return ORDER_NEEDS_CODE_MESSAGE

    logger.info(
        "Tool get_order :: store=%s account=%s inbox=%s conversation=%s",
        store.store_name, account_id, inbox_id, chatwoot_conversation_id,
    )
    try:
        order = await get_order_remote(
            account_id, inbox_id, chatwoot_conversation_id, str(code).strip()
        )
    except Exception as e:  # noqa: BLE001
        logger.error("get_order: fallo inesperado del cliente de identidad: %s", e)
        return ORDER_NOT_IDENTIFIED_MESSAGE

    if order is None:
        return ORDER_NOT_FOUND_MESSAGE
    return _format_order(order)


def _history_mentions_products(history: list[dict]) -> bool:
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if "👉" in content or "http" in content or "Producto:" in content:
            return True
    return False


async def _create_completion(vk: str, model: str, messages: list[dict]):
    """Llamada cruda a Bifrost (OpenAI-compat) con la VK de la tienda."""
    return await get_llm(vk).chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.2,
        max_tokens=1024,
    )


async def _create_with_model_fallback(
    vk: str, messages: list[dict], model: str, preference: list[str]
) -> tuple[object, str]:
    """Crea la completación; ante 403 (modelo no permitido por el plan)
    refresca el descubrimiento de modelos de esa VK y reintenta UNA vez con
    el mejor modelo disponible. Automático, sin intervención manual.

    Devuelve `(completion, modelo_realmente_usado)`.
    """
    try:
        return await _create_completion(vk, model, messages), model
    except Exception as e:  # noqa: BLE001
        if getattr(e, "status_code", None) != 403:
            raise
        invalidate_allowed_models(vk)
        retry_model = await pick_model(vk, preference)
        if retry_model == model:
            retry_model = settings.BIFROST_CHAT_MODEL_FALLBACK
        if retry_model == model:
            raise
        logger.warning(
            "Modelo '%s' bloqueado para la VK (%s caracteres); reintentando con '%s'",
            model, len(vk or ""), retry_model,
        )
        return await _create_completion(vk, retry_model, messages), retry_model


@observe(as_type="generation")
async def answer(
    query: str,
    store: StoreConfig,
    context_items: list[dict],
    intent: str,
    history: list[dict],
) -> str:
    """Herramienta terminal: genera la respuesta final con Bifrost.

    El modelo depende del plan de la tienda: se prefiere el más grande que la
    VK permite (Phi-4) y, si el plan no lo tiene (free/tienda), se usa el
    mini. Replica los fallbacks de siempre: contenido vacío y errores
    devuelven mensajes cordiales sin romper el flujo. Trazada como una
    generación (LLM) para capturar latencia y consumo de tokens.
    """
    has_prior_products = _history_mentions_products(history)
    messages = build_prompt(
        query=query,
        intent=intent,
        context_items=context_items,
        history=history,
        store_prompt=store.system_prompt,
        nocontext_guard=not context_items and not has_prior_products,
        few_shot=store.few_shot,
    )
    vk = store.bifrost_vk
    preference = [settings.BIFROST_CHAT_MODEL, settings.BIFROST_CHAT_MODEL_FALLBACK]
    client = get_client()
    try:
        model = await pick_model(vk, preference)
        observation = client.start_as_current_observation(
            name="bifrost_chat",
            as_type="generation",
            model=model,
            input={"messages": messages},
        )
        with observation as generation:
            completion, model = await _create_with_model_fallback(
                vk, messages, model, preference
            )
            raw = completion.choices[0].message.content or ""
            text = raw.encode("utf-8", errors="replace").decode("utf-8")
            text = text.replace("\n", " ").replace("\r", "").strip()
            text, grounding_issues = apply_grounding(
                text, context_items, history, intent=intent
            )
            added_links: list[str] = []
            if not grounding_issues and text:
                clean_text, mentioned = annotate_products(text, context_items, intent=intent)
                footer = render_product_footer(mentioned)
                text = f"{clean_text} {footer}".strip() if footer else clean_text
                added_links = [p.url for p in mentioned]
            generation.update(
                output=text,
                usage_details=usage_details_from_completion(completion),
                metadata={
                    "model_used": model,
                    "finish_reason": completion.choices[0].finish_reason,
                    "grounded": not grounding_issues,
                    "grounding_issues": grounding_issues,
                    "links_added": added_links,
                },
            )
        if not text:
            logger.warning(
                "Bifrost devolvió contenido vacío (plan=%s)",
                store.bifrost_plan or "sin-plan",
            )
            return "Lo siento, no pude generar una respuesta en este momento. ¿Podrías reformular tu pregunta?"
        return text
    except Exception as e:
        logger.error("Error generando respuesta: %s", e)
        return "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."


@observe(as_type="tool")
async def escalate(store: StoreConfig) -> str:
    """Herramienta terminal: escalada a humano (fallback fail-closed cordial).

    Aún no se conecta a Chatwoot; el flag `escalado` del resultado queda como
    gancho para el handoff a un agente humano en fases posteriores.
    """
    logger.warning("Tool escalate :: tienda no mapeada=%s", store.store_name)
    return ESCALATION_MESSAGE


TOOL_REGISTRY: dict[str, Callable[..., Awaitable[object]]] = {
    "search_catalogo": search_catalogo,
    "search_docs": search_docs,
    "list_categorias": list_categorias,
    "get_memory": get_memory,
    "get_cart": get_cart,
    "get_order": get_order,
    "answer": answer,
    "escalate": escalate,
}