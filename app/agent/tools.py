"""Fase 3 — Herramientas del agente.

Son wrappers delgados sobre los módulos existentes de eia-rag: no duplican
lógica de búsqueda, memoria ni generación. `answer` y `escalate` son
herramientas terminales: producen el texto final de la respuesta.

`TOOL_REGISTRY` centraliza el catálogo de herramientas; el núcleo del agente
(`core.py`) decide cuáles invocar según la intención detectada.
"""

import logging
from typing import Awaitable, Callable

from groq import AsyncGroq
from langfuse import get_client, observe

from app.config import settings
from app.grounding import apply_grounding
from app.llm_generator import build_prompt
from app.memory import get_history
from app.observability import usage_details_from_groq
from app.renderer import annotate_products, render_product_footer
from app.retriever import search_context, list_categories
from app.store_resolver import StoreConfig

logger = logging.getLogger(__name__)

_groq_client: AsyncGroq | None = None

# Fallback cordial de escalada a humano (sin integración con Chatwoot aún).
# Mantener el término "configurada": el golden set (gs_025/gs_026) valida que
# la respuesta fail-closed lo incluya.
ESCALATION_MESSAGE = (
    "Aun no tengo informacion configurada para esta tienda. "
    "Un miembro del equipo te va a contactar pronto."
)


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _groq_client


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
    """Busca productos del catálogo (intención CATALOGO)."""
    logger.info("Tool search_catalogo :: store=%s query='%s'", store.store_name, query)
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


def _history_mentions_products(history: list[dict]) -> bool:
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if "👉" in content or "http" in content or "Producto:" in content:
            return True
    return False


@observe(as_type="generation")
async def answer(
    query: str,
    store: StoreConfig,
    context_items: list[dict],
    intent: str,
    history: list[dict],
) -> str:
    """Herramienta terminal: genera la respuesta final con Groq.

    Replica los fallbacks de `/chat`: contenido vacío y errores de Groq
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
    client = get_client()
    try:
        observation = client.start_as_current_observation(
            name="groq_chat",
            as_type="generation",
            model=settings.GROQ_CHAT_MODEL,
            input={"messages": messages},
        )
        with observation as generation:
            completion = await _get_groq().chat.completions.create(
                messages=messages,
                model=settings.GROQ_CHAT_MODEL,
                temperature=0.2,
                max_tokens=1024,
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
                usage_details=usage_details_from_groq(completion),
                metadata={
                    "finish_reason": completion.choices[0].finish_reason,
                    "grounded": not grounding_issues,
                    "grounding_issues": grounding_issues,
                    "links_added": added_links,
                },
            )
        if not text:
            logger.warning(
                "Groq devolvió contenido vacío",
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
    "answer": answer,
    "escalate": escalate,
}