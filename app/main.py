import logging
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import ValidationError

from app.config import settings, validate_settings
from app.schemas import ChatRequest, ChatResponse, AgentChatResponse
from app.store_resolver import resolve_store, init_stores, reload_stores, get_inbox_map
from app.store_loader import list_stores_summary
from app.intent_classifier import classify_intent
from app.retriever import search_context, list_categories
from app.memory import get_history, save_message
from app.llm_generator import build_prompt
from app.grounding import apply_grounding
from app.renderer import annotate_products, render_product_footer
from app.agent.core import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EIA RAG Gateway",
    description="Unified RAG: intent classification + vector search + memory + LLM generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_groq_client: AsyncGroq | None = None

_CATEGORIES_QUERY_RE = re.compile(r"categor[ií]as", re.IGNORECASE)


def _is_categories_query(query: str) -> bool:
    return bool(_CATEGORIES_QUERY_RE.search(query))


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _groq_client


@app.on_event("startup")
async def startup() -> None:
    try:
        validate_settings()
        init_stores()
        logger.info("eia-rag gateway iniciado (collection=%s)", settings.COLLECTION_NAME)
    except ValueError as e:
        logger.error("Configuración inválida: %s", e)
        raise


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    store = resolve_store(request.inbox_id)
    logger.info(
        "Consulta: '%s' | tienda=%s | canal=%s | conversation=%s",
        query, store.store_name, store.channel_name, request.conversation_id,
    )

    intents = await classify_intent(query)
    intent_str = ", ".join(intents)
    logger.info("Intenciones: %s", intent_str)

    if "CATALOGO" in intents and _is_categories_query(query):
        categories = await list_categories(store)
        context_items = (
            [{"score": 1.0, "payload": {
                "metadata": {"categories": categories, "name": "Catálogo de la tienda"},
                "text": f"Categorías disponibles en la tienda: {', '.join(categories)}.",
            }}]
            if categories
            else []
        )
    else:
        context_items = await search_context(query, store, intents)

    history = get_history(request.conversation_id)

    messages = build_prompt(
        query=query,
        intent=intent_str,
        context_items=context_items,
        history=history,
        store_prompt=store.system_prompt,
    )

    try:
        client = _get_groq()
        completion = await client.chat.completions.create(
            messages=messages,
            model=settings.GROQ_CHAT_MODEL,
            temperature=0.2,
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content or ""
        answer = raw.encode("utf-8", errors="replace").decode("utf-8")
        answer = answer.replace("\n", " ").replace("\r", "").strip()
        answer, grounding_issues = apply_grounding(
            answer, context_items, history, intent=intent_str
        )
        if grounding_issues:
            logger.warning("/chat :: grounding bloqueó la respuesta: %s", grounding_issues)
        elif answer:
            clean_text, mentioned = annotate_products(answer, context_items, intent=intent_str)
            footer = render_product_footer(mentioned)
            answer = f"{clean_text} {footer}".strip() if footer else clean_text
            if mentioned:
                logger.info("/chat :: links agregados: %s", [p.url for p in mentioned])
        if not answer:
            logger.warning(
                "Groq devolvió contenido vacío (finish_reason=%s)",
                completion.choices[0].finish_reason,
            )
            answer = "Lo siento, no pude generar una respuesta en este momento. ¿Podrías reformular tu pregunta?"
    except Exception as e:
        logger.error("Error generando respuesta: %s", e)
        answer = "Lo siento, estoy teniendo problemas técnicos en este momento para procesar tu solicitud."

    save_message(request.conversation_id, "user", query)
    save_message(request.conversation_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        intent_detected=intent_str,
        sources_used=len(context_items),
        conversation_id=request.conversation_id,
    )


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest):
    """Fase 3/6 — Endpoint del agente conversacional (a mano, sin framework).

    Mismo contrato que /chat. Devuelve additionally `escalado` y `tools_used`.
    No reemplaza a /chat hasta que supere el baseline del golden set.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        result = await run_agent(
            query=query,
            conversation_id=request.conversation_id or "",
            inbox_id=request.inbox_id,
            user_id=request.user_id,
            channel=request.channel,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = resolve_store(request.inbox_id)

    logger.info(
        "Agente resultado: '%s' | tienda=%s (%s) | escalado=%s | tools=%s",
        query, store.store_name, store.channel_name, result.escalado, result.tools_used,
    )

    return AgentChatResponse(
        answer=result.answer,
        intent_detected=result.intent_detected,
        sources_used=result.sources_used,
        conversation_id=result.conversation_id,
        escalado=result.escalado,
        tools_used=result.tools_used,
    )


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": "EIA RAG Gateway",
        "collection": settings.COLLECTION_NAME,
    }


@app.get("/stores")
async def list_stores() -> dict:
    loaded = get_inbox_map()
    summary = list_stores_summary()
    return {
        "total_inboxes": len(loaded),
        "stores": summary,
        "inbox_ids_loaded": sorted(loaded.keys()),
    }


@app.post("/stores/reload")
async def reload_stores_endpoint() -> dict:
    result = reload_stores()
    return {"status": "ok", "message": "Tiendas recargadas", "inboxes_loaded": result["loaded"]}
