import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.clients import close_clients
from app.config import settings, validate_settings
from app.rate_limit import RateLimitMiddleware
from app.schemas import ChatRequest, AgentChatResponse
from app.store_resolver import resolve_store, reload_stores
from app.store_repository import get_all_stores
from app.agent.core import run_agent
from app.db import init_db_pool, close_db_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        validate_settings()
        await init_db_pool()
        logger.info("eia-rag gateway iniciado (collection=%s)", settings.COLLECTION_NAME)
        yield
    except ValueError as e:
        logger.error("Configuración inválida: %s", e)
        raise
    finally:
        await close_db_pool()
        await close_clients()
        logger.info("eia-rag gateway detenido")


app = FastAPI(
    title="EIA RAG Gateway",
    description="Unified RAG: intent classification + vector search + memory + LLM generation",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting primero (protege los endpoints costosos incluso antes de CORS).
if settings.RATE_LIMIT_PER_MINUTE and settings.RATE_LIMIT_PER_MINUTE > 0:
    app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_PER_MINUTE)

# CORS: orígenes explícitos desde configuración. `allow_credentials=True` y
# orígenes wildcard son incompatibles (los browsers lo rechazan), por eso
# credentials solo se activan si no hay "*".
allow_credentials = "*" not in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest):
    """Fase 3/6 — Endpoint del agente conversacional (a mano, sin framework)."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        result = await run_agent(
            query=query,
            conversation_id=request.conversation_id or "",
            account_id=request.account_id,
            inbox_id=request.inbox_id,
            user_id=request.user_id,
            channel=request.channel,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = await resolve_store(request.account_id, request.inbox_id)

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
    stores = await get_all_stores()
    return {
        "total_stores": len(stores),
        "stores": [
            {
                "store_name": s.store_name,
                "account_id": s.account_id,
                "is_global": s.is_global,
                "audience": s.audience,
                "channel_tokens": s.channel_tokens,
                "inbox_count": len(s.inbox_map),
                "inbox_ids": s.inbox_map,
            }
            for s in stores
        ],
        "account_ids_loaded": sorted([s.account_id for s in stores]),
    }


@app.post("/stores/reload")
async def reload_stores_endpoint() -> dict:
    result = await reload_stores()
    return {"status": "ok", "message": "Tiendas recargadas", **result}