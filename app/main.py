from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
import logging

from app.schemas import ChatRequest, ChatResponse
from app.rag_client import get_context
from app.llm_generator import generate_response
from app.auth import create_access_token, get_current_user_tenant
from app.db_client import (
    get_pool,
    close_pool,
    upsert_contact,
    get_or_create_conversation,
    get_recent_messages,
    save_message,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y cierra recursos al arrancar/apagar."""
    logger.info("🚀 Iniciando EIA Chat Gateway...")
    await get_pool()  # Inicializar pool PostgreSQL
    logger.info("✅ App lista.")
    yield
    await close_pool()  # Cerrar pool al apagar
    logger.info("🛑 App cerrada.")


app = FastAPI(
    title="EIA Chat Gateway - Multi-canal",
    description="Gateway que orquesta RAG, LLM y conversaciones multi-canal",
    version="2.0.0",
    lifespan=lifespan,
)


# ────────────────────────────────────────────
# ENDPOINT: Login
# ────────────────────────────────────────────


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Obtiene un JWT token usando credenciales básicas."""
    if (
        form_data.username != settings.ADMIN_USER
        or form_data.password != settings.ADMIN_PASSWORD
    ):
        raise HTTPException(
            status_code=401,
            detail="Credenciales de acceso incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": form_data.username, "tenant_id": "mi_tienda_01"}
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ────────────────────────────────────────────
# ENDPOINT: Chat Multi-canal
# ────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal que orquesta:
    1. Upsert de contacto
    2. Get/create conversación
    3. Cargar historial
    4. Recuperar contexto RAG
    5. Generar respuesta con LLM
    6. Persistir mensajes
    """
    try:
        # ── 1. Identificar / crear contacto ──────────────────────────
        logger.info(
            f"📡 Nueva consulta | canal: {request.channel} | usuario: {request.external_id}"
        )
        contact_id = await upsert_contact(
            tenant_id=request.tenant_id,
            channel=request.channel,
            external_id=request.external_id,
            name=request.contact_name,
        )

        # ── 2. Obtener / crear conversación ───────────────────────────
        conversation_id = await get_or_create_conversation(
            tenant_id=request.tenant_id,
            contact_id=contact_id,
            channel=request.channel,
        )

        # ── 3. Cargar historial (últimos 5 mensajes) ──────────────────
        history = await get_recent_messages(conversation_id, limit=5)

        # ── 4. Recuperar contexto RAG ─────────────────────────────────
        rag_result = await get_context(request.query)
        context = rag_result.get("context", [])
        intent = rag_result.get("intent", "GENERAL")

        # ── 5. Generar respuesta con historial + contexto ─────────────
        answer = await generate_response(
            query=request.query,
            context=context,
            intent=intent,
            channel=request.channel,
            history=history,
        )

        # ── 6. Persistir mensajes ─────────────────────────────────────
        await save_message(conversation_id, "user", request.query)
        await save_message(conversation_id, "assistant", answer)

        return ChatResponse(
            answer=answer,
            intent=intent,
            channel=request.channel,
            conversation_id=conversation_id,
            contact_id=contact_id,
        )

    except Exception as e:
        logger.error(f"❌ Error en /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "EIA Chat Gateway"}


# Importar settings para el endpoint de login
from app.config import settings