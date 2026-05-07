from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas import ChatRequest, ChatResponse
from app.rag_client import fetch_context_from_rag
from app.llm_generator import generate_final_response
from app.config import settings

# --- 1. NUEVAS IMPORTACIONES PARA SEGURIDAD ---
from app.auth import create_access_token, get_current_user_tenant

app = FastAPI(
    title="Gateway Bot API - E-commerce",
    description="API frontal que atiende al usuario y orquesta con el servicio RAG",
    version="1.0.0"
)

# --- 2. NUEVO ENDPOINT: Necesario para que el frontend obtenga el token ---
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Comparamos directamente con lo que hay en Settings
    if form_data.username != settings.ADMIN_USER or form_data.password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401, 
            detail="Credenciales de acceso incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(
        data={"sub": form_data.username, "tenant_id": "mi_tienda_01"}
    )
    return {"access_token": access_token, "token_type": "bearer"}
# -------------------------------------------------------------------------

# --- 3. CAMBIO MÍNIMO: Se inyecta la dependencia de seguridad ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    current_user: dict = Depends(get_current_user_tenant) # ¡Esto bloquea la ruta!
):
    query = request.query
    
    # Opcional: Ahora tienes acceso al tenant_id extraído del token
    # tenant_id = current_user["tenant_id"] 
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")
    
    # 1. Hablar con API 1 para obtener Contexto e Intención
    # (Si luego modificas fetch_context_from_rag, le podrías pasar el tenant_id aquí)
    rag_data = await fetch_context_from_rag(query)
    intent = rag_data.get("intent", "GENERAL")
    context_items = rag_data.get("context", [])
    
    # 2. Generar respuesta final con el LLM
    final_answer = generate_final_response(query, intent, context_items)
    
    # 3. Retornar al usuario/frontend
    return ChatResponse(
        answer=final_answer,
        intent_detected=intent,
        sources_used=len(context_items)
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Gateway Bot API"}