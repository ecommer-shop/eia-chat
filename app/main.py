from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.rag_client import fetch_context_from_rag
from app.llm_generator import generate_final_response

app = FastAPI(
    title="Gateway Bot API - E-commerce",
    description="API frontal que atiende al usuario y orquesta con el servicio RAG",
    version="1.0.0"
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")
    
    # 1. Hablar con API 1 para obtener Contexto e Intención
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