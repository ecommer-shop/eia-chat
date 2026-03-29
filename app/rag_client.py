import httpx
from app.config import settings

async def fetch_context_from_rag(query: str) -> dict:
    """
    Hace una petición POST a la API 1 (RAG Query) para obtener contexto e intención.
    """
    url = f"{settings.RAG_API_URL}/retrieve_context"
    payload = {"query": query}
    
    print(f"📡 [RAG CLIENT] Consultando API 1 en: {url}")
    
    try:
        # Usamos timeout de 15s por si Qdrant o Groq tardan en la API 1
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ [RAG CLIENT] Éxito. Intención detectada: {data.get('intent')}")
            return data
            
    except httpx.HTTPError as e:
        print(f"❌ [RAG CLIENT ERROR] Fallo al conectar con API 1: {e}")
        # Si falla, devolvemos un esquema vacío seguro
        return {"intent": "GENERAL", "context": []}