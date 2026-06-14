import httpx
from app.config import settings
import logging

logger = logging.getLogger(__name__)

async def get_context(query: str) -> dict:
    """
    Hace una petición POST a la API de RAG (eia-query) para obtener contexto e intención.
    """
    url = f"{settings.RAG_API_URL}/retrieve_context"
    payload = {"query": query}
    
    logger.info(f"📡 Consultando API RAG en: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ Contexto obtenido. Intención: {data.get('intent')}")
            return data
            
    except httpx.HTTPError as e:
        logger.error(f"❌ Error conectando con RAG: {e}")
        # Esquema seguro si falla
        return {"intent": "GENERAL", "context": []}