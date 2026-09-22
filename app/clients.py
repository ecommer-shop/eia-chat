"""Contenedor de clientes externos (singletons lazy).

Centraliza la creación de los clientes HTTP de Groq, Qdrant, Azure OpenAI y
Redis/Valkey que antes se repetía en `main.py`, `intent_classifier.py`,
`retriever.py`, `memory.py` y `agent/tools.py`.

Los clientes se crean una sola vez (patrón lazy singleton) y se pueden cerrar
en el shutdown del servicio con `close_clients()`.
"""

import logging
import time

from groq import AsyncGroq
from openai import AsyncAzureOpenAI
from qdrant_client import AsyncQdrantClient
from redis import Redis

from app.config import settings

logger = logging.getLogger(__name__)

_groq: AsyncGroq | None = None
_qdrant: AsyncQdrantClient | None = None
_azure: AsyncAzureOpenAI | None = None

_redis: Redis | None = None
_redis_last_attempt: float = 0.0
_REDIS_RETRY_INTERVAL = 30


def get_groq() -> AsyncGroq:
    global _groq
    if _groq is None:
        _groq = AsyncGroq(api_key=settings.GROQ_API_KEY)
    return _groq


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    return _qdrant


def get_azure() -> AsyncAzureOpenAI:
    global _azure
    if _azure is None:
        _azure = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
    return _azure


def get_redis() -> Redis | None:
    """Cliente Redis/Valkey opcional con reconexión automática.

    Devuelve `None` si Redis no está configurado (`.env`) o no está
    disponible. Si una conexión falla, se reintenta como máximo una vez cada
    30 segundos (`_REDIS_RETRY_INTERVAL`) para no martillar el endpoint.
    """
    global _redis, _redis_last_attempt
    if _redis is not None:
        return _redis
    if not settings.REDIS_URL:
        return None
    if time.time() - _redis_last_attempt < _REDIS_RETRY_INTERVAL:
        return None
    _redis_last_attempt = time.time()
    try:
        _redis = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis.ping()
        logger.info("Redis conectado correctamente")
        return _redis
    except Exception as e:
        logger.warning("Redis no disponible — reintento en %ds (%s)", _REDIS_RETRY_INTERVAL, e)
        _redis = None
        return None


def reset_redis() -> None:
    """Fuerza a `get_redis()` a intentar reconectar en la próxima llamada."""
    global _redis
    _redis = None


async def close_clients() -> None:
    """Cierra los clientes asíncronos en el shutdown del servicio.

    Los clientes que nunca se crearon no se tocan. Redis es síncrono y se
    cierra solo si existe.
    """
    global _groq, _qdrant, _azure, _redis
    if _groq is not None:
        try:
            await _groq.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error cerrando cliente Groq: %s", e)
        _groq = None
    if _qdrant is not None:
        try:
            await _qdrant.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error cerrando cliente Qdrant: %s", e)
        _qdrant = None
    if _azure is not None:
        try:
            await _azure.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error cerrando cliente Azure: %s", e)
        _azure = None
    if _redis is not None:
        try:
            _redis.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error cerrando cliente Redis: %s", e)
        _redis = None