"""Contenedor de clientes externos (singletons lazy).

Centraliza la creación de los clientes HTTP de Bifrost (LLM), Qdrant, Azure
OpenAI (embeddings) y Redis/Valkey.

Los clientes se crean una sola vez por VK (patrón cacheado) y se pueden cerrar
en el shutdown del servicio con `close_clients()`.

Seguridad: la VK de Bifrost viaja SOLO en el header `x-bf-vk`. Nunca se
loguea, nunca se expone en respuestas HTTP ni en las trazas de Langfuse.
"""

import logging
import os
import time

from cachetools import TTLCache
from openai import AsyncAzureOpenAI, AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from redis import Redis

from app.config import settings

logger = logging.getLogger(__name__)

# Test mode: sin llamadas de red a Bifrost (descubrimiento de modelos).
_TEST_MODE = os.getenv("EIA_TEST_MODE", "").lower() in ("1", "true", "yes")

_qdrant: AsyncQdrantClient | None = None
_azure: AsyncAzureOpenAI | None = None

# Clientes LLM cacheados por VK (una VK = un plan; acotado por nº de tiendas).
_llm_clients: dict[str, AsyncOpenAI] = {}

# Modelos permitidos por VK (GET /v1/models), TTL corto para reflejar cambios
# de plan o rotación de claves sin reiniciar el servicio.
_allowed_models: TTLCache[str, list[str]] = TTLCache(
    maxsize=200, ttl=settings.BIFROST_MODELS_CACHE_TTL
)

_redis: Redis | None = None
_redis_last_attempt: float = 0.0
_REDIS_RETRY_INTERVAL = 30


def get_llm(vk: str | None) -> AsyncOpenAI:
    """Cliente OpenAI-compatible apuntando a la pasarela Bifrost.

    `vk` es la clave virtual de la tienda (o la de Ecommer para account_id 1).
    La VK se envía como header `x-bf-vk` y como `Authorization: Bearer`
    (Bifrost ignora el segundo). No se admite VK vacía: sin clave no hay LLM.
    """
    if not vk:
        raise ValueError("VK de Bifrost requerida: no hay llave para esta tienda")
    client = _llm_clients.get(vk)
    if client is None:
        client = AsyncOpenAI(
            base_url=settings.BIFROST_BASE_URL.rstrip("/") + "/v1",
            api_key=vk,
            default_headers={"x-bf-vk": vk},
            timeout=settings.BIFROST_TIMEOUT,
            # El default del SDK (2) apila intentos de 60s y hace que un solo
            # mensaje tarde minutos; eia-bot corta a los 60s y el usuario
            # no recibe nada. Un reintento basta.
            max_retries=1,
        )
        _llm_clients[vk] = client
    return client


async def get_allowed_models(vk: str) -> list[str]:
    """Modelos que la VK puede usar según su plan (cacheado, TTL configurable).

    Devuelve `[]` si el descubrimiento falla: en ese caso `pick_model` usa la
    preferencia por defecto y la llamada real dirá si el modelo está bloqueado.
    """
    if not vk:
        return []
    if _TEST_MODE:
        # Sin red en tests: se asume que todos los modelos preferidos existen.
        return list(
            dict.fromkeys(
                [
                    settings.BIFROST_ROUTER_MODEL,
                    settings.BIFROST_CHAT_MODEL,
                    settings.BIFROST_CHAT_MODEL_FALLBACK,
                ]
            )
        )
    if vk in _allowed_models:
        return _allowed_models[vk] or []
    try:
        page = await get_llm(vk).models.list()
        ids = sorted({m.id for m in getattr(page, "data", []) or [] if getattr(m, "id", None)})
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "No se pudieron listar modelos de Bifrost (VK de %s caracteres): %s",
            len(vk), type(e).__name__,
        )
        ids = []
    _allowed_models[vk] = ids
    return ids


def invalidate_allowed_models(vk: str) -> None:
    """Fuerza a re-descubrir los modelos de esa VK (tras un 403 model_blocked)."""
    if vk:
        _allowed_models.pop(vk, None)


def reset_llm_model_cache() -> None:
    """Invalida el descubrimiento de modelos de todas las VK (POST /stores/reload)."""
    _allowed_models.clear()


async def pick_model(vk: str, preference: list[str]) -> str:
    """Elige el mejor modelo que la VK permite, según el orden de `preference`.

    Reglas:
    - Clasificación (router): `preference=[Phi-4-mini, ...]` → el más pequeño.
    - Respuesta: `preference=[Phi-4, Phi-4-mini]` → el más grande permitido
      por el plan (free/tienda terminan usando el mini).
    - Si el descubrimiento falla, se asume la primera preferencia.
    """
    allowed = await get_allowed_models(vk)
    prefs = list(dict.fromkeys(preference))
    if not allowed:
        return prefs[0]
    for model in prefs:
        if model in allowed:
            return model
    return prefs[-1]


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
    cierra solo si existe. Nunca se loguea ninguna VK.
    """
    global _qdrant, _azure, _redis
    for vk_client in list(_llm_clients.values()):
        try:
            await vk_client.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("Error cerrando cliente LLM de Bifrost: %s", e)
    _llm_clients.clear()
    _allowed_models.clear()
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
