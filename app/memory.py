import json
import logging

from redis.exceptions import ConnectionError as RedisConnectionError

from app.clients import get_redis, reset_redis
from app.config import settings

logger = logging.getLogger(__name__)

MEMORY_PREFIX = "eia-rag"
MAX_MESSAGES = 12
TTL_SECONDS = 86400

# Estado interno del flujo de compra (place_order en dos turnos). Vive en una
# clave propia, NUNCA en el historial de mensajes: el historial se inyecta tal
# cual en el prompt del LLM y el estado pendiente no debe formar parte de él.
PENDING_ORDER_SUFFIX = "pending-order"


def _key(conversation_id: str) -> str:
    return f"{MEMORY_PREFIX}:conversation:{conversation_id}:messages"


def _pending_key(conversation_id: str) -> str:
    return f"{MEMORY_PREFIX}:conversation:{conversation_id}:{PENDING_ORDER_SUFFIX}"


def get_history(conversation_id: str) -> list[dict]:
    client = get_redis()
    if client is None:
        return []
    try:
        raw = client.lrange(_key(conversation_id), 0, -1)
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis no disponible — historial vacío (%s)", e)
        reset_redis()
        return []
    messages: list[dict] = []
    for item in raw:
        try:
            messages.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return messages


def save_message(conversation_id: str, role: str, content: str) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        key = _key(conversation_id)
        item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        client.rpush(key, item)
        client.ltrim(key, -MAX_MESSAGES, -1)
        client.expire(key, TTL_SECONDS)
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis no disponible — mensaje no guardado (%s)", e)
        reset_redis()


def save_pending_order(conversation_id: str, data: dict) -> None:
    """Guarda el estado "pendiente de place_order" de la conversación.

    Es un JSON suelto (no una lista) con TTL igual al del historial: el estado
    de compra no tiene que vivir más que la propia conversación. Best-effort,
    como el resto de la memoria: si Redis no está no se guarda nada y el turno
    siguiente simplemente se comporta como uno nuevo.
    """
    client = get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(data, ensure_ascii=False)
        client.set(_pending_key(conversation_id), payload, ex=TTL_SECONDS)
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis no disponible — pending-order no guardado (%s)", e)
        reset_redis()
    except (TypeError, ValueError) as e:
        logger.warning("pending-order no serializable (%s)", e)


def get_pending_order(conversation_id: str) -> dict | None:
    """Estado pendiente de la conversación, o `None` si no hay o expiró."""
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_pending_key(conversation_id))
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis no disponible — pending-order vacío (%s)", e)
        reset_redis()
        return None
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def clear_pending_order(conversation_id: str) -> None:
    """Descarta el estado pendiente (fin del flujo o cambio de tema)."""
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(_pending_key(conversation_id))
    except (RedisConnectionError, OSError) as e:
        logger.warning("Redis no disponible — pending-order no borrado (%s)", e)
        reset_redis()
