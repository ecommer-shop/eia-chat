import json
import logging

from redis.exceptions import ConnectionError as RedisConnectionError

from app.clients import get_redis, reset_redis
from app.config import settings

logger = logging.getLogger(__name__)

MEMORY_PREFIX = "eia-rag"
MAX_MESSAGES = 12
TTL_SECONDS = 86400


def _key(conversation_id: str) -> str:
    return f"{MEMORY_PREFIX}:conversation:{conversation_id}:messages"


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