import json
import logging
from pathlib import Path
from app.store_resolver import StoreConfig

logger = logging.getLogger(__name__)

_STORES_FILE = Path(__file__).parent / "stores.json"


def load_stores() -> dict[int, StoreConfig]:
    if not _STORES_FILE.exists():
        logger.error("stores.json no encontrado en %s", _STORES_FILE)
        return {}
    try:
        raw = json.loads(_STORES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("Error parseando stores.json: %s", e)
        return {}
    stores_raw = raw.get("stores", [])
    inbox_map: dict[int, StoreConfig] = {}
    for store_def in stores_raw:
        store_name = store_def["store_name"]
        is_global = store_def.get("is_global", False)
        audience = store_def.get("audience", "CLIENTE")
        language = store_def.get("language", "es")
        channel_tokens = store_def.get("channel_tokens", [])
        few_shot = store_def.get("few_shot_examples", [])
        prompts = store_def.get("system_prompt", {})
        default_prompt = prompts.get("default", f"Eres el asistente virtual de {store_name}.")
        inbox_map_raw = store_def.get("inbox_map", {})
        for inbox_id_str, channel_name in inbox_map_raw.items():
            inbox_id = int(inbox_id_str)
            prompt = prompts.get(channel_name, default_prompt)
            inbox_map[inbox_id] = StoreConfig(
                store_name=store_name,
                channel_name=channel_name,
                channel_tokens=list(channel_tokens),
                is_global=is_global,
                audience=audience,
                system_prompt=prompt,
                language=language,
                few_shot=list(few_shot),
            )
        logger.info(
            "Tienda '%s' cargada: %d inboxes, global=%s, tokens=%s",
            store_name, len(inbox_map_raw), is_global, channel_tokens,
        )
    logger.info("Total inboxes mapeados: %d", len(inbox_map))
    return inbox_map


def list_stores_summary() -> list[dict]:
    try:
        raw = json.loads(_STORES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    stores_raw = raw.get("stores", [])
    result = []
    for store_def in stores_raw:
        inbox_map_raw = store_def.get("inbox_map", {})
        result.append({
            "store_name": store_def["store_name"],
            "is_global": store_def.get("is_global", False),
            "audience": store_def.get("audience", "CLIENTE"),
            "channel_tokens": store_def.get("channel_tokens", []),
            "inbox_count": len(inbox_map_raw),
            "inbox_ids": {int(k): v for k, v in inbox_map_raw.items()},
        })
    return result
