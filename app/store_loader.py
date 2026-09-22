import json
import logging
from pathlib import Path
from app.models import StoreConfig

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
    account_map: dict[int, StoreConfig] = {}
    for store_def in stores_raw:
        store_name = store_def["store_name"]
        account_id_raw = store_def.get("account_id")
        if account_id_raw is None:
            logger.warning("Tienda '%s' sin account_id — se omite", store_name)
            continue
        account_id = int(account_id_raw)
        is_global = store_def.get("is_global", False)
        audience = store_def.get("audience", "CLIENTE")
        language = store_def.get("language", "es")
        channel_tokens = store_def.get("channel_tokens", [])
        few_shot = store_def.get("few_shot_examples", [])
        prompts = store_def.get("system_prompt", {})
        default_prompt = prompts.get("default", f"Eres el asistente virtual de {store_name}.")
        inbox_map_raw = store_def.get("inbox_map", {})
        inbox_map = {int(k): v for k, v in inbox_map_raw.items()}
        account_map[account_id] = StoreConfig(
            store_name=store_name,
            account_id=account_id,
            channel_tokens=list(channel_tokens),
            is_global=is_global,
            audience=audience,
            system_prompt=default_prompt,
            language=language,
            few_shot=list(few_shot),
            inbox_map=inbox_map,
            prompts=dict(prompts),
        )
        logger.info(
            "Tienda '%s' cargada: account_id=%s, %d inboxes, global=%s, tokens=%s",
            store_name, account_id, len(inbox_map_raw), is_global, channel_tokens,
        )
    logger.info("Total tiendas mapeadas: %d", len(account_map))
    return account_map


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
            "account_id": store_def.get("account_id"),
            "is_global": store_def.get("is_global", False),
            "audience": store_def.get("audience", "CLIENTE"),
            "channel_tokens": store_def.get("channel_tokens", []),
            "inbox_count": len(inbox_map_raw),
            "inbox_ids": {int(k): v for k, v in inbox_map_raw.items()},
        })
    return result
