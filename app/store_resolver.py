import logging
from dataclasses import replace

from app.models import StoreConfig
from app.store_loader import load_stores

logger = logging.getLogger(__name__)

_ACCOUNT_MAP: dict[int, StoreConfig] = {}


def _fallback_store(account_id: int | None, inbox_id: int | None) -> StoreConfig:
    key = account_id if account_id is not None else inbox_id
    key_str = str(key) if key is not None else "desconocida"
    return StoreConfig(
        store_name=f"tienda-{key_str}",
        account_id=account_id,
        channel_name="unknown",
        channel_tokens=[f"__unmapped_{key_str}__"],
        is_global=False,
        audience="CLIENTE",
        is_mapped=False,
        system_prompt=(
            "Aun no tengo informacion configurada para esta tienda. "
            "Un miembro del equipo te va a contactar pronto. "
            "Habla en espanol con un tono cercano y profesional."
        ),
        language="es",
    )


def resolve_store(account_id: int | None, inbox_id: int | None = None) -> StoreConfig:
    if account_id is not None and account_id in _ACCOUNT_MAP:
        base = _ACCOUNT_MAP[account_id]
        channel = base.inbox_map.get(inbox_id, "default")
        prompt = base.prompts.get(channel, base.prompts.get("default", base.system_prompt))
        return replace(base, channel_name=channel, system_prompt=prompt)
    return _fallback_store(account_id, inbox_id)


def get_account_map() -> dict[int, StoreConfig]:
    return dict(_ACCOUNT_MAP)


def set_account_map(new_map: dict[int, StoreConfig]) -> None:
    global _ACCOUNT_MAP
    _ACCOUNT_MAP = new_map
    logger.info("ACCOUNT_MAP actualizado: %d tiendas", len(_ACCOUNT_MAP))


def reload_stores() -> dict[str, int]:
    new_map = load_stores()
    set_account_map(new_map)
    return {"loaded": len(new_map)}


def init_stores() -> None:
    new_map = load_stores()
    set_account_map(new_map)
