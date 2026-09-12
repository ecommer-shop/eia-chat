import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StoreConfig:
    store_name: str
    channel_name: str
    channel_tokens: list[str] = field(default_factory=list)
    is_global: bool = False
    audience: str = "CLIENTE"
    system_prompt: str = ""
    language: str = "es"
    is_mapped: bool = True
    few_shot: list[dict] = field(default_factory=list)


_INBOX_MAP: dict[int, StoreConfig] = {}


def _fallback_store(inbox_id: int) -> StoreConfig:
    return StoreConfig(
        store_name=f"tienda-{inbox_id}",
        channel_name="unknown",
        channel_tokens=[f"__unmapped_{inbox_id}__"],
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


def resolve_store(inbox_id: int) -> StoreConfig:
    if inbox_id in _INBOX_MAP:
        return _INBOX_MAP[inbox_id]
    return _fallback_store(inbox_id)


def get_inbox_map() -> dict[int, StoreConfig]:
    return dict(_INBOX_MAP)


def set_inbox_map(new_map: dict[int, StoreConfig]) -> None:
    global _INBOX_MAP
    _INBOX_MAP = new_map
    logger.info("INBOX_MAP actualizado: %d inboxes", len(_INBOX_MAP))


def reload_stores() -> dict[str, int]:
    from app.store_loader import load_stores
    new_map = load_stores()
    set_inbox_map(new_map)
    return {"loaded": len(new_map)}


def init_stores() -> None:
    from app.store_loader import load_stores
    new_map = load_stores()
    set_inbox_map(new_map)
