"""Resuelve tienda por account_id + inbox_id (usa BD + cache + JSON prompts)."""
import logging
import os
from dataclasses import replace
from app.models import StoreConfig
from app.store_repository import get_store_by_account_id, invalidate_cache as repo_invalidate

logger = logging.getLogger(__name__)

# Test mode flag - allows tests to bypass database
_TEST_MODE = os.getenv("EIA_TEST_MODE", "").lower() in ("1", "true", "yes")


# Mock data for test mode
_TEST_STORES = {
    1: StoreConfig(
        store_name="ecommer",
        account_id=1,
        channel_tokens=[],
        is_global=True,
        audience="CLIENTE",
        system_prompt="Eres Simetria, el asistente virtual de Ecommer SAS.",
        language="es",
        few_shot=[
            {"role": "user", "content": "¡Hola!"},
            {"role": "assistant", "content": "¡Hola! Qué gusto saludarte 😊 Soy Simetria, tu asistente de Ecommer. ¿Qué estás buscando hoy: productos, envíos o algo más?"},
            {"role": "user", "content": "¿Qué productos tienen?"},
            {"role": "assistant", "content": "Hay de todo un poco: panela orgánica para endulzar de forma natural, champiñones orellana para recetas, y más selecciones de la plataforma. Para recomendarte los mejores, dime: ¿qué estás buscando (un postre, algo para cocinar, un regalo)?"},
        ],
        inbox_map={},
        prompts={"default": "Eres Simetria, el asistente virtual de Ecommer SAS."},
    ),
    5: StoreConfig(
        store_name="ziru-acoustics",
        account_id=5,
        channel_tokens=["ziru-acoustics-token"],
        is_global=False,
        audience="CLIENTE",
        system_prompt="Eres el asistente virtual de Ziru Acoustics.",
        language="es",
        few_shot=[],
        inbox_map={13: "shop"},
        prompts={"default": "Eres el asistente virtual de Ziru Acoustics."},
    ),
}


def _fallback_store(account_id: int | None, inbox_id: int | None) -> StoreConfig:
    """Tienda no mapeada -> token unmapped (cero resultados catálogo)."""
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


async def resolve_store(account_id: int | None, inbox_id: int | None = None) -> StoreConfig:
    """Versión async que usa BD + cache + JSON prompts."""
    if _TEST_MODE:
        # En modo test, devolvemos datos mockeados para account_ids conocidos
        if account_id is not None and account_id in _TEST_STORES:
            store = _TEST_STORES[account_id]
            channel = "default"
            prompt = store.prompts.get("default", store.system_prompt)
            return replace(store, channel_name=channel, system_prompt=prompt)
        return _fallback_store(account_id, inbox_id)
    
    if account_id is not None:
        store = await get_store_by_account_id(account_id)
        if store:
            # inbox_map simplificado: un solo canal "default"
            channel = "default"
            prompt = store.prompts.get("default", store.system_prompt)
            return replace(store, channel_name=channel, system_prompt=prompt)
    return _fallback_store(account_id, inbox_id)


# Compatibilidad hacia atrás (deprecated)
def get_account_map() -> dict[int, StoreConfig]:
    from app.store_repository import _store_cache
    return dict(_store_cache)


def set_account_map(new_map: dict[int, StoreConfig]) -> None:
    logger.warning("set_account_map() deprecated: BD es source of truth")


async def reload_stores() -> dict[str, int]:
    """Invalida cache completo (POST /stores/reload)."""
    repo_invalidate()
    from app.store_repository import get_all_stores
    stores = await get_all_stores()
    return {"loaded": len(stores)}


def init_stores() -> None:
    logger.info("init_stores(): BD es source of truth, no se carga al inicio")