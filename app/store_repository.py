"""Repository para configuración de tiendas (BD Vendure + Cache TTL)."""
import json
import os
from pathlib import Path
from cachetools import TTLCache
from app.models import StoreConfig
from app.db import acquire_db

# Cache: account_id -> StoreConfig (TTL 30s)
_store_cache: TTLCache[int, StoreConfig] = TTLCache(maxsize=200, ttl=30)

# Test mode flag - allows tests to bypass database
_TEST_MODE = os.getenv("EIA_TEST_MODE", "").lower() in ("1", "true", "yes")

# Mock data for test mode
_TEST_STORES = {
    1: {
        "store_name": "ecommer",
        "chatwoot_account_id": 1,
        "channel_token": "",
        "seller_name": "Ecommer",
        "language": "es",
    },
    5: {
        "store_name": "ziru-acoustics",
        "chatwoot_account_id": 5,
        "channel_token": "ziru-acoustics-token",
        "seller_name": "Ziru Acoustics",
        "language": "es",
    },
}

# Query SQL validada para una tienda
STORE_QUERY = """
    SELECT
        ca.chatwoot_account_id,
        ch.code as store_name,
        ch.token as channel_token,
        s.name as seller_name,
        ch."defaultLanguageCode" as language
    FROM public.chatwoot_account ca
    JOIN public.administrator a ON a.id = ca.administrator_id
    JOIN public."user" u ON u.id = a."userId"
    JOIN public.user_roles_role urr ON urr."userId" = u.id
    JOIN public.role r ON r.id = urr."roleId"
    JOIN public.role_channels_channel rcc ON rcc."roleId" = r.id
    JOIN public.channel ch ON ch.id = rcc."channelId" AND ch."sellerId" IS NOT NULL
    JOIN public.seller s ON s.id = ch."sellerId"
    WHERE ca.chatwoot_account_id = $1
      AND r.code LIKE '%-admin'
      AND ca.status = 'active'
    LIMIT 1
"""

# Query para listar todas las tiendas activas
ALL_STORES_QUERY = """
    SELECT
        ca.chatwoot_account_id,
        ch.code as store_name,
        ch.token as channel_token,
        s.name as seller_name,
        ch."defaultLanguageCode" as language
    FROM public.chatwoot_account ca
    JOIN public.administrator a ON a.id = ca.administrator_id
    JOIN public."user" u ON u.id = a."userId"
    JOIN public.user_roles_role urr ON urr."userId" = u.id
    JOIN public.role r ON r.id = urr."roleId"
    JOIN public.role_channels_channel rcc ON rcc."roleId" = r.id
    JOIN public.channel ch ON ch.id = rcc."channelId" AND ch."sellerId" IS NOT NULL
    JOIN public.seller s ON s.id = ch."sellerId"
    WHERE r.code LIKE '%-admin'
      AND ca.status = 'active'
"""

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_json(store_name: str) -> dict:
    """Carga prompt desde app/prompts/{store_name}.json"""
    prompt_file = _PROMPTS_DIR / f"{store_name}.json"
    if prompt_file.exists():
        return json.loads(prompt_file.read_text(encoding="utf-8"))
    # Fallback genérico
    return {
        "system_prompt": f"Eres el asistente virtual de {store_name}. Habla en español con un tono cercano y profesional.",
        "is_global": False,
        "audience": "CLIENTE",
        "language": "es",
        "few_shot_examples": [],
    }


def _row_to_store_config(row) -> StoreConfig:
    """Convierte row de BD a StoreConfig dataclass."""
    prompt_data = _load_prompt_json(row["store_name"])
    return StoreConfig(
        store_name=row["store_name"],
        account_id=row["chatwoot_account_id"],
        channel_tokens=[row["channel_token"]],
        is_global=prompt_data.get("is_global", False),
        audience=prompt_data.get("audience", "CLIENTE"),
        language=prompt_data.get("language", row["language"] or "es"),
        system_prompt=prompt_data.get("system_prompt", ""),
        inbox_map={},
        prompts={"default": prompt_data.get("system_prompt", "")},
        few_shot=prompt_data.get("few_shot_examples", []),
    )


async def get_store_by_account_id(account_id: int) -> StoreConfig | None:
    """Obtiene config de tienda por chatwoot_account_id."""
    if _TEST_MODE:
        # En modo test, devolvemos datos mockeados
        if account_id in _TEST_STORES:
            mock = _TEST_STORES[account_id]
            prompt_data = _load_prompt_json(mock["store_name"])
            return StoreConfig(
                store_name=mock["store_name"],
                account_id=mock["chatwoot_account_id"],
                channel_tokens=[mock["channel_token"]] if mock["channel_token"] else [],
                is_global=prompt_data.get("is_global", False),
                audience=prompt_data.get("audience", "CLIENTE"),
                language=prompt_data.get("language", mock["language"] or "es"),
                system_prompt=prompt_data.get("system_prompt", ""),
                inbox_map={},
                prompts={"default": prompt_data.get("system_prompt", "")},
                few_shot=prompt_data.get("few_shot_examples", []),
            )
        return None
    
    # 1. Cache hit
    if account_id in _store_cache:
        return _store_cache[account_id]

    # 2. Cache miss -> Query BD
    async with acquire_db() as conn:
        row = await conn.fetchrow(STORE_QUERY, account_id)

    if not row:
        return None

    # 3. Construir StoreConfig
    store = _row_to_store_config(row)

    # 4. Guardar en cache
    _store_cache[account_id] = store
    return store


async def get_all_stores() -> list[StoreConfig]:
    """Lista todas las tiendas activas (para GET /stores)."""
    if _TEST_MODE:
        return [
            await get_store_by_account_id(1),
            await get_store_by_account_id(5),
        ]
    
    async with acquire_db() as conn:
        rows = await conn.fetch(ALL_STORES_QUERY)

    return [_row_to_store_config(row) for row in rows]


def invalidate_cache(account_id: int | None = None) -> None:
    """Invalida cache (POST /stores/reload)."""
    if account_id:
        _store_cache.pop(account_id, None)
    else:
        _store_cache.clear()