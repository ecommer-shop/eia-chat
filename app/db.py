"""Pool asyncpg a Vendure PG (read-only)."""
from contextlib import asynccontextmanager
import asyncpg
import os
from app.config import settings

_pool: asyncpg.Pool | None = None

# Test mode flag
_TEST_MODE = os.getenv("EIA_TEST_MODE", "").lower() in ("1", "true", "yes")


def _get_ssl_mode() -> str:
    """Convierte DB_SSL a sslmode válido para asyncpg."""
    ssl_val = str(settings.PG_SSL).lower()
    if ssl_val in ("true", "1", "yes", "require"):
        return "require"
    if ssl_val in ("false", "0", "no", "disable"):
        return "disable"
    if ssl_val in ("allow", "prefer", "verify-ca", "verify-full"):
        return ssl_val
    # Default seguro para Azure PostgreSQL
    return "require"


async def init_db_pool() -> None:
    global _pool
    if _TEST_MODE:
        # En modo test, no inicializamos el pool real
        return
    _pool = await asyncpg.create_pool(
        host=settings.PG_HOST,
        port=settings.PG_PORT,
        database=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PASSWORD,
        ssl=_get_ssl_mode(),
        min_size=1,
        max_size=5,
        command_timeout=10,
    )


async def close_db_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def acquire_db():
    if _TEST_MODE:
        raise RuntimeError("DB pool not available in test mode")
    if not _pool:
        raise RuntimeError("DB pool not initialized. Call init_db_pool() first.")
    async with _pool.acquire() as conn:
        yield conn