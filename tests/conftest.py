"""Configuración global de pytest y fixtures compartidos."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.models import StoreConfig
from app.db import _pool, acquire_db
from app.store_repository import _store_cache, invalidate_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Limpia el cache entre tests."""
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture
def mock_db_pool(monkeypatch):
    """Mock del pool de base de datos para tests unitarios."""
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    # Set the global pool
    import app.db as db_module
    monkeypatch.setattr(db_module, "_pool", mock_pool)
    
    yield mock_conn, mock_pool


@pytest.fixture(autouse=True)
def mock_store_resolver(monkeypatch):
    """Mock del store resolver para tests unitarios (aplicado automáticamente)."""
    async def mock_resolve_store(account_id, inbox_id=None):
        if account_id == 99 or account_id is None:
            return StoreConfig(
                store_name=f"tienda-{account_id or 'desconocida'}",
                account_id=account_id,
                channel_name="unknown",
                channel_tokens=[f"__unmapped_{account_id or 'desconocida'}__"],
                is_global=False,
                audience="CLIENTE",
                is_mapped=False,
                system_prompt="Aun no tengo informacion configurada para esta tienda.",
                language="es",
            )
        
        stores = {
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
        
        store = stores.get(account_id)
        if store:
            return store
        
        return StoreConfig(
            store_name=f"tienda-{account_id}",
            account_id=account_id,
            channel_name="unknown",
            channel_tokens=[f"__unmapped_{account_id}__"],
            is_global=False,
            audience="CLIENTE",
            is_mapped=False,
            system_prompt="Aun no tengo informacion configurada para esta tienda.",
            language="es",
        )
    
    # Patch the resolve_store function where it's used
    monkeypatch.setattr("app.store_resolver.resolve_store", mock_resolve_store)
    monkeypatch.setattr("app.main.resolve_store", mock_resolve_store)
    monkeypatch.setattr("app.agent.core.resolve_store", mock_resolve_store)
    
    # Patch the repository functions
    async def mock_get_store_by_account_id(account_id):
        if account_id == 99:
            return None
        return await mock_resolve_store(account_id)
    
    async def mock_get_all_stores():
        return [await mock_resolve_store(1), await mock_resolve_store(5)]
    
    monkeypatch.setattr("app.store_repository.get_store_by_account_id", mock_get_store_by_account_id)
    monkeypatch.setattr("app.store_repository.get_all_stores", mock_get_all_stores)
    monkeypatch.setattr("app.store_repository.invalidate_cache", lambda *args: None)
    
    yield mock_resolve_store