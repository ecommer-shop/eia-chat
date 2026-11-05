"""Tests del cliente HTTP hacia el endpoint de catálogo (Fase 0.5 / T0.5.3).

Offline: mockean `httpx.AsyncClient` y `app.catalog_client.settings` para
verificar el esquema de firma HMAC y el mapeo de la respuesta al shape de
contexto RAG, sin llamadas de red.
"""

import asyncio
import json

import pytest

from app import catalog_client


def _frozen_settings(monkeypatch, **overrides):
    """Sobrescribe settings del cliente de catálogo para el test."""
    base = {
        "ASSISTANT_CATALOG_ENABLED": True,
        "ASSISTANT_CATALOG_BASE_URL": "https://admin-stg.ecommer.shop",
        "ASSISTANT_CATALOG_SECRET": "test-secret",
        "ASSISTANT_CATALOG_KEY_ID": "eia-chat",
        "ASSISTANT_CATALOG_TIMESTAMP_TOLERANCE_SECONDS": 300,
        "ASSISTANT_CATALOG_TIMEOUT_SECONDS": 5,
        "ASSISTANT_CATALOG_STOREFRONT_BASE_URL": "https://ecommer.shop",
    }
    base.update(overrides)
    for key, value in base.items():
        monkeypatch.setattr(catalog_client.settings, key, value, raising=False)


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _RecordingClient:
    """Reemplaza httpx.AsyncClient y captura headers/body enviados."""

    instances = []

    def __init__(self, base_url, timeout):
        self.base_url = base_url
        self.timeout = timeout
        self.calls = []
        _RecordingClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, path, content, headers):
        self.calls.append({"path": path, "content": content, "headers": headers})
        return _FakeResponse(200, {"items": []})


@pytest.fixture
def recording_client(monkeypatch):
    _RecordingClient.instances = []
    monkeypatch.setattr(catalog_client, "httpx", type("_H", (), {"AsyncClient": _RecordingClient})())
    yield _RecordingClient


def _search_response():
    return {
        "items": [
            {
                "id": "1",
                "name": "KZ Castor Pro",
                "slug": "kz-castor-pro",
                "description": "Audífonos IEM",
                "enabled": True,
                "featuredAsset": {"id": "a1", "preview": "https://cdn.ecommer.shop/kz.jpg"},
                "priceRange": {"min": 12345, "minDecimal": "123.45", "max": 23456,
                               "maxDecimal": "234.56", "currencyCode": "COP"},
            },
            {
                "id": "2",
                "name": "Panela orgánica",
                "slug": "panela-organica",
                "description": "Panela",
                "enabled": True,
                "featuredAsset": None,
                "priceRange": None,
            },
        ],
        "totalItems": 2,
        "limit": 5,
        "offset": 0,
    }


class TestSigning:
    def test_sign_formato_esperado(self, monkeypatch):
        _frozen_settings(monkeypatch)
        sig = catalog_client._sign(1700000000, b'{"query":"x"}')
        assert sig.startswith("sha256=")
        assert len(sig) == 7 + 64  # "sha256=" + 64 hex chars

    def test_sign_varia_con_timestamp(self, monkeypatch):
        _frozen_settings(monkeypatch)
        a = catalog_client._sign(1, b"body")
        b = catalog_client._sign(2, b"body")
        assert a != b

    def test_sign_varia_con_body(self, monkeypatch):
        _frozen_settings(monkeypatch)
        a = catalog_client._sign(1, b'{"a":1}')
        b = catalog_client._sign(1, b'{"a":2}')
        assert a != b


class TestEnabled:
    def test_disabled_sin_flag(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_ENABLED=False)
        assert catalog_client._enabled() is False

    def test_disabled_sin_secret(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_SECRET="")
        assert catalog_client._enabled() is False

    def test_disabled_sin_base_url(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_BASE_URL=None)
        assert catalog_client._enabled() is False

    def test_enabled_con_todo(self, monkeypatch):
        _frozen_settings(monkeypatch)
        assert catalog_client._enabled() is True


class TestMapping:
    def test_to_context_item_shape(self, monkeypatch):
        _frozen_settings(monkeypatch)
        item = {
            "id": "1",
            "name": "KZ Castor Pro",
            "slug": "kz-castor-pro",
            "description": "Audífonos IEM",
            "enabled": True,
            "featuredAsset": {"id": "a1", "preview": "https://cdn.ecommer.shop/kz.jpg"},
            "priceRange": {"min": 12345, "minDecimal": "123.45"},
        }
        ctx = catalog_client._to_context_item(item, 1.0)
        assert ctx["score"] == 1.0
        assert ctx["payload"]["name"] == "KZ Castor Pro"
        assert ctx["payload"]["url"] == "https://ecommer.shop/es/product/kz-castor-pro"
        assert ctx["payload"]["slug"] == "kz-castor-pro"
        assert ctx["payload"]["image"] == "https://cdn.ecommer.shop/kz.jpg"
        assert ctx["payload"]["priceRange"]["min"] == 12345

    def test_to_context_item_sin_slug_sin_url(self, monkeypatch):
        _frozen_settings(monkeypatch)
        ctx = catalog_client._to_context_item({"name": "X"}, 1.0)
        assert ctx["payload"]["url"] == ""


class TestSearchProducts:
    def test_search_mapea_items(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)

        async def _post(path, payload):
            return _search_response()

        monkeypatch.setattr(catalog_client, "_post", _post)
        items = asyncio.run(catalog_client.search_products("kz", limit=5))
        assert len(items) == 2
        assert items[0]["payload"]["name"] == "KZ Castor Pro"

    def test_search_sin_resultado_devuelve_vacio(self, monkeypatch):
        _frozen_settings(monkeypatch)

        async def _post(path, payload):
            return {"items": [], "totalItems": 0, "limit": 5, "offset": 0}

        monkeypatch.setattr(catalog_client, "_post", _post)
        items = asyncio.run(catalog_client.search_products("nada"))
        assert items == []

    def test_search_offline_no_explota(self, monkeypatch):
        _frozen_settings(monkeypatch)

        async def _post(path, payload):
            raise catalog_client.CatalogClientUnavailable("red caida")

        monkeypatch.setattr(catalog_client, "_post", _post)
        items = asyncio.run(catalog_client.search_products("kz"))
        assert items == []


class TestPost:
    def test_post_firma_con_headers_correctos(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        monkeypatch.setattr(catalog_client, "_now_seconds", lambda: 1700000000)

        asyncio.run(catalog_client._post("/api/assistant/catalog", {"query": "kz"}))
        call = recording_client.instances[0].calls[0]
        assert call["path"] == "/api/assistant/catalog"
        assert call["headers"]["x-assistant-key-id"] == "eia-chat"
        assert call["headers"]["x-assistant-timestamp"] == "1700000000"
        assert call["headers"]["x-assistant-signature"].startswith("sha256=")

    def test_post_no_configurado_lanza(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_ENABLED=False)

        with pytest.raises(catalog_client.CatalogClientUnavailable):
            asyncio.run(catalog_client._post("/api/assistant/catalog", {}))


class TestGetProduct:
    def test_get_product_mapea(self, monkeypatch):
        _frozen_settings(monkeypatch)

        async def _post(path, payload):
            assert path == "/api/assistant/catalog/product"
            return {"product": {"id": "9", "name": "P", "slug": "p", "description": "",
                                "enabled": True, "featuredAsset": None}}

        monkeypatch.setattr(catalog_client, "_post", _post)
        result = asyncio.run(catalog_client.get_product(product_id="9"))
        assert result["payload"]["name"] == "P"

    def test_get_product_sin_resultado_devuelve_none(self, monkeypatch):
        _frozen_settings(monkeypatch)

        async def _post(path, payload):
            return {"product": None, "message": "not found"}

        monkeypatch.setattr(catalog_client, "_post", _post)
        assert asyncio.run(catalog_client.get_product(slug="zzz")) is None

    def test_get_product_sin_id_ni_slug_devuelve_none(self, monkeypatch):
        _frozen_settings(monkeypatch)
        assert asyncio.run(catalog_client.get_product()) is None
