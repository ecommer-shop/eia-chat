"""Tests del cliente HTTP de identidad (Fase 1b / T2).

Offline: mockean `httpx.AsyncClient` y `app.identity_client.settings` para
verificar el esquema de firma HMAC (idéntico al del catálogo), la forma del
body enviado y el mapeo de `{ order: {...}|null, message? }`, sin red.
"""

import asyncio
import json

import pytest

from app import identity_client


def _frozen_settings(monkeypatch, **overrides):
    """Sobrescribe settings del cliente de identidad para el test."""
    base = {
        "ASSISTANT_CATALOG_ENABLED": True,
        "ASSISTANT_CATALOG_BASE_URL": "https://admin-stg.ecommer.shop",
        "ASSISTANT_CATALOG_SECRET": "test-secret",
        "ASSISTANT_CATALOG_KEY_ID": "eia-chat",
        "ASSISTANT_CATALOG_TIMEOUT_SECONDS": 5,
    }
    base.update(overrides)
    for key, value in base.items():
        monkeypatch.setattr(identity_client.settings, key, value, raising=False)


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

    instances: list = []
    response: _FakeResponse = _FakeResponse(200, {"order": None})
    raises: Exception | None = None

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
        if _RecordingClient.raises is not None:
            raise _RecordingClient.raises
        return _RecordingClient.response


@pytest.fixture
def recording_client(monkeypatch):
    _RecordingClient.instances = []
    _RecordingClient.raises = None
    _RecordingClient.response = _FakeResponse(200, {"order": None})
    monkeypatch.setattr(
        identity_client, "httpx", type("_H", (), {"AsyncClient": _RecordingClient})()
    )
    yield _RecordingClient


def _order():
    return {
        "id": 10,
        "code": "CART-10",
        "state": "AddingItems",
        "active": True,
        "total": 18500,
        "totalDecimal": "185.00",
        "totalWithTax": 18500,
        "totalWithTaxDecimal": "185.00",
        "currencyCode": "COP",
        "totalQuantity": 2,
        "couponCodes": [],
        "lines": [
            {
                "id": 1,
                "quantity": 2,
                "linePriceWithTax": 18500,
                "linePriceWithTaxDecimal": "185.00",
                "productVariant": {"id": 7, "name": "KZ Castor Pro", "sku": "KZ-1"},
            }
        ],
    }


class TestSigning:
    def test_sign_formato_esperado(self, monkeypatch):
        _frozen_settings(monkeypatch)
        sig = identity_client._sign(1700000000, b'{"conversation_id":1}')
        assert sig.startswith("sha256=")
        assert len(sig) == 7 + 64

    def test_sign_identica_al_catalogo(self, monkeypatch):
        """Mismo secreto, misma firma: un solo esquema para el backend."""
        from app import catalog_client

        _frozen_settings(monkeypatch)
        for key in (
            "ASSISTANT_CATALOG_SECRET",
            "ASSISTANT_CATALOG_BASE_URL",
            "ASSISTANT_CATALOG_KEY_ID",
        ):
            monkeypatch.setattr(catalog_client.settings, key, getattr(identity_client.settings, key))
        body = b'{"conversation_id":42,"account_id":1}'
        assert identity_client._sign(1700000000, body) == catalog_client._sign(1700000000, body)

    def test_sign_varia_con_body(self, monkeypatch):
        _frozen_settings(monkeypatch)
        assert identity_client._sign(1, b'{"a":1}') != identity_client._sign(1, b'{"a":2}')


class TestEnabled:
    def test_disabled_sin_flag(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_ENABLED=False)
        assert identity_client._enabled() is False

    def test_disabled_sin_secret(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_SECRET="")
        assert identity_client._enabled() is False

    def test_disabled_sin_base_url(self, monkeypatch):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_BASE_URL=None)
        assert identity_client._enabled() is False

    def test_enabled_con_todo(self, monkeypatch):
        _frozen_settings(monkeypatch)
        assert identity_client._enabled() is True


class TestGetCart:
    def test_exito_devuelve_order(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(200, {"order": _order()})

        order = asyncio.run(identity_client.get_cart(1, 2, 42))

        assert order is not None
        assert order["code"] == "CART-10"
        assert order["totalWithTaxDecimal"] == "185.00"

    def test_body_y_headers_firmados(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        monkeypatch.setattr(identity_client, "_now_seconds", lambda: 1700000000)

        asyncio.run(identity_client.get_cart(1, 2, 42))

        call = recording_client.instances[0].calls[0]
        assert call["path"] == "/api/assistant/identity/cart"
        assert json.loads(call["content"]) == {
            "conversation_id": 42,
            "account_id": 1,
            "inbox_id": 2,
        }
        assert call["headers"]["x-assistant-key-id"] == "eia-chat"
        assert call["headers"]["x-assistant-timestamp"] == "1700000000"
        assert call["headers"]["x-assistant-signature"] == identity_client._sign(
            1700000000, call["content"]
        )

    def test_sin_carrito_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(200, {"order": None, "message": "no cart"})

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None

    def test_404_anonimo_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(404, None, text="anonymous")

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None

    def test_401_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(401, None, text="bad signature")

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None

    def test_error_de_red_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.raises = RuntimeError("boom")

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None

    def test_sin_config_no_llama_http(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_ENABLED=False)

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None
        assert recording_client.instances == []

    def test_respuesta_no_json_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(200, None, text="<html>")

        assert asyncio.run(identity_client.get_cart(1, 2, 42)) is None


class TestGetOrder:
    def test_exito_devuelve_order(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(200, {"order": _order()})

        order = asyncio.run(identity_client.get_order(1, 2, 42, "CART-10"))

        assert order is not None
        assert order["code"] == "CART-10"

    def test_body_incluye_code(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        monkeypatch.setattr(identity_client, "_now_seconds", lambda: 1700000000)

        asyncio.run(identity_client.get_order(1, 2, 42, "  CART-10  "))

        call = recording_client.instances[0].calls[0]
        assert call["path"] == "/api/assistant/identity/order"
        assert json.loads(call["content"]) == {
            "conversation_id": 42,
            "account_id": 1,
            "inbox_id": 2,
            "code": "CART-10",
        }

    def test_sin_code_no_llama_http(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)

        assert asyncio.run(identity_client.get_order(1, 2, 42, "   ")) is None
        assert recording_client.instances == []

    def test_codigo_ajeno_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(
            200, {"order": None, "message": "not visible"}
        )

        assert asyncio.run(identity_client.get_order(1, 2, 42, "NOPE")) is None

    def test_404_anonimo_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch)
        recording_client.response = _FakeResponse(404, None, text="anonymous")

        assert asyncio.run(identity_client.get_order(1, 2, 42, "CART-10")) is None

    def test_sin_config_devuelve_none(self, monkeypatch, recording_client):
        _frozen_settings(monkeypatch, ASSISTANT_CATALOG_SECRET="")

        assert asyncio.run(identity_client.get_order(1, 2, 42, "CART-10")) is None
        assert recording_client.instances == []
