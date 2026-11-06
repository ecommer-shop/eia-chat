"""Tests del estado pendiente de compra en memoria (Redis/Valkey).

Offline: reemplazan `app.memory.get_redis` por un Redis falso en proceso para
verificar la clave dedicada, el JSON guardado, el TTL y —sobre todo— que el
estado NO se mezcla con el historial de mensajes (`get_history`/`save_message`).
"""

import json

import pytest

from app import memory


class _FakeRedis:
    """Subconjunto de redis.Redis que usa `app.memory`."""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.expires: dict[str, int] = {}

    # listas (historial)
    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        self.lists[key] = items[start:] if end == -1 else items[start:end + 1]

    def expire(self, key, seconds):
        self.expires[key] = seconds

    # strings (estado pendiente)
    def set(self, key, value, ex=None):
        self.values[key] = value
        if ex is not None:
            self.expires[key] = ex
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        self.values.pop(key, None)
        self.expires.pop(key, None)
        return 1


@pytest.fixture
def fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(memory, "get_redis", lambda: fake)
    return fake


def _pending():
    return {
        "status": "confirmation_required",
        "paymentMethodCode": "wompi",
        "account_id": 1,
        "inbox_id": 2,
        "chatwoot_conversation_id": 42,
        "order": {
            "code": "CART-10",
            "totalWithTaxDecimal": "185.00",
            "currencyCode": "COP",
        },
    }


class TestPendingOrder:
    def test_clave_dedicada(self, fake_redis):
        memory.save_pending_order("conv-1", _pending())

        assert memory._pending_key("conv-1") == (
            "eia-rag:conversation:conv-1:pending-order"
        )
        assert list(fake_redis.values) == ["eia-rag:conversation:conv-1:pending-order"]

    def test_guardar_y_leer_roundtrip(self, fake_redis):
        memory.save_pending_order("conv-1", _pending())

        assert memory.get_pending_order("conv-1") == _pending()

    def test_usa_el_ttl_del_historial(self, fake_redis):
        memory.save_pending_order("conv-1", _pending())

        assert fake_redis.expires["eia-rag:conversation:conv-1:pending-order"] == (
            memory.TTL_SECONDS
        )

    def test_sin_pendiente_devuelve_none(self, fake_redis):
        assert memory.get_pending_order("conv-vacia") is None

    def test_clear_borra_el_estado(self, fake_redis):
        memory.save_pending_order("conv-1", _pending())
        memory.clear_pending_order("conv-1")

        assert memory.get_pending_order("conv-1") is None

    def test_clear_sin_estado_no_rompe(self, fake_redis):
        memory.clear_pending_order("conv-nunca-usada")

    def test_json_invalido_devuelve_none(self, fake_redis):
        fake_redis.values["eia-rag:conversation:conv-1:pending-order"] = "{no-json"

        assert memory.get_pending_order("conv-1") is None

    def test_respuesta_no_dict_devuelve_none(self, fake_redis):
        fake_redis.values["eia-rag:conversation:conv-1:pending-order"] = json.dumps([1, 2])

        assert memory.get_pending_order("conv-1") is None

    def test_bytes_se_decodifican(self, fake_redis):
        fake_redis.values["eia-rag:conversation:conv-1:pending-order"] = json.dumps(
            _pending()
        ).encode("utf-8")

        assert memory.get_pending_order("conv-1") == _pending()

    def test_no_contamina_el_historial(self, fake_redis):
        """El estado interno NO debe aparecer en el historial que ve el LLM."""
        memory.save_message("conv-1", "user", "quiero finalizar mi compra")
        memory.save_pending_order("conv-1", _pending())

        history = memory.get_history("conv-1")

        assert history == [{"role": "user", "content": "quiero finalizar mi compra"}]
        assert all("wompi" not in m["content"] for m in history)

    def test_historial_no_contamina_el_pendiente(self, fake_redis):
        memory.save_pending_order("conv-1", _pending())
        memory.save_message("conv-1", "user", "hola")

        assert memory.get_pending_order("conv-1") == _pending()


class TestPendingOrderSinRedis:
    def test_save_sin_cliente_no_rompe(self, monkeypatch):
        monkeypatch.setattr(memory, "get_redis", lambda: None)
        memory.save_pending_order("conv-1", _pending())

    def test_get_sin_cliente_devuelve_none(self, monkeypatch):
        monkeypatch.setattr(memory, "get_redis", lambda: None)
        assert memory.get_pending_order("conv-1") is None

    def test_clear_sin_cliente_no_rompe(self, monkeypatch):
        monkeypatch.setattr(memory, "get_redis", lambda: None)
        memory.clear_pending_order("conv-1")

    def test_redis_caido_no_rompe(self, monkeypatch):
        from redis.exceptions import ConnectionError as RedisConnectionError

        class _Broken:
            def set(self, *a, **k):
                raise RedisConnectionError("down")

            def get(self, *a, **k):
                raise RedisConnectionError("down")

            def delete(self, *a, **k):
                raise RedisConnectionError("down")

        reset_calls = []
        monkeypatch.setattr(memory, "get_redis", lambda: _Broken())
        monkeypatch.setattr(memory, "reset_redis", lambda: reset_calls.append(True))

        memory.save_pending_order("conv-1", _pending())
        assert memory.get_pending_order("conv-1") is None
        memory.clear_pending_order("conv-1")

        assert len(reset_calls) == 3
