"""Tests del clasificador de intención (T2.1/T2.2).

Offline: mockean `get_llm` y `pick_model` de `app.clients` para no tocar Bifrost.
Verifican el contrato nuevo (`classify_intent_detailed` con intents+params+confidence)
y la compatibilidad hacia atrás de `classify_intent` (solo lista de intenciones).
"""

import asyncio

import pytest

from app.intent_classifier import (
    IntentClassification,
    classify_intent,
    classify_intent_detailed,
    _extract_order_code,
    _keyword_fallback,
    _parse_code_param,
    _parse_intent_output,
    _parse_query_param,
)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    async def create(self, **kwargs):
        choice = type("C", (), {})()
        choice.finish_reason = "stop"
        choice.message = type("M", (), {"content": self._content})()
        return type("R", (), {"choices": [choice]})()


class _FakeLLM:
    def __init__(self, content):
        self.content = content

    def with_options(self, timeout):
        return self

    @property
    def chat(self):
        return type("Chat", (), {"completions": _FakeCompletions(self.content)})()


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock de pick_model y get_llm para un clasificador controlado."""
    async def _pick(vk, preference):
        return preference[0]

    monkeypatch.setattr("app.intent_classifier.pick_model", _pick)
    return _FakeLLM


def _set_llm(monkeypatch, content):
    fake = _FakeLLM(content)
    monkeypatch.setattr("app.intent_classifier.get_llm", lambda vk: fake)
    return fake


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_intents(self):
        assert _parse_intent_output("BUSCAR, CATALOGO\nQUERY: audifonos kz") == ["BUSCAR", "CATALOGO"]

    def test_parse_ignora_invalidos(self):
        assert _parse_intent_output("CATALOGO, COSAS\n") == ["CATALOGO"]

    def test_parse_deduplica(self):
        assert _parse_intent_output("CATALOGO, CATALOGO") == ["CATALOGO"]

    def test_parse_query_param(self):
        assert _parse_query_param("CATALOGO\nQUERY: audifonos kz") == "audifonos kz"

    def test_parse_query_param_ausente(self):
        assert _parse_query_param("CATALOGO") is None

    def test_parse_code_param(self):
        assert _parse_code_param("PEDIDO\nCODE: CART-10") == "CART-10"

    def test_parse_code_param_ausente(self):
        assert _parse_code_param("PEDIDO") is None

    def test_parse_intents_carrito_pedido(self):
        assert _parse_intent_output("CARRITO, PEDIDO") == ["CARRITO", "PEDIDO"]


# ---------------------------------------------------------------------------
# Fase 1b — extracción de código de pedido
# ---------------------------------------------------------------------------

class TestOrderCodeExtraction:
    def test_codigo_con_guion(self):
        assert _extract_order_code("¿dónde está mi pedido ABC-1234?") == "ABC-1234"

    def test_codigo_con_prefijo_hash(self):
        assert _extract_order_code("consultá el pedido #98765") == "98765"

    def test_codigo_tras_dos_puntos(self):
        assert _extract_order_code("mi pedido: 3F9A2B") == "3F9A2B"

    def test_sin_codigo(self):
        assert _extract_order_code("¿dónde está mi pedido?") is None

    def test_no_confunde_palabras_con_codigo(self):
        assert _extract_order_code("estado de mi pedido") is None


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def test_buscar_detecta_buscar(self):
        intents = _keyword_fallback("busca audífonos kz")
        assert "BUSCAR" in intents

    def test_buscar_detecta_encuentrame(self):
        intents = _keyword_fallback("encuéntrame una cámara")
        assert "BUSCAR" in intents

    def test_comprar_va_a_catalogo(self):
        intents = _keyword_fallback("quiero comprar unos audífonos")
        assert "CATALOGO" in intents
        assert "BUSCAR" not in intents

    def test_sin_coincidencia_conversacional(self):
        assert _keyword_fallback("hola buenos días") == ["CONVERSACIONAL"]

    # --- Fase 1b: CARRITO / PEDIDO ---

    def test_mi_carrito_detecta_carrito(self):
        assert "CARRITO" in _keyword_fallback("¿qué tengo en mi carrito?")

    def test_ver_carrito_detecta_carrito(self):
        assert "CARRITO" in _keyword_fallback("quiero ver el carrito")

    def test_carrito_no_detecta_catalogo(self):
        assert "CATALOGO" not in _keyword_fallback("mostrame mi carrito")

    def test_mi_pedido_detecta_pedido(self):
        assert "PEDIDO" in _keyword_fallback("¿dónde está mi pedido?")

    def test_estado_de_pedido_detecta_pedido(self):
        assert "PEDIDO" in _keyword_fallback("quiero saber el estado de mi pedido")

    def test_seguimiento_de_pedido_detecta_pedido(self):
        assert "PEDIDO" in _keyword_fallback("necesito el seguimiento de mi pedido")

    def test_quiero_hacer_un_pedido_es_catalogo_no_pedido(self):
        intents = _keyword_fallback("quiero hacer un pedido")
        assert "PEDIDO" not in intents
        assert "CATALOGO" in intents

    def test_comprar_no_es_pedido(self):
        intents = _keyword_fallback("quiero comprar unos audífonos")
        assert "PEDIDO" not in intents


# ---------------------------------------------------------------------------
# classify_intent_detailed
# ---------------------------------------------------------------------------

class TestDetailed:
    def test_sin_vk_baja_confianza_y_fallback(self):
        result = asyncio.run(classify_intent_detailed("quiero comprar audífonos", vk=None))
        assert isinstance(result, IntentClassification)
        assert result.intents == ["CATALOGO"]
        assert result.confidence < 0.4

    def test_con_vk_buscar_extrae_query(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "BUSCAR\nQUERY: audifonos kz")
        result = asyncio.run(
            classify_intent_detailed("encuéntrame audífonos kz", vk="test-vk")
        )
        assert "BUSCAR" in result.intents
        assert result.params["query"] == "audifonos kz"

    def test_con_vk_catalogo_sin_query_usa_original(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "CATALOGO")
        result = asyncio.run(
            classify_intent_detailed("quiero una cámara instantánea", vk="test-vk")
        )
        assert "CATALOGO" in result.intents
        assert result.params["query"] == "quiero una cámara instantánea"

    def test_error_llm_degradar_keywords(self, monkeypatch, mock_llm):
        class _Boom:
            def with_options(self, timeout):
                raise RuntimeError("boom")

        monkeypatch.setattr("app.intent_classifier.get_llm", lambda vk: _Boom())
        result = asyncio.run(
            classify_intent_detailed("hola buenos días", vk="test-vk")
        )
        assert result.intents == ["CONVERSACIONAL"]
        assert result.confidence < 0.4


# ---------------------------------------------------------------------------
# Fase 1b — CARRITO / PEDIDO en classify_intent_detailed
# ---------------------------------------------------------------------------

class TestIdentityIntents:
    def test_con_vk_carrito(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "CARRITO")
        result = asyncio.run(
            classify_intent_detailed("¿qué tengo en mi carrito?", vk="test-vk")
        )
        assert "CARRITO" in result.intents
        assert "code" not in result.params

    def test_con_vk_pedido_con_codigo_del_llm(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "PEDIDO\nCODE: CART-10")
        result = asyncio.run(
            classify_intent_detailed("¿dónde está mi pedido CART-10?", vk="test-vk")
        )
        assert "PEDIDO" in result.intents
        assert result.params["code"] == "CART-10"

    def test_con_vk_pedido_con_codigo_en_la_query(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "PEDIDO")
        result = asyncio.run(
            classify_intent_detailed("estado de mi pedido ABC-1234", vk="test-vk")
        )
        assert "PEDIDO" in result.intents
        assert result.params["code"] == "ABC-1234"

    def test_con_vk_pedido_sin_codigo_no_setea_code(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "PEDIDO")
        result = asyncio.run(
            classify_intent_detailed("¿dónde está mi pedido?", vk="test-vk")
        )
        assert "PEDIDO" in result.intents
        assert "code" not in result.params

    def test_sin_vk_pedido_con_codigo_lo_extrae(self):
        result = asyncio.run(
            classify_intent_detailed("mi pedido CART-10", vk=None)
        )
        assert "PEDIDO" in result.intents
        assert result.params["code"] == "CART-10"

    def test_sin_vk_carrito(self):
        result = asyncio.run(classify_intent_detailed("ver mi carrito", vk=None))
        assert result.intents == ["CARRITO"]

    def test_error_llm_extrae_codigo_en_fallback(self, monkeypatch, mock_llm):
        class _Boom:
            def with_options(self, timeout):
                raise RuntimeError("boom")

        monkeypatch.setattr("app.intent_classifier.get_llm", lambda vk: _Boom())
        result = asyncio.run(
            classify_intent_detailed("mi pedido CART-10", vk="test-vk")
        )
        assert "PEDIDO" in result.intents
        assert result.params["code"] == "CART-10"


# ---------------------------------------------------------------------------
# backward-compat classify_intent
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_sin_vk_devuelve_lista(self):
        intents = asyncio.run(classify_intent("quiero comprar audífonos", vk=None))
        assert isinstance(intents, list)
        assert "CATALOGO" in intents

    def test_con_vk_devuelve_lista(self, monkeypatch, mock_llm):
        _set_llm(monkeypatch, "BUSCAR\nQUERY: kz")
        intents = asyncio.run(classify_intent("busca kz", vk="test-vk"))
        assert isinstance(intents, list)
        assert "BUSCAR" in intents
