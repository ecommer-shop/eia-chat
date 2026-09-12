import pytest
from app.store_resolver import resolve_store, StoreConfig, init_stores, reload_stores, get_inbox_map
from app.store_loader import load_stores, list_stores_summary
from app.intent_classifier import _keyword_fallback, _parse_intent_output
from app.schemas import ChatRequest, ChatResponse


@pytest.fixture(autouse=True)
def _setup_stores():
    init_stores()


class TestStoreResolver:
    def test_global_store(self):
        store = resolve_store(2)
        assert store.store_name == "ecommer"
        assert store.is_global is True
        assert store.channel_tokens == []
        assert store.audience == "CLIENTE"
        assert store.channel_name == "whatsapp"

    def test_tenant_store(self):
        store = resolve_store(10)
        assert store.store_name == "sol-y-luna"
        assert store.is_global is False
        assert "sol-y-luna-token" in store.channel_tokens
        assert store.audience == "CLIENTE"
        assert store.channel_name == "whatsapp"

    def test_unknown_store_fallback(self):
        store = resolve_store(99)
        assert store.store_name == "tienda-99"
        assert store.is_global is False
        assert store.audience == "CLIENTE"
        assert store.channel_tokens == ["__unmapped_99__"]

    def test_other_stores_loaded(self):
        assert resolve_store(2).store_name == "ecommer"
        assert resolve_store(10).store_name == "sol-y-luna"

    def test_ecommer_has_few_shot_examples(self):
        store = resolve_store(2)
        assert len(store.few_shot) >= 4
        roles = {e["role"] for e in store.few_shot}
        assert {"user", "assistant"} == roles

    def test_store_without_few_shot_defaults_empty(self):
        store = resolve_store(99)
        assert store.few_shot == []


class TestStoreLoader:
    def test_load_stores(self):
        stores = load_stores()
        assert len(stores) > 0
        assert 2 in stores
        assert 10 in stores

    def test_list_stores_summary(self):
        summary = list_stores_summary()
        names = [s["store_name"] for s in summary]
        assert "ecommer" in names
        assert "sol-y-luna" in names

    def test_reload_stores(self):
        result = reload_stores()
        assert "loaded" in result
        assert result["loaded"] > 0


class TestStoreConfigInboxMap:
    def test_ecommer_has_five_channels(self):
        store = resolve_store(2)
        assert store.channel_name == "whatsapp"
        store = resolve_store(5)
        assert store.channel_name == "instagram"
        store = resolve_store(4)
        assert store.channel_name == "admin"

    def test_sol_y_luna_has_five_channels(self):
        store = resolve_store(10)
        assert store.channel_name == "whatsapp"
        store = resolve_store(11)
        assert store.channel_name == "instagram"
        store = resolve_store(14)
        assert store.channel_name == "admin"

    def test_tenant_stores_different_tokens(self):
        stores = load_stores()
        sol = stores[10]
        ziru = [s for s in stores.values() if s.store_name == "ziru-acoustics"]
        if ziru:
            assert sol.channel_tokens != ziru[0].channel_tokens


class TestIntentClassifier:
    def test_catalogo_keywords(self):
        result = _keyword_fallback("quiero comprar adidas superstar")
        assert "CATALOGO" in result

    def test_catalogo_keywords_tiene(self):
        result = _keyword_fallback("¿tiene adidas superstar?")
        assert "CATALOGO" in result

    def test_politicas_keywords(self):
        result = _keyword_fallback("¿cuál es la política de devoluciones?")
        assert "POLITICAS" in result

    def test_info_general_keywords(self):
        result = _keyword_fallback("¿qué es Ecommer?")
        assert "INFO_GENERAL" in result

    def test_conversacional_keywords(self):
        result = _keyword_fallback("hola, buenos días")
        assert "CONVERSACIONAL" in result

    def test_mixed_intents(self):
        result = _keyword_fallback("quiero comprar adidas y cuál es su política de devoluciones?")
        assert "CATALOGO" in result
        assert "POLITICAS" in result

    def test_accents_normalized(self):
        result = _keyword_fallback("¿cuál es la política de envíos a Bogotá?")
        assert "POLITICAS" in result

    def test_catalogo_necesito(self):
        result = _keyword_fallback("necesito unos audífonos para la escuela")
        assert "CATALOGO" in result

    def test_catalogo_venden_talla(self):
        result = _keyword_fallback("¿venden camisas talla M?")
        assert "CATALOGO" in result

    def test_politicas_envios(self):
        result = _keyword_fallback("¿tienen envíos a Bogotá?")
        assert "POLITICAS" in result

    def test_greeting_never_overrides_real_intent(self):
        result = _keyword_fallback("hola, quiero comprar un producto")
        assert "CONVERSACIONAL" not in result
        assert "CATALOGO" in result

    def test_parse_intent_output_valid(self):
        result = _parse_intent_output("CATALOGO, POLITICAS")
        assert result == ["CATALOGO", "POLITICAS"]

    def test_parse_intent_output_invalid(self):
        result = _parse_intent_output("OTRA_COSA")
        assert result == []

    def test_parse_intent_output_single(self):
        result = _parse_intent_output("CONVERSACIONAL")
        assert result == ["CONVERSACIONAL"]


class TestSchemas:
    def test_chat_request_valid(self):
        req = ChatRequest(query="hola", conversation_id="c1", inbox_id=1)
        assert req.query == "hola"
        assert req.user_id is None

    def test_chat_request_with_user(self):
        req = ChatRequest(query="test", conversation_id="c1", inbox_id=2, user_id=123)
        assert req.user_id == 123

    def test_chat_response(self):
        resp = ChatResponse(
            answer="Hola!",
            intent_detected="CONVERSACIONAL",
            sources_used=0,
            conversation_id="c1",
        )
        assert resp.sources_used == 0
