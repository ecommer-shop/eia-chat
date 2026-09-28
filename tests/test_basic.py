import pytest
from app.store_resolver import resolve_store, StoreConfig, init_stores, reload_stores, get_account_map
from app.store_repository import get_store_by_account_id, get_all_stores, invalidate_cache
from app.intent_classifier import _keyword_fallback, _parse_intent_output
from app.schemas import ChatRequest, ChatResponse


@pytest.fixture(autouse=True)
def _setup_stores(mock_store_resolver):
    init_stores()
    invalidate_cache()


class TestStoreResolver:
    @pytest.mark.asyncio
    async def test_global_store(self, mock_store_resolver):
        store = await resolve_store(1, 2)
        assert store.store_name == "ecommer"
        assert store.account_id == 1
        assert store.is_global is True
        assert store.channel_tokens == []
        assert store.audience == "CLIENTE"
        assert store.channel_name == "default"

    @pytest.mark.asyncio
    async def test_tenant_store(self, mock_store_resolver):
        store = await resolve_store(5, 13)
        assert store.store_name == "ziru-acoustics"
        assert store.account_id == 5
        assert store.is_global is False
        assert "ziru-acoustics-token" in store.channel_tokens
        assert store.audience == "CLIENTE"
        assert store.channel_name == "default"

    @pytest.mark.asyncio
    async def test_unknown_store_fallback(self, mock_store_resolver):
        store = await resolve_store(99)
        assert store.store_name == "tienda-99"
        assert store.is_global is False
        assert store.audience == "CLIENTE"
        assert store.channel_tokens == ["__unmapped_99__"]

    @pytest.mark.asyncio
    async def test_other_stores_loaded(self, mock_store_resolver):
        assert (await resolve_store(1, 2)).store_name == "ecommer"
        assert (await resolve_store(5, 13)).store_name == "ziru-acoustics"

    @pytest.mark.asyncio
    async def test_ecommer_has_few_shot_examples(self, mock_store_resolver):
        store = await resolve_store(1, 2)
        assert len(store.few_shot) >= 4
        roles = {e["role"] for e in store.few_shot}
        assert {"user", "assistant"} == roles

    @pytest.mark.asyncio
    async def test_store_without_few_shot_defaults_empty(self, mock_store_resolver):
        store = await resolve_store(99)
        assert store.few_shot == []


class TestStoreRepository:
    @pytest.mark.asyncio
    async def test_get_store_by_account_id(self, mock_store_resolver):
        store = await get_store_by_account_id(1)
        assert store is not None
        assert store.store_name == "ecommer"

    @pytest.mark.asyncio
    async def test_get_all_stores(self, mock_store_resolver):
        stores = await get_all_stores()
        assert len(stores) > 0
        names = [s.store_name for s in stores]
        assert "ecommer" in names
        assert "ziru-acoustics" in names

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, mock_store_resolver):
        invalidate_cache()
        store = await get_store_by_account_id(1)
        assert store is not None


class TestStoreConfigInboxMap:
    @pytest.mark.asyncio
    async def test_ecommer_has_five_channels(self, mock_store_resolver):
        store = await resolve_store(1, 2)
        assert store.channel_name == "default"
        store = await resolve_store(1, 5)
        assert store.channel_name == "default"

    @pytest.mark.asyncio
    async def test_ziru_has_channel(self, mock_store_resolver):
        store = await resolve_store(5, 13)
        assert store.channel_name == "default"

    @pytest.mark.asyncio
    async def test_tenant_stores_different_tokens(self, mock_store_resolver):
        stores = await get_all_stores()
        ecommer = next(s for s in stores if s.store_name == "ecommer")
        ziru = next(s for s in stores if s.store_name == "ziru-acoustics")
        assert ecommer.channel_tokens != ziru.channel_tokens


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
        assert req.account_id is None

    def test_chat_request_with_user(self):
        req = ChatRequest(query="test", conversation_id="c1", account_id=1, inbox_id=2, user_id=123)
        assert req.user_id == 123
        assert req.account_id == 1

    def test_chat_response(self):
        resp = ChatResponse(
            answer="Hola!",
            intent_detected="CONVERSACIONAL",
            sources_used=0,
            conversation_id="c1",
        )
        assert resp.sources_used == 0