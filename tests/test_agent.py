"""Tests de Fase 3 — agente conversacional (core + herramientas).

Usan mocks: no hay llamadas a Qdrant, Azure OpenAI, Groq ni Redis.
Se ejecutan de forma síncrona con `asyncio.run`.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.agent import tools
from app.agent.core import run_agent, TRIVIAL_REPLY
from app.grounding import FALLBACK_UNGROUNDED
from app.llm_generator import build_prompt
from app.store_resolver import resolve_store, init_stores


@pytest.fixture(autouse=True)
def _setup_stores():
    init_stores()


@pytest.fixture(autouse=True)
def _fake_memory(monkeypatch):
    """Aísla la memoria: nada de Redis, historia en proceso por conversación.

    Cada test puede sobrescribir `tools.get_history` / `core.save_message`
    según lo que quiera verificar.
    """
    history: dict[str, list[dict]] = {}

    def _get_history(cid):
        return [dict(m) for m in history.get(cid, [])]

    monkeypatch.setattr(tools, "get_history", _get_history)
    monkeypatch.setattr("app.agent.core.save_message",
                        lambda cid, role, content: history.setdefault(cid, []).append(
                            {"role": role, "content": content}))
    return history


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _product(score=0.9, name="KZ Castor Pro"):
    return {
        "score": score,
        "payload": {
            "name": name,
            "url": "https://ecommer.shop/es/product/kz-castor-pro-bass-edition",
        },
    }


def _doc(score=0.8):
    return {
        "score": score,
        "payload": {"text": "Los compradores tendran derecho a la reversión del pago."},
    }


def _stub_classify(result):
    async def _classify(query, *args, **kwargs):
        return SimpleNamespace(intents=list(result), params={})
    return _classify


def _recording_search(calls):
    async def _search(query, store, intents, limit=10):
        calls.append(list(intents))
        return [_product()] if "CATALOGO" in intents else [_doc()]
    return _search


def _recording_conditional_search(calls, good_terms=("productos",)):
    async def _search(query, store, intents, limit=10):
        calls.append(query)
        if any(t in query.lower() for t in good_terms):
            return [_product(score=0.95, name="Panela orgánica")]
        return []
    return _search


def _has_guard(messages) -> bool:
    return any(
        m.get("role") == "system"
        and "no afirmes que no existen productos" in m.get("content", "").lower()
        for m in messages
    )


def _fail_never(*args, **kwargs):
    raise AssertionError("No debería invocarse")


class _FakeGroq:
    def __init__(self, content="", finish_reason="stop", error=None):
        self._content = content
        self._finish_reason = finish_reason
        self._error = error
        self.records = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, parent):
            self._parent = parent

        async def create(self, **kwargs):
            self._parent.records.append(kwargs)
            if self._parent._error is not None:
                raise self._parent._error
            choice = SimpleNamespace(finish_reason=self._parent._finish_reason)
            choice.message = SimpleNamespace(content=self._parent._content)
            return SimpleNamespace(choices=[choice])


def _run(query, conversation_id="conv-1", account_id=1, inbox_id=2, chatwoot_conversation_id=None):
    return asyncio.run(
        run_agent(
            query,
            conversation_id,
            account_id,
            inbox_id,
            chatwoot_conversation_id=chatwoot_conversation_id,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_expected_tools_registered(self):
        expected = {
            "search_catalogo", "search_docs", "get_memory", "answer", "escalate",
            "get_cart", "get_order",
        }
        assert expected.issubset(tools.TOOL_REGISTRY.keys())

    def test_identity_tools_are_the_expected_callables(self):
        assert tools.TOOL_REGISTRY["get_cart"] is tools.get_cart
        assert tools.TOOL_REGISTRY["get_order"] is tools.get_order

    def test_escalation_message_keeps_fail_closed_token(self):
        assert "configurada" in tools.ESCALATION_MESSAGE


class TestStoreMappingFlag:
    def test_mapped_store_is_mapped(self):
        import asyncio
        assert asyncio.run(resolve_store(1, 2)).is_mapped is True
        assert asyncio.run(resolve_store(5, 13)).is_mapped is True

    def test_unknown_store_not_mapped(self):
        import asyncio
        assert asyncio.run(resolve_store(99)).is_mapped is False


class TestEscalate:
    def test_unknown_store_escalates_without_llm(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        def _boom_search(*args, **kwargs):
            raise AssertionError("No debe buscar contexto en tienda no mapeada")
        monkeypatch.setattr(tools, "search_context", _boom_search)

        saved = []
        monkeypatch.setattr("app.agent.core.save_message",
                            lambda cid, role, content: saved.append((role, content)))

        result = _run("¿tienen audífonos?", conversation_id="c99", account_id=99)

        assert result.escalado is True
        assert result.escalado
        assert result.sources_used == 0
        assert result.tools_used == ["escalate"]
        assert "configurada" in result.answer
        assert saved == [("user", "¿tienen audífonos?"), ("assistant", result.answer)]

    def test_escalate_keeps_conversation_id(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["INFO_GENERAL"]))
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: None)
        result = _run("¿qué es Ecommer?", conversation_id="c99-bis", account_id=99)
        assert result.conversation_id == "c99-bis"
        assert result.escalado is True


class TestDispatch:
    def test_catalogo_uses_search_catalogo(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta del agente."))

        result = _run("¿tienen audífonos KZ Castor Pro?", inbox_id=2)

        assert result.escalado is False
        assert result.sources_used == 1
        assert result.answer == "Respuesta del agente."
        assert "search_catalogo" in result.tools_used
        assert "answer" in result.tools_used
        assert calls == [["CATALOGO"]]

    def test_docs_uses_search_docs(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["POLITICAS"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Política."))

        result = _run("¿cuál es la política de devoluciones?", inbox_id=2)

        assert result.sources_used == 1
        assert "search_docs" in result.tools_used
        assert calls == [["POLITICAS"]]

    def test_mixed_intents_use_both_search_tools(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _stub_classify(["CATALOGO", "POLITICAS"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Mixta."))

        result = _run("¿venden café y cuál es la política?", inbox_id=2)

        assert result.sources_used == 2
        assert "search_catalogo" in result.tools_used
        assert "search_docs" in result.tools_used
        assert calls == [["CATALOGO"], ["POLITICAS"]]

    def test_conversacional_skips_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CONVERSACIONAL"]))

        def _boom_search(*args, **kwargs):
            raise AssertionError("CONVERSACIONAL no debe llamar al retriever")
        monkeypatch.setattr(tools, "search_context", _boom_search)
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("¡Hola!"))

        result = _run("hola, buenos días", inbox_id=2)

        assert result.escalado is False
        assert result.sources_used == 0
        assert not any(n.startswith("search_") for n in result.tools_used)
        assert result.answer == "¡Hola!"

    def test_buscar_dispatcha_search_catalogo(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["BUSCAR"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta."))

        result = _run("busca audífonos kz", inbox_id=2)

        assert result.escalado is False
        assert "search_catalogo" in result.tools_used
        assert calls == [["CATALOGO"]]  # degradación del dispatch normaliza BUSCAR→CATALOGO

    def test_buscar_usa_query_extraida_por_el_clasificador(self, monkeypatch):
        async def _classify(query, *args, **kwargs):
            return SimpleNamespace(intents=["BUSCAR"], params={"query": "audifonos kz"})
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _classify)

        queries = []

        async def _search(query, store, intents, limit=10):
            queries.append(query)
            return [_product()]

        monkeypatch.setattr(tools, "search_context", _search)
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta."))

        result = _run("busca audifonos kz asdf foo bar", inbox_id=2)

        assert "search_catalogo" in result.tools_used
        assert queries == ["audifonos kz"]
        assert "busca audifonos kz asdf foo bar" not in queries


class TestLLMFallbacks:
    def test_empty_content_fallback(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq(content="", finish_reason="stop"))

        result = _run("¿cuánto cuesta el KZ?", inbox_id=2)

        assert "reformular" in result.answer

    def test_llm_error_fallback(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm",
                            lambda *a, **k: _FakeGroq(error=RuntimeError("boom")))

        result = _run("¿tienen café orgánico?", inbox_id=2)

        assert "problemas técnicos" in result.answer


class TestNoContextGuard:
    def test_no_context_injects_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        fake = _FakeGroq("Respuesta sin contexto.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        _run("¿tienen audífonos?", inbox_id=2)

        assert _has_guard(fake.records[0]["messages"])

    def test_with_context_omits_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        fake = _FakeGroq("Respuesta con contexto.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        _run("¿tienen KZ Castor Pro?", inbox_id=2)

        assert not _has_guard(fake.records[0]["messages"])

    def test_no_context_with_prior_products_omits_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        monkeypatch.setattr(
            tools, "get_history",
            lambda cid: [{"role": "assistant",
                          "content": "👉 Panela orgánica 🔗 https://ecommer.shop/es/product/panela-organica"}],
        )
        fake = _FakeGroq("Puedo recomendarte la panela de antes.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        _run("¿qué marcas hay?", inbox_id=2)

        assert not _has_guard(fake.records[0]["messages"])


class TestFollowUpRescue:
    def test_no_results_rescues_with_previous_query(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("productos",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "user", "content": "¿qué productos tienes?"},
        ])
        fake = _FakeGroq("Te recomiendo la panela orgánica.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("Busco poder comprar algún cage", conversation_id="rescue-1", inbox_id=2)

        assert calls == ["Busco poder comprar algún cage", "¿qué productos tienes?"]
        assert result.sources_used == 1
        assert result.tools_used.count("search_catalogo") == 2
        assert not _has_guard(fake.records[0]["messages"])

    def test_no_previous_query_no_rescue(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("productos",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [])
        fake = _FakeGroq("No encontramos coincidencias exactas.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("Busco poder comprar algún cage", conversation_id="rescue-2", inbox_id=2)

        assert calls == ["Busco poder comprar algún cage"]
        assert result.sources_used == 0
        assert result.tools_used.count("search_catalogo") == 1
        assert _has_guard(fake.records[0]["messages"])

    def test_same_previous_query_no_rescue_loop(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("jaula",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "user", "content": "¿tienen jaulas?"},
        ])
        monkeypatch.setattr(tools, "get_llm",
                            lambda *a, **k: _FakeGroq("No encontramos coincidencias exactas."))

        _run("¿tienen jaulas?", conversation_id="rescue-3", inbox_id=2)

        assert calls == ["¿tienen jaulas?"]


class TestFollowUpUsesPriorProducts:
    HISTORY = [{"role": "assistant",
                "content": "👉 Panela orgánica 🔗 https://ecommer.shop/es/product/panela-organica"}]

    def test_conversacional_followup_skips_guard_and_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "get_history", lambda cid: self.HISTORY)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("Te recomiendo la panela orgánica.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿cuál me recomiendas?", conversation_id="followup-1", inbox_id=2)

        assert result.answer == "Te recomiendo la panela orgánica."
        assert not _has_guard(fake.records[0]["messages"])
        assert not any(n.startswith("search_") for n in result.tools_used)


class TestFewShotInjection:
    EXAMPLES = [
        {"role": "user", "content": "¡Hola!"},
        {"role": "assistant", "content": "¡Hola! ¿Qué estás buscando hoy?"},
    ]

    def test_build_prompt_injects_few_shot_before_history(self):
        messages = build_prompt(
            query="¿tienen panela?",
            intent="CATALOGO",
            context_items=[_product()],
            history=[{"role": "user", "content": "turno previo"}],
            store_prompt="Eres el asistente.",
            few_shot=self.EXAMPLES,
        )
        assert messages[:3] == [
            {"role": "system", "content": messages[0]["content"]},
            self.EXAMPLES[0],
            self.EXAMPLES[1],
        ]
        assert {"role": "user", "content": "turno previo"} in messages
        assert messages[-1] == {"role": "user", "content": "¿tienen panela?"}

    def test_build_prompt_without_few_shot(self):
        messages = build_prompt(
            query="hola",
            intent="CONVERSACIONAL",
            context_items=[],
            history=[],
            store_prompt="Eres el asistente.",
        )
        assert all(m.get("role") == "system" or m == {"role": "user", "content": "hola"}
                   for m in messages)
        assert messages[-1] == {"role": "user", "content": "hola"}

    def test_agent_wires_store_few_shot(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        fake = _FakeGroq("Respuesta.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        _run("¿tienen panela?", inbox_id=2)

        msgs = fake.records[0]["messages"]
        assert any(m.get("role") == "assistant" and "Simetria" in m.get("content", "")
                   and "buscando" in m.get("content", "") for m in msgs)


class TestTrivialMessages:
    def test_dot_reply_short_without_llm_or_memory(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "get_llm", _fail_never)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        saved = []
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: saved.append(a))

        result = _run(".", conversation_id="triv-1", inbox_id=2)

        assert result.answer == TRIVIAL_REPLY
        assert result.sources_used == 0
        assert result.tools_used == ["get_memory"]
        assert saved == []

    def test_emoji_reply_short(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "get_llm", _fail_never)
        saved = []
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: saved.append(a))

        result = _run("😀", conversation_id="triv-2", inbox_id=2)

        assert result.answer == TRIVIAL_REPLY
        assert saved == []


class TestMemory:
    def test_history_flows_into_prompt(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_history",
                            lambda cid: [{"role": "user", "content": "turno anterior"}])
        fake = _FakeGroq("Respuesta.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        _run("¿y trae estuche?", conversation_id="mem-1", inbox_id=2)

        messages = fake.records[0]["messages"]
        assert {"role": "user", "content": "turno anterior"} in messages
        assert messages[-1] == {"role": "user", "content": "¿y trae estuche?"}

    def test_messages_saved_user_and_assistant(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta."))
        saved = []
        monkeypatch.setattr("app.agent.core.save_message",
                            lambda cid, role, content: saved.append((role, content)))

        _run("¿tienen amigurumis?", conversation_id="mem-2")

        assert saved == [("user", "¿tienen amigurumis?"), ("assistant", "Respuesta.")]


class TestGroundingGuardrail:
    def test_hallucinated_product_is_blocked(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        fake = _FakeGroq("👉 iPhone 15 🔗 https://ecommer.shop/product/iphone-15")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿tienen iPhone 15?", inbox_id=2)

        assert result.answer == FALLBACK_UNGROUNDED

    def test_grounded_answer_passes_through(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        url = "https://ecommer.shop/es/product/kz-castor-pro-bass-edition"
        fake = _FakeGroq(f"Te recomiendo el KZ Castor Pro 🔗 {url}")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿tienen KZ Castor Pro?", inbox_id=2)

        assert url in result.answer
        assert result.answer.count(url) == 1
        assert f"🔗 {url}" not in result.answer

    def test_followup_product_from_history_passes(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "assistant",
             "content": "👉 Panela orgánica 🔗 https://ecommer.shop/product/panela-organica"},
        ])
        url = "https://ecommer.shop/product/panela-organica"
        fake = _FakeGroq(f"Puedes ir por la de antes 👉 Panela orgánica 🔗 {url}")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿me recomiendas la panela?", inbox_id=2)

        assert "Panela orgánica" in result.answer

    def test_missing_product_link_appended(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        fake = _FakeGroq("Te recomiendo el KZ Castor Pro, ideal para escuchar música.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿me recomiendas el KZ?", inbox_id=2)

        url = "https://ecommer.shop/es/product/kz-castor-pro-bass-edition"
        assert url in result.answer


class TestCategoriesDispatch:
    def test_categories_query_uses_list_categorias_not_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)

        async def _fake_cats(store):
            return ["Audio", "Libros"]
        monkeypatch.setattr(tools, "list_categorias", _fake_cats)
        fake = _FakeGroq("Tenemos las categorías Audio y Libros.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿qué categorías hay en tu tienda?", inbox_id=2)

        assert result.answer == "Tenemos las categorías Audio y Libros."
        assert "list_categorias" in result.tools_used
        assert "search_catalogo" not in result.tools_used
        assert result.sources_used == 1

    def test_non_categories_query_uses_search_catalogo(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "list_categorias", _fail_never)
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Ok."))

        _run("¿tienen amigurumis?", inbox_id=2)

        assert calls == [["CATALOGO"]]

    def test_categories_query_with_tenants(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)

        async def _fake_cats(store):
            return ["Aromaterapia"]
        monkeypatch.setattr(tools, "list_categorias", _fake_cats)
        fake = _FakeGroq("En Sol y Luna tenemos Aromaterapia.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿qué categorías manejan?", conversation_id="syL-1", account_id=5, inbox_id=13)

        assert result.answer == "En Sol y Luna tenemos Aromaterapia."
        assert "list_categorias" in result.tools_used


class TestInputs:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            _run("   ", inbox_id=2)

    def test_missing_conversation_id_generates_one(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta."))
        result = asyncio.run(run_agent("hola", "", 1, 2))
        assert result.conversation_id.startswith("agent-")


# ---------------------------------------------------------------------------
# Fase 1b — dispatch de identidad (get_cart / get_order) y degradación
# ---------------------------------------------------------------------------

_CART_ORDER = {
    "code": "CART-10",
    "state": "AddingItems",
    "currencyCode": "COP",
    "totalQuantity": 2,
    "totalWithTaxDecimal": "185.00",
    "lines": [
        {
            "quantity": 2,
            "linePriceWithTaxDecimal": "185.00",
            "productVariant": {"name": "KZ Castor Pro"},
        }
    ],
}


class TestIdentityDispatch:
    def test_carrito_dispatcha_get_cart(self, monkeypatch):
        calls = []

        async def _fake_remote(account_id, inbox_id, conversation_id):
            calls.append((account_id, inbox_id, conversation_id))
            return dict(_CART_ORDER)

        monkeypatch.setattr(tools, "get_cart_remote", _fake_remote)
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CARRITO"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("Tu carrito tiene 2 productos por $185.00.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿qué tengo en mi carrito?", chatwoot_conversation_id=42)

        assert calls == [(1, 2, 42)]
        assert "get_cart" in result.tools_used
        assert result.sources_used == 1
        assert not _has_guard(fake.records[0]["messages"])
        context_text = fake.records[0]["messages"][0]["content"]
        assert "KZ Castor Pro" in context_text

    def test_pedido_con_codigo_dispatcha_get_order(self, monkeypatch):
        calls = []

        async def _fake_remote(account_id, inbox_id, conversation_id, code):
            calls.append((account_id, inbox_id, conversation_id, code))
            return {
                "code": code,
                "state": "Shipped",
                "currencyCode": "COP",
                "totalWithTaxDecimal": "90.00",
                "lines": [{"quantity": 1, "productVariant": {"name": "KZ Castor"}}],
            }

        async def _classify(query, *args, **kwargs):
            return SimpleNamespace(intents=["PEDIDO"], params={"code": "CART-10"})

        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _classify)
        monkeypatch.setattr(tools, "get_order_remote", _fake_remote)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("Tu pedido está en camino.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿dónde está mi pedido CART-10?", chatwoot_conversation_id=42)

        assert calls == [(1, 2, 42, "CART-10")]
        assert "get_order" in result.tools_used
        assert "KZ Castor" in fake.records[0]["messages"][0]["content"]

    def test_pedido_sin_codigo_pide_el_codigo(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise AssertionError("Sin código no debe llamar al backend")

        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["PEDIDO"]))
        monkeypatch.setattr(tools, "get_order_remote", _boom)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("¿Me pasás el número de pedido?")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿dónde está mi pedido?", chatwoot_conversation_id=42)

        assert "get_order" in result.tools_used
        assert "número de pedido" in fake.records[0]["messages"][0]["content"]

    def test_sin_chatwoot_conversation_id_degrada_sin_llamar_backend(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise AssertionError("Sin chatwoot_conversation_id no hay llamada de identidad")

        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CARRITO"]))
        monkeypatch.setattr(tools, "get_cart_remote", _boom)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("No puedo ver tu carrito porque no estás identificado.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿qué tengo en mi carrito?")  # sin chatwoot_conversation_id

        assert "get_cart" in result.tools_used
        assert "no estás identificado" in fake.records[0]["messages"][0]["content"]

    def test_carrito_none_degrada_a_mensaje_cordial(self, monkeypatch):
        async def _fake_remote(*args, **kwargs):
            return None

        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CARRITO"]))
        monkeypatch.setattr(tools, "get_cart_remote", _fake_remote)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("No veo un carrito activo.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("mi carrito", chatwoot_conversation_id=42)

        assert result.escalado is False
        assert "get_cart" in result.tools_used
        assert "carrito activo" in fake.records[0]["messages"][0]["content"]

    def test_carrito_y_catalogo_conviven(self, monkeypatch):
        async def _fake_remote(*args, **kwargs):
            return dict(_CART_ORDER)

        calls = []

        async def _search(query, store, intents, limit=10):
            calls.append(list(intents))
            return [_product()]

        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _stub_classify(["CARRITO", "CATALOGO"]))
        monkeypatch.setattr(tools, "get_cart_remote", _fake_remote)
        monkeypatch.setattr(tools, "search_context", _search)
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Ok."))

        result = _run("mirá mi carrito y buscame audífonos", chatwoot_conversation_id=42)

        assert "get_cart" in result.tools_used
        assert "search_catalogo" in result.tools_used
        assert calls == [["CATALOGO"]]
        assert result.sources_used == 2

    def test_error_del_cliente_no_rompe_el_turno(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("red caída")

        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CARRITO"]))
        monkeypatch.setattr(tools, "get_cart_remote", _boom)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Seguimos."))

        result = _run("mi carrito", chatwoot_conversation_id=42)

        assert result.answer == "Seguimos."
        assert result.escalado is False

    def test_back_compat_sin_el_kwarg(self, monkeypatch):
        """`run_agent` sin `chatwoot_conversation_id` sigue funcionando igual."""
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Respuesta."))

        result = asyncio.run(run_agent("¿tienen audífonos?", "bc-1", 1, 2))

        assert result.answer == "Respuesta."
        assert not any(t in result.tools_used for t in ("get_cart", "get_order"))

# ---------------------------------------------------------------------------
# COMPRAR — place_order en dos turnos (preview + confirmación)
# ---------------------------------------------------------------------------

from app.identity_client import PlaceOrderResult  # noqa: E402

_PREVIEW_ORDER = {
    "code": "CART-10",
    "state": "AddingItems",
    "currencyCode": "COP",
    "totalQuantity": 2,
    "totalWithTaxDecimal": "185.00",
    "lines": [
        {
            "quantity": 2,
            "linePriceWithTaxDecimal": "185.00",
            "productVariant": {"name": "KZ Castor Pro"},
        }
    ],
}

_PLACED_ORDER = {
    **_PREVIEW_ORDER,
    "code": "MP-1234",
    "state": "PaymentSettled",
}

_STORE = SimpleNamespace(store_name="ecommer", system_prompt="Eres Simetria.")


def _preview(**overrides):
    base = dict(
        status="confirmation_required",
        confirmed=False,
        order=dict(_PREVIEW_ORDER),
    )
    base.update(overrides)
    return PlaceOrderResult(**base)


def _placed(**overrides):
    base = dict(
        status="placed",
        confirmed=True,
        order=dict(_PLACED_ORDER),
    )
    base.update(overrides)
    return PlaceOrderResult(**base)


@pytest.fixture
def pending_store(monkeypatch):
    """Estado pendiente en proceso: reemplaza Redis para el flujo multi-turno."""
    state: dict[str, dict] = {}

    def _save(cid, data):
        state[cid] = dict(data)

    def _get(cid):
        data = state.get(cid)
        return dict(data) if data else None

    def _clear(cid):
        state.pop(cid, None)

    monkeypatch.setattr("app.agent.core.save_pending_order", _save)
    monkeypatch.setattr("app.agent.core.get_pending_order", _get)
    monkeypatch.setattr("app.agent.core.clear_pending_order", _clear)
    return state


def _recording_place_order(calls, result):
    """Doble de `identity_client.place_order` (la firma que llama `tools`)."""
    async def _fake(account_id, inbox_id, conversation_id, payment_method_code,
                    confirm, payment_metadata=None):
        calls.append({
            "account_id": account_id,
            "inbox_id": inbox_id,
            "chatwoot_conversation_id": conversation_id,
            "confirm": confirm,
            "payment_method_code": payment_method_code,
        })
        if callable(result):
            return result(confirm=confirm, payment_method_code=payment_method_code)
        return result
    return _fake


def _classify_with_params(intents, params=None):
    async def _classify(query, *args, **kwargs):
        return SimpleNamespace(intents=list(intents), params=dict(params or {}))
    return _classify


class TestPlaceOrderTool:
    def test_registrada_en_el_registry(self):
        assert "place_order" in tools.TOOL_REGISTRY
        assert tools.TOOL_REGISTRY["place_order"] is tools.place_order

    def test_sin_identidad_degrada_sin_llamar_backend(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise AssertionError("Sin chatwoot_conversation_id no hay llamada de identidad")

        monkeypatch.setattr(tools, "place_order_remote", _boom)

        result = asyncio.run(tools.place_order(None, 1, 2, None, confirm=False))

        assert result.status == "error"
        assert result.order is None
        assert "no estás identificado" in result.message

    def test_usa_el_metodo_de_pago_pedido(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote", _recording_place_order(calls, _preview()))

        asyncio.run(
            tools.place_order(_STORE, 1, 2, 42, confirm=False, payment_method_code="wompi")
        )

        assert calls[0]["payment_method_code"] == "wompi"
        assert calls[0]["confirm"] is False

    def test_sin_metodo_de_pago_usa_el_fallback_de_config(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote", _recording_place_order(calls, _preview()))
        monkeypatch.setattr(tools.settings, "ASSISTANT_PAYMENT_METHOD_CODE", "wompi", raising=False)

        asyncio.run(tools.place_order(_STORE, 1, 2, 42, confirm=False))

        assert calls[0]["payment_method_code"] == "wompi"

    def test_error_inesperado_devuelve_resultado_cordial(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("red caída")

        monkeypatch.setattr(tools, "place_order_remote", _boom)

        result = asyncio.run(tools.place_order(_STORE, 1, 2, 42, confirm=True))

        assert result.status == "error"
        assert "no se cobró nada" in result.message

    def test_preview_formatea_total_y_pregunta(self):
        text = tools.format_place_order_preview(_preview())

        assert "185.00" in text
        assert "KZ Castor Pro" in text
        assert "¿Confirmás la compra?" in text

    def test_preview_sin_carrito_pide_agregar(self):
        text = tools.format_place_order_preview(
            PlaceOrderResult(status="confirmation_required", confirmed=False, order=None)
        )
        assert "carrito activo" in text

    def test_placed_formatea_codigo_y_total(self):
        text = tools.format_place_order_result(_placed())

        assert "MP-1234" in text
        assert "185.00" in text
        assert "confirmada" in text

    def test_awaiting_payment_formatea_pendiente(self):
        text = tools.format_place_order_result(
            PlaceOrderResult(
                status="awaiting_payment", confirmed=True, order=dict(_PLACED_ORDER)
            )
        )
        assert "pendiente de pago" in text

    def test_error_formatea_accionable(self):
        text = tools.format_place_order_result(
            PlaceOrderResult(status="error", confirmed=False, message="Pago rechazado.")
        )
        assert "Pago rechazado." in text
        assert "no se cobró nada" in text


class TestMultiTurnoCompra:
    def test_turno_1_es_solo_preview(self, monkeypatch, pending_store):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote", _recording_place_order(calls, _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        result = _run("quiero finalizar mi compra", conversation_id="buy-1",
                      chatwoot_conversation_id=42)

        # Nunca se cobra en el turno 1.
        assert calls == [{
            "account_id": 1,
            "inbox_id": 2,
            "chatwoot_conversation_id": 42,
            "confirm": False,
            "payment_method_code": "wompi",
        }]
        assert "place_order" in result.tools_used
        assert "answer" not in result.tools_used
        assert "185.00" in result.answer
        assert "¿Confirmás la compra?" in result.answer
        assert result.escalado is False

    def test_turno_1_guarda_estado_pendiente(self, monkeypatch, pending_store):
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-2", chatwoot_conversation_id=42)

        pending = pending_store["buy-2"]
        assert pending["paymentMethodCode"] == "wompi"
        assert pending["account_id"] == 1
        assert pending["inbox_id"] == 2
        assert pending["chatwoot_conversation_id"] == 42
        assert pending["order"]["totalWithTaxDecimal"] == "185.00"
        assert pending["status"] == "confirmation_required"

    def test_turno_2_confirma_y_ejecuta_el_pago(self, monkeypatch, pending_store):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order(calls, _placed()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"], {"confirmation": True}))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        # Turno 1: preview y estado pendiente.
        _run("quiero finalizar mi compra", conversation_id="buy-3",
             chatwoot_conversation_id=42)
        calls.clear()

        # Turno 2: confirmación explícita.
        result = _run("sí, confirmo", conversation_id="buy-3",
                      chatwoot_conversation_id=42)

        assert calls == [{
            "account_id": 1,
            "inbox_id": 2,
            "chatwoot_conversation_id": 42,
            "confirm": True,
            "payment_method_code": "wompi",
        }]
        assert "place_order" in result.tools_used
        assert "answer" not in result.tools_used
        assert "MP-1234" in result.answer
        # El pendiente se limpia al terminar el flujo.
        assert "buy-3" not in pending_store

    def test_turno_2_toma_el_metodo_mencionado(self, monkeypatch, pending_store):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order(calls, _placed()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"], {"confirmation": True}))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-4", chatwoot_conversation_id=42)
        calls.clear()

        async def _classify_with_payment(query, *args, **kwargs):
            return SimpleNamespace(
                intents=["COMPRAR"], params={"confirmation": True, "payment_method": "wompi"}
            )
        monkeypatch.setattr("app.agent.core.classify_intent_detailed", _classify_with_payment)

        _run("sí, con Wompi", conversation_id="buy-4", chatwoot_conversation_id=42)

        assert calls[0]["confirm"] is True
        assert calls[0]["payment_method_code"] == "wompi"

    def test_turno_2_awaiting_payment(self, monkeypatch, pending_store):
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-5", chatwoot_conversation_id=42)

        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], PlaceOrderResult(
                                status="awaiting_payment",
                                confirmed=True,
                                order=dict(_PLACED_ORDER),
                            )))

        result = _run("dale", conversation_id="buy-5", chatwoot_conversation_id=42)

        assert "pendiente de pago" in result.answer
        assert "buy-5" not in pending_store

    def test_turno_2_sin_confirmacion_limpia_y_sigue_normal(self, monkeypatch, pending_store):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order(calls, _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-6", chatwoot_conversation_id=42)
        calls.clear()

        # El comprador cambia de tema: no es confirmación.
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["POLITICAS"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        fake = _FakeGroq("Los envíos tardan 3 días.")
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: fake)

        result = _run("¿cuánto tarda el envío?", conversation_id="buy-6",
                      chatwoot_conversation_id=42)

        assert calls == []  # no se ejecutó ningún pago
        assert result.answer == "Los envíos tardan 3 días."
        assert "answer" in result.tools_used
        assert "buy-6" not in pending_store  # el pendiente se descartó

    def test_turno_2_error_de_pago_limpia_y_avisa(self, monkeypatch, pending_store):
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-7", chatwoot_conversation_id=42)

        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], PlaceOrderResult(
                                status="error",
                                confirmed=False,
                                message="El pago fue rechazado por Wompi.",
                            )))

        result = _run("confirmo", conversation_id="buy-7", chatwoot_conversation_id=42)

        assert "El pago fue rechazado por Wompi." in result.answer
        assert "no se cobró nada" in result.answer
        assert "buy-7" not in pending_store

    def test_sin_carrito_no_guarda_pendiente(self, monkeypatch, pending_store):
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], _preview(order=None)))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        result = _run("finalizar compra", conversation_id="buy-8",
                      chatwoot_conversation_id=42)

        assert "carrito activo" in result.answer
        assert "buy-8" not in pending_store
        assert result.sources_used == 0

    def test_sin_identidad_degrada_cordial(self, monkeypatch, pending_store):
        async def _boom(*args, **kwargs):
            raise AssertionError("Sin chatwoot_conversation_id no hay llamada de identidad")
        monkeypatch.setattr(tools, "place_order_remote", _boom)
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        result = _run("finalizar compra", conversation_id="buy-9")  # sin chatwoot id

        assert "no estás identificado" in result.answer
        assert result.escalado is False
        assert "buy-9" not in pending_store

    def test_pendiente_de_otra_identidad_se_descarta(self, monkeypatch, pending_store):
        calls = []
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order(calls, _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["CATALOGO"], {"confirmation": True}))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Hola."))

        pending_store["buy-10"] = {
            "account_id": 999,
            "inbox_id": 2,
            "chatwoot_conversation_id": 999,
            "paymentMethodCode": "wompi",
            "order": dict(_PREVIEW_ORDER),
        }

        result = _run("sí, confirmo", conversation_id="buy-10",
                      chatwoot_conversation_id=42)

        # Nunca se ejecuta un cobro con las coordenadas del pendiente ajeno.
        assert [c for c in calls if c["confirm"]] == []
        assert "buy-10" not in pending_store
        # Sin pendiente válido, el mensaje sigue como turno normal (sin cobrar).
        assert result.answer == "Hola."

    def test_conversacion_sin_pendiente_no_toca_place_order(self, monkeypatch, pending_store):
        async def _boom(*args, **kwargs):
            raise AssertionError("Sin pendiente y sin COMPRAR no hay place_order")
        monkeypatch.setattr(tools, "place_order_remote", _boom)
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Ok."))

        result = _run("¿tienen audífonos?", conversation_id="buy-11",
                      chatwoot_conversation_id=42)

        assert "place_order" not in result.tools_used

    def test_turno_2_no_confirmacion_con_intencion_comprar_no_reabre_preview(
        self, monkeypatch, pending_store
    ):
        """Un "no" tras el preview no vuelve a mostrar otro preview."""
        calls = []
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order(calls, _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeGroq("Ok, lo dejo."))

        pending_store["buy-13"] = {
            "account_id": 1,
            "inbox_id": 2,
            "chatwoot_conversation_id": 42,
            "paymentMethodCode": "wompi",
            "order": dict(_PREVIEW_ORDER),
        }

        result = _run("no, mejor después", conversation_id="buy-13",
                      chatwoot_conversation_id=42)

        assert calls == []  # ni cobro ni preview nuevo
        assert result.answer == "Ok, lo dejo."
        assert "buy-13" not in pending_store

    def test_historial_no_contiene_el_estado_pendiente(self, monkeypatch, pending_store,
                                                       _fake_memory):
        monkeypatch.setattr(tools, "place_order_remote",
                            _recording_place_order([], _preview()))
        monkeypatch.setattr("app.agent.core.classify_intent_detailed",
                            _classify_with_params(["COMPRAR"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)
        monkeypatch.setattr(tools, "get_llm", _fail_never)

        _run("finalizar compra", conversation_id="buy-12", chatwoot_conversation_id=42)

        saved = _fake_memory["buy-12"]
        assert [m["role"] for m in saved] == ["user", "assistant"]
        assert all("pending" not in m["content"].lower() for m in saved)
