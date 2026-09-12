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
    async def _classify(query):
        return list(result)
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


def _run(query, conversation_id="conv-1", inbox_id=2):
    return asyncio.run(run_agent(query, conversation_id, inbox_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_expected_tools_registered(self):
        expected = {"search_catalogo", "search_docs", "get_memory", "answer", "escalate"}
        assert expected.issubset(tools.TOOL_REGISTRY.keys())

    def test_escalation_message_keeps_fail_closed_token(self):
        assert "configurada" in tools.ESCALATION_MESSAGE


class TestStoreMappingFlag:
    def test_mapped_store_is_mapped(self):
        assert resolve_store(2).is_mapped is True
        assert resolve_store(10).is_mapped is True

    def test_unknown_store_not_mapped(self):
        assert resolve_store(99).is_mapped is False


class TestEscalate:
    def test_unknown_store_escalates_without_llm(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        def _boom_search(*args, **kwargs):
            raise AssertionError("No debe buscar contexto en tienda no mapeada")
        monkeypatch.setattr(tools, "search_context", _boom_search)

        saved = []
        monkeypatch.setattr("app.agent.core.save_message",
                            lambda cid, role, content: saved.append((role, content)))

        result = _run("¿tienen audífonos?", conversation_id="c99", inbox_id=99)

        assert result.escalado is True
        assert result.escalado
        assert result.sources_used == 0
        assert result.tools_used == ["escalate"]
        assert "configurada" in result.answer
        assert saved == [("user", "¿tienen audífonos?"), ("assistant", result.answer)]

    def test_escalate_keeps_conversation_id(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["INFO_GENERAL"]))
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: None)
        result = _run("¿qué es Ecommer?", conversation_id="c99-bis", inbox_id=99)
        assert result.conversation_id == "c99-bis"
        assert result.escalado is True


class TestDispatch:
    def test_catalogo_uses_search_catalogo(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Respuesta del agente."))

        result = _run("¿tienen audífonos KZ Castor Pro?", inbox_id=2)

        assert result.escalado is False
        assert result.sources_used == 1
        assert result.answer == "Respuesta del agente."
        assert "search_catalogo" in result.tools_used
        assert "answer" in result.tools_used
        assert calls == [["CATALOGO"]]

    def test_docs_uses_search_docs(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["POLITICAS"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Política."))

        result = _run("¿cuál es la política de devoluciones?", inbox_id=2)

        assert result.sources_used == 1
        assert "search_docs" in result.tools_used
        assert calls == [["POLITICAS"]]

    def test_mixed_intents_use_both_search_tools(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent",
                            _stub_classify(["CATALOGO", "POLITICAS"]))
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Mixta."))

        result = _run("¿venden café y cuál es la política?", inbox_id=2)

        assert result.sources_used == 2
        assert "search_catalogo" in result.tools_used
        assert "search_docs" in result.tools_used
        assert calls == [["CATALOGO"], ["POLITICAS"]]

    def test_conversacional_skips_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CONVERSACIONAL"]))

        def _boom_search(*args, **kwargs):
            raise AssertionError("CONVERSACIONAL no debe llamar al retriever")
        monkeypatch.setattr(tools, "search_context", _boom_search)
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("¡Hola!"))

        result = _run("hola, buenos días", inbox_id=2)

        assert result.escalado is False
        assert result.sources_used == 0
        assert not any(n.startswith("search_") for n in result.tools_used)
        assert result.answer == "¡Hola!"


class TestLLMFallbacks:
    def test_empty_content_fallback(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq(content="", finish_reason="stop"))

        result = _run("¿cuánto cuesta el KZ?", inbox_id=2)

        assert "reformular" in result.answer

    def test_groq_error_fallback(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "_get_groq",
                            lambda: _FakeGroq(error=RuntimeError("boom")))

        result = _run("¿tienen café orgánico?", inbox_id=2)

        assert "problemas técnicos" in result.answer


class TestNoContextGuard:
    def test_no_context_injects_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        fake = _FakeGroq("Respuesta sin contexto.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        _run("¿tienen audífonos?", inbox_id=2)

        assert _has_guard(fake.records[0]["messages"])

    def test_with_context_omits_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        fake = _FakeGroq("Respuesta con contexto.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        _run("¿tienen KZ Castor Pro?", inbox_id=2)

        assert not _has_guard(fake.records[0]["messages"])

    def test_no_context_with_prior_products_omits_guard(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        monkeypatch.setattr(
            tools, "get_history",
            lambda cid: [{"role": "assistant",
                          "content": "👉 Panela orgánica 🔗 https://ecommer.shop/es/product/panela-organica"}],
        )
        fake = _FakeGroq("Puedo recomendarte la panela de antes.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        _run("¿qué marcas hay?", inbox_id=2)

        assert not _has_guard(fake.records[0]["messages"])


class TestFollowUpRescue:
    def test_no_results_rescues_with_previous_query(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("productos",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "user", "content": "¿qué productos tienes?"},
        ])
        fake = _FakeGroq("Te recomiendo la panela orgánica.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("Busco poder comprar algún cage", conversation_id="rescue-1", inbox_id=2)

        assert calls == ["Busco poder comprar algún cage", "¿qué productos tienes?"]
        assert result.sources_used == 1
        assert result.tools_used.count("search_catalogo") == 2
        assert not _has_guard(fake.records[0]["messages"])

    def test_no_previous_query_no_rescue(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("productos",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [])
        fake = _FakeGroq("No encontramos coincidencias exactas.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("Busco poder comprar algún cage", conversation_id="rescue-2", inbox_id=2)

        assert calls == ["Busco poder comprar algún cage"]
        assert result.sources_used == 0
        assert result.tools_used.count("search_catalogo") == 1
        assert _has_guard(fake.records[0]["messages"])

    def test_same_previous_query_no_rescue_loop(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        calls = []
        monkeypatch.setattr(tools, "search_context",
                            _recording_conditional_search(calls, good_terms=("jaula",)))
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "user", "content": "¿tienen jaulas?"},
        ])
        monkeypatch.setattr(tools, "_get_groq",
                            lambda: _FakeGroq("No encontramos coincidencias exactas."))

        _run("¿tienen jaulas?", conversation_id="rescue-3", inbox_id=2)

        assert calls == ["¿tienen jaulas?"]


class TestFollowUpUsesPriorProducts:
    HISTORY = [{"role": "assistant",
                "content": "👉 Panela orgánica 🔗 https://ecommer.shop/es/product/panela-organica"}]

    def test_conversacional_followup_skips_guard_and_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "get_history", lambda cid: self.HISTORY)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        fake = _FakeGroq("Te recomiendo la panela orgánica.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

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
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        fake = _FakeGroq("Respuesta.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        _run("¿tienen panela?", inbox_id=2)

        msgs = fake.records[0]["messages"]
        assert any(m.get("role") == "assistant" and "Simetria" in m.get("content", "")
                   and "buscando" in m.get("content", "") for m in msgs)


class TestTrivialMessages:
    def test_dot_reply_short_without_llm_or_memory(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "_get_groq", _fail_never)
        monkeypatch.setattr(tools, "search_context", _fail_never)
        saved = []
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: saved.append(a))

        result = _run(".", conversation_id="triv-1", inbox_id=2)

        assert result.answer == TRIVIAL_REPLY
        assert result.sources_used == 0
        assert result.tools_used == ["get_memory"]
        assert saved == []

    def test_emoji_reply_short(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CONVERSACIONAL"]))
        monkeypatch.setattr(tools, "_get_groq", _fail_never)
        saved = []
        monkeypatch.setattr("app.agent.core.save_message", lambda *a: saved.append(a))

        result = _run("😀", conversation_id="triv-2", inbox_id=2)

        assert result.answer == TRIVIAL_REPLY
        assert saved == []


class TestMemory:
    def test_history_flows_into_prompt(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "get_history",
                            lambda cid: [{"role": "user", "content": "turno anterior"}])
        fake = _FakeGroq("Respuesta.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        _run("¿y trae estuche?", conversation_id="mem-1", inbox_id=2)

        messages = fake.records[0]["messages"]
        assert {"role": "user", "content": "turno anterior"} in messages
        assert messages[-1] == {"role": "user", "content": "¿y trae estuche?"}

    def test_messages_saved_user_and_assistant(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Respuesta."))
        saved = []
        monkeypatch.setattr("app.agent.core.save_message",
                            lambda cid, role, content: saved.append((role, content)))

        _run("¿tienen amigurumis?", conversation_id="mem-2", inbox_id=10)

        assert saved == [("user", "¿tienen amigurumis?"), ("assistant", "Respuesta.")]


class TestGroundingGuardrail:
    def test_hallucinated_product_is_blocked(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _no_context(*a, **k):
            return []
        monkeypatch.setattr(tools, "search_context", _no_context)
        fake = _FakeGroq("👉 iPhone 15 🔗 https://ecommer.shop/product/iphone-15")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿tienen iPhone 15?", inbox_id=2)

        assert result.answer == FALLBACK_UNGROUNDED

    def test_grounded_answer_passes_through(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        url = "https://ecommer.shop/es/product/kz-castor-pro-bass-edition"
        fake = _FakeGroq(f"Te recomiendo el KZ Castor Pro 🔗 {url}")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿tienen KZ Castor Pro?", inbox_id=2)

        assert url in result.answer
        assert result.answer.count(url) == 1
        assert f"🔗 {url}" not in result.answer

    def test_followup_product_from_history_passes(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        monkeypatch.setattr(tools, "get_history", lambda cid: [
            {"role": "assistant",
             "content": "👉 Panela orgánica 🔗 https://ecommer.shop/product/panela-organica"},
        ])
        url = "https://ecommer.shop/product/panela-organica"
        fake = _FakeGroq(f"Puedes ir por la de antes 👉 Panela orgánica 🔗 {url}")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿me recomiendas la panela?", inbox_id=2)

        assert "Panela orgánica" in result.answer

    def test_missing_product_link_appended(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))

        async def _with_context(*a, **k):
            return [_product(score=0.9, name="KZ Castor Pro")]
        monkeypatch.setattr(tools, "search_context", _with_context)
        fake = _FakeGroq("Te recomiendo el KZ Castor Pro, ideal para escuchar música.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿me recomiendas el KZ?", inbox_id=2)

        url = "https://ecommer.shop/es/product/kz-castor-pro-bass-edition"
        assert url in result.answer


class TestCategoriesDispatch:
    def test_categories_query_uses_list_categorias_not_search(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)

        async def _fake_cats(store):
            return ["Audio", "Libros"]
        monkeypatch.setattr(tools, "list_categorias", _fake_cats)
        fake = _FakeGroq("Tenemos las categorías Audio y Libros.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿qué categorías hay en tu tienda?", inbox_id=2)

        assert result.answer == "Tenemos las categorías Audio y Libros."
        assert "list_categorias" in result.tools_used
        assert "search_catalogo" not in result.tools_used
        assert result.sources_used == 1

    def test_non_categories_query_uses_search_catalogo(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "list_categorias", _fail_never)
        calls = []
        monkeypatch.setattr(tools, "search_context", _recording_search(calls))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Ok."))

        _run("¿tienen amigurumis?", inbox_id=2)

        assert calls == [["CATALOGO"]]

    def test_categories_query_with_tenants(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _fail_never)

        async def _fake_cats(store):
            return ["Aromaterapia"]
        monkeypatch.setattr(tools, "list_categorias", _fake_cats)
        fake = _FakeGroq("En Sol y Luna tenemos Aromaterapia.")
        monkeypatch.setattr(tools, "_get_groq", lambda: fake)

        result = _run("¿qué categorías manejan?", conversation_id="syL-1", inbox_id=10)

        assert result.answer == "En Sol y Luna tenemos Aromaterapia."
        assert "list_categorias" in result.tools_used


class TestInputs:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            _run("   ", inbox_id=2)

    def test_missing_conversation_id_generates_one(self, monkeypatch):
        monkeypatch.setattr("app.agent.core.classify_intent", _stub_classify(["CATALOGO"]))
        monkeypatch.setattr(tools, "search_context", _recording_search([]))
        monkeypatch.setattr(tools, "_get_groq", lambda: _FakeGroq("Respuesta."))
        result = asyncio.run(run_agent("hola", "", 2))
        assert result.conversation_id.startswith("agent-")