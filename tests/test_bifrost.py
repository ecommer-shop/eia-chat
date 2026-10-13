"""Tests de la integración con Bifrost (VK por tienda + modelos por plan).

Todo es offline: se mockean `get_llm`, `get_allowed_models` y `pick_model`.
Cubren: resolución de VK (ecommer por .env vs tienda por BD), selección de
modelo según plan, retry automático ante 403 model_blocked, el error 503 de
una tienda sin clave y que la VK nunca aparezca en `repr`.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from app import clients, store_repository
from app.agent import tools
from app.exceptions import StoreWithoutBifrostKeyError
from app.intent_classifier import classify_intent
from app.models import StoreConfig

# Capturado al importar (antes de que conftest parchee el mock autouse).
_real_get_store_by_account_id = store_repository.get_store_by_account_id


# ---------------------------------------------------------------------------
# Resolución de VK (repository)
# ---------------------------------------------------------------------------

class TestResolveVK:
    def test_ecommer_usa_vk_del_env(self, monkeypatch):
        monkeypatch.setattr(store_repository.settings, "BIFROST_VK_ECOMMER", "vk-env-ecommer")
        vk, plan = store_repository._resolve_vk(1, "vk-de-la-bd", "free")
        assert vk == "vk-env-ecommer"
        assert plan == "ecommer"

    def test_otra_tienda_usa_vk_de_la_bd(self):
        vk, plan = store_repository._resolve_vk(22, "vk-de-la-bd", "free")
        assert vk == "vk-de-la-bd"
        assert plan == "free"

    def test_tienda_sin_vk_en_bd_devuelve_none(self):
        vk, plan = store_repository._resolve_vk(18, None, None)
        assert vk is None
        assert plan == ""


# ---------------------------------------------------------------------------
# Selección de modelo por plan
# ---------------------------------------------------------------------------

class TestPickModel:
    def _pick(self, monkeypatch, allowed, preference):
        async def _fake_allowed(vk):
            return list(allowed)

        monkeypatch.setattr(clients, "get_allowed_models", _fake_allowed)
        return asyncio.run(clients.pick_model("vk-test", preference))

    def test_plan_con_phi4_usa_phi4_para_respuesta(self, monkeypatch):
        model = self._pick(
            monkeypatch,
            ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
            ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
        )
        assert model == "azure/Phi-4"

    def test_plan_free_solo_mini_usa_mini_para_respuesta(self, monkeypatch):
        model = self._pick(
            monkeypatch,
            ["azure/Phi-4-mini-instruct"],
            ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
        )
        assert model == "azure/Phi-4-mini-instruct"

    def test_router_prefiere_el_modelo_mas_pequeno(self, monkeypatch):
        model = self._pick(
            monkeypatch,
            ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
            ["azure/Phi-4-mini-instruct", "azure/Phi-4-mini-instruct"],
        )
        assert model == "azure/Phi-4-mini-instruct"

    def test_descubrimiento_falla_usa_primera_preferencia(self, monkeypatch):
        model = self._pick(monkeypatch, [], ["azure/Phi-4", "azure/Phi-4-mini-instruct"])
        assert model == "azure/Phi-4"

    def test_sin_preferencias_lanza_error(self, monkeypatch):
        with pytest.raises((IndexError, TypeError)):
            self._pick(monkeypatch, [], [])


# ---------------------------------------------------------------------------
# Cliente LLM
# ---------------------------------------------------------------------------

class TestGetLLM:
    def test_sin_vk_lanza_error(self):
        with pytest.raises(ValueError):
            clients.get_llm(None)

    def test_cliente_cacheado_por_vk(self):
        c1 = clients.get_llm("vk-uno")
        c2 = clients.get_llm("vk-uno")
        c3 = clients.get_llm("vk-dos")
        assert c1 is c2
        assert c1 is not c3
        clients._llm_clients.clear()

    def test_max_retries_acotado(self):
        # El default del SDK (2) apila intentos de 60s: un solo LLM call podía
        # tardar minutos y eia-bot cortaba a los 60s (usuario sin respuesta).
        clients._llm_clients.clear()
        client = clients.get_llm("vk-retries")
        assert client.max_retries == 1
        clients._llm_clients.clear()


# ---------------------------------------------------------------------------
# Retry automático ante 403 (model_blocked)
# ---------------------------------------------------------------------------

class _Blocked(Exception):
    status_code = 403


class _FakeCompletion:
    def __init__(self, content="Respuesta"):
        choice = type("C", (), {})()
        choice.finish_reason = "stop"
        choice.message = type("M", (), {"content": content})()
        self.choices = [choice]


class _FakeLLM:
    """Primer create falla con 403; los siguientes responden bien."""

    def __init__(self, fail_times=1):
        self.calls = []
        self._fail_times = fail_times

        class _Completions:
            def __init__(self, parent):
                self._parent = parent

            async def create(self, **kwargs):
                self._parent.calls.append(kwargs)
                if len(self._parent.calls) <= self._parent._fail_times:
                    raise _Blocked("Model 'Phi-4' is not allowed for this virtual key")
                return _FakeCompletion()

        self.chat = type("Chat", (), {"completions": _Completions(self)})()


class TestModelFallback403:
    def test_tras_403_refresca_y_reintenta_con_fallback(self, monkeypatch):
        fake = _FakeLLM(fail_times=1)
        invalidated = []

        async def _fake_pick(vk, preference):
            # descubrimiento refrescado: solo el mini
            return "azure/Phi-4-mini-instruct"

        monkeypatch.setattr(tools, "get_llm", lambda vk: fake)
        monkeypatch.setattr(tools, "pick_model", _fake_pick)
        monkeypatch.setattr(tools, "invalidate_allowed_models", invalidated.append)

        completion, model = asyncio.run(
            tools._create_with_model_fallback(
                "vk-test",
                [{"role": "user", "content": "hola"}],
                "azure/Phi-4",
                ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
            )
        )

        assert model == "azure/Phi-4-mini-instruct"
        assert invalidated == ["vk-test"]
        assert len(fake.calls) == 2
        assert fake.calls[1]["model"] == "azure/Phi-4-mini-instruct"
        assert completion.choices[0].message.content == "Respuesta"

    def test_si_ya_usa_el_mini_no_buclea(self, monkeypatch):
        class _AlwaysBlocked:
            def __init__(self):
                self.calls = 0

                class _Completions:
                    def __init__(self, parent):
                        self._parent = parent

                    async def create(self, **kwargs):
                        self._parent.calls += 1
                        raise _Blocked("nope")

                self.chat = type("Chat", (), {"completions": _Completions(self)})()

        fake = _AlwaysBlocked()
        monkeypatch.setattr(tools, "get_llm", lambda vk: fake)
        monkeypatch.setattr(tools, "invalidate_allowed_models", lambda vk: None)

        async def _same_model(vk, preference):
            return "azure/Phi-4-mini-instruct"

        monkeypatch.setattr(tools, "pick_model", _same_model)

        with pytest.raises(_Blocked):
            asyncio.run(
                tools._create_with_model_fallback(
                    "vk-test",
                    [{"role": "user", "content": "hola"}],
                    "azure/Phi-4-mini-instruct",
                    ["azure/Phi-4", "azure/Phi-4-mini-instruct"],
                )
            )
        # un solo intento: sin bucle de reintentos
        assert fake.calls == 1


# ---------------------------------------------------------------------------
# Clasificador sin VK (fail-closed: solo keywords)
# ---------------------------------------------------------------------------

class TestClassifierSinVK:
    def test_sin_vk_no_usa_llm(self):
        intents = asyncio.run(classify_intent("quiero comprar unos audífonos", vk=None))
        assert "CATALOGO" in intents


# ---------------------------------------------------------------------------
# Error 503 para tienda sin VK
# ---------------------------------------------------------------------------

class TestSinVK503:
    def test_run_agent_lanza_error_si_tienda_mapeada_sin_vk(self, monkeypatch):
        from app.agent.core import run_agent

        async def _sin_vk(account_id, inbox_id=None):
            return StoreConfig(
                store_name="tienda-sin-vk",
                account_id=account_id,
                is_mapped=True,
                bifrost_vk=None,
            )

        monkeypatch.setattr("app.agent.core.resolve_store", _sin_vk)

        with pytest.raises(StoreWithoutBifrostKeyError):
            asyncio.run(run_agent("hola", "conv-sin-vk", 18, 2))

    def test_endpoint_devuelve_503(self, monkeypatch):
        from app.main import app

        async def _sin_vk(account_id, inbox_id=None):
            return StoreConfig(
                store_name="tienda-sin-vk",
                account_id=account_id,
                is_mapped=True,
                bifrost_vk=None,
            )

        monkeypatch.setattr("app.agent.core.resolve_store", _sin_vk)

        with TestClient(app) as client:
            resp = client.post(
                "/agent/chat",
                json={"query": "hola", "account_id": 18, "inbox_id": 2},
            )

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "clave Bifrost" in detail
        assert "tienda-sin-vk" in detail


# ---------------------------------------------------------------------------
# Ecommer global (account_id 1): no existe en la BD -> override local
# ---------------------------------------------------------------------------

class _FakeConn:
    async def fetchrow(self, *args, **kwargs):
        return None


@asynccontextmanager
async def _fake_acquire_db():
    yield _FakeConn()


class TestEcommerGlobalSinFilaBD:
    """La cuenta 1 (ecommer, la plataforma) no está en chatwoot_account.

    Debe construirse localmente desde app/prompts/ecommer.json + VK del env;
    sin esto respondía el mensaje de escalada "no tengo informacion configurada".
    """

    def _patch(self, monkeypatch):
        from app import store_repository
        from app import store_resolver

        # conftest instala mocks autouse: hay que forzar las funciones REALES
        # (capturadas al importar el módulo) para probar el camino de verdad.
        monkeypatch.setattr(store_repository, "_TEST_MODE", False)
        monkeypatch.setattr(store_repository, "acquire_db", _fake_acquire_db)
        monkeypatch.setattr(store_repository, "get_store_by_account_id", _real_get_store_by_account_id)
        monkeypatch.setattr(store_resolver, "_TEST_MODE", False)
        monkeypatch.setattr(store_resolver, "get_store_by_account_id", _real_get_store_by_account_id)
        store_repository.invalidate_cache()
        return store_repository, store_resolver

    def test_get_store_cuenta_1_construye_ecommer(self, monkeypatch):
        repo, _ = self._patch(monkeypatch)
        store = asyncio.run(repo.get_store_by_account_id(1))
        assert store is not None
        assert store.store_name == "ecommer"
        assert store.is_global is True
        assert store.channel_tokens == []
        assert store.bifrost_plan == "ecommer"
        assert store.bifrost_vk  # VK del .env (no vacía)

    def test_resolve_store_cuenta_1_no_escala(self, monkeypatch):
        _, resolver = self._patch(monkeypatch)
        store = asyncio.run(resolver.resolve_store(1, 2))
        assert store.store_name == "ecommer"
        assert store.is_mapped is True
        assert "informacion configurada" not in (store.system_prompt or "")
        assert "Simetria" in store.system_prompt

    def test_otra_cuenta_sin_fila_sigue_devolviendo_none(self, monkeypatch):
        repo, _ = self._patch(monkeypatch)
        assert asyncio.run(repo.get_store_by_account_id(999)) is None


# ---------------------------------------------------------------------------
# Seguridad: la VK nunca aparece en repr / serializaciones
# ---------------------------------------------------------------------------

class TestVKNeverExposed:
    def test_repr_store_config_no_expone_vk(self):
        store = StoreConfig(store_name="x", bifrost_vk="SECRETA-SUPER-VK")
        assert "SECRETA-SUPER-VK" not in repr(store)
        assert "SECRETA-SUPER-VK" not in str(store)

    def test_repr_incluye_el_resto_de_campos(self):
        store = StoreConfig(store_name="tienda-una", bifrost_vk="SECRETA")
        assert "tienda-una" in repr(store)


# ---------------------------------------------------------------------------
# Router: keywords-first (sin LLM) + Phi-4 solo para mensajes ambiguos
# ---------------------------------------------------------------------------

class TestClassifierKeywordsFirst:
    """El router debe resolver los mensajes típicos con regex (~0ms) y solo
    invocar al LLM cuando los keywords no detectan nada."""

    def _counting_llm(self, monkeypatch, allowed=None, respuesta="CATALOGO"):
        calls = []

        async def _fake_pick(vk, preference):
            return preference[0]

        class _FakeCompletion:
            def __init__(self):
                self.choices = [type("C", (), {"message": type("M", (), {"content": respuesta})()})()]

        class _FakeLLM:
            class _Completions:
                async def create(self, **kwargs):
                    calls.append(kwargs)
                    return _FakeCompletion()

            chat = type("Chat", (), {"completions": _Completions()})()

            def with_options(self, **kwargs):
                return self

        monkeypatch.setattr("app.intent_classifier.pick_model", _fake_pick)
        monkeypatch.setattr("app.intent_classifier.get_llm", lambda vk: _FakeLLM())
        return calls

    def test_keywords_catalogo_no_llama_llm(self, monkeypatch):
        calls = self._counting_llm(monkeypatch)
        intents = asyncio.run(classify_intent("¿tienen Adidas Superstar?", vk="vk-test"))
        assert intents == ["CATALOGO"]
        assert calls == []

    def test_keywords_politicas_no_llama_llm(self, monkeypatch):
        calls = self._counting_llm(monkeypatch)
        intents = asyncio.run(classify_intent("quiero devolver el producto", vk="vk-test"))
        assert "POLITICAS" in intents
        assert calls == []

    def test_keywords_conversacional_no_llama_llm(self, monkeypatch):
        calls = self._counting_llm(monkeypatch)
        intents = asyncio.run(classify_intent("hola", vk="vk-test"))
        assert intents == ["CONVERSACIONAL"]
        assert calls == []

    def test_ambiguo_con_phi4_llama_llm_en_rol_user(self, monkeypatch):
        calls = self._counting_llm(monkeypatch, respuesta="CATALOGO")
        intents = asyncio.run(classify_intent("me atiende una persona?", vk="vk-test"))
        assert intents == ["CATALOGO"]
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 32
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "CATALOGO" in messages[0]["content"]
        assert "me atiende una persona?" in messages[0]["content"]

    def test_ambiguo_plan_sin_phi4_no_llama_llm(self, monkeypatch):
        async def _solo_mini(vk, preference):
            return "azure/Phi-4-mini-instruct"

        monkeypatch.setattr("app.intent_classifier.pick_model", _solo_mini)

        def _boom(vk):
            raise AssertionError("no debe llamar al LLM sin Phi-4")

        monkeypatch.setattr("app.intent_classifier.get_llm", _boom)
        intents = asyncio.run(classify_intent("me atiende una persona?", vk="vk-test"))
        assert intents == ["CONVERSACIONAL"]

    def test_llm_devuelve_basura_cae_a_conversacional(self, monkeypatch):
        self._counting_llm(monkeypatch, respuesta="Si no puedes encontrar el boton")
        intents = asyncio.run(classify_intent("me atiende una persona?", vk="vk-test"))
        assert intents == ["CONVERSACIONAL"]


# ---------------------------------------------------------------------------
# Prompt: sin links al LLM, docs truncados, historial/few-shot acotados
# ---------------------------------------------------------------------------

class TestPromptSize:
    def _doc_item(self, chars=2000):
        return {"score": 0.9, "payload": {"text": "palabra " * (chars // 8)}}

    def test_format_context_no_incluye_links(self):
        from app.llm_generator import _format_context

        item = {
            "score": 0.9,
            "payload": {
                "name": "Adidas Superstar",
                "url": "https://shop.example/adidas",
                "attributes": ["talla 42", "cuero"],
            },
        }
        text = _format_context([item], "CATALOGO")
        assert "Adidas Superstar" in text
        assert "talla 42" in text
        assert "https://" not in text
        assert "Link" not in text

    def test_format_context_trunca_documentos(self, monkeypatch):
        from app import llm_generator
        from app.llm_generator import _format_context

        monkeypatch.setattr(llm_generator.settings, "RETRIEVER_DOC_MAX_CHARS", 100)
        text = _format_context([self._doc_item()], "POLITICAS")
        assert len(text) < 200
        assert text.endswith("…")

    def test_build_prompt_limita_historial_a_6(self):
        from app.llm_generator import build_prompt

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(12)
        ]
        messages = build_prompt(
            query="hola",
            intent="CONVERSACIONAL",
            context_items=[],
            history=history,
            store_prompt="Eres x.",
        )
        contents = [m["content"] for m in messages]
        assert "msg5" not in contents
        assert "msg6" in contents

    def test_build_prompt_limita_few_shot_a_4(self):
        from app.llm_generator import build_prompt

        few_shot = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"ej{i}"}
            for i in range(10)
        ]
        messages = build_prompt(
            query="hola",
            intent="CONVERSACIONAL",
            context_items=[],
            history=[],
            store_prompt="Eres x.",
            few_shot=few_shot,
        )
        ej = [m for m in messages if m["content"].startswith("ej")]
        assert len(ej) == 4
        assert ej[0]["content"] == "ej6"
