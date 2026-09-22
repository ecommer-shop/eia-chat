"""Tests de integración del gateway (endpoints HTTP + rate limiting).

Todas las dependencias externas (Groq, Qdrant, Azure, Redis) se mockean: no
se hacen llamadas de red. Se verifica el contrato HTTP de `/chat`,
`/agent/chat`, `/health`, `/stores` y `/stores/reload`, además del middleware
de rate limiting.
"""

from typing import Any
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------- fakes groq


class _FakeCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)


class _FakeCompletions:
    def __init__(self, completion: _FakeCompletion):
        self._completion = completion

    async def create(self, **_: Any) -> _FakeCompletion:
        return self._completion


class _FakeChat:
    def __init__(self, completion: _FakeCompletion):
        self.completions = _FakeCompletions(completion)


class _FakeGroq:
    def __init__(self, completion: _FakeCompletion):
        self.chat = _FakeChat(completion)


def _fake_groq(content: str = "Respuesta de prueba") -> _FakeGroq:
    return _FakeGroq(_FakeCompletion(content))


# ----------------------------------------------------------- fixtures/mocks


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Mockea el pipeline completo: clasificador, retriever, memoria y Groq."""

    async def _fake_classify(query: str) -> list[str]:
        return ["CATALOGO"]

    async def _fake_search(query: str, store, intents: list[str], limit: int = 10) -> list[dict]:
        return [{"score": 0.9, "payload": {"metadata": {"name": "Zapatos"}, "text": "Zapatos de prueba"}}]

    async def _fake_categories(store, limit: int = 1000) -> list[str]:
        return ["Calzado", "Ropa"]

    monkeypatch.setattr("app.main.classify_intent", _fake_classify)
    monkeypatch.setattr("app.main.search_context", _fake_search)
    monkeypatch.setattr("app.main.list_categories", _fake_categories)
    monkeypatch.setattr("app.main.get_history", lambda cid: [])
    monkeypatch.setattr("app.main.save_message", lambda cid, role, content: None)
    monkeypatch.setattr("app.main.get_groq", lambda: _fake_groq())
    return monkeypatch


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------------------- health


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["collection"]


# --------------------------------------------------------------------- stores


def test_stores_list(client):
    resp = client.get("/stores")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_stores" in body
    assert "stores" in body
    assert "account_ids_loaded" in body


def test_stores_reload(client):
    resp = client.post("/stores/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["stores_loaded"] > 0


# ----------------------------------------------------------------------- chat


def test_chat_returns_answer(client, mock_pipeline):
    resp = client.post("/chat", json={"query": "¿Tienen zapatos?", "account_id": 1, "inbox_id": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Respuesta de prueba"
    assert body["intent_detected"] == "CATALOGO"
    assert body["sources_used"] == 1
    assert body["conversation_id"]


def test_chat_empty_query_returns_400(client, mock_pipeline):
    resp = client.post("/chat", json={"query": "   "})
    assert resp.status_code == 400


def test_chat_accepts_legacy_message_key(client, mock_pipeline):
    resp = client.post("/chat", json={"message": "¿Envían a Bogotá?", "account_id": 1, "inbox_id": 2})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Respuesta de prueba"


def test_chat_groq_error_falls_back(client, mock_pipeline):
    class _BrokenGroq:
        class _BrokenCompletions:
            async def create(self, **_: Any):
                raise RuntimeError("Groq caído")

        chat = _BrokenCompletions()

    mock_pipeline.setattr("app.main.get_groq", lambda: _BrokenGroq())
    resp = client.post("/chat", json={"query": "¿Tienen zapatos?", "account_id": 1, "inbox_id": 2})
    assert resp.status_code == 200
    assert "problemas técnicos" in resp.json()["answer"]


def test_chat_empty_groq_content_falls_back(client, mock_pipeline):
    def _empty_groq():
        completion = _FakeCompletion(content="", finish_reason="length")
        return _FakeGroq(completion)

    mock_pipeline.setattr("app.main.get_groq", _empty_groq)
    resp = client.post("/chat", json={"query": "¿Tienen zapatos?", "account_id": 1, "inbox_id": 2})
    assert resp.status_code == 200
    assert "no pude generar" in resp.json()["answer"]


# ---------------------------------------------------------------- agent/chat


def test_agent_chat_returns_result(client, mock_pipeline, monkeypatch):
    from app.agent.core import AgentResult

    async def _fake_run_agent(**_: Any):
        return AgentResult(
            answer="Respuesta del agente",
            intent_detected="CATALOGO",
            sources_used=2,
            conversation_id="conv-1",
            escalado=False,
            tools_used=["search_catalogo", "answer"],
        )

    monkeypatch.setattr("app.main.run_agent", _fake_run_agent)
    resp = client.post("/agent/chat", json={"query": "¿Qué hay en oferta?", "account_id": 1, "inbox_id": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Respuesta del agente"
    assert body["escalado"] is False
    assert "search_catalogo" in body["tools_used"]


def test_agent_chat_escalated(client, mock_pipeline, monkeypatch):
    from app.agent.core import AgentResult

    async def _fake_escalate(**_: Any):
        return AgentResult(
            answer="Un agente te contactará.",
            intent_detected="CONVERSACIONAL",
            sources_used=0,
            conversation_id="conv-9",
            escalado=True,
            tools_used=["escalate"],
        )

    monkeypatch.setattr("app.main.run_agent", _fake_escalate)
    resp = client.post("/agent/chat", json={"query": "Hola", "account_id": 999, "inbox_id": 999, "user_id": 5})
    assert resp.status_code == 200
    assert resp.json()["escalado"] is True


def test_agent_chat_empty_query_returns_400(client, mock_pipeline):
    resp = client.post("/agent/chat", json={"query": ""})
    assert resp.status_code == 400


# ------------------------------------------------------------- rate limiting


def test_rate_limiter_allows_up_to_limit_then_rejects():
    from app.rate_limit import RateLimiter

    class _FakeClient:
        host = "192.168.1.1"

    class _FakeRequest:
        headers = {}
        client = _FakeClient()

    limiter = RateLimiter(max_requests=2)
    req = _FakeRequest()
    assert limiter.allow(req) is True
    assert limiter.allow(req) is True
    assert limiter.allow(req) is False


def test_rate_limiter_resets_after_window(monkeypatch):
    from app.rate_limit import RateLimiter

    class _FakeClient:
        host = "192.168.1.2"

    class _FakeRequest:
        headers = {}
        client = _FakeClient()

    import app.rate_limit as rate_limit

    now = 1000.0
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now)

    limiter = RateLimiter(max_requests=1, window_seconds=60)
    req = _FakeRequest()
    assert limiter.allow(req) is True
    assert limiter.allow(req) is False

    now = 1061.0
    assert limiter.allow(req) is True


def test_rate_limiter_keys_by_forwarded_ip():
    from app.rate_limit import RateLimiter

    class _FakeRequest:
        def __init__(self, fwd: str):
            self.headers = {"x-forwarded-for": fwd}
            self.client = None

    limiter = RateLimiter(max_requests=1)
    assert limiter.allow(_FakeRequest("10.0.0.1")) is True
    assert limiter.allow(_FakeRequest("10.0.0.1")) is False
    assert limiter.allow(_FakeRequest("10.0.0.2")) is True