"""Offline tests for safe, request-correlated /agent/chat timings."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import tools
from app.main import app


class _Blocked(Exception):
    status_code = 403


class _FakeLLM:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _Blocked("provider error contains PRIVATE-EXCEPTION-MARKER")
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Safe answer"), finish_reason="stop"
        )
        return SimpleNamespace(choices=[choice], usage=None)


def _timing_events(caplog):
    return [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("timing_event ")]


def _fields(event):
    return dict(field.split("=", 1) for field in event.split()[1:])


@pytest.fixture
def offline_client(monkeypatch):
    """Exercise HTTP handling without deployment validation or DB startup."""
    from app import main

    async def _noop():
        pass

    monkeypatch.setattr(main, "validate_settings", lambda: None)
    monkeypatch.setattr(main, "init_db_pool", _noop)
    monkeypatch.setattr(main, "close_db_pool", _noop)
    monkeypatch.setattr(main, "close_clients", _noop)
    with TestClient(app) as client:
        yield client


def test_chat_timings_are_correlated_safe_and_keep_response_contract(
    monkeypatch, caplog, offline_client
):
    from app import clients
    from app.agent import core

    fake_llm = _FakeLLM()

    async def _classify(*args, **kwargs):
        return ["CATALOGO"]

    async def _memory(*args, **kwargs):
        return []

    async def _search(*args, **kwargs):
        return []

    monkeypatch.setattr(core, "classify_intent", _classify)
    monkeypatch.setattr(tools, "get_memory", _memory)
    monkeypatch.setattr(core, "save_message", lambda *args: None)
    monkeypatch.setattr(tools, "search_context", _search)
    monkeypatch.setattr(tools, "get_llm", lambda *_: fake_llm)
    monkeypatch.setattr(clients, "_TEST_MODE", True)
    caplog.set_level("INFO")

    response = offline_client.post("/agent/chat", json={
        "query": "PRIVATE-QUERY-MARKER",
        "account_id": 1,
        "inbox_id": 2,
    })

    assert response.status_code == 200
    assert set(response.json()) == {
        "answer", "intent_detected", "sources_used", "conversation_id",
        "escalado", "tools_used",
    }
    assert response.json()["answer"] == "Safe answer"
    assert len(fake_llm.calls) == 2

    events = _timing_events(caplog)
    parsed = [_fields(event) for event in events]
    totals = [event for event in parsed if event.get("stage") == "request_total"]
    assert len(totals) == 1
    request_id = totals[0]["request_id"]
    assert request_id
    assert totals[0]["outcome"] == "success"
    assert all(event.get("request_id") == request_id for event in parsed)

    stages = [event.get("stage") for event in parsed]
    for stage in (
        "store_resolution", "intent_classification", "memory_read", "memory_write",
        "context_retrieval", "model_discovery", "model_selection",
        "llm_completion_attempt", "answer_post_processing", "response_handling",
        "request_total",
    ):
        assert stage in stages
    attempts = [event for event in parsed if event.get("stage") == "llm_completion_attempt"]
    assert [event["outcome"] for event in attempts] == ["failure", "success"]
    assert [event["attempt"] for event in attempts] == ["1", "2"]
    assert [event["model"] for event in attempts] == [
        "azure/Phi-4", "azure/Phi-4-mini-instruct"
    ]
    assert all(event.get("model_role") == "answer" for event in attempts)
    selections = [event for event in parsed if event.get("stage") == "model_selection"]
    assert any(event.get("model") == "azure/Phi-4" for event in selections)
    assert any(event.get("model") == "azure/Phi-4-mini-instruct" for event in selections)

    for event in events:
        assert "PRIVATE-QUERY-MARKER" not in event
        assert "PRIVATE-EXCEPTION-MARKER" not in event
        assert "Safe answer" not in event
        assert "test-vk" not in event


def test_request_total_waits_for_final_asgi_body_send(caplog):
    from app.observability import _REQUEST_ID

    async def _exercise():
        body_send_started = asyncio.Event()
        finish_body_send = asyncio.Event()
        sent = []

        async def receive():
            return {"type": "http.request", "body": b'{"query":""}', "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                body_send_started.set()
                await finish_body_send.wait()
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/agent/chat",
            "raw_path": b"/agent/chat",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(body_send_started.wait(), timeout=2)
            assert not [event for event in _timing_events(caplog)
                        if "stage=request_total" in event]
        finally:
            finish_body_send.set()
            await task

        assert sent[0]["status"] == 400
        headers = dict(sent[0]["headers"])
        request_id = headers[b"x-request-id"].decode()
        events = [_fields(event) for event in _timing_events(caplog)]
        total = next(event for event in events if event.get("stage") == "request_total")
        assert total["request_id"] == request_id
        assert total["outcome"] == "failure"
        assert total["status_code"] == "400"
        assert _REQUEST_ID.get() is None

    caplog.set_level("INFO")
    asyncio.run(_exercise())


def test_handled_chat_failure_emits_correlated_total(
    monkeypatch, caplog, offline_client
):
    async def _fail(**kwargs):
        from app.exceptions import StoreWithoutBifrostKeyError
        raise StoreWithoutBifrostKeyError(kwargs.get("account_id"), "safe-store-label")

    monkeypatch.setattr("app.main.run_agent", _fail)
    caplog.set_level("INFO")

    response = offline_client.post("/agent/chat", json={"query": "failure marker", "account_id": 18})

    assert response.status_code == 503
    totals = [_fields(event) for event in _timing_events(caplog)
              if "stage=request_total" in event]
    assert len(totals) == 1
    assert totals[0]["outcome"] == "failure"
    assert totals[0]["request_id"]
    assert "failure marker" not in " ".join(_timing_events(caplog))


class _TerminalStream:
    def __init__(self, is_tty):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def _format_log(message, *, is_tty, monkeypatch, level=logging.INFO):
    from app.observability import TerminalColorFormatter

    formatter = TerminalColorFormatter("%(message)s", stream=_TerminalStream(is_tty))
    record = logging.LogRecord("test", level, __file__, 1, message, (), None)
    return formatter.format(record)


def test_timing_outcomes_use_terminal_colors_without_changing_message(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    success = "timing_event request_id=abc stage=llm_completion_attempt outcome=success"
    failure = "timing_event request_id=abc stage=llm_completion_attempt outcome=failure"

    colored_success = _format_log(success, is_tty=True, monkeypatch=monkeypatch)
    colored_failure = _format_log(failure, is_tty=True, monkeypatch=monkeypatch)

    assert "\x1b[32m" in colored_success and "\x1b[0m" in colored_success
    assert "\x1b[31m" in colored_failure and "\x1b[0m" in colored_failure
    assert colored_success.replace("\x1b[32m", "").replace("\x1b[0m", "") == success
    assert colored_failure.replace("\x1b[31m", "").replace("\x1b[0m", "") == failure


def test_timing_outcomes_stay_plain_without_terminal_or_with_no_color(monkeypatch):
    event = "timing_event request_id=abc stage=request_total outcome=success"
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "\x1b[" not in _format_log(event, is_tty=False, monkeypatch=monkeypatch)

    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in _format_log(event, is_tty=True, monkeypatch=monkeypatch)


def test_unrelated_info_logs_are_not_colored_as_success(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    message = "application started successfully"
    assert _format_log(message, is_tty=True, monkeypatch=monkeypatch) == message


def test_warning_and_error_logs_use_severity_colors(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    warning = _format_log("warning", is_tty=True, monkeypatch=monkeypatch, level=logging.WARNING)
    error = _format_log("error", is_tty=True, monkeypatch=monkeypatch, level=logging.ERROR)

    assert "\x1b[33mwarning\x1b[0m" == warning
    assert "\x1b[31merror\x1b[0m" == error


def test_classifier_completion_emits_safe_timing(monkeypatch, caplog):
    from app import intent_classifier
    from app.observability import _REQUEST_ID

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def with_options(self, **kwargs):
            return self

        async def create(self, **kwargs):
            choice = SimpleNamespace(
                message=SimpleNamespace(content="CATALOGO"),
            )
            return SimpleNamespace(choices=[choice])

    async def _pick_model(*args, **kwargs):
        return "azure/Phi-4-mini-instruct"

    monkeypatch.setattr(intent_classifier, "pick_model", _pick_model)
    monkeypatch.setattr(intent_classifier, "get_llm", lambda _vk: _FakeClient())
    caplog.set_level("INFO")
    token = _REQUEST_ID.set("classifier-test-request")
    try:
        import asyncio
        result = asyncio.run(intent_classifier.classify_intent(
            "PRIVATE-CLASSIFIER-QUERY", vk="fake-vk"
        ))
    finally:
        _REQUEST_ID.reset(token)

    assert result == ["CATALOGO"]
    events = _timing_events(caplog)
    selection = next(event for event in events if "stage=model_selection" in event)
    completion = next(
        event for event in events if "stage=llm_completion_attempt" in event
    )
    for event in (selection, completion):
        assert "request_id=classifier-test-request" in event
        assert "model_role=classifier" in event
        assert "model=azure/Phi-4-mini-instruct" in event
        assert "PRIVATE-CLASSIFIER-QUERY" not in event
        assert "fake-vk" not in event
