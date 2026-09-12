import asyncio
from types import SimpleNamespace

import pytest

from app.retriever import _apply_min_score, expand_catalog_query, list_categories, search_context
from app.store_resolver import StoreConfig


def _make(score: float) -> dict:
    return {"score": score, "payload": {"source_id": f"s{score}"}}


def _global_store() -> StoreConfig:
    return StoreConfig(
        store_name="ecommer",
        channel_name="whatsapp",
        is_global=True,
        audience="CLIENTE",
    )


class TestApplyMinScore:
    def test_keeps_score_at_or_above_threshold(self, monkeypatch):
        monkeypatch.setattr("app.retriever.settings.RETRIEVER_MIN_SCORE", 0.5)
        out = _apply_min_score([_make(0.6), _make(0.5), _make(0.49)])
        assert [r["score"] for r in out] == [0.6, 0.5]

    def test_all_below_returns_empty(self, monkeypatch):
        monkeypatch.setattr("app.retriever.settings.RETRIEVER_MIN_SCORE", 0.5)
        assert _apply_min_score([_make(0.3), _make(0.2)]) == []

    def test_zero_disables_filter(self, monkeypatch):
        monkeypatch.setattr("app.retriever.settings.RETRIEVER_MIN_SCORE", 0.0)
        assert len(_apply_min_score([_make(0.3), _make(0.2)])) == 2

    def test_preserves_order(self, monkeypatch):
        monkeypatch.setattr("app.retriever.settings.RETRIEVER_MIN_SCORE", 0.5)
        out = _apply_min_score([_make(0.4), _make(0.7), _make(0.9)])
        assert [r["score"] for r in out] == [0.7, 0.9]

    def test_missing_score_is_falsy(self, monkeypatch):
        monkeypatch.setattr("app.retriever.settings.RETRIEVER_MIN_SCORE", 0.5)
        assert _apply_min_score([{"payload": {}}, _make(0.6)]) == [_make(0.6)]


class TestExpandCatalogQuery:
    def test_tecnologia_expands_with_domain_terms(self):
        out = expand_catalog_query("quiero algo de tecnologia")
        assert out != "quiero algo de tecnologia"
        assert "electronica" in out
        assert "audio" in out

    def test_accents_normalized_before_matching(self):
        assert "electronica" in expand_catalog_query("busco tecnología")

    def test_irrelevant_query_unchanged(self):
        assert expand_catalog_query("¿tienen panela orgánica?") == "¿tienen panela orgánica?"


class TestSearchContextExpansion:
    def test_catalogo_retries_with_expanded_query_on_empty(self, monkeypatch):
        embedded = []

        async def _fake_embed(text):
            embedded.append(text)
            return [1.0 if "electronica" in text else 0.0]

        class _FakeClient:
            async def query_points(self, **kwargs):
                if kwargs["query"] == [0.0]:
                    return SimpleNamespace(points=[
                        SimpleNamespace(score=0.2, payload={"source_id": "weak"})
                    ])
                return SimpleNamespace(points=[
                    SimpleNamespace(score=0.9, payload={"source_id": "kz"})
                ])

        monkeypatch.setattr("app.retriever._embed", _fake_embed)
        monkeypatch.setattr("app.retriever._get_qdrant", lambda: _FakeClient())

        results = asyncio.run(
            search_context("quiero tecnologia", _global_store(), ["CATALOGO"])
        )

        assert [r["payload"]["source_id"] for r in results] == ["kz"]
        assert len(embedded) == 2
        assert "electronica" in embedded[1]

    def test_non_catalogo_does_not_retry(self, monkeypatch):
        embedded = []

        async def _fake_embed(text):
            embedded.append(text)
            return [0.0]

        class _FakeClient:
            async def query_points(self, **kwargs):
                return SimpleNamespace(points=[
                    SimpleNamespace(score=0.2, payload={"source_id": "weak"})
                ])

        monkeypatch.setattr("app.retriever._embed", _fake_embed)
        monkeypatch.setattr("app.retriever._get_qdrant", lambda: _FakeClient())

        results = asyncio.run(
            search_context("¿política de envíos?", _global_store(), ["POLITICAS"])
        )

        assert results == []
        assert len(embedded) == 1


class TestListCategories:
    def test_collects_distinct_categories_from_scroll(self, monkeypatch):
        class _FakeClient:
            async def scroll(self, **kwargs):
                return [
                    SimpleNamespace(payload={
                        "metadata": {"categories": ["Audio", "Electrónica"]},
                    }),
                    SimpleNamespace(payload={
                        "metadata": {"categories": ["Libros"]},
                    }),
                    SimpleNamespace(payload={"text": "Categorías: Juguetes."}),
                ], None

        monkeypatch.setattr("app.retriever._get_qdrant", lambda: _FakeClient())

        result = asyncio.run(list_categories(_global_store()))

        assert result == ["Audio", "Electrónica", "Juguetes", "Libros"]

    def test_empty_collection_returns_empty(self, monkeypatch):
        class _FakeClient:
            async def scroll(self, **kwargs):
                return [], None

        monkeypatch.setattr("app.retriever._get_qdrant", lambda: _FakeClient())
        assert asyncio.run(list_categories(_global_store())) == []

    def test_error_returns_empty(self, monkeypatch):
        class _FakeClient:
            async def scroll(self, **kwargs):
                raise RuntimeError("qds no disponible")

        monkeypatch.setattr("app.retriever._get_qdrant", lambda: _FakeClient())
        assert asyncio.run(list_categories(_global_store())) == []