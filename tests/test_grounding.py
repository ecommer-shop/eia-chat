"""Tests del guardrail de grounding (app/grounding.py).

Sin dependencias externas: solo manipulan payloads, historial y texto.
"""

from app.grounding import (
    apply_grounding,
    collect_known,
    find_ungrounded,
    FALLBACK_UNGROUNDED,
    _normalize,
)


def _product(name="KZ Castor Pro", url="https://ecommer.shop/es/product/kz-castor-pro-bass-edition"):
    return {"score": 0.92, "payload": {"metadata": {"name": name, "url": url}}}


class TestNormalize:
    def test_strips_accents_and_case(self):
        assert _normalize("Tecnología  ") == "tecnologia"

    def test_keeps_spaces(self):
        assert _normalize("KZ  Castor Pro") == "kz  castor pro"


class TestCollectKnown:
    def test_extracts_url_and_name_from_metadata(self):
        known = collect_known([_product()], [])
        assert known["urls"] == {"https://ecommer.shop/es/product/kz-castor-pro-bass-edition"}
        assert "kz castor pro" in known["names"]

    def test_accepts_top_level_name_url_payload(self):
        item = {"payload": {
            "name": "Panela orgánica",
            "url": "https://ecommer.shop/product/panela-organica",
        }}
        known = collect_known([item], [])
        assert "panela organica" in known["names"]

    def test_history_assistant_mentions_become_known(self):
        history = [{"role": "assistant", "content": (
            "👉 Panela orgánica 🔗 https://ecommer.shop/product/panela-organica"
        )}]
        known = collect_known([], history)
        assert "https://ecommer.shop/product/panela-organica" in known["urls"]
        assert "panela organica" in known["names"]

    def test_user_history_ignored(self):
        history = [{"role": "user", "content": "👉 iPhone 15 🔗 https://fake/iphone"}]
        known = collect_known([], history)
        assert known["urls"] == set()
        assert known["names"] == set()


class TestFindUngrounded:
    def test_unknown_url_is_flagged(self):
        known = collect_known([_product()], [])
        assert find_ungrounded(
            "Mira el 👉 iPhone 15 🔗 https://ecommer.shop/product/iphone-15", known
        ) == ["url:https://ecommer.shop/product/iphone-15", "producto:iPhone 15"]

    def test_known_url_passes(self):
        known = collect_known([_product()], [])
        text = f"Te recomiendo el KZ Castor Pro 🔗 {_product()['payload']['metadata']['url']}"
        assert find_ungrounded(text, known) == []

    def test_arrow_name_not_in_catalog_flagged(self):
        known = collect_known([_product()], [])
        assert "producto:Vision Pro" in find_ungrounded("👉 Vision Pro", known)

    def test_substring_abbreviation_passes(self):
        known = collect_known([_product(name="KZ Castor Pro Bass Edition")], [])
        assert find_ungrounded("👉 KZ Castor Pro", known) == []

    def test_plain_text_without_links_or_arrows_passes(self):
        assert find_ungrounded("Te recomiendo los audífonos que vimos antes", collect_known([], [])) == []


class TestApplyGrounding:
    def test_hallucinated_product_replaced_with_fallback(self):
        text = "👉 iPhone 15 🔗 https://ecommer.shop/product/iphone-15"
        out, issues = apply_grounding(text, [_product()], [], intent="CATALOGO")
        assert out == FALLBACK_UNGROUNDED
        assert len(issues) == 2

    def test_grounded_answer_passes_through(self):
        url = _product()["payload"]["metadata"]["url"]
        text = f"Te recomiendo el 👉 KZ Castor Pro 🔗 {url}"
        out, issues = apply_grounding(text, [_product()], [], intent="CATALOGO")
        assert out == text
        assert issues == []

    def test_non_catalog_answer_not_blocked_without_product_signals(self):
        text = "La política de devoluciones es de 15 días hábiles."
        out, issues = apply_grounding(text, [], [], intent="POLITICAS")
        assert out == text
        assert issues == []

    def test_non_catalog_answer_still_blocked_for_unknown_url(self):
        text = "Detalles en https://www.google.com/pagina-inventada"
        out, issues = apply_grounding(text, [], [], intent="POLITICAS")
        assert out == FALLBACK_UNGROUNDED
        assert issues == ["url:https://www.google.com/pagina-inventada"]

    def test_history_product_allows_followup_mention(self):
        history = [{"role": "assistant", "content": (
            "👉 Panela orgánica 🔗 https://ecommer.shop/product/panela-organica"
        )}]
        text = "La panela de antes es ideal para endulzar 👉 Panela orgánica 🔗 https://ecommer.shop/product/panela-organica"
        out, issues = apply_grounding(text, [], history, intent="CATALOGO")
        assert out == text
        assert issues == []