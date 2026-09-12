"""Tests del renderer determinístico de links (app/renderer.py).

Sin dependencias externas: solo manipulan payloads y texto.
"""

from app.renderer import (
    annotate_products,
    match_mentioned_products,
    render_product_footer,
    strip_catalog_urls,
)


def _product(name="KZ Castor Pro", url="https://ecommer.shop/es/product/kz-castor-pro-bass-edition"):
    return {"score": 0.92, "payload": {"metadata": {"name": name, "url": url}}}


class TestMatchMentionedProducts:
    def test_matches_name_as_substring(self):
        item = _product(name="Café orgánico Alem",
                        url="https://ecommer.shop/es/product/cafe-organico-alem")
        mentioned = match_mentioned_products("El café orgánico Alem es intenso.", [item], intent="CATALOGO")
        assert len(mentioned) == 1
        assert mentioned[0].name == "Café orgánico Alem"
        assert mentioned[0].url == "https://ecommer.shop/es/product/cafe-organico-alem"

    def test_normalizes_accents_and_case(self):
        item = _product(name="Panela orgánica", url="https://ecommer.shop/es/product/panela")
        mentioned = match_mentioned_products("Me encanta la PANELA ORGANICA.", [item], intent="CATALOGO")
        assert len(mentioned) == 1

    def test_non_catalog_intent_returns_nothing(self):
        item = _product()
        mentioned = match_mentioned_products("El KZ Castor Pro suena bien.", [item], intent="POLITICAS")
        assert mentioned == []

    def test_unknown_product_not_mentioned(self):
        item = _product(name="Café orgánico Alem", url="https://ecommer.shop/es/product/cafe")
        mentioned = match_mentioned_products("Busco un teléfono nuevo.", [item], intent="CATALOGO")
        assert mentioned == []

    def test_product_without_url_in_context_is_ignored(self):
        item = {"score": 0.9, "payload": {"metadata": {"name": "Panela orgánica"}}}
        mentioned = match_mentioned_products("La panela orgánica es rica.", [item], intent="CATALOGO")
        assert mentioned == []

    def test_deduplicates_same_url(self):
        ctx = [
            _product(name="Café orgánico Alem", url="https://ecommer.shop/es/product/cafe"),
            _product(name="Café orgánico", url="https://ecommer.shop/es/product/cafe"),
        ]
        mentioned = match_mentioned_products("Quiero el café orgánico.", ctx, intent="CATALOGO")
        assert len(mentioned) == 1


class TestStripCatalogUrls:
    def test_removes_context_url_written_inline(self):
        item = _product()
        text = f"Te recomiendo el KZ Castor Pro 🔗 {item['payload']['metadata']['url']}"
        out = strip_catalog_urls(text, [item])
        assert item["payload"]["metadata"]["url"] not in out
        assert "KZ Castor Pro" in out

    def test_keeps_urls_not_from_context(self):
        text = "Más info en https://www.ecommer.shop/blog/reviews"
        out = strip_catalog_urls(text, [_product()])
        assert "https://www.ecommer.shop/blog/reviews" in out

    def test_empty_context_unchanged(self):
        text = "Hola, ¿cómo estás?"
        assert strip_catalog_urls(text, []) == text


class TestRenderProductFooter:
    def test_empty_products_returns_empty(self):
        assert render_product_footer([]) == ""

    def test_single_product_fixed_format(self):
        item = _product(name="Café orgánico Alem",
                        url="https://ecommer.shop/es/product/cafe-organico-alem")
        mentioned = match_mentioned_products("El café orgánico Alem.", [item], intent="CATALOGO")
        footer = render_product_footer(mentioned)
        assert footer == "🛒 Encuéntralos aquí: 🔗 Café orgánico Alem — https://ecommer.shop/es/product/cafe-organico-alem"

    def test_multiple_products_joined(self):
        ctx = [
            _product(name="Café orgánico Alem", url="https://ecommer.shop/es/product/cafe"),
            _product(name="Panela orgánica", url="https://ecommer.shop/es/product/panela"),
        ]
        mentioned = match_mentioned_products(
            "Quiero el café orgánico alem y la panela orgánica.", ctx, intent="CATALOGO"
        )
        footer = render_product_footer(mentioned)
        assert footer.count("🔗") == 2
        assert "https://ecommer.shop/es/product/cafe" in footer
        assert "https://ecommer.shop/es/product/panela" in footer


class TestAnnotateProducts:
    def test_appends_footer_once_and_strips_inline_url(self):
        item = _product(name="KZ Castor Pro", url="https://ecommer.shop/es/product/kz-castor")
        text = (f"Te recomiendo el KZ Castor Pro 🔗 {item['payload']['metadata']['url']} "
                "y el KZ Castor Pro es ideal para ti.")
        clean, mentioned = annotate_products(text, [item], intent="CATALOGO")
        assert item["payload"]["metadata"]["url"] not in clean
        footer = render_product_footer(mentioned)
        assert footer == "🛒 Encuéntralos aquí: 🔗 KZ Castor Pro — https://ecommer.shop/es/product/kz-castor"
        assert clean.count("KZ Castor Pro") == 2

    def test_no_mentions_returns_text_intact(self):
        item = _product()
        text = "¿Me das más información?"
        clean, mentioned = annotate_products(text, [item], intent="CATALOGO")
        assert clean == text
        assert mentioned == []

    def test_empty_list_without_duplicates_when_model_only_named_once(self):
        item = _product(name="Panela orgánica", url="https://ecommer.shop/es/product/panela")
        text = "Puedes comprar la panela orgánica, está deliciosa."
        clean, mentioned = annotate_products(text, [item], intent="CATALOGO")
        footer = render_product_footer(mentioned)
        completed = f"{clean} {footer}".strip()
        assert completed.count("https://ecommer.shop/es/product/panela") == 1
        assert completed.lower().count("panela orgánica") == 2