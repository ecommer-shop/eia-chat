"""Capa de renderizado determinística (separación de responsabilidades).

- QUÉ decir lo decide el LLM (texto natural, mencionando nombres de
  producto, sin escribir URLs nunca).
- CÓMO se muestran los links lo decide este módulo en código:

    1. `match_mentioned_products`: detecta de forma determinística qué
       productos mencionados en el texto corresponden a `context_items`
       del turno (matching por nombre normalizado). Esto es además el
       grounding estructural: solo se puede enlazar lo que existe en el
       contexto de esa tool call.
    2. `strip_catalog_urls`: quita del cuerpo las URLs de catálogo que el
       modelo haya escrito igualmente, para que el link nunca se duplique.
    3. `render_product_footer`: adjunta el link real de cada producto
       mencionado, con un formato fijo, una sola vez, independiente de qué
       tan bien redactó el modelo ese turno.

Con esto un link roto, duplicado o inventado deja de ser posible: solo se
emiten URLs que existen en el contexto del turno.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.grounding import _URL_RE, _extract_names, _extract_urls, _normalize

logger = logging.getLogger(__name__)

LINK_FORMAT = "🔗 {name} — {url}"
FOOTER_JOIN = " · "
FOOTER_PREFIXES = ("🛒 Encuéntralos aquí:", "🛒 Aquí tienes los links:", "🔗 Links:")
FOOTER_PREFIX = FOOTER_PREFIXES[0]


@dataclass
class MentionedProduct:
    name: str
    url: str
    score: float


def _display_name(payload: dict) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("name", "title", "producto", "product"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
    for key in ("name", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    names = sorted(_extract_names(payload))
    return names[0] if names else "este producto"


def match_mentioned_products(
    text: str,
    context_items: list[dict],
    intent: str = "",
) -> list[MentionedProduct]:
    """Productos del turno que el texto menciona (matching determinista).

    Un producto se considera mencionado si alguna de sus variantes de nombre
    (normalizada: sin acentos, minúsculas) aparece como subcadena del texto.
    Si el mencionado no matchea ningún `context_items`, no aparece en el
    resultado — y por tanto no se enlaza.
    """
    if "CATALOGO" not in intent or not text:
        return []

    normalized = _normalize(text)
    mentioned: list[MentionedProduct] = []
    seen_urls: set[str] = set()

    for item in context_items or []:
        payload = item.get("payload", {}) or {}
        urls = sorted(_extract_urls(payload))
        if not urls:
            continue
        names = _extract_names(payload)
        if not any(name and name in normalized for name in names):
            continue
        url = urls[0]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        mentioned.append(MentionedProduct(
            name=_display_name(payload),
            url=url,
            score=float(item.get("score") or 0.0),
        ))

    return mentioned


def strip_catalog_urls(text: str, context_items: list[dict]) -> str:
    """Quita del cuerpo las URLs del catálogo de este turno.

    Evita la duplicación: el modelo puede escribir igualmente un link del
    contexto; se elimina del cuerpo para que el footer sea la única fuente.
    Nunca toca URLs que no pertenezcan a `context_items`.
    """
    known: set[str] = set()
    for item in context_items or []:
        known |= _extract_urls(item.get("payload", {}) or {})
    if not known:
        return text

    clean = text
    for url in known:
        clean = re.sub(rf"🔗\s*{re.escape(url)}(?:\s|$)", "", clean)
        clean = re.sub(rf"{re.escape(url)}(?:\s|$)", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean


def render_product_footer(products: list[MentionedProduct]) -> str:
    """Bloque de links con formato fijo (una sola vez, sin duplicados)."""
    if not products:
        return ""
    parts = [LINK_FORMAT.format(name=p.name, url=p.url) for p in products]
    return f"{FOOTER_PREFIX} {FOOTER_JOIN.join(parts)}"


def annotate_products(
    text: str,
    context_items: list[dict],
    intent: str = "",
) -> tuple[str, list[MentionedProduct]]:
    """(texto sin URLs de catálogo en el cuerpo, productos mencionados).

    El footer se arma con `render_product_footer(mentioned)`. Si no hubo
    menciones, devuelve el texto intacto.
    """
    mentioned = match_mentioned_products(text, context_items, intent)
    if not mentioned:
        return text, []
    clean = strip_catalog_urls(text, context_items)
    return clean, mentioned