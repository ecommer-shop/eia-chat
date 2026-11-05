"""Cliente HTTP del asistente hacia el endpoint de catálogo del backend Vendure.

Fase 0.5 (T0.5.3) — consume el endpoint de LECTURA `POST /api/assistant/catalog`
(y su variante `/api/assistant/catalog/product`) implementado en `Shop/shop`
(backend admin, Vendure 3.7.3). El endpoint corre sobre un contexto Shop
anónimo, así que solo expone catálogo público; no recibe identidad de comprador.

Autenticación servicio a servicio (mismo esquema que el webhook de Chatwoot):

    x-assistant-key-id:    <identificador del caller, p. ej. eia-chat>
    x-assistant-timestamp: <unix segundos>
    x-assistant-signature: "sha256=" + hex(HMAC_SHA256(secret, `${timestamp}.${rawBody}`))

El `rawBody` firmado es exactamente el body JSON enviado (bytes, sin reformatear).
Si el secreto/base-url no está configurado (o el flag está apagado), el cliente
devuelve resultados vacíos para que el caller degrade al RAG (Qdrant).

El cliente MApea la respuesta del catálogo al shape de `context_items` que el
pipeline RAG espera: `[{ "score": float, "payload": { "name", "url", ... } }]`.
La URL del producto de storefront se construye desde `slug` con la base
configurable `ASSISTANT_CATALOG_STOREFRONT_BASE_URL` (default ecommer.shop).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Base del storefront para armar la URL de producto a partir del `slug`.
# La ruta del producto es `/product/{slug}` con prefijo de locale (`/es`).
_STOREFRONT_BASE = (
    settings.ASSISTANT_CATALOG_STOREFRONT_BASE_URL or "https://ecommer.shop"
).rstrip("/")
_PRODUCT_PATH = "/es/product/{}"


class CatalogClientUnavailable(Exception):
    """El endpoint de catálogo no está disponible o no está configurado."""


def _now_seconds() -> int:
    return int(time.time())


def _sign(timestamp: int, raw_body: bytes) -> str:
    """Firma HMAC-SHA256 del body raw con el secreto compartido."""
    secret = settings.ASSISTANT_CATALOG_SECRET or ""
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + raw_body,
        hashlib.sha256,
    ).hexdigest()
    return "sha256=" + digest


def _enabled() -> bool:
    """El cliente queda deshabilitado si el flag o las credenciales no están."""
    return bool(
        settings.ASSISTANT_CATALOG_ENABLED
        and settings.ASSISTANT_CATALOG_BASE_URL
        and settings.ASSISTANT_CATALOG_SECRET
    )


def _product_url(slug: str) -> str:
    return f"{_STOREFRONT_BASE}{_PRODUCT_PATH.format(slug)}"


def _to_context_item(
    item: dict, score: float
) -> dict:
    """Convierte un `CatalogProductListItem` a un item de contexto RAG."""
    slug = item.get("slug") or ""
    name = item.get("name") or "Producto sin nombre"
    description = item.get("description") or ""
    featured = item.get("featuredAsset") or {}
    price_range = item.get("priceRange")

    payload: dict = {
        "name": name,
        "url": _product_url(slug) if slug else "",
        "slug": slug,
    }
    if description:
        payload["description"] = description
    if featured.get("preview"):
        payload["image"] = featured["preview"]
    if price_range:
        payload["priceRange"] = price_range

    return {"score": score, "payload": payload}


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ASSISTANT_CATALOG_BASE_URL.rstrip("/"),
        timeout=settings.ASSISTANT_CATALOG_TIMEOUT_SECONDS,
    )


async def _post(path: str, payload: dict) -> dict:
    """POST firmado al endpoint; lanza CatalogClientUnavailable si falla."""
    if not _enabled():
        raise CatalogClientUnavailable("Catálogo deshabilitado o sin credenciales")

    raw_body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    timestamp = _now_seconds()
    headers = {
        "content-type": "application/json",
        "x-assistant-key-id": settings.ASSISTANT_CATALOG_KEY_ID,
        "x-assistant-timestamp": str(timestamp),
        "x-assistant-signature": _sign(timestamp, raw_body),
    }

    try:
        async with _client() as client:
            resp = await client.post(path, content=raw_body, headers=headers)
    except Exception as e:  # noqa: BLE001
        logger.error("Catálogo: error de red/tipo en %s: %s", path, type(e).__name__)
        raise CatalogClientUnavailable(str(e)) from e

    if resp.status_code >= 400:
        logger.warning(
            "Catálogo: %s respondió %d: %s",
            path, resp.status_code, resp.text[:200],
        )
        raise CatalogClientUnavailable(f"HTTP {resp.status_code}")

    try:
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.error("Catálogo: respuesta no-JSON en %s: %s", path, e)
        raise CatalogClientUnavailable("respuesta no-JSON") from e


async def search_products(query: str, limit: int = 5) -> list[dict]:
    """Equivale a `search_products` del MCP: busca productos por query.

    Devuelve items de contexto RAG (lista vacía si no hay resultados o el
    endpoint no está disponible). El caller decide si degrada al RAG.
    """
    payload: dict = {"limit": limit, "offset": 0}
    if query:
        payload["query"] = query

    try:
        data = await _post("/api/assistant/catalog", payload)
    except CatalogClientUnavailable:
        return []

    items = data.get("items") or []
    return [_to_context_item(item, 1.0) for item in items if isinstance(item, dict)]


async def get_product(product_id: str | None = None, slug: str | None = None) -> dict | None:
    """Equivale a `get_product` del MCP: detalle de un producto por id o slug.

    Devuelve un item de contexto RAG o `None` si no existe/no está disponible.
    """
    payload: dict = {}
    if product_id:
        payload["id"] = str(product_id)
    elif slug:
        payload["slug"] = slug
    else:
        return None

    try:
        data = await _post("/api/assistant/catalog/product", payload)
    except CatalogClientUnavailable:
        return None

    product = data.get("product")
    if not isinstance(product, dict):
        return None
    return _to_context_item(product, 1.0)
