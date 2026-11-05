"""Cliente HTTP del asistente hacia los endpoints de identidad del backend.

Fase 1b (T2) — consume `POST /api/assistant/identity/cart` y
`POST /api/assistant/identity/order` implementados en `Shop/shop`
(plugin `assistant-identity`). Es un espejo deliberado de
`app/catalog_client.py`: mismo backend, mismo secreto y misma firma HMAC,
solo cambia el path y el shape de la respuesta.

Autenticación servicio a servicio:

    x-assistant-key-id:    <identificador del caller, p. ej. eia-chat>
    x-assistant-timestamp: <unix segundos>
    x-assistant-signature: "sha256=" + hex(HMAC_SHA256(secret, `${timestamp}.${rawBody}`))

El `rawBody` firmado es exactamente el body JSON enviado (bytes, sin reformatear).

A diferencia del catálogo, la identidad del comprador la resuelve el backend
leyendo `storefront_chat_session.customer_id` para el `conversation_id` de
Chatwoot (entero puro) que enviamos. Nosotros nunca mandamos un customer id.

Fail-open, igual que el catálogo: si el flag está apagado, faltan credenciales,
el endpoint falla o responde 404 (comprador anónimo / sin sesión), las
funciones devuelven `None`. Nunca lanzan hacia el agente: el turno degrada a
un mensaje cordial (ver `app/agent/tools.py`).
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

_CART_PATH = "/api/assistant/identity/cart"
_ORDER_PATH = "/api/assistant/identity/order"


class IdentityClientUnavailable(Exception):
    """El endpoint de identidad no está disponible o no está configurado."""


def _now_seconds() -> int:
    return int(time.time())


def _sign(timestamp: int, raw_body: bytes) -> str:
    """Firma HMAC-SHA256 del body raw con el secreto compartido.

    Idéntica a `catalog_client._sign` (mismo esquema y mismo secreto).
    """
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


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ASSISTANT_CATALOG_BASE_URL.rstrip("/"),
        timeout=settings.ASSISTANT_CATALOG_TIMEOUT_SECONDS,
    )


def _identity_payload(account_id: int, inbox_id: int | None, conversation_id: int) -> dict:
    """Body común: solo las coordenadas de Chatwoot (nunca identidad)."""
    payload: dict = {
        "conversation_id": int(conversation_id),
        "account_id": int(account_id),
    }
    if inbox_id is not None:
        payload["inbox_id"] = int(inbox_id)
    return payload


async def _post(path: str, payload: dict) -> dict:
    """POST firmado al endpoint; lanza IdentityClientUnavailable si falla."""
    if not _enabled():
        raise IdentityClientUnavailable("Identidad deshabilitada o sin credenciales")

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
        logger.error("Identidad: error de red/tipo en %s: %s", path, type(e).__name__)
        raise IdentityClientUnavailable(str(e)) from e

    if resp.status_code >= 400:
        logger.warning(
            "Identidad: %s respondió %d: %s",
            path, resp.status_code, resp.text[:200],
        )
        raise IdentityClientUnavailable(f"HTTP {resp.status_code}")

    try:
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.error("Identidad: respuesta no-JSON en %s: %s", path, e)
        raise IdentityClientUnavailable("respuesta no-JSON") from e


def _extract_order(data: dict) -> dict | None:
    """Mapea `{ order: {...}|null, message? }` al dict del pedido."""
    order = (data or {}).get("order")
    return order if isinstance(order, dict) else None


async def get_cart(
    account_id: int, inbox_id: int | None, conversation_id: int
) -> dict | None:
    """Carrito activo del comprador de esa conversación de Chatwoot.

    Devuelve el dict serializado del pedido o `None` si no hay carrito, el
    comprador es anónimo (404), el endpoint no está configurado o falla.
    """
    try:
        data = await _post(
            _CART_PATH, _identity_payload(account_id, inbox_id, conversation_id)
        )
    except IdentityClientUnavailable:
        return None
    return _extract_order(data)


async def get_order(
    account_id: int, inbox_id: int | None, conversation_id: int, code: str
) -> dict | None:
    """Un pedido del comprador, por `code`.

    El backend exige `code` (400 sin él) y verifica la propiedad del pedido;
    un código ajeno y uno inexistente responden el mismo 200 con `order: null`.
    """
    if not code or not str(code).strip():
        return None
    payload = _identity_payload(account_id, inbox_id, conversation_id)
    payload["code"] = str(code).strip()

    try:
        data = await _post(_ORDER_PATH, payload)
    except IdentityClientUnavailable:
        return None
    return _extract_order(data)
