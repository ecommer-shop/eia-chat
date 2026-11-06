"""Cliente HTTP del asistente hacia los endpoints de identidad del backend.

Fase 1b (T2) — consume `POST /api/assistant/identity/cart` y
`POST /api/assistant/identity/order` implementados en `Shop/shop`
(plugin `assistant-identity`). Es un espejo deliberado de
`app/catalog_client.py`: mismo backend, mismo secreto y misma firma HMAC,
solo cambia el path y el shape de la respuesta.

`POST /api/assistant/identity/place-order` (mismo plugin) crea el pago del
carrito activo. Su contrato es en dos tiempos: sin `confirm=true` devuelve un
preview (`status="confirmation_required"`) con el total exacto y sin cobrar;
con `confirm=true` ejecuta el pago y devuelve `placed` o `awaiting_payment`.
Por eso `place_order` no reutiliza `_post` para el camino de error: necesita
distinguir un 200 normal (que el agente interpreta) del fallo accionable del
backend (400 por pago/transición, 404 si el carrito desapareció).

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
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_CART_PATH = "/api/assistant/identity/cart"
_ORDER_PATH = "/api/assistant/identity/order"
_PLACE_ORDER_PATH = "/api/assistant/identity/place-order"

# `paymentMethodCode` es el `code` del PaymentMethod de Vendure: corto y
# estable ("wompi"). El backend rechaza con 400 un valor vacío o > 64 chars,
# así que se valida antes de salir a la red.
_MAX_PAYMENT_METHOD_CODE_LEN = 64


class IdentityClientUnavailable(Exception):
    """El endpoint de identidad no está disponible o no está configurado."""


@dataclass
class PlaceOrderResult:
    """Resultado interpretado de `place-order` (nunca lanza).

    `status` es el del backend (`confirmation_required`, `placed`,
    `awaiting_payment`) o `error`, que es la degradación cordial del cliente:
    el agente recibe siempre un objeto, no una excepción. En el camino
    `error` `order` es `None` salvo que el backend haya devuelto igual un
    preview utilizable, y `message` trae el texto del backend cuando lo hubo.
    """

    status: str
    confirmed: bool
    order: dict | None = None
    message: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "error"

    @property
    def is_confirmation_required(self) -> bool:
        return self.status == "confirmation_required"

    @property
    def is_placed(self) -> bool:
        return self.status == "placed"

    @property
    def is_awaiting_payment(self) -> bool:
        return self.status == "awaiting_payment"


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
    data, status_code = await _post_with_status(path, payload)
    if status_code >= 400:
        logger.warning(
            "Identidad: %s respondió %d: %s",
            path, status_code, str(data)[:200],
        )
        raise IdentityClientUnavailable(f"HTTP {status_code}")
    return data


async def _post_with_status(path: str, payload: dict) -> tuple[dict | None, int]:
    """POST firmado que devuelve `(body_json, status_code)` sin interpretar.

    A diferencia de `_post`, no convierte los >= 400 en excepción: el caller
    decide (place-order necesita el status y el mensaje del backend para
    degradar con un texto útil). Sí lanza `IdentityClientUnavailable` ante
    errores de red o de transporte, donde no hay nada que interpretar.
    """
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

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        data = None

    if data is not None and not isinstance(data, dict):
        data = None
    return data, resp.status_code


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


def _place_order_error(message: str | None = None, raw: dict | None = None) -> PlaceOrderResult:
    """Degradación cordial: `status="error"`, sin lanzar al agente."""
    return PlaceOrderResult(
        status="error",
        confirmed=False,
        order=_extract_order(raw or {}),
        message=message,
        raw=raw or {},
    )


async def place_order(
    account_id: int,
    inbox_id: int | None,
    conversation_id: int,
    payment_method_code: str,
    confirm: bool,
    payment_metadata: dict | None = None,
) -> PlaceOrderResult:
    """Crea el pago del carrito activo (preview o ejecución según `confirm`).

    `confirm=False` devuelve el preview (`confirmation_required`) con el total
    exacto y sin cobrar; `confirm=True` ejecuta el pago y devuelve `placed` o
    `awaiting_payment`. Sin carrito activo el backend responde 200 con
    `order: null` y un mensaje.

    Nunca lanza: cualquier fallo (flag apagado, credenciales, red, 400/404 del
    backend, respuesta no-JSON) se degrada a `PlaceOrderResult(status="error")`
    con el `message` del backend cuando existe, para que el agente pueda
    ofrecer una salida accionable en vez de un 500.
    """
    code = (payment_method_code or "").strip()
    if not code:
        logger.info("place_order: sin paymentMethodCode — no se llama al backend")
        return _place_order_error("Método de pago no especificado.")
    if len(code) > _MAX_PAYMENT_METHOD_CODE_LEN:
        logger.warning("place_order: paymentMethodCode demasiado largo (%d)", len(code))
        return _place_order_error("Método de pago inválido.")

    payload = _identity_payload(account_id, inbox_id, conversation_id)
    payload["paymentMethodCode"] = code
    payload["confirm"] = bool(confirm)
    if isinstance(payment_metadata, dict) and payment_metadata:
        payload["paymentMetadata"] = payment_metadata

    try:
        data, status_code = await _post_with_status(_PLACE_ORDER_PATH, payload)
    except IdentityClientUnavailable as e:
        logger.info("place_order: cliente no disponible (%s)", e)
        return _place_order_error()
    except Exception as e:  # noqa: BLE001
        logger.error("place_order: fallo inesperado del cliente: %s", e)
        return _place_order_error()

    if status_code >= 400:
        message = None
        if isinstance(data, dict):
            raw_message = data.get("message")
            if isinstance(raw_message, str) and raw_message.strip():
                message = raw_message.strip()
        logger.warning(
            "Identidad: %s respondió %d (confirm=%s): %s",
            _PLACE_ORDER_PATH, status_code, bool(confirm), (message or "")[:200],
        )
        return _place_order_error(message, data)

    if not isinstance(data, dict):
        logger.error("Identidad: respuesta no-JSON en %s", _PLACE_ORDER_PATH)
        return _place_order_error()

    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        # Un 200 sin `status` no es interpretable: se trata como preview para
        # no dar una compra por hecha sin evidencia del backend.
        logger.warning("place_order: 200 sin status — se asume preview")
        status = "confirmation_required"
    else:
        status = status.strip()

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        message = None
    else:
        message = message.strip()

    return PlaceOrderResult(
        status=status,
        confirmed=bool(data.get("confirmed")),
        order=_extract_order(data),
        message=message,
        raw=data,
    )
