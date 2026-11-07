"""Autenticación por clave de API para los endpoints sensibles."""
import hmac
import logging

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


def _extract_key(x_api_key: str | None, authorization: str | None) -> str | None:
    """La clave puede venir en `X-API-Key` o como `Authorization: Bearer <clave>` (lo que ya envía eia-bot)."""
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Exige la clave solo si `EIA_API_KEY` está configurada.

    Con la variable vacía no cambia nada, así que se puede desplegar antes de activarla.
    La comparación es de tiempo constante y el mensaje de error no revela cuál falló.
    """
    expected = settings.EIA_API_KEY
    if not expected:
        return
    provided = _extract_key(x_api_key, authorization)
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        logger.warning("Acceso rechazado: clave de API ausente o inválida")
        raise HTTPException(status_code=401, detail="No autorizado")
