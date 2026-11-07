"""Autenticación por clave de API para los endpoints sensibles."""
import hmac
import logging

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Exige `X-API-Key` solo si `EIA_API_KEY` está configurada.

    Con la variable vacía no cambia nada, así que se puede desplegar antes de activarla.
    La comparación es de tiempo constante y el mensaje de error no revela cuál falló.
    """
    expected = settings.EIA_API_KEY
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        logger.warning("Acceso rechazado: X-API-Key ausente o inválida")
        raise HTTPException(status_code=401, detail="No autorizado")
