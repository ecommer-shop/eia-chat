"""Rate limiting por IP (ventana deslizante en memoria).

Simple y sin dependencias: limita el número de requests por cliente IP en una
ventana de 60 segundos. Aplica únicamente a los endpoints que disparan
llamadas LLM/embeddings (costosas): `/chat`, `/agent/chat` y `/stores/reload`.

Consideraciones:
- Es un límite por proceso (en memoria). Si se escala a N réplicas, cada una
  mantiene su propio contador; para límites globales estrictos habría que
  mover el contador a Redis.
- `_RESPECTED_METHODS`/rutas: solo los POST seleccionados para no penalizar
  health checks ni lecturas baratas.
"""

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_WINDOW_SECONDS = 60
_LIMITED_PATHS = {"/chat", "/agent/chat", "/stores/reload"}


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    """Mantiene contadores IP → timestamps en ventana deslizante."""

    def __init__(self, max_requests: int, window_seconds: int = _WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        if request.client is not None:
            return request.client.host
        return "unknown"

    def allow(self, request: Request) -> bool:
        if self.max_requests <= 0:
            return True
        key = self._client_key(request)
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] >= self.window_seconds:
            window.popleft()
        if len(window) >= self.max_requests:
            return False
        window.append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rechaza con 429 los requests que superan el límite configurado."""

    def __init__(self, app, max_requests: int):
        super().__init__(app)
        self._limiter = RateLimiter(max_requests=max_requests)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and request.url.path in _LIMITED_PATHS:
            if not self._limiter.allow(request):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Se superó el límite de solicitudes. Intenta de nuevo en un minuto."},
                )
        return await call_next(request)