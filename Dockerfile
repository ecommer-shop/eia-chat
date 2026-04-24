# Usa una imagen base de Python 3.13 slim
FROM python:3.13-slim AS runtime

# Instalamos uv copiando los binarios desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Variables de entorno para optimizar uv y python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Instalamos dependencias del sistema necesarias (gcc para algunas librerías si fuera necesario)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos solo los archivos de dependencias primero para aprovechar el cache de capas
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias (sin instalar el proyecto aún y sin dependencias de desarrollo)
RUN uv sync --frozen --no-install-project --no-dev

# Copiamos el resto de la aplicación
COPY . .

# Sincronizamos el proyecto
RUN uv sync --frozen --no-dev

# Colocamos el entorno virtual en el PATH para que uvicorn sea ejecutable directamente
ENV PATH="/app/.venv/bin:$PATH"

# Configuración de usuario no-root por seguridad
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app
RUN chown -R app:app /app

# Variables de entorno por defecto
ENV PORT=8000
ENV UVICORN_WORKERS=1

USER app

EXPOSE ${PORT}

# Ejecutamos con uvicorn directamente (ya está en el PATH gracias al venv de uv)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
