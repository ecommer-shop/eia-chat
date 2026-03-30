FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /src

# Install build deps only in builder (minimize runtime image size)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest(s) first to leverage docker layer cache
COPY requirements.txt ./

# Upgrade pip and install dependencies into /install (isolated target)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --target=/install -r requirements.txt


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Copy installed packages from builder into site-packages
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application source (use .dockerignore to keep context small)
COPY . /app

# Ensure app directory is owned by non-root user
RUN chown -R app:app /app

# Default environment placeholders for configuration injection
ENV PORT=8000
ENV RAG_API_URL=""
ENV VECTOR_DB_URL=""
ENV OTHER_API_URL=""

# Switch to non-root user
USER app

# Railway provides a PORT env var; use it at runtime. Expose for documentation.
EXPOSE ${PORT}

# Entrypoint: use sh -c to allow environment variable expansion for workers
# Tune workers via UVICORN_WORKERS env var (set in Railway to scale horizontally)
# Cambiamos "uvicorn" por "python -m uvicorn"
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001} --workers ${UVICORN_WORKERS:-1}"]