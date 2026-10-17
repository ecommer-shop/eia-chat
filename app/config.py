import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Qdrant ---
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "COMPLETA")

    # --- Azure OpenAI (embeddings) ---
    AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-small")

    # --- Groq (solo eval offline; los LLM de producción van por Bifrost) ---
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_ROUTER_MODEL: str = os.getenv("GROQ_ROUTER_MODEL")
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL")

    # --- Bifrost (pasarela LLM propia, OpenAI-compatible) ---
    BIFROST_BASE_URL: str = os.getenv(
        "BIFROST_BASE_URL", "https://bifrost-stage.up.railway.app"
    )
    # VK de la plataforma Ecommer: SOLO para chatwoot account_id == 1.
    # Se acepta la variante exacta BIFROST_VK_Ecommer (case-sensitive en Linux).
    BIFROST_VK_ECOMMER: str | None = (
        os.getenv("BIFROST_VK_ECOMMER") or os.getenv("BIFROST_VK_Ecommer")
    )
    # Preferencias de modelo: la primera que la VK del plan permite es la usada.
    BIFROST_ROUTER_MODEL: str = os.getenv(
        "BIFROST_ROUTER_MODEL", "azure/Phi-4"
    )
    BIFROST_CHAT_MODEL: str = os.getenv("BIFROST_CHAT_MODEL", "azure/Phi-4")
    # Planes sin Phi-4 (free/tienda) usan el mini para respuesta y retry de 403.
    BIFROST_CHAT_MODEL_FALLBACK: str = os.getenv(
        "BIFROST_CHAT_MODEL_FALLBACK", "azure/Phi-4-mini-instruct"
    )
    BIFROST_TIMEOUT: float = float(os.getenv("BIFROST_TIMEOUT", "60"))
    # El router (Phi-4-mini) debe fallar rápido al keywords: si se toma su
    # presupuesto completo, la respuesta final ya no cabe en el timeout de
    # eia-bot (60s) y el usuario queda sin respuesta.
    BIFROST_ROUTER_TIMEOUT: float = float(os.getenv("BIFROST_ROUTER_TIMEOUT", "25"))
    # TTL del descubrimiento de modelos por VK (GET /v1/models).
    BIFROST_MODELS_CACHE_TTL: int = int(os.getenv("BIFROST_MODELS_CACHE_TTL", "30"))

    # --- Redis/Valkey (memory) ---
    REDIS_URL: str | None = os.getenv("REDIS_URL") or os.getenv("VALKEY_URL")

    # --- Retriever ---
    RETRIEVER_MIN_SCORE: float = float(os.getenv("RETRIEVER_MIN_SCORE", "0.4"))

    # --- CORS (orígenes separados por coma; p. ej. "https://a.com,https://b.com") ---
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]

    # --- Rate limiting (por IP, ventana deslizante de 1 minuto) ---
    # 0 desactiva el límite (útil para tests).
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # --- Langfuse (observabilidad / tracing) ---
    LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_BASE_URL: str | None = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )

    # --- Gateway ---
    PORT: int = int(os.getenv("PORT", "8080"))

    # --- PostgreSQL Vendure (read-only) ---
    PG_HOST: str = os.getenv("DB_HOST", "")
    PG_PORT: int = int(os.getenv("DB_PORT", "5432"))
    PG_DB: str = os.getenv("DB_NAME", "")
    PG_USER: str = os.getenv("DB_USERNAME", "")
    PG_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    PG_SSL: str = os.getenv("DB_SSL", "require")


settings = Settings()


def validate_settings() -> None:
    required = [
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "BIFROST_BASE_URL",
        "BIFROST_VK_ECOMMER",
    ]
    missing = [name for name in required if not getattr(settings, name, None)]
    if missing:
        raise ValueError(f"Faltan variables en .env: {', '.join(missing)}")
