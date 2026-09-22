import os

# Desactiva el tracing de Langfuse durante tests.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

# Valores dummy para que `validate_settings()` del lifespan de la app pase
# sin depender de credenciales reales (.env no se usa en CI).
os.environ.setdefault("QDRANT_URL", "http://qdrant-test:6333")
os.environ.setdefault("QDRANT_API_KEY", "dummy")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "dummy")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GROQ_ROUTER_MODEL", "dummy-router")
os.environ.setdefault("GROQ_CHAT_MODEL", "dummy-chat")

# Desactiva el rate limiting en tests (cada suite hace muchas llamadas desde
# el mismo "client" de TestClient).
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"