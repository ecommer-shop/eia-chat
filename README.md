# EIA RAG Gateway

Unified RAG gateway for Ecommer — intent classification, vector search, conversational memory, and LLM generation.

## Architecture

```
User (Chatwoot) → POST /chat → Intent Classifier (Groq)
                                      ↓
                              Vector Search (Qdrant + Azure OpenAI embeddings)
                                      ↓
                              Memory Retrieval (Redis)
                                      ↓
                              LLM Generation (Groq gpt-oss-120b)
                                      ↓
                                ChatResponse
```

## Stack

| Component | Service |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Intent Classification | Groq (gpt-oss-20b, reasoning) |
| Embeddings | Azure OpenAI (text-embedding-3-small) |
| Vector Database | Qdrant |
| LLM Generation | Groq (gpt-oss-120b, reasoning) |
| Conversational Memory | Redis |
| Package Manager | uv |
| Deployment | Railway (Docker) |

## Project Structure

```
eia-rag/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── config.py            # Environment variables
│   ├── schemas.py           # Pydantic models
│   ├── store_resolver.py    # Multi-tenant store config
│   ├── intent_classifier.py # LLM-based intent detection
│   ├── retriever.py         # Qdrant vector search
│   ├── memory.py            # Redis chat history
│   ├── llm_generator.py     # Prompt builder
│   └── templates/
│       └── prompts.py       # System prompt template
├── tests/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── railway.json
├── .env.example
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USER/eia-rag.git
cd eia-rag

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
uv sync

# Run development server
uv run dev
```

The API will be available at `http://localhost:8000`.

### Production (Docker)

```bash
# Build image
docker build -t eia-rag .

# Run container
docker run -p 8000:8000 --env-file .env eia-rag
```

## API Endpoints

### `POST /chat`

Main RAG endpoint.

**Request:**

```json
{
  "query": "Tienen envíos a todo el país?",
  "conversation_id": "cw_12345",
  "inbox_id": 2,
  "user_id": 67890,
  "channel": "whatsapp"
}
```

**Response:**

```json
{
  "answer": "Sí, realizamos envíos a todo el país...",
  "intent_detected": "POLITICAS",
  "sources_used": 3,
  "conversation_id": "cw_12345"
}
```

### `GET /health`

Health check.

```json
{
  "status": "ok",
  "service": "EIA RAG Gateway",
  "collection": "COMPLETA"
}
```

## Multi-Tenant Support

Each store is identified by `inbox_id` (Chatwoot inbox). See `app/store_resolver.py` for the full mapping.

| inbox_id | Store | Channel |
|---|---|---|
| 2 | ecommer | whatsapp |
| 3 | ecommer | shop |
| 4 | ecommer | admin |
| 5 | ecommer | instagram |
| 6 | ecommer | messenger |
| 10 | sol-y-luna | whatsapp |
| 11 | sol-y-luna | instagram |
| 12 | sol-y-luna | messenger |
| 13 | sol-y-luna | shop |
| 14 | sol-y-luna | admin |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | Yes | Qdrant cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `COLLECTION_NAME` | No | Qdrant collection (default: `COMPLETA`) |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | No | Embedding model (default: `text-embedding-3-small`) |
| `GROQ_API_KEY` | Yes | Groq API key |
| `GROQ_ROUTER_MODEL` | Yes | Router model (e.g. `openai/gpt-oss-20b`). No default since 2026-08-29. Reasoning model → use generous max_tokens (256) |
| `GROQ_CHAT_MODEL` | Yes | Generation model (e.g. `openai/gpt-oss-120b`). No default since 2026-08-29. Reasoning model → use generous max_tokens (1024) |
| `REDIS_URL` | Yes | Redis connection URL |
| `PORT` | No | Server port (default: `8080`; `8000` via `uv run dev`) |

## Deployment to Railway

1. Push this repo to GitHub
2. Create a new project in Railway
3. Connect your GitHub repo
4. Railway will auto-detect the `Dockerfile`
5. Add environment variables in Railway dashboard (Settings → Variables)
6. Deploy

Health check is configured at `/health`.

## License

Proprietary — Ecommer SAS
