# EIA Chat Gateway

Unified RAG gateway for Ecommer — intent classification, vector search, conversational memory, and LLM generation.

## Architecture

```
User (Chatwoot) → POST /agent/chat → Intent Classifier (Bifrost, Phi-4-mini)
                                      ↓
                              Vector Search (Qdrant + Azure OpenAI embeddings)
                                      ↓
                              Memory Retrieval (Redis)
                                      ↓
                              LLM Generation (Bifrost, Phi-4 según el plan)
                                      ↓
                                ChatResponse
```

## Stack

| Component | Service |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Intent Classification | Bifrost (`azure/Phi-4-mini-instruct`, VK por tienda) |
| Embeddings | Azure OpenAI (text-embedding-3-small) |
| Vector Database | Qdrant |
| LLM Generation | Bifrost (`azure/Phi-4` si el plan lo permite; si no `Phi-4-mini-instruct`) |
| Conversational Memory | Redis |
| Package Manager | uv |
| Deployment | Railway (Docker) |

## Project Structure

```
eia-chat/
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
git clone https://github.com/ecommer-shop/eia-chat.git
cd eia-chat

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
docker build -t eia-chat .

# Run container
docker run -p 8000:8000 --env-file .env eia-chat
```

## API Endpoints

### `POST /agent/chat`

Main RAG endpoint (`/chat` was removed).

**Request:**

```json
{
  "query": "Tienen envíos a todo el país?",
  "conversation_id": "cw_12345",
  "account_id": 1,
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

Each store is identified by `account_id` (Chatwoot account). The `inbox_id` only selects the channel (prompt) within the store. Store data (name, channel token, language, Bifrost VK) is loaded read-only from the Vendure database (`app/store_repository.py`, 30 s cache, reload via `POST /stores/reload`); system prompts live in `app/prompts/{store_name}.json`. Use `GET /stores` for the live mapping.

Stores without an active Bifrost key respond `503` (no shared/pooled keys).

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | Yes | Qdrant cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `COLLECTION_NAME` | No | Qdrant collection (default: `COMPLETA`) |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | No | Embedding model (default: `text-embedding-3-small`) |
| `BIFROST_BASE_URL` | Yes | Bifrost gateway URL (OpenAI-compatible). E.g. `https://bifrost-stage.up.railway.app` |
| `BIFROST_VK_ECOMMER` | Yes | Bifrost VK of the global `ecommer` store (account_id 1). Other stores resolve their VK from the database |
| `BIFROST_ROUTER_MODEL` | No | Intent classifier model (default: `azure/Phi-4-mini-instruct`) |
| `BIFROST_CHAT_MODEL` | No | Preferred answer model (default: `azure/Phi-4`) |
| `BIFROST_CHAT_MODEL_FALLBACK` | No | Used when the plan blocks `BIFROST_CHAT_MODEL` (default: `azure/Phi-4-mini-instruct`) |
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
