import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    query: str = Field(description="Mensaje del usuario")
    conversation_id: str = Field(
        default_factory=lambda: f"shop-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
        description="ID de conversación",
    )
    inbox_id: int = Field(default=1, description="ID del inbox de Chatwoot (→ tienda + canal)")
    user_id: int | None = Field(None, description="ID del usuario en Chatwoot")
    channel: str | None = Field(None, description="Canal de origen: whatsapp, instagram, messenger, shop, admin")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_message_key(cls, data):
        if isinstance(data, dict):
            payload = dict(data)
            if "query" not in payload and "message" in payload:
                payload["query"] = payload["message"]
            return payload
        return data


class ChatResponse(BaseModel):
    answer: str
    intent_detected: str
    sources_used: int
    conversation_id: str


class AgentChatResponse(ChatResponse):
    escalado: bool = Field(default=False, description="True si la conversación fue escalada a humano")
    tools_used: list[str] = Field(default_factory=list, description="Herramientas invocadas por el agente")
