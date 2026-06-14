from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

# Canales soportados
ChannelType = Literal["web", "whatsapp", "instagram"]

class ChatRequest(BaseModel):
    query: str = Field(..., description="Mensaje del usuario")
    tenant_id: str = Field(..., description="UUID del tenant (tienda)")
    channel: ChannelType = Field(..., description="Canal de origen: web, whatsapp, instagram")
    external_id: str = Field(
        ...,
        description="ID único del usuario en ese canal. Ej: número de cel, ID de Instagram, o session UUID para web"
    )
    contact_name: Optional[str] = Field(None, description="Nombre del contacto (opcional)")

class ChatResponse(BaseModel):
    answer: str
    intent: str
    channel: str
    conversation_id: str
    contact_id: str