from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    intent_detected: str
    sources_used: int