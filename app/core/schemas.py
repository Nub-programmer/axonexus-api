from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class Message(BaseModel):
    role: str = Field(..., description="The role of the message author (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="The content of the message")


class ChatRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: datetime
