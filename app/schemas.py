"""Pydantic models defining the small public HTTP contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def trim_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le contenu ne peut pas être vide.")
        return value


class RouteDecision(BaseModel):
    """The first LLM call may route, but never answer a legal question."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["legal", "conversational", "out_of_scope"]
    reply: str = Field(default="", max_length=1000)
    search_query: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_route_payload(self) -> "RouteDecision":
        if self.intent == "legal" and not self.search_query.strip():
            raise ValueError("A legal route requires a search query.")
        if self.intent != "legal" and not self.reply.strip():
            raise ValueError("A non-legal route requires a reply.")
        return self


class GeneratedAnswer(BaseModel):
    """Validated structured output of the grounded answer LLM call."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=8000)
    used_chunk_ids: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("answer")
    @classmethod
    def trim_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The answer cannot be empty.")
        return value

    @field_validator("used_chunk_ids")
    @classmethod
    def normalize_chunk_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le message ne peut pas être vide.")
        return value


class SourceCard(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str = ""
    category: str = ""
    content_type: str = ""
    article_number: str = ""
    creation_date: str = ""
    score: float
    excerpt: str


class ChatMetrics(BaseModel):
    retrieved: int
    retained: int
    rewritten: bool
    embedding_ms: int
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class ChatResponse(BaseModel):
    answer: str
    standalone_query: str
    sources: list[SourceCard]
    metrics: ChatMetrics


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str
    embedding_model: str
    collection: str
    points: int
    llm_model: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str
