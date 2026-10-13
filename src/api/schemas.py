from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=10, ge=1, le=50)
    language: Literal["en", "it"] = "en"


class SourceResponse(BaseModel):
    chunk_id: str
    article_number: str
    article_title: str | None
    paragraph_number: str | None
    text: str
    source_url: str | None
    reranker_score: float | None


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str

class PostgresReadiness(BaseModel):
    status: Literal["ok", "unavailable"]
    chunks: int | None = None


class OllamaReadiness(BaseModel):
    status: Literal["ok", "unavailable"]
    model: str | None = None


class ReadinessServices(BaseModel):
    postgres: PostgresReadiness
    ollama: OllamaReadiness


class ReadinessResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    services: ReadinessServices