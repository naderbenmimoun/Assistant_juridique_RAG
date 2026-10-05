"""Typed application configuration loaded from the environment."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration contract shared by Docker Compose and the application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    hf_dataset: str = "AgentPublic/dole"
    hf_dataset_split: str = "train"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "dole_bge_m3_v1"
    vector_size: int = Field(default=1024, gt=0)
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_embedding_model: str = "bge-m3"
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_num_ctx: int = Field(default=8192, gt=0)
    ollama_num_predict: int = Field(default=768, gt=0)
    retrieval_max_k: int = Field(default=5, ge=1, le=5)
    retrieval_min_score: float = Field(default=0.40, ge=-1.0, le=1.0)
    retrieval_gap_cutoff: float = Field(default=0.04, ge=0.0, le=2.0)
    max_history_turns: int = Field(default=4, ge=0, le=20)
    max_question_chars: int = Field(default=2000, ge=1, le=10000)
    max_context_chars: int = Field(default=22000, ge=100)
    auto_ingest: bool = True

    @field_validator("ollama_chat_model", "ollama_embedding_model", "qdrant_collection")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
