"""Typed settings loaded from environment / .env file.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Loaded from .env in the project root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    # GOOGLE_API_KEY (or GEMINI_API_KEY) is required: it's used both by the
    # Gemini embedder and by ADK whenever LLM_MODEL points at a Gemini model.
    google_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    # Provider keys for non-Gemini LLM_MODEL choices. LiteLLM reads these env vars directly when invoked.
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Flockjay MCP
    flockjay_mcp_url: str = Field(
        default="https://api-demo.flockjay.com/mcp", alias="FLOCKJAY_MCP_URL"
    )

    # API auth — required; callers must send `x-api-key: <api_key>` on /chat.
    api_key: str = Field(..., alias="API_KEY")

    # Server
    port: int = Field(default=8000, alias="PORT")

    # Observability (Opik)
    opik_api_key: str | None = Field(default=None, alias="OPIK_API_KEY")
    opik_workspace: str = Field(default="default", alias="OPIK_WORKSPACE")
    opik_project_name: str = Field(default="flockjay-agents", alias="OPIK_PROJECT_NAME")


settings = Settings()
