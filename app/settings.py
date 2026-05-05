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

    # API keys — each provider's key is optional at the schema level. The
    # embedder factory and ADK's LLM resolver instantiate clients lazily;
    # missing keys surface as a clear SDK error at startup. Whichever
    # provider(s) your configured LLM_MODEL and EMBEDDING_MODEL target must
    # have its corresponding key present.
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
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
