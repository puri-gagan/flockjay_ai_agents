"""x-api-key gate for protected routes.
"""

from __future__ import annotations

from secrets import compare_digest

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.settings import settings

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency — raises 401 if the ``x-api-key`` header is missing
    or doesn't match the configured ``API_KEY``."""
    if not provided or not compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x-api-key",
        )
