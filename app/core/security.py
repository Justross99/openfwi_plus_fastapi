import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
logger = logging.getLogger(__name__)


async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency that verifies the X-API-Key header against the secret key.

    Args:
        api_key: The API key passed in the request header.

    Raises:
        HTTPException: If the API key is missing or invalid.

    Returns:
        The validated API key if it is correct.
    """
    if not api_key or api_key != settings.SECRET_API_KEY:
        logger.warning("Failed API key validation attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
