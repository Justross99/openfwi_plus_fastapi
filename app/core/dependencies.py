"""
Shared dependencies for the FastAPI application.
"""

import logging
from redis import asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_redis_client():
    """
    Provides a Redis client connection.
    This dependency is used for caching and other Redis-based operations.
    """
    try:
        redis = await aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            encoding="utf-8",
            decode_responses=True,
        )
        yield redis
    finally:
        if "redis" in locals() and redis:
            await redis.close()
            logger.info("Redis connection closed.")
