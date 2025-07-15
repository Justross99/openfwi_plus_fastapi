# tests/test_main.py
"""
Unit tests for the main FastAPI application endpoints (e.g., /health).
"""
import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
@patch("app.main.settings")
@patch("os.path.isdir")
@patch("redis.asyncio.from_url")
async def test_health_check_healthy(
    mock_from_url, mock_isdir, mock_settings, async_client: AsyncClient
):
    """
    Tests the /health endpoint when all dependencies are healthy.
    """
    # Mock the dependencies to be in a healthy state
    mock_settings.MODEL_ARTIFACTS_DIR = "/fake/dir"
    mock_isdir.return_value = True

    # Mock the redis client and its async methods
    mock_redis_client = MagicMock()
    mock_redis_client.ping = AsyncMock(return_value=True)
    mock_redis_client.close = AsyncMock()

    # The from_url function itself is awaited, so it needs to be an AsyncMock
    # that resolves to our client mock.
    mock_from_url.return_value = mock_redis_client

    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "details": "Service is healthy.",
        "dependencies": {"redis": "ok", "model_artifacts": "ok"},
    }


@pytest.mark.asyncio
@patch("app.main.settings")
@patch("os.path.isdir")
@patch("redis.asyncio.from_url")
async def test_health_check_unhealthy_redis(
    mock_from_url, mock_isdir, mock_settings, async_client: AsyncClient
):
    """
    Tests the /health endpoint when the Redis connection is down.
    """
    mock_settings.MODEL_ARTIFACTS_DIR = "/fake/dir"
    mock_isdir.return_value = True

    # Simulate Redis connection failure by having from_url raise an exception
    mock_from_url.side_effect = ConnectionError("Redis is down")

    response = await async_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "details": "Redis connection failed.",
    }


@pytest.mark.asyncio
@patch("app.main.settings")
@patch("os.path.isdir")
@patch("redis.asyncio.from_url")
async def test_health_check_unhealthy_model_dir(
    mock_from_url, mock_isdir, mock_settings, async_client: AsyncClient
):
    """
    Tests the /health endpoint when the model artifacts directory is inaccessible.
    """
    mock_settings.MODEL_ARTIFACTS_DIR = "/non_existent_dir"
    mock_isdir.return_value = False  # Simulate directory not found

    # Mock the redis client to be healthy
    mock_redis_client = MagicMock()
    mock_redis_client.ping = AsyncMock(return_value=True)
    mock_redis_client.close = AsyncMock()
    mock_from_url.return_value = mock_redis_client

    response = await async_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "details": "Model artifacts directory not found.",
    }


@pytest.mark.asyncio
async def test_read_root(async_client: AsyncClient):
    """
    Tests the root endpoint to ensure it returns the correct welcome message.
    """
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the OpenFWI Model Server API!",
        "docs_url": "/docs",
    }
