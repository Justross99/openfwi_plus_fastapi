# tests/conftest.py
"""
Configuration and fixtures for the pytest test suite.
"""
import pytest
from typing import Generator, Any, AsyncGenerator
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

# Import the main FastAPI application instance
from app.main import app
from app.core.security import get_api_key


def pytest_configure(config):
    """Register a custom marker for tests that should not use the API key override."""
    config.addinivalue_line(
        "markers", "no_override: mark test to run without the API key override"
    )


@pytest.fixture(scope="session", autouse=True)
def initialize_cache():
    """Fixture to initialize the cache for the test session."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    yield
    FastAPICache.reset()


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, Any, None]:
    """
    Pytest fixture to provide a synchronous TestClient for the application.
    This is useful for simple, non-async tests.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, Any]:
    """
    Pytest fixture to provide an asynchronous AsyncClient for the application.
    This is essential for testing async endpoints and middleware.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def override_api_key_dependency(request):
    """
    Fixture to automatically override the API key dependency for all API tests,
    unless the test is marked with 'no_override'.
    """
    # Check if the test is part of the API test suite
    # and does not have the 'no_override' marker.
    is_api_test = "api" in str(request.node.path)
    if not is_api_test or "no_override" in request.keywords:
        yield
        return

    async def mock_get_api_key():
        """A mock function that bypasses the actual API key check."""
        return "test-key"

    # Override the dependency in the main app instance
    app.dependency_overrides[get_api_key] = mock_get_api_key
    yield
    # Clean up the override after the test runs
    app.dependency_overrides.clear()
