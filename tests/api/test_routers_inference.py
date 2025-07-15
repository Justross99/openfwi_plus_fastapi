# tests/api/test_routers_inference.py
"""
Unit tests for the inference router (`app/routers/inference.py`).
"""
import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock
import numpy as np

from app.main import app
from app.routers.dependencies import get_model_assets


# A mock model that can be used in tests
class MockModel:
    """A mock model with a `predict` method for testing."""

    def apply(self, variables, data, rng_key=None, training=False):
        """Returns a dummy numpy array and mock latents."""
        # The mock now accepts the arguments passed by the router
        return np.random.rand(1, 64, 64, 1), None, None


# Mock model assets, similar to how it's done in test_routers_models.py
MOCK_MODEL_ASSETS = {
    "model": MockModel(),
    "config": {"input_shape": [1, 64, 64, 1]},
    "params": MagicMock(),
    "batch_stats": MagicMock(),
    "input_scaler": MagicMock(),
    "label_scaler": MagicMock(),
}

# Mock scalers that perform a no-op transform
MOCK_MODEL_ASSETS["input_scaler"].transform = lambda x: x
MOCK_MODEL_ASSETS["label_scaler"].inverse_transform = lambda x: x


@pytest.fixture(autouse=True)
def override_model_assets_dependency():
    """
    Overrides the get_model_assets dependency for all tests in this module
    to prevent real model loading and ensure consistent mock assets are used.
    """
    app.dependency_overrides[get_model_assets] = lambda model_name: MOCK_MODEL_ASSETS
    yield
    # Cleanup is handled by the autouse fixture in conftest.py which clears all overrides


@pytest.mark.asyncio
async def test_predict_success(
    async_client: AsyncClient,
):
    """
    Tests POST /predict for a successful prediction.
    """
    model_name = "test_model"
    # The mock model expects a 64x64x1 input
    input_data = np.random.rand(64, 64, 1).tolist()
    request_payload = {"data": input_data}

    response = await async_client.post(
        f"/api/v1/predict/{model_name}", json=request_payload
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == model_name
    assert "output_map" in json_response
    assert "request_id" in json_response
    # Check the shape of the output map (it's a list of lists)
    assert len(json_response["output_map"]) == 64
    assert len(json_response["output_map"][0]) == 64


@pytest.mark.asyncio
async def test_predict_model_not_found(async_client: AsyncClient):
    """
    Tests that the /predict endpoint returns a 404 Not Found error when the
    model or its assets are not found.
    """
    # Override the dependency for this specific test to simulate a model not found scenario
    app.dependency_overrides[get_model_assets] = lambda model_name: None

    model_name = "non_existent_model"
    input_data = np.random.rand(64, 64, 1).tolist()
    request_payload = {"data": input_data}

    response = await async_client.post(
        f"/api/v1/predict/{model_name}", json=request_payload
    )

    assert response.status_code == 404
    assert f"Model '{model_name}' not found" == response.json()["detail"]

    # Clean up the override after the test
    del app.dependency_overrides[get_model_assets]


@pytest.mark.asyncio
async def test_predict_rate_limit_exceeded(async_client: AsyncClient):
    """
    Tests that the /predict endpoint returns a 429 error when the rate limit is exceeded.
    """
    model_name = "test_model_rate_limited"
    input_data = np.random.rand(64, 64, 1).tolist()
    request_payload = {"data": input_data}

    # Make more requests than allowed by the rate limit (set to 10/min in router)
    # We send 11 requests to be sure to trigger the limit.
    for i in range(11):
        response = await async_client.post(
            f"/api/v1/predict/{model_name}", json=request_payload
        )
        if response.status_code == 429:
            break

    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]


@pytest.mark.no_override
@pytest.mark.asyncio
async def test_predict_unauthorized_missing_key(async_client: AsyncClient):
    """Tests POST /predict returns 401 if the X-API-Key header is missing."""
    model_name = "test_model"
    input_data = np.random.rand(64, 64, 1).tolist()
    request_payload = {"data": input_data}

    response = await async_client.post(
        f"/api/v1/predict/{model_name}", json=request_payload
    )

    assert response.status_code == 401
    # FastAPI returns this detail for a missing Security dependency
    assert response.json()["detail"] == "Invalid or missing API Key"


@pytest.mark.no_override
@pytest.mark.asyncio
async def test_predict_unauthorized_invalid_key(async_client: AsyncClient):
    """Tests POST /predict returns 401 if the X-API-Key is invalid."""
    model_name = "test_model"
    input_data = np.random.rand(64, 64, 1).tolist()
    request_payload = {"data": input_data}

    response = await async_client.post(
        f"/api/v1/predict/{model_name}",
        json=request_payload,
        headers={"X-API-Key": "invalid-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, error_message_part",
    [
        ({}, "Field required"),  # Missing data field
        ({"data": "not a list"}, "Input should be a valid list"),  # Wrong type
        (
            {"data": [[[["a"]]]]},
            "Input should be a valid number",
        ),  # Wrong inner type
        (
            {"data": [1, 2, 3]},
            "Input should be a valid list",
        ),  # Wrong dimension/structure
    ],
)
async def test_predict_invalid_input_validation(
    async_client: AsyncClient, payload: dict, error_message_part: str
):
    """
    Tests that the /predict endpoint returns a 422 Unprocessable Entity error
    for malformed request payloads, verifying Pydantic validation.
    """
    model_name = "test_model"

    response = await async_client.post(f"/api/v1/predict/{model_name}", json=payload)

    assert response.status_code == 422
    json_response = response.json()
    assert "detail" in json_response
    # FastAPI's validation error details are a list of dicts.
    # We check if our expected message part is in any of the error messages.
    assert any(error_message_part in error["msg"] for error in json_response["detail"])
