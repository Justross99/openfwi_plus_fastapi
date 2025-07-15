# tests/test_routers_models.py
"""
Unit tests for the model information router (`app/routers/models.py`).
"""
import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.routers.dependencies import get_model_assets
from app.core.exceptions import ModelNotConfiguredError

# A sample model configuration to be used in mocks
MOCK_MODEL_CONFIG = {
    "description": "A mock model for testing.",
    "input_shape": [1, 64, 64, 1],
    "output_shape_expected": [1, 64, 64, 1],
    "model_config": {
        "encoder_name": "unet",
        "encoder_config": {"some_param": 1},
        "latent_dim": 32,
        "decoder_config": {"some_other_param": 2},
    },
}


@pytest.fixture(autouse=True)
def override_dependencies():
    """
    Fixture to override global dependencies for the tests in this module.
    This uses app.dependency_overrides, which is the standard way for FastAPI tests.
    """
    # Mock the get_model_assets dependency
    mock_assets = {
        "model": MagicMock(),
        "config": MOCK_MODEL_CONFIG,
        "params": MagicMock(),
    }
    # The model mock needs a `tabulate` method for the tabulation test
    mock_assets["model"].tabulate.return_value = "This is the tabulated model summary."

    def mock_get_model_assets(model_name: str):
        if model_name == "non_existent_model":
            raise ModelNotConfiguredError(f"Model '{model_name}' not configured.")
        return mock_assets

    app.dependency_overrides[get_model_assets] = mock_get_model_assets
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@patch(
    "app.routers.models.model_registry.MODEL_REGISTRY",
    {"mock_model_1": MagicMock(), "mock_model_2": MagicMock()},
)
async def test_list_models(async_client: AsyncClient):
    """
    Tests the GET /models endpoint to ensure it lists all available models.
    """
    response = await async_client.get("/api/v1/models")
    assert response.status_code == 200
    json_response = response.json()
    assert "models" in json_response
    assert isinstance(json_response["models"], list)
    assert "mock_model_1" in json_response["models"]
    assert "mock_model_2" in json_response["models"]


@pytest.mark.asyncio
async def test_get_model_info_success(async_client: AsyncClient):
    """
    Tests the GET /models/{model_name} endpoint for a successful case.
    """
    model_name = "test_model"
    response = await async_client.get(f"/api/v1/models/{model_name}")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == model_name
    assert json_response["model_configuration"] == MOCK_MODEL_CONFIG["model_config"]
    assert json_response["input_shape"] == MOCK_MODEL_CONFIG["input_shape"]
    assert json_response["output_shape"] == MOCK_MODEL_CONFIG["output_shape_expected"]


@pytest.mark.asyncio
async def test_get_model_info_not_found(async_client: AsyncClient):
    """
    Tests the GET /models/{model_name} endpoint when the model is not found.
    """
    model_name = "non_existent_model"
    response = await async_client.get(f"/api/v1/models/{model_name}")

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Model inference error: Model '{model_name}' not configured."
    }


@pytest.mark.asyncio
async def test_get_model_tabulated_summary_success(async_client: AsyncClient):
    """
    Tests the GET /models/{model_name}/tabulate endpoint for a successful case.
    """
    model_name = "tabulated_model"
    tabulated_string = "This is the tabulated model summary."

    response = await async_client.get(f"/api/v1/models/{model_name}/tabulate")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["model_name"] == model_name
    assert json_response["tabulated_summary"] == tabulated_string
    assert json_response["model_configuration"] == MOCK_MODEL_CONFIG["model_config"]
    # Verify that the `tabulate` method was called correctly
    # We can access the mock from the dependency overrides
    mock_model = app.dependency_overrides[get_model_assets](model_name)["model"]
    mock_model.tabulate.assert_called_once()
