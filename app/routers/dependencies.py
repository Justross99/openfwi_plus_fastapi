import logging
from typing import Any, Dict

from fastapi import HTTPException

from app.services.model_registry import MODEL_REGISTRY
from app.services.model_loader import load_model_from_artifacts

logger = logging.getLogger(__name__)


def get_model_assets(model_name: str) -> Dict[str, Any]:
    """
    Loads and returns the assets for a given model name, utilizing a cache.

    This function acts as a gateway to the cached `load_model_from_artifacts`.
    It ensures a model is loaded from disk only once and raises appropriate
    HTTP exceptions for the API layer.

    Args:
        model_name: The name of the model to load.

    Returns:
        A dictionary containing the loaded model assets.

    Raises:
        HTTPException: If the model_name is not found in the configuration
                       or if the artifacts fail to load.
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not configured."
        )

    artifact_path = MODEL_REGISTRY[model_name]
    try:
        # This call is cached via @functools.lru_cache in model_loader.py
        model_assets = load_model_from_artifacts(artifact_path)
        return model_assets
    except FileNotFoundError as e:
        logger.error(
            f"Artifacts not found for model '{model_name}' at path '{artifact_path}': {e}"
        )
        raise HTTPException(
            status_code=404, detail=f"Artifacts for model '{model_name}' not found."
        )
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not load model '{model_name}'."
        )
