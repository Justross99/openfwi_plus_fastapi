import logging
from typing import List

import jax
import jax.numpy as jnp
from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache

from app.core.config import settings
from app.services import model_registry
from app.schemas.model import ModelInfo, ModelList, ModelTabulatedInfo
from app.routers.dependencies import get_model_assets

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/models",
    response_model=ModelList,
    summary="List Available Models",
    tags=["Model Information"],
    responses={200: {"description": "A list of available model names."}},
)
@cache(expire=settings.CACHE_EXPIRE_SECONDS)
async def list_models():
    """
    Retrieves a list of all model names that are configured and available for inference.
    These names can be used in other endpoints to get detailed information or run predictions.
    """
    return ModelList(models=list(model_registry.MODEL_REGISTRY.keys()))


@router.get(
    "/models/{model_name}",
    response_model=ModelInfo,
    summary="Get Model Details",
    tags=["Model Information"],
    responses={
        200: {"description": "Detailed configuration for the specified model."},
        404: {"description": "Model not found or artifacts are missing."},
    },
)
@cache(expire=settings.CACHE_EXPIRE_SECONDS)
async def get_model_info(
    model_name: str, model_assets: dict = Depends(get_model_assets)
):
    """
    Returns detailed configuration information for a specific model, including its
    description, expected input/output shapes, and other metadata.
    """
    config = model_assets["config"]
    return ModelInfo(
        model_name=model_name,
        model_configuration=config["model_config"],
        input_shape=tuple(config["input_shape"]),
        output_shape=tuple(config["output_shape_expected"]),
    )


@router.get(
    "/models/{model_name}/tabulate",
    response_model=ModelTabulatedInfo,
    summary="Get a Tabulated Summary of a Model's Architecture",
    description="""
    Retrieves a detailed, tabulated summary of a specific model's architecture,
    including layers, output shapes, and parameter counts, powered by `flax.linen.summary`.
    This endpoint is useful for developers and researchers who need to inspect the
    internal structure of a model. The response is cached to ensure fast subsequent lookups.
    """,
    tags=["Model Information"],
    responses={
        200: {"description": "A tabulated summary of the model's architecture."},
        404: {"description": "Model not found or artifacts are missing."},
        500: {"description": "Internal server error if summary generation fails."},
    },
)
@cache(expire=settings.CACHE_EXPIRE_SECONDS)
async def get_model_summary(
    model_name: str, model_assets: dict = Depends(get_model_assets)
):
    """
    Generates and returns a tabulated summary of the model's architecture.
    """
    try:
        model = model_assets["model"]
        config = model_assets["config"]
        input_shape = config["input_shape"]

        # Flax's `tabulate` method initializes the model with random keys
        # and traces a forward pass to generate the architecture summary.
        # It does not use the pre-loaded parameters, as its goal is to
        # describe the model's structure.

        # 1. Define the necessary random number generator keys for initialization.
        rng_keys = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }

        # 2. Create a dummy latent key required by the model's forward pass signature.
        dummy_latent_key = jax.random.PRNGKey(42)

        # 3. Generate the tabulated summary string.
        tabulated_string = model.tabulate(
            rngs=rng_keys,
            x=jnp.ones(input_shape),  # Dummy input tensor
            z_rng=dummy_latent_key,  # Dummy latent key
            training=False,
            compute_flops=True,
            compute_vjp_flops=True,
        )

        return ModelTabulatedInfo(
            model_name=model_name,
            tabulated_summary=tabulated_string,
            model_configuration=config["model_config"],
        )
    except Exception as e:
        logger.error(
            f"Failed to generate model summary for '{model_name}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate model summary. Ensure the model is correctly loaded and supports tabulation.",
        )


@router.get(
    "/models/details",
    response_model=List[ModelInfo],
    summary="Get All Model Details",
    tags=["Model Information"],
    responses={
        200: {
            "description": "A list of detailed configurations for all loadable models."
        },
        503: {"description": "Service unavailable because no models could be loaded."},
    },
)
@cache(expire=settings.CACHE_EXPIRE_SECONDS)
async def get_all_models_info():
    """
    Returns a list of detailed configurations for all available models.

    This endpoint attempts to load each configured model to retrieve its details.
    If a specific model fails to load, it will be omitted from the response.
    If no models can be loaded at all, it will return a 503 error.
    """
    models_details = []
    for model_name in model_registry.MODEL_REGISTRY.keys():
        try:
            # Manually call the dependency function for each model
            model_assets = get_model_assets(model_name)
            config = model_assets["config"]
            model_info = ModelInfo(
                model_name=model_name,
                model_configuration=config["model_config"],
                input_shape=tuple(config["input_shape"]),
                output_shape=tuple(config["output_shape_expected"]),
            )
            models_details.append(model_info)
        except HTTPException as e:
            logger.warning(
                f"Could not retrieve details for model '{model_name}': {e.detail}"
            )

    if not models_details:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: No valid models could be loaded.",
        )
    return models_details
