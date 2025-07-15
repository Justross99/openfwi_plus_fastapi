import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, Depends, Request, HTTPException

from app.core.exceptions import InputValidationError, ModelInferenceError
from app.schemas.inference import InferenceInput, InferenceOutput
from app.core.security import get_api_key
from app.core.limiter import limiter
from .dependencies import get_model_assets

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predict/{model_name}",
    response_model=InferenceOutput,
    summary="Perform Inference",
    tags=["Inference"],
    responses={
        200: {"description": "Inference successful, returning the velocity map."},
        401: {"description": "Unauthorized. Invalid or missing API Key."},
        429: {"description": "Too Many Requests. Rate limit exceeded."},
        400: {"description": "Bad Request: Invalid input data, shape, or channels."},
        404: {"description": "Model not found or its artifacts are missing."},
        500: {"description": "Internal Server Error during model inference."},
    },
    dependencies=[Depends(get_api_key)],
)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    model_name: str,
    input_data: InferenceInput,
    model_assets: dict = Depends(get_model_assets),
):
    """
    Performs inference using a specified trained model to generate a velocity map.

    **Input Data Handling:**
    - The input is a nested list representing the seismic data.
    - The API automatically handles adding a batch dimension.
    - **Automatic Resizing**: If the spatial dimensions (e.g., height and width)
      of the input data do not match the model's expected input shape, the API
      will automatically resize the data using bilinear interpolation. A warning
      will be logged.
    - **Channel Mismatch**: If the number of channels in the input does not match
      the model's expectation, the request will be rejected with a 400 error.

    The endpoint returns a nested list containing the model's output velocity map,
    which has been inverse-scaled to its original physical units.
    """
    if model_assets is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    try:
        model = model_assets["model"]
        params = model_assets["params"]
        batch_stats = model_assets["batch_stats"]
        input_scaler = model_assets["input_scaler"]
        label_scaler = model_assets["label_scaler"]
        config = model_assets["config"]

        # 1. Preprocess input data
        np_data = np.array(input_data.data, dtype=np.float32)
        expected_shape_no_batch = tuple(config["input_shape"][1:])

        # --- Input Validation and Reshaping ---

        # Check for channel mismatch (assuming channels are the last dimension)
        if np_data.shape[-1] != expected_shape_no_batch[-1]:
            raise InputValidationError(
                f"Channel mismatch. Model '{model_name}' expects {expected_shape_no_batch[-1]} channels, "
                f"but input has {np_data.shape[-1]} channels."
            )

        # Check for spatial dimension mismatch and resize if necessary
        input_spatial_shape = np_data.shape[:-1]
        expected_spatial_shape = expected_shape_no_batch[:-1]

        if input_spatial_shape != expected_spatial_shape:
            logger.warning(
                f"Input spatial dimensions {input_spatial_shape} do not match model's "
                f"expected {expected_spatial_shape}. Resizing input..."
            )
            np_data_batched = jnp.expand_dims(np_data, axis=0)
            resized_data = jax.image.resize(
                np_data_batched,
                shape=(1, *expected_spatial_shape, np_data.shape[-1]),
                method="bilinear",
            )
            np_data = np.array(resized_data[0])

        original_shape = np_data.shape
        np_data_reshaped = np_data.reshape(-1, original_shape[-1])
        scaled_data = input_scaler.transform(np_data_reshaped)
        scaled_data_reshaped = scaled_data.reshape(original_shape)

        jax_data = jnp.expand_dims(scaled_data_reshaped, axis=0)

        expected_input_shape = tuple(config["input_shape"])
        if jax_data.shape != expected_input_shape:
            raise ModelInferenceError(
                f"Internal error: Invalid input shape after processing. Expected {expected_input_shape}, got {jax_data.shape}."
            )

        # 2. Perform inference
        rng_key = jax.random.PRNGKey(int(time.time()))
        variables = {"params": params, "batch_stats": batch_stats}
        output_map, _, _ = model.apply(variables, jax_data, rng_key, training=False)

        # 3. Postprocess output
        output_map_np = np.array(output_map[0])
        output_original_shape = output_map_np.shape
        output_reshaped = output_map_np.reshape(-1, output_original_shape[-1])
        inversed_output = label_scaler.inverse_transform(output_reshaped)
        final_output = inversed_output.reshape(output_original_shape)

        # Squeeze the last dimension if it's 1, to match the 2D output schema
        if final_output.shape[-1] == 1:
            final_output = final_output.squeeze(axis=-1)

        return InferenceOutput(
            output_map=final_output.tolist(),
            model_name=model_name,
            request_id=request.state.request_id,
        )

    except (InputValidationError, ModelInferenceError) as e:
        raise e
    except Exception as e:
        logger.error(
            f"Unhandled exception during prediction for model '{model_name}': {e}",
            exc_info=True,
        )
        raise ModelInferenceError(
            "An unexpected internal server error occurred during inference."
        )
