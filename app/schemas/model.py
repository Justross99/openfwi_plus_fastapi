# app/schemas/model.py
"""
Pydantic models for API model metadata responses.
"""
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


class ModelInfo(BaseModel):
    """
    Pydantic model for returning detailed information about a single model.
    """

    model_name: str = Field(..., description="The unique name of the model.")
    model_configuration: Dict[str, Any] = Field(
        ..., description="The internal configuration dictionary of the model."
    )
    input_shape: Tuple[int, ...] = Field(
        ..., description="The expected shape of a single input sample (N, H, W, C)."
    )
    output_shape: Tuple[int, ...] = Field(
        ..., description="The shape of the model's output velocity map."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "small_test_model",
                "model_configuration": {
                    "backbone": "UNet",
                    "latent_dim": 128,
                    "dropout_rate": 0.1,
                },
                "input_shape": (1, 64, 64, 1),
                "output_shape": (64, 64),
            }
        }
    )


class ModelList(BaseModel):
    """
    Pydantic model for listing all available models.
    """

    models: List[str] = Field(
        ..., description="A list of names of the available models."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"models": ["inversion_net_v1", "inversion_net_v2"]}
        }
    )


class ModelTabulatedInfo(BaseModel):
    """
    Pydantic model for returning a tabulated summary of a model's architecture.
    """

    model_name: str = Field(..., description="The unique name of the model.")
    tabulated_summary: str = Field(
        ..., description="A string containing the Flax model summary."
    )
    model_configuration: Dict[str, Any] = Field(
        {}, description="The configuration dictionary used to build the model."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "small_test_model",
                "tabulated_summary": (
                    "UNet Summary\n"
                    "==================================================================================================\n"
                    "Layer (type)                           Output Shape         Param #     FLOPs #      VJP FLOPs #\n"
                    "--------------------------------------------------------------------------------------------------\n"
                    "UNet                                   (1, 64, 64, 1)       --          --           --\n"
                    "    Encoder_0 (EncoderBlock)           (1, 32, 32, 64)      6,080       100,044,544  100,044,544\n"
                    "    Encoder_1 (EncoderBlock)           (1, 16, 16, 128)     221,568     113,508,352  113,508,352\n"
                    "    ... (layers omitted) ...\n"
                    "--------------------------------------------------------------------------------------------------\n"
                    "Total Parameters: 31,037,633\n"
                    "Total FLOPs: 21,855,932,416\n"
                    "Total VJP FLOPs: 21,855,932,416\n"
                    "=================================================================================================="
                ),
                "model_configuration": {
                    "encoder_name": "unet",
                    "encoder_config": {"num_features": [64, 128]},
                    "latent_dim": 256,
                    "decoder_config": {"num_features": [128, 64]},
                },
            }
        }
    )
