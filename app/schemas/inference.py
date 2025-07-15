# app/schemas/inference.py
"""
Pydantic models for API inference request and response validation.
"""
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class InferenceInput(BaseModel):
    """
    Pydantic model for validating the input data for an inference request.
    """

    # Expects a single data sample as a list of lists of lists
    data: List[List[List[float]]] = Field(
        ...,
        title="Input Seismic Data",
        description="A single seismic data sample, typically with dimensions (time_steps, num_receivers, channels).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    [[0.1], [0.2], [0.3]],
                    [[0.4], [0.5], [0.6]],
                    [[0.7], [0.8], [0.9]],
                ]
            }
        }
    )


class InferenceOutput(BaseModel):
    """
    Pydantic model for the response of a successful inference request.
    """

    output_map: List[List[float]] = Field(
        ..., description="The predicted velocity map."
    )
    model_name: str = Field(
        ..., description="The name of the model used for inference."
    )
    request_id: str = Field(..., description="The unique ID of the inference request.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "output_map": [
                    [2500.5, 2600.1, 2750.9],
                    [2800.0, 2950.2, 3100.3],
                ],
                "model_name": "example_model",
                "request_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            }
        }
    )
