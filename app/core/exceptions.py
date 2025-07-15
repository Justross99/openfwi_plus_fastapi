# app/core/exceptions.py
"""
Custom exception classes for the FastAPI application.
"""


class ModelInferenceError(Exception):
    """Base exception for errors during model inference."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InputValidationError(ModelInferenceError):
    """Exception raised for errors in the input data validation."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ModelLoadError(ModelInferenceError):
    """Exception raised when a model or its artifacts cannot be loaded."""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class ModelNotFoundError(ModelInferenceError):
    """Exception raised when a model or its assets are not found."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ModelNotConfiguredError(ModelInferenceError):
    """Exception raised when a model is registered but not configured."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)
