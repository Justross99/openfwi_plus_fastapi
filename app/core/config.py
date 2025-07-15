"""
Application-wide settings, loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import List


class Settings(BaseSettings):
    """
    Manages application settings using Pydantic, loading from environment variables.
    """

    APP_NAME: str = Field(
        "OpenFWI Model Server", description="The title of the application."
    )
    LOG_LEVEL: str = Field(
        "INFO", description="Logging level (e.g., DEBUG, INFO, WARNING)."
    )

    # The base directory where all model artifacts are stored.
    # Each subdirectory within this path is expected to be a model version,
    # containing the necessary 'model.pkl', 'scaler.pkl', and 'config.json' files.
    MODEL_ARTIFACTS_DIR: str = Field(
        ...,
        description="The absolute path to the directory containing model artifacts.",
    )

    # A secret, random string used to validate API requests.
    # This key must be present in the `X-API-Key` header of protected routes.
    SECRET_API_KEY: str = Field(
        ...,
        description="The secret key for API authentication.",
    )

    # Rate limit for protected endpoints.
    # Format: "requests/period" (e.g., "10/minute", "100/hour").
    API_RATE_LIMIT: str = Field(
        "10/minute",
        description="Rate limit for protected API endpoints.",
    )

    # The expiration time for cached API responses, in seconds.
    CACHE_EXPIRE_SECONDS: int = Field(
        3600,
        description="The default expiration time for cached API responses.",
    )

    # Redis configuration for caching
    REDIS_HOST: str = Field(
        "localhost",
        description="The hostname for the Redis server.",
    )
    REDIS_PORT: int = Field(6379, description="The port for the Redis server.")

    # --- Security ---
    API_KEY: str = "you-shall-not-pass"
    API_KEY_NAME: str = "X-API-Key"

    # --- CORS ---
    # A list of allowed origins. Use ["*"] for public access (less secure).
    # Example for production: ["https://your-frontend.com", "https://your-other-domain.org"]
    CORS_ALLOWED_ORIGINS: List[str] = ["*"]

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# Create a single, importable instance of the settings
settings = Settings()
