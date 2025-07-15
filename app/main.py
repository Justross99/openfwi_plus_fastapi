import logging
import time
import uuid
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_prometheus import PrometheusMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from app.core.api_metadata import api_description, tags_metadata
from app.core.exceptions import ModelInferenceError
from app.core.logging import setup_logging
from app.services.model_loader import load_model_from_artifacts
from app.routers import inference, models
from app.core.config import settings
from app.core.limiter import limiter  # Import the limiter instance

# --- Logging Setup ---
# Apply the structured logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# --- Global State ---
# This dictionary is deprecated. Use app.state for robust state management.
lifespan_context: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Actions to perform on application startup
    logger.info("Application startup: Initializing resources.")

    # Initialize Redis connection and cache
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    logger.info(f"Connecting to Redis at {redis_url} for caching.")
    try:
        redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await redis.ping()
        FastAPICache.init(RedisBackend(redis), prefix="/api/v1/cache")
        app.state.redis = redis  # Store redis client on app.state
        logger.info("Redis cache initialized successfully.")
    except Exception as e:
        logger.error(
            f"Failed to connect to Redis: {e}. Caching will be disabled.", exc_info=True
        )
        app.state.redis = None

    yield

    # Actions to perform on application shutdown
    logger.info("Application shutdown: Releasing resources.")
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.close()
        logger.info("Redis connection closed.")

    logger.info("Clearing model loader cache...")
    load_model_from_artifacts.cache_clear()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description=api_description,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    contact={
        "name": "Ross John",
        "email": "rossdjohn14@gmail.com",
    },
)


# --- Middleware ---
# Add Prometheus middleware to expose /metrics endpoint
app.add_middleware(PrometheusMiddleware)

# Add rate limiting state to the app
app.state.limiter = limiter


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add custom middleware for logging and request ID
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id  # Add request_id to state

        # Create a logger adapter to add request_id to all log records
        adapter = logging.LoggerAdapter(logger, {"request_id": request_id})

        adapter.info(f"Request started: {request.method} {request.url.path}")
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        adapter.info(
            f"Request finished in {process_time:.4f}s with status {response.status_code}"
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# Add custom middleware for security headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# --- Exception Handlers ---
# Rate limit exceeded handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors.
    """
    logger.warning(f"Rate limit exceeded for {request.client.host}: {exc.detail}")
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Custom exception handler for model inference errors
@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request: Request, exc: ModelInferenceError):
    request_id = request.headers.get("X-Request-ID", "N/A")
    logger.error(
        f"Model inference error for request {request_id}: {exc.message}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": f"Model inference error: {exc.message}"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "N/A")
    logger.error(
        f"Unhandled exception for request {request_id}: {exc}",
        exc_info=True,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal server error occurred.",
            "request_id": request_id,
        },
    )


# --- Routers ---
# Include the routers for different parts of the API
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(inference.router, prefix="/api/v1", tags=["Inference"])


# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message and a link to the API documentation.
    """
    return {
        "message": "Welcome to the OpenFWI Model Server API!",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the service and its dependencies are operational.
    Checks for:
    - Service availability
    - Model artifacts directory existence
    - Redis connectivity
    """
    # Check if the model artifacts directory exists
    if not os.path.isdir(settings.MODEL_ARTIFACTS_DIR):
        logger.error(
            f"Health check failed: Model artifacts directory not found at '{settings.MODEL_ARTIFACTS_DIR}'"
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "details": "Model artifacts directory not found.",
            },
        )

    # Check Redis connection
    redis_status = "ok"
    redis = None
    try:
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await redis.ping()
    except Exception as e:
        logger.error(f"Health check failed: Redis connection error: {e}")
        redis_status = "error"
    finally:
        if redis:
            await redis.close()

    if redis_status == "error":
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "details": "Redis connection failed.",
            },
        )

    return {
        "status": "ok",
        "details": "Service is healthy.",
        "dependencies": {"redis": redis_status, "model_artifacts": "ok"},
    }
