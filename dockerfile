# Stage 1: Builder
# This stage installs all the Python dependencies.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim as builder

WORKDIR /app

# Install gunicorn for production serving, in addition to app dependencies.
# Using a requirements file is often more explicit for Docker builds.
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system --no-cache -r pyproject.toml gunicorn

# Stage 2: Production
# This stage builds the final, lean image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Create a non-privileged user to run the application for security.
RUN useradd --create-home appuser
ENV HOME=/home/appuser

# Copy installed dependencies from the builder stage.
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Set the environment variable for the model artifacts directory inside the container.
# This path must match where the models are copied to below.
ENV MODEL_ARTIFACTS_DIR="/app/app/pretrained"
# Set PYTHONPATH to include the working directory for absolute imports.
ENV PYTHONPATH="/app"

# Copy the application source code.
# Note: The 'app/pretrained' directory is copied here.
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser data/ ./data/

# Switch to the non-root user.
USER appuser

# Expose the port the app will run on.
EXPOSE 8000

# Define the command to run the application using Gunicorn with Uvicorn workers.
# This is a production-ready command.
# -w 4: Spawns 4 worker processes.
# -k uvicorn.workers.UvicornWorker: Specifies the worker class to use.
# -b 0.0.0.0:8000: Binds the server to all available network interfaces on port 8000.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]

