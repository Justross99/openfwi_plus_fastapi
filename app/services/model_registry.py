# app/services/model_registry.py
"""
Dynamically discovers and registers available models for the application.

This module scans the directory specified by the `MODEL_ARTIFACTS_DIR`
environment variable. It identifies valid model subdirectories and makes them
available to the application through a central registry.
"""

import os
import logging
from typing import Dict

from app.core.config import settings

logger = logging.getLogger(__name__)


def discover_models() -> Dict[str, str]:
    """
    Scans the `MODEL_ARTIFACTS_DIR` to dynamically discover available models.

    Each subdirectory in the artifacts directory is considered a model. The name
    of the subdirectory is used as the model name. A model is considered valid
    only if it contains the required artifact files (`model.pkl`, `scaler.pkl`, `config.json`).

    Returns:
        A dictionary mapping model names to their absolute artifact paths.
        Returns an empty dictionary if the directory doesn't exist or is empty.
    """
    artifacts_dir = settings.MODEL_ARTIFACTS_DIR
    if not os.path.isdir(artifacts_dir):
        logger.error(
            f"Model artifacts directory not found at: {artifacts_dir}. "
            "Please set the MODEL_ARTIFACTS_DIR environment variable correctly."
        )
        return {}

    model_paths = {}
    for model_name in os.listdir(artifacts_dir):
        model_path = os.path.join(artifacts_dir, model_name)
        # Ensure it's a directory and not a file like '.gitkeep'
        if os.path.isdir(model_path):
            # Check for essential files before adding the model
            required_files = ["model.pkl", "scaler.pkl", "config.json"]
            if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                model_paths[model_name] = model_path
                logger.info(f"Discovered model '{model_name}' at {model_path}")
            else:
                logger.warning(
                    f"Skipping directory '{model_name}' as it lacks one or more required artifacts: {required_files}"
                )

    if not model_paths:
        logger.warning(f"No valid models discovered in {artifacts_dir}.")

    return model_paths


# MODEL_REGISTRY is dynamically populated by scanning the artifacts directory.
# This dictionary serves as the single source of truth for what models are available.
MODEL_REGISTRY = discover_models()
