# app/services/model_loader.py
"""
Contains helper functions for loading model artifacts.
"""
import os
import logging
import json
import joblib
import functools
from typing import Dict, Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.core import FrozenDict

from model.full_model_defs.models import UnifiedVAE

# The following import is necessary for joblib to unpickle the scalers
from data.scalers import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_model_from_artifacts(artifact_dir: str) -> Dict[str, Any]:
    """Loads a model, its parameters, and scalers from a training artifact directory."""
    logger.info(f"Loading model artifacts from: {artifact_dir}")
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    # 1. Load training configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        train_config = json.load(f)

    model_config = train_config["model_config"]
    input_shape = train_config["input_shape"]

    # 2. Load scalers
    input_scaler_path = os.path.join(artifact_dir, "scalers", "input_scaler.joblib")
    label_scaler_path = os.path.join(artifact_dir, "scalers", "label_scaler.joblib")
    input_scaler: StandardScaler = joblib.load(input_scaler_path)
    label_scaler: MinMaxScaler = joblib.load(label_scaler_path)

    # 3. Initialize model
    model = UnifiedVAE(**model_config)

    # 4. Load checkpointed parameters
    # Point to the directory containing all checkpoints; Orbax will find the latest one.
    checkpoint_dir = os.path.join(artifact_dir, "checkpoints")
    checkpointer = ocp.PyTreeCheckpointer()

    # Create a dummy state to restore into. The optimizer state is not needed for inference.
    dummy_input_for_init = jnp.ones(input_shape, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    params_init_key, dropout_init_key, latent_init_key = jax.random.split(key, 3)
    init_rngs = {"params": params_init_key, "dropout": dropout_init_key}
    variables = model.init(
        init_rngs, dummy_input_for_init, latent_init_key, training=False
    )

    # Restore only the parameters and batch_stats
    restored_target = {
        "params": variables["params"],
        "batch_stats": variables.get("batch_stats", FrozenDict({})),
    }

    restored_checkpoint = checkpointer.restore(checkpoint_dir, item=restored_target)
    params = restored_checkpoint["params"]
    batch_stats = restored_checkpoint.get("batch_stats", FrozenDict({}))

    logger.info(f"Successfully loaded model and artifacts from {artifact_dir}")

    return {
        "model": model,
        "params": params,
        "batch_stats": batch_stats,
        "input_scaler": input_scaler,
        "label_scaler": label_scaler,
        "config": train_config,
    }
