import os
import logging
import math
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import joblib
from flax.core import FrozenDict

from .utils import TrainStateWithBatchStats
from criterion.combined_loss import create_combined_loss_fn
from data.dataloader import create_hf_dataset_jax_wrapper
from data.scalers import StandardScaler, MinMaxScaler
from model.full_model_defs.models import UnifiedVAE

logger = logging.getLogger(__name__)


def setup_training(config: Dict[str, Any]) -> Tuple:
    """Initializes everything needed for training."""
    # Create and resolve output directory to be absolute
    output_dir = os.path.abspath(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize PRNG keys
    key = jax.random.PRNGKey(config["seed"])
    (
        params_init_key,
        dropout_init_key,
        latent_init_key,
        train_dropout_key_base,
        eval_dropout_key_base,
    ) = jax.random.split(key, 5)

    # --- Dataloaders and Scalers ---
    logger.info("Initializing dataloaders...")
    scaler_dir = os.path.join(output_dir, "scalers")
    os.makedirs(scaler_dir, exist_ok=True)
    input_scaler_path = os.path.join(scaler_dir, "input_scaler.joblib")
    label_scaler_path = os.path.join(scaler_dir, "label_scaler.joblib")

    input_scaler, label_scaler = None, None
    if os.path.exists(input_scaler_path) and os.path.exists(label_scaler_path):
        try:
            input_scaler: StandardScaler = joblib.load(input_scaler_path)
            label_scaler: MinMaxScaler = joblib.load(label_scaler_path)
            logger.info(f"Successfully loaded pre-fitted scalers from {scaler_dir}")
        except Exception as e:
            logger.warning(
                f"Failed to load scalers from {scaler_dir}. Will re-fit them. Error: {e}"
            )
            input_scaler, label_scaler = None, None

    scalers_were_loaded = input_scaler is not None and label_scaler is not None

    (
        train_loader,
        num_train_samples,
        val_loader,
        num_val_samples,
        input_scaler,
        label_scaler,
    ) = create_hf_dataset_jax_wrapper(
        data_dir=config["data_dir"],
        label_dir=config["label_dir"],
        val_file_ratio=config.get("val_file_ratio", 0.2),
        shuffle_files_before_split=config.get("shuffle_files_before_split", True),
        seed=config["seed"],
        batch_size=config["batch_size"],
        input_scaler=input_scaler,
        label_scaler=label_scaler,
    )

    if not scalers_were_loaded:
        joblib.dump(input_scaler, input_scaler_path)
        joblib.dump(label_scaler, label_scaler_path)
        logger.info(f"Saved newly fitted input and label scalers to {scaler_dir}")

    # --- Model ---
    logger.info(f"Initializing model: {config['model_name']}")
    model = UnifiedVAE(**config["model_config"])
    dummy_input_for_init = jnp.ones(config["input_shape"], dtype=jnp.float32)
    init_rngs = {"params": params_init_key, "dropout": dropout_init_key}
    variables = model.init(
        init_rngs, dummy_input_for_init, latent_init_key, training=False
    )
    params = variables["params"]
    batch_stats = variables.get("batch_stats", FrozenDict({}))
    logger.info("Model initialized.")

    # --- Optimizer, Schedulers ---
    logger.info("Initializing optimizer and schedulers...")
    if num_train_samples > 0:
        num_batches_per_epoch = math.ceil(num_train_samples / config["batch_size"])
    else:
        num_batches_per_epoch = config.get("num_batches_per_epoch_fallback", 1000)

    total_train_steps = num_batches_per_epoch * config["num_train_epochs"]
    warmup_steps = num_batches_per_epoch * config["lr_warmup_epochs"]
    cosine_decay_steps = max(1, total_train_steps - warmup_steps)

    kl_anneal_total_steps = num_batches_per_epoch * config.get("kl_anneal_epochs", 0)
    kl_final_weight = config.get("loss_config", {}).get("kld_weight", 0.01)
    kl_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=kl_final_weight,
        transition_steps=kl_anneal_total_steps,
        transition_begin=warmup_steps,
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=warmup_steps,
        decay_steps=cosine_decay_steps,
        end_value=config["learning_rate"] * 0.01,
    )
    optimizer = optax.adamw(
        learning_rate=lr_schedule, weight_decay=config["weight_decay"]
    )

    # --- TrainState ---
    loss_config = config.get("loss_config", {})
    loss_components = [
        key.replace("_weight", "_loss")
        for key, value in loss_config.items()
        if key.endswith("_weight") and value > 0
    ]
    initial_loss_averages = {key: jnp.array(0.0) for key in loss_components}
    initial_loss_averages["count"] = jnp.array(0.0)

    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        loss_averages=initial_loss_averages,
    )
    logger.info("TrainState created.")

    # --- Loss Function ---
    loss_fn_closure = create_combined_loss_fn(config)

    # --- 9. Return all components ---
    return (
        state,
        model,
        train_loader,
        val_loader,
        lr_schedule,
        kl_schedule,
        loss_fn_closure,
        loss_components,
        train_dropout_key_base,
        eval_dropout_key_base,
        output_dir,
        num_val_samples,
    )
