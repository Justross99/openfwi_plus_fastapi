import os
import logging
import sys
from typing import Any, Dict
import functools
import json

import jax
import orbax.checkpoint as ocp
from clu import metric_writers

from .metrics import TrainMetrics, EvalMetrics
from .config import CONFIG
from .utils import train_step, eval_step, make_serializable
from .setup import setup_training

logger = logging.getLogger(__name__)


def train_and_evaluate(config: Dict[str, Any]):
    """Main training and evaluation loop."""
    logger.info("Starting training and evaluation...")
    logger.info(f"Using JAX backend: {jax.default_backend()}")
    logger.info(f"Available JAX devices: {jax.devices()}")

    # --- 1. Setup ---
    (
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
    ) = setup_training(config)

    # Ensure output_dir is a standard string, not a JAX array.
    # This can happen if the config dict is processed by JAX.
    output_dir = str(config["output_dir"])

    # Orbax requires absolute paths for checkpointing.
    output_dir = os.path.abspath(output_dir)

    # --- Log Dataloader Info ---
    if not val_loader or num_val_samples == 0:
        logger.warning(
            "Validation loader is not available or is empty. Validation steps will be skipped."
        )

    # --- Save Configuration to JSON ---
    config_save_path = os.path.join(output_dir, "config.json")

    try:
        serializable_config = make_serializable(config)
        with open(config_save_path, "w") as f:
            json.dump(serializable_config, f, indent=4)
        logger.info(f"Saved training configuration to {config_save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

    # Prepare JITted step functions with static arguments bound
    train_step_jit = functools.partial(
        train_step,
        model_apply_fn=model.apply,
        loss_fn_closure=loss_fn_closure,
        kl_schedule=kl_schedule,
        loss_components=tuple(loss_components),
    )
    eval_step_jit = functools.partial(
        eval_step,
        model_apply_fn=model.apply,
        loss_fn_closure=loss_fn_closure,
    )

    # --- 2. Setup Logging & Checkpointing ---
    writer = metric_writers.create_default_writer(logdir=output_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(output_dir, "checkpoints"), checkpointer, options=options
    )
    best_checkpoint_manager = ocp.CheckpointManager(
        os.path.join(output_dir, "best_checkpoint"),
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )

    # --- 3. Early Stopping State ---
    early_stopping = {
        "best_metric": (
            float("inf")
            if config.get("early_stopping_mode", "min") == "min"
            else float("-inf")
        ),
        "epochs_no_improve": 0,
        "best_epoch": 0,
        "stop_training": False,
    }

    # --- 4. Training Loop ---
    logger.info("Starting main training loop...")
    global_step = 0
    current_train_dropout_key = train_dropout_key_base
    current_eval_dropout_key = eval_dropout_key_base

    # Try to load checkpoint if exists
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        # The structure of the saved checkpoint is needed for restoration.
        checkpoint_target_for_restore = {
            "params": state.params,
            "opt_state": state.opt_state,
            "batch_stats": state.batch_stats,
            "loss_averages": state.loss_averages,
            "train_dropout_key": current_train_dropout_key,
            "global_step": global_step,
        }
        loaded_checkpoint = checkpoint_manager.restore(
            latest_step, items=checkpoint_target_for_restore
        )
        state = state.replace(
            params=loaded_checkpoint["params"],
            opt_state=loaded_checkpoint["opt_state"],
            batch_stats=loaded_checkpoint.get(
                "batch_stats", state.batch_stats
            ),  # Safely load batch_stats
            loss_averages=loaded_checkpoint.get(
                "loss_averages", state.loss_averages
            ),  # Safely load loss averages
        )
        current_train_dropout_key = loaded_checkpoint["train_dropout_key"]
        global_step = loaded_checkpoint["global_step"]
        logger.info(f"Restored checkpoint. Resuming from step {global_step + 1}.")
    else:
        logger.info("No checkpoint found, starting from scratch.")

    for epoch in range(1, config["num_train_epochs"] + 1):
        logger.info(f"Epoch {epoch}/{config['num_train_epochs']}")
        train_metrics_accumulator = TrainMetrics.empty()

        for batch in train_loader():
            current_train_dropout_key, dropout_key_for_step = jax.random.split(
                current_train_dropout_key
            )

            state, step_metrics = train_step_jit(
                state=state,
                batch=batch,
                dropout_rng=dropout_key_for_step,
            )

            update_dict = {**step_metrics, "learning_rate": lr_schedule(global_step)}
            train_metrics_accumulator = train_metrics_accumulator.update(**update_dict)
            global_step += 1

            if global_step % config["log_every_steps"] == 0:
                computed_metrics = train_metrics_accumulator.compute()
                writer.write_scalars(global_step, computed_metrics)
                train_metrics_accumulator = TrainMetrics.empty()

            if global_step % config["checkpoint_every_steps"] == 0:
                save_state = state
                checkpoint_data_to_save = {
                    "params": save_state.params,
                    "opt_state": save_state.opt_state,
                    "batch_stats": save_state.batch_stats,  # Save batch_stats
                    "loss_averages": save_state.loss_averages,  # Save loss averages
                    "train_dropout_key": current_train_dropout_key,
                    "global_step": global_step,
                }
                checkpoint_manager.save(
                    global_step, args=ocp.args.PyTreeSave(checkpoint_data_to_save)
                )

        # --- Evaluation ---
        if val_loader and num_val_samples > 0:
            eval_metrics_accumulator = EvalMetrics.empty()
            for batch in val_loader():
                current_eval_dropout_key, dropout_key_for_step = jax.random.split(
                    current_eval_dropout_key
                )
                eval_metrics = eval_step_jit(
                    state=state,
                    batch=batch,
                    dropout_rng=dropout_key_for_step,
                )
                eval_metrics_accumulator = eval_metrics_accumulator.update(
                    **eval_metrics
                )
            computed_eval_metrics = eval_metrics_accumulator.compute()
            writer.write_scalars(global_step, computed_eval_metrics)
            logger.info(f"Epoch {epoch} evaluation metrics: {computed_eval_metrics}")

            # --- Early Stopping ---
            metric_to_check = computed_eval_metrics.get(
                config.get("early_stopping_metric", "eval_loss")
            )
            if metric_to_check is not None:
                mode = config.get("early_stopping_mode", "min")
                improved = (
                    mode == "min" and metric_to_check < early_stopping["best_metric"]
                ) or (mode == "max" and metric_to_check > early_stopping["best_metric"])

                if improved:
                    early_stopping["best_metric"] = metric_to_check
                    early_stopping["epochs_no_improve"] = 0
                    early_stopping["best_epoch"] = epoch
                    # Save the best model checkpoint
                    save_state = state
                    checkpoint_data_to_save = {
                        "params": save_state.params,
                        "opt_state": save_state.opt_state,
                        "batch_stats": save_state.batch_stats,
                        "loss_averages": save_state.loss_averages,
                        "train_dropout_key": current_train_dropout_key,
                        "global_step": global_step,
                    }
                    best_checkpoint_manager.save(
                        global_step, args=ocp.args.PyTreeSave(checkpoint_data_to_save)
                    )
                    logger.info(
                        f"New best model saved at epoch {epoch} with {config.get('early_stopping_metric', 'eval_loss')}: {metric_to_check:.4f}"
                    )
                else:
                    early_stopping["epochs_no_improve"] += 1

                if early_stopping["epochs_no_improve"] >= config.get(
                    "early_stopping_patience", 5
                ):
                    early_stopping["stop_training"] = True
                    logger.info(
                        f"Early stopping triggered after {early_stopping['epochs_no_improve']} epochs with no improvement. Best epoch was {early_stopping['best_epoch']}."
                    )
                    break  # Exit the epoch loop

        if early_stopping["stop_training"]:
            break

    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger.info("Starting training script...")
    train_and_evaluate(CONFIG)
