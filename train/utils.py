from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training import train_state
from functools import partial


# --- Custom TrainState to handle batch_stats ---
class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: FrozenDict
    loss_averages: Dict[str, jnp.ndarray]


# --- Training Step ---
@partial(
    jax.jit,
    static_argnames={
        "model_apply_fn",
        "loss_fn_closure",
        "kl_schedule",
        "loss_components",
    },
)
def train_step(
    state: TrainStateWithBatchStats,
    batch: tuple,
    dropout_rng: jax.random.PRNGKey,
    model_apply_fn: callable,
    loss_fn_closure: callable,
    kl_schedule: callable,
    loss_components: List[str],
):
    """A single training step, JIT-compiled for performance."""
    data_input, true_velocity_map = batch

    def loss_for_grad(params):
        # Forward pass, making batch_stats mutable
        variables, updated_model_state = model_apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            data_input,
            dropout_rng,
            training=True,
            mutable=["batch_stats"],
        )
        output_map, mu, log_var = variables

        # The loss function returns individual, unweighted losses.
        _, individual_losses = loss_fn_closure(
            predicted_velocity_map=output_map,
            true_velocity_map=true_velocity_map,
            mu=mu,
            log_var=log_var,
        )

        # --- KL Annealing ---
        current_kld_weight = kl_schedule(state.step)
        individual_losses["kld_loss"] = (
            individual_losses["kld_loss"] * current_kld_weight
        )

        # --- Adaptive Loss Scaling ---
        loss_avgs = state.loss_averages
        count = loss_avgs["count"]
        beta = 0.98  # Smoothing factor for EMA

        # Update running averages for each loss component
        new_loss_avgs = {}
        for key, value in individual_losses.items():
            if key in loss_components:  # Use the passed loss_components list
                detached_value = jax.lax.stop_gradient(value)
                new_loss_avgs[key] = loss_avgs[key] * beta + detached_value * (1 - beta)
        new_loss_avgs["count"] = count + 1

        # Calculate scaling factors using old averages for stability
        scaling_factors = {
            key: 1.0 / (jax.lax.stop_gradient(loss_avgs[key]) + 1e-8)
            for key in loss_components
        }

        # Normalize scaling factors
        num_losses = len(loss_components)
        sum_of_inv_squares = jnp.sum(
            jnp.array([v**2 for v in scaling_factors.values()])
        )
        norm_factor = jnp.sqrt(num_losses) / jnp.sqrt(sum_of_inv_squares + 1e-8)

        # Apply scaling to get weighted losses
        weighted_losses = {
            key: individual_losses[key] * scaling_factors[key] * norm_factor
            for key in scaling_factors
        }

        # Final loss is the sum of adaptively weighted losses
        total_loss = jnp.sum(jnp.array(list(weighted_losses.values())))

        aux_data = {
            "individual_losses": individual_losses,
            "updated_model_state": updated_model_state,
            "new_loss_averages": new_loss_avgs,
        }
        return total_loss, aux_data

    (loss, aux_data), grads = jax.value_and_grad(loss_for_grad, has_aux=True)(
        state.params
    )

    # Update state with new gradients, batch_stats, and loss averages
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=aux_data["updated_model_state"]["batch_stats"],
        loss_averages=aux_data["new_loss_averages"],
    )

    # Return metrics as a dictionary for accumulation
    metrics = {
        "loss": loss,
        **aux_data["individual_losses"],
    }
    return new_state, metrics


# --- Evaluation Step ---
@partial(jax.jit, static_argnames={"model_apply_fn", "loss_fn_closure"})
def eval_step(
    state: TrainStateWithBatchStats,
    batch: tuple,
    dropout_rng: jax.random.PRNGKey,
    model_apply_fn: callable,
    loss_fn_closure: callable,
):
    """A single evaluation step, JIT-compiled for performance."""
    data_input, true_velocity_map = batch

    output_map, mu, log_var = model_apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        data_input,
        dropout_rng,
        training=False,
    )

    # Calculate loss using static weights from the config
    total_loss, individual_losses = loss_fn_closure(
        predicted_velocity_map=output_map,
        true_velocity_map=true_velocity_map,
        mu=mu,
        log_var=log_var,
    )

    # Return metrics as a dictionary for accumulation
    metrics = {"loss": total_loss, **individual_losses}
    return metrics


def make_serializable(obj):
    """Recursively converts non-JSON-serializable types."""
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # Add other type conversions as needed
    return obj
