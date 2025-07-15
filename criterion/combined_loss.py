import jax.numpy as jnp
from flax import linen as nn
from functools import partial


# Import individual loss implementations
from .perceptual_loss import perceptual_loss
from .structural_similarity_loss import ssim_loss
from .kl_div import kl_divergence_loss


# --- Individual Loss Component Wrappers ---
# These wrappers standardize the function signatures for the factory.


def mae_loss_wrapper(predicted_velocity_map, true_velocity_map, **kwargs):
    """Computes Mean Absolute Error."""
    return jnp.mean(jnp.abs(predicted_velocity_map - true_velocity_map))


def ssim_loss_wrapper(
    predicted_velocity_map, true_velocity_map, max_val, window_size, **kwargs
):
    """Computes Structural Similarity Index Measure loss."""
    return ssim_loss(
        predicted_velocity_map,
        true_velocity_map,
        max_val=max_val,
        window_size=window_size,
    )


def perceptual_loss_wrapper(
    predicted_velocity_map,
    true_velocity_map,
    feature_extractor_params,
    feature_extractor_model,
    **kwargs,
):
    """Computes Perceptual Loss using a pre-trained feature extractor."""
    # Perceptual loss expects batch and channel dims, e.g., (B, H, W, C).
    # The inputs from the training loop should already be in this 4D format.
    return perceptual_loss(
        predicted_velocity_map,
        true_velocity_map,
        feature_extractor_params,
        feature_extractor_model,
    )


def kld_loss_wrapper(mu, log_var, **kwargs):
    """Computes KL Divergence loss for VAEs."""
    return kl_divergence_loss(mu, log_var)


# --- Loss Function Factory ---


def create_combined_loss_fn(
    config: dict,
) -> callable:
    """
    A factory that creates a combined loss function based on a configuration.

    This approach composes a loss function from active components specified in the
    config. This avoids runtime conditional logic based on weights inside the
    loss function, making it more amenable to JIT compilation.

    Args:
        config: The main configuration dictionary, expected to contain a "loss_config"
                sub-dictionary with weights and parameters for each loss component.

    Returns:
        A callable loss function that takes all necessary tensors and returns
        the total weighted loss and a dictionary of individual loss values.
    """
    loss_components = []
    loss_config = config.get("loss_config", {})

    # --- Configure components from config ---
    def add_component(name, wrapper_fn, weight_key, **static_kwargs):
        weight = loss_config.get(weight_key, 0.0)
        if weight > 0:
            loss_components.append(
                {
                    "name": name,
                    "fn": (
                        partial(wrapper_fn, **static_kwargs)
                        if static_kwargs
                        else wrapper_fn
                    ),
                    "weight": weight,
                }
            )

    add_component("mae_loss", mae_loss_wrapper, "mae_weight")
    add_component(
        "ssim_loss",
        ssim_loss_wrapper,
        "ssim_weight",
        max_val=loss_config.get("max_velocity_val", 4500.0),
        window_size=loss_config.get("ssim_window_size", 7),
    )
    add_component("percept_loss", perceptual_loss_wrapper, "percept_weight")
    add_component("kld_loss", kld_loss_wrapper, "kld_weight")

    # --- The dynamically constructed, JIT-friendly loss function ---
    def final_loss_fn(
        predicted_velocity_map: jnp.ndarray,
        true_velocity_map: jnp.ndarray,
        feature_extractor_params: dict | None = None,
        feature_extractor_model: nn.Module | None = None,
        mu: jnp.ndarray | None = None,
        log_var: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Calculates the total weighted loss based on the configuration provided
        to the parent factory.
        """
        total_loss = 0.0
        # Initialize all possible losses to 0.0 for consistent logging
        individual_losses = {
            "mae_loss": 0.0,
            "ssim_loss": 0.0,
            "percept_loss": 0.0,
            "kld_loss": 0.0,
        }

        kwargs = {
            "predicted_velocity_map": predicted_velocity_map,
            "true_velocity_map": true_velocity_map,
            "feature_extractor_params": feature_extractor_params,
            "feature_extractor_model": feature_extractor_model,
            "mu": mu,
            "log_var": log_var,
        }

        for component in loss_components:
            loss_val = component["fn"](**kwargs)
            individual_losses[component["name"]] = loss_val
            total_loss += component["weight"] * loss_val

        return total_loss, individual_losses

    return final_loss_fn
