# filepath: criterion/kl_div.py
import jax.numpy as jnp


def kl_divergence_loss(mu: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the KL divergence between a learned Gaussian distribution
    (parameterized by mu and log_var) and a standard normal distribution N(0, I).

    This is commonly used as a regularization term in Variational Autoencoders (VAEs).

    The formula for KL divergence for a single data point is:
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    where the sum is over the latent dimensions.

    This function computes the mean KL divergence across the batch and any spatial dimensions.

    Args:
        mu: The mean of the learned Gaussian distribution.
            Expected shape: (batch_size, [height, width,] latent_dim).
        log_var: The log variance of the learned Gaussian distribution.
                 Expected shape: (batch_size, [height, width,] latent_dim).

    Returns:
        A scalar JAX array representing the mean KL divergence loss.
    """
    if mu.shape != log_var.shape:
        raise ValueError(
            f"mu and log_var must have the same shape, but got {mu.shape} and {log_var.shape}"
        )

    # Calculate KL divergence per element
    # The sum is typically over the latent dimensions. If spatial dimensions are present,
    # the KL divergence is often summed over these as well, or averaged.
    # Here, we sum over all dimensions except the batch dimension (axis 0).
    # For example, if shape is (batch, H, W, D_latent), sum over H, W, D_latent.
    # If shape is (batch, D_latent), sum over D_latent.
    axis_to_sum = tuple(range(1, mu.ndim))

    kl_div = -0.5 * jnp.sum(
        1 + log_var - jnp.square(mu) - jnp.exp(log_var), axis=axis_to_sum
    )

    # Return the mean KL divergence across the batch
    return jnp.mean(kl_div)


# Example Usage (conceptual):
if __name__ == "__main__":
    import jax

    key = jax.random.PRNGKey(42)

    # Example 1: Latent space without spatial dimensions
    batch_size, latent_dim = 4, 16
    dummy_mu = jax.random.normal(key, (batch_size, latent_dim))
    dummy_log_var = (
        jax.random.normal(key, (batch_size, latent_dim)) * 0.1
    )  # Smaller log_var for stability

    loss1 = kl_divergence_loss(dummy_mu, dummy_log_var)
    print(
        f"KL Divergence (no spatial dims, batch={batch_size}, latent_dim={latent_dim}): {loss1}"
    )

    # Example 2: Latent space with spatial dimensions (e.g., from a ConvVAE)
    batch_size, height, width, latent_dim_conv = 4, 8, 8, 3
    dummy_mu_conv = jax.random.normal(key, (batch_size, height, width, latent_dim_conv))
    dummy_log_var_conv = (
        jax.random.normal(key, (batch_size, height, width, latent_dim_conv)) * 0.1
    )

    loss2 = kl_divergence_loss(dummy_mu_conv, dummy_log_var_conv)
    print(
        f"KL Divergence (with spatial dims, batch={batch_size}, H={height}, W={width}, latent_dim={latent_dim_conv}): {loss2}"
    )
