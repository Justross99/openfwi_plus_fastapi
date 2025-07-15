# filepath: model/latents/gaussian_latents.py
import flax.linen as nn
import jax
import jax.numpy as jnp


class GaussianLatent(nn.Module):
    """
    A module to compute latent variables using the reparameterization trick
    for a Gaussian distribution. Typically used in VAEs.

    Takes backbone features, projects them to mu and log_var, and samples z.
    """

    latent_dim: int  # Number of channels in the latent space

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, rng_key: jax.random.PRNGKey, training: bool = True
    ):
        """
        Forward pass for the GaussianLatent module.

        Args:
            x: Input tensor from the backbone/encoder.
               Expected shape: (batch_size, height, width, features).
            rng_key: JAX PRNG key for random number generation (for sampling epsilon).
            training: Boolean indicating if the model is in training mode.
                      Not directly used in this module but good practice to include.

        Returns:
            A tuple (z, mu, log_var):
            - z: The sampled latent variable.
            - mu: The mean of the latent distribution.
            - log_var: The log variance of the latent distribution.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Input features x must be 4D (batch, height, width, channels), got shape {x.shape}"
            )

        # Project backbone features to mu and log_var
        # Using 1x1 convolutions to maintain spatial dimensions if present,
        # and project to the latent_dim.
        mu = nn.Conv(
            features=self.latent_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name="mu_conv",
        )(x)

        log_var = nn.Conv(
            features=self.latent_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name="log_var_conv",
        )(x)

        # Reparameterization trick: z = mu + std * epsilon
        std = jnp.exp(0.5 * log_var)
        epsilon = jax.random.normal(rng_key, std.shape, dtype=std.dtype)
        z = mu + std * epsilon

        return z, mu, log_var


# Example Usage (conceptual, not run):
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    dummy_key, params_key, sample_key = jax.random.split(key, 3)

    batch_size, height, width, features_in = 4, 32, 32, 64
    latent_dimension = 16

    # Dummy input features from a backbone
    dummy_features = jnp.ones((batch_size, height, width, features_in))

    # Initialize the model
    model = GaussianLatent(latent_dim=latent_dimension)
    variables = model.init(params_key, dummy_features, sample_key)
    params = variables["params"]

    # Forward pass
    z_sample, mu_latent, log_var_latent = model.apply(
        {"params": params}, dummy_features, sample_key
    )

    print("Input features shape:", dummy_features.shape)
    print("Sampled z shape:", z_sample.shape)
    print("Mu shape:", mu_latent.shape)
    print("Log_var shape:", log_var_latent.shape)

    # z_sample would then be passed to a decoder.
    # mu_latent and log_var_latent would be used to calculate the KL divergence loss term in a VAE.
    # For example, KL divergence with a standard normal prior N(0,I):
    # kl_divergence = -0.5 * jnp.sum(1 + log_var_latent - jnp.square(mu_latent) - jnp.exp(log_var_latent), axis=(1,2,3))
    # kl_loss = jnp.mean(kl_divergence)
    # print("Calculated KL loss (example):", kl_loss)
