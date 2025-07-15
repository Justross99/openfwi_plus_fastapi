import pytest
import jax
import jax.numpy as jnp

# Import the module to be tested
try:
    from model.latents.gaussian_latents import GaussianLatent
except ImportError as e:
    pytest.fail(f"Failed to import GaussianLatent model: {e}")

# Common test parameters
BATCH_SIZE = 2
H, W, C_IN = 16, 16, 64  # Example input feature dimensions
LATENT_DIM = 8
KEY = jax.random.PRNGKey(0)


# --- Helper Function ---
def _get_dummy_features(key, batch_size, h, w, c):
    """Generates a dummy 4D feature tensor."""
    return jax.random.normal(key, (batch_size, h, w, c))


# --- Test Class for GaussianLatent ---
class TestGaussianLatent:
    def test_initialization_and_forward_pass_shapes(self):
        """
        Tests that the module initializes correctly and that the outputs
        (z, mu, log_var) have the expected shapes.
        """
        key_input, key_init, key_reparam = jax.random.split(KEY, 3)
        dummy_features = _get_dummy_features(key_input, BATCH_SIZE, H, W, C_IN)

        model = GaussianLatent(latent_dim=LATENT_DIM)

        # Test initialization: The rng_key is a positional argument to __call__.
        variables = model.init(key_init, dummy_features, rng_key=key_reparam)
        assert "params" in variables
        assert "mu_conv" in variables["params"]
        assert "log_var_conv" in variables["params"]

        # Test forward pass
        # The rng_key is also passed as a positional argument to apply.
        z, mu, log_var = model.apply(variables, dummy_features, rng_key=key_reparam)

        # All outputs should have the same shape, with the new latent dimension
        expected_shape = (BATCH_SIZE, H, W, LATENT_DIM)
        assert z.shape == expected_shape
        assert mu.shape == expected_shape
        assert log_var.shape == expected_shape

    def test_reparameterization_trick(self):
        """
        Verifies that the reparameterization trick is applied, meaning z is
        stochastically sampled and not just equal to mu.
        """
        key_input, key_init, key_reparam = jax.random.split(KEY, 3)
        dummy_features = _get_dummy_features(key_input, BATCH_SIZE, H, W, C_IN)

        model = GaussianLatent(latent_dim=LATENT_DIM)
        variables = model.init(key_init, dummy_features, rng_key=key_reparam)

        z, mu, log_var = model.apply(variables, dummy_features, rng_key=key_reparam)

        # z should be mu + some random noise (std * epsilon), so they should not be equal.
        assert not jnp.allclose(z, mu)

        # Manually perform the reparameterization to verify the output
        std = jnp.exp(0.5 * log_var)
        # Use the same key to generate the same random numbers
        epsilon = jax.random.normal(key_reparam, std.shape)
        expected_z = mu + std * epsilon
        assert jnp.allclose(z, expected_z)

    def test_input_ndim_validation(self):
        """
        Tests that the module raises a ValueError if the input is not 4D.
        """
        key_input, key_init, key_reparam = jax.random.split(KEY, 3)
        # Create an invalid 3D input
        dummy_invalid_features = jax.random.normal(key_input, (BATCH_SIZE, H, C_IN))

        model = GaussianLatent(latent_dim=LATENT_DIM)

        # The error should be raised during init or apply where the shape is checked.
        with pytest.raises(ValueError, match="Input features x must be 4D"):
            model.init(key_init, dummy_invalid_features, rng_key=key_reparam)
