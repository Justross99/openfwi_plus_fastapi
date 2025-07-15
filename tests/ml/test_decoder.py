import pytest
import jax
import jax.numpy as jnp

# Assuming pytest is run from the root of Openfwi_src directory
try:
    from model.decoder.decoder_unet import ConvBlock, ResizingDecoder
except ImportError as e:
    pytest.fail(f"Failed to import decoder models, check PYTHONPATH or imports: {e}")


# Common dummy input parameters
BATCH_SIZE = 2
KEY = jax.random.PRNGKey(0)


# --- Helper Functions ---
def _get_dummy_spatial_input(key, batch_size, h, w, c, dtype=jnp.float32):
    """Generates a dummy 4D tensor."""
    return jax.random.normal(key, (batch_size, h, w, c), dtype=dtype)


# --- Tests for ConvBlock (from decoder_unet.py) ---
class TestDecoderConvBlock:
    @pytest.mark.parametrize("training", [True, False])
    def test_initialization_and_forward_pass(self, training):
        key_input, key_init, key_dropout = jax.random.split(KEY, 3)
        h, w, c_in, c_out = 16, 16, 32, 64
        dummy_input = _get_dummy_spatial_input(key_input, BATCH_SIZE, h, w, c_in)

        model = ConvBlock(features=c_out, dropout_rate=0.1)

        # Initialize and check for batch_stats (should always be present)
        variables = model.init(
            {"params": key_init, "dropout": key_dropout}, dummy_input, training=training
        )
        assert "params" in variables
        assert "batch_stats" in variables  # BatchNorm creates this collection

        # Apply the model
        output, updated_state = model.apply(
            variables,
            dummy_input,
            training=training,
            rngs={"dropout": key_dropout},
            mutable=["batch_stats"],  # Make sure batch_stats can be updated
        )

        # Check output shape
        assert output.shape == (BATCH_SIZE, h, w, c_out)


# --- Tests for ResizingDecoder ---
class TestResizingDecoder:
    latent_h, latent_w, latent_c = 8, 8, 128  # Example latent dimensions

    @pytest.mark.parametrize(
        "upsample_blocks, final_activation",
        [
            # Test case 1: Standard 3-block upsampling
            (
                [(64, (16, 16)), (32, (32, 32)), (16, (64, 64))],
                "identity",
            ),
            # Test case 2: Fewer blocks, different activation
            (
                [(64, (16, 16)), (32, (32, 32))],
                "sigmoid",
            ),
            # Test case 3: Single block
            (
                [(64, (16, 16))],
                "tanh",
            ),
            # Test case 4: No upsampling blocks (only initial and final conv)
            (
                [],
                "identity",
            ),
        ],
    )
    @pytest.mark.parametrize("training", [True, False])
    def test_initialization_and_forward_pass(
        self, upsample_blocks, final_activation, training
    ):
        key_input, key_init, key_dropout = jax.random.split(KEY, 3)
        output_channels = 3
        start_features = 64

        dummy_latent_z = _get_dummy_spatial_input(
            key_input, BATCH_SIZE, self.latent_h, self.latent_w, self.latent_c
        )

        model = ResizingDecoder(
            start_features=start_features,
            upsample_blocks=upsample_blocks,
            output_channels=output_channels,
            final_activation=final_activation,
            dropout_rate=0.1,
        )

        variables = model.init(
            {"params": key_init, "dropout": key_dropout},
            dummy_latent_z,
            training=training,
        )
        output, updated_state = model.apply(
            variables,
            dummy_latent_z,
            training=training,
            rngs={"dropout": key_dropout},
            mutable=["batch_stats"],
        )

        # Determine expected output shape
        if upsample_blocks:
            final_h, final_w = upsample_blocks[-1][1]
        else:
            # If no upsampling, shape is determined by the initial conv layer
            # which doesn't change spatial dims.
            final_h, final_w = self.latent_h, self.latent_w

        assert output.shape == (BATCH_SIZE, final_h, final_w, output_channels)

        # Check activation function effect
        if final_activation == "sigmoid":
            assert jnp.all(output >= 0) and jnp.all(output <= 1)
        elif final_activation == "tanh":
            assert jnp.all(output >= -1) and jnp.all(output <= 1)

    def test_invalid_activation_function(self):
        """Test that an invalid activation function string raises a ValueError."""
        key_input, key_init = jax.random.split(KEY)
        dummy_latent_z = _get_dummy_spatial_input(
            key_input, BATCH_SIZE, self.latent_h, self.latent_w, self.latent_c
        )

        model = ResizingDecoder(
            start_features=64,
            upsample_blocks=[],
            output_channels=1,
            final_activation="invalid_function_name",
        )

        with pytest.raises(ValueError, match="Unknown activation function"):
            # The training flag is required for the call signature, even if not used
            # by the logic that raises the error.
            model.init(key_init, dummy_latent_z, training=False)
