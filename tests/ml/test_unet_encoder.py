import pytest
import jax

try:
    from model.backbone.UNet import UNetEncoder, DownBlock, ConvBlock
except ImportError as e:
    pytest.fail(f"Failed to import UNet models, check PYTHONPATH or imports: {e}")

# Common dummy input parameters
BATCH_SIZE = 2
IMG_H = 128
IMG_W = 128
CHANNELS_IN = 3
KEY = jax.random.PRNGKey(0)


def _get_dummy_input(key, batch_size, h, w, c):
    return jax.random.normal(key, (batch_size, h, w, c))


class TestUNetEncoder:
    def test_conv_block(self):
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, IMG_H, IMG_W, CHANNELS_IN)
        features = 64

        model = ConvBlock(features=features)
        # Test in training mode - requires mutable batch_stats
        variables_train = model.init({"params": key_init}, dummy_input, training=True)
        output_train, updated_vars = model.apply(
            variables_train, dummy_input, training=True, mutable=["batch_stats"]
        )
        assert "batch_stats" in updated_vars
        assert output_train.shape == (BATCH_SIZE, IMG_H, IMG_W, features)

        # Test in eval mode - use the updated batch_stats
        output_eval = model.apply(
            {**variables_train, "batch_stats": updated_vars["batch_stats"]},
            dummy_input,
            training=False,
        )
        assert output_eval.shape == (BATCH_SIZE, IMG_H, IMG_W, features)

    def test_down_block(self):
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, IMG_H, IMG_W, CHANNELS_IN)
        features = 64

        model = DownBlock(features=features)
        variables = model.init({"params": key_init}, dummy_input, training=True)
        (output, skip), updated_vars = model.apply(
            variables, dummy_input, training=True, mutable=["batch_stats"]
        )

        assert "batch_stats" in updated_vars
        assert output.shape == (BATCH_SIZE, IMG_H // 2, IMG_W // 2, features)
        assert skip.shape == (BATCH_SIZE, IMG_H, IMG_W, features)

    @pytest.mark.parametrize(
        "encoder_channels, bottleneck_features",
        [
            ((16, 32), 64),
            ((64, 128, 256), 512),
        ],
    )
    def test_unet_encoder_full_initialization_and_forward(
        self, encoder_channels, bottleneck_features
    ):
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, IMG_H, IMG_W, CHANNELS_IN)

        model = UNetEncoder(
            encoder_channels=encoder_channels,
            bottleneck_features=bottleneck_features,
        )
        variables = model.init({"params": key_init}, dummy_input, training=True)
        (output, skips), updated_vars = model.apply(
            variables, dummy_input, training=True, mutable=["batch_stats"]
        )

        # Check that batch stats were updated
        assert "batch_stats" in updated_vars

        # Check bottleneck output shape
        final_encoder_h = IMG_H // (2 ** len(encoder_channels))
        final_encoder_w = IMG_W // (2 ** len(encoder_channels))
        assert output.shape == (
            BATCH_SIZE,
            final_encoder_h,
            final_encoder_w,
            bottleneck_features,
        )

        # Check skip connections shapes
        assert len(skips) == len(encoder_channels)
        for i, (skip, ch) in enumerate(zip(skips, encoder_channels)):
            expected_h = IMG_H // (2**i)
            expected_w = IMG_W // (2**i)
            assert skip.shape == (BATCH_SIZE, expected_h, expected_w, ch)
