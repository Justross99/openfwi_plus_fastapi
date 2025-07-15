import pytest
import jax
import jax.numpy as jnp

# Attempt to import backbone models
# These paths assume pytest is run from the root of the Openfwi_src directory
try:
    from model.backbone.resnet import (
        ResNetDeepLabBackbone,
        BottleneckBlock as ResNetBottleneckBlock,
    )  # Alias to avoid name clash
    from model.backbone.vision_mamba import (
        VisionMambaBackbone,
    )
except ImportError as e:
    pytest.fail(f"Failed to import backbone models, check PYTHONPATH or imports: {e}")

# Common dummy input parameters
BATCH_SIZE = 2
IMG_H = 128
IMG_W = 128
CHANNELS_IN = 3
KEY = jax.random.PRNGKey(0)


# --- Helper Functions ---
def _get_dummy_input(
    key: jax.random.PRNGKey, batch_size: int, height: int, width: int, channels: int
) -> jnp.ndarray:
    """Generates a dummy input tensor."""
    return jax.random.normal(key, (batch_size, height, width, channels))


# ==============================================================================
# ResNet Tests
# ==============================================================================
# --- Test Cases ---
def test_bottleneck_block_initialization_and_forward():
    key_input, key_init = jax.random.split(KEY)
    dummy_input = _get_dummy_input(
        key_input, BATCH_SIZE, IMG_H // 4, IMG_W // 4, 64
    )  # Example input after initial layers

    # Test without projection, no stride/dilation change
    model_no_proj = ResNetBottleneckBlock(features=64, use_projection=False)
    variables_no_proj = model_no_proj.init(key_init, dummy_input, training=False)
    output_no_proj = model_no_proj.apply(variables_no_proj, dummy_input, training=False)
    assert output_no_proj.shape[-1] == 256  # 64 * 4

    # Test with projection to change feature dimensions
    model_with_proj = ResNetBottleneckBlock(features=128, use_projection=True)
    variables_with_proj = model_with_proj.init(key_init, dummy_input, training=False)
    output_with_proj = model_with_proj.apply(
        variables_with_proj, dummy_input, training=False
    )
    assert output_with_proj.shape[-1] == 512  # 128 * 4

    # Test with stride to downsample
    model_with_stride = ResNetBottleneckBlock(
        features=64, strides=2, use_projection=True
    )
    variables_with_stride = model_with_stride.init(
        key_init, dummy_input, training=False
    )
    output_with_stride = model_with_stride.apply(
        variables_with_stride, dummy_input, training=False
    )
    assert output_with_stride.shape[1] == dummy_input.shape[1] // 2
    assert output_with_stride.shape[2] == dummy_input.shape[2] // 2


@pytest.mark.parametrize("output_stride", [8, 16, 32])
def test_resnet_backbone_initialization_and_forward(output_stride):
    key_input, key_init = jax.random.split(KEY)
    dummy_input = _get_dummy_input(key_input, BATCH_SIZE, IMG_H, IMG_W, CHANNELS_IN)

    model = ResNetDeepLabBackbone(output_stride=output_stride)
    variables = model.init(key_init, dummy_input, training=False)
    output = model.apply(variables, dummy_input, training=False)

    assert isinstance(output, dict)
    assert "out" in output

    # The output shape depends on the output_stride parameter.
    expected_h = IMG_H // output_stride
    expected_w = IMG_W // output_stride
    # The final number of channels in the ResNet backbone is 2048
    assert output["out"].shape == (BATCH_SIZE, expected_h, expected_w, 2048)


def test_resnet_backbone_invalid_output_stride():
    with pytest.raises(ValueError, match="output_stride must be 8, 16, or 32"):
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, IMG_H, IMG_W, CHANNELS_IN)
        model = ResNetDeepLabBackbone(output_stride=99)
        # We need to call the model to trigger the check
        model.init(key_init, dummy_input, training=False)


# ==============================================================================
# Vision Mamba Tests
# ==============================================================================


@pytest.mark.parametrize("bidirectional", [True, False])
def test_vision_mamba_backbone_initialization_and_forward(bidirectional):
    key_input, key_init = jax.random.split(KEY)
    # Adjust input size to be divisible by patch_size
    patch_size = 16
    img_h, img_w = 64, 64
    dummy_input = _get_dummy_input(key_input, BATCH_SIZE, img_h, img_w, CHANNELS_IN)

    model = VisionMambaBackbone(
        patch_size=patch_size,
        embed_dim=192,
        depth=4,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        bidirectional=bidirectional,
    )

    variables = model.init(key_init, dummy_input, training=False)
    output = model.apply(variables, dummy_input, training=False)

    assert isinstance(output, dict)
    assert "out" in output

    expected_h = img_h // patch_size
    expected_w = img_w // patch_size
    expected_channels = 192  # embed_dim
    assert output["out"].shape == (
        BATCH_SIZE,
        expected_h,
        expected_w,
        expected_channels,
    )


def test_vision_mamba_backbone_invalid_config():
    # Invalid config: depth 0
    with pytest.raises(ValueError, match="Depth must be a positive integer."):
        model = VisionMambaBackbone(
            patch_size=16,
            embed_dim=192,
            depth=0,  # Invalid depth
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        )
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, 64, 64, CHANNELS_IN)
        model.init(key_init, dummy_input, training=False)

    # Invalid config: non-square patches
    with pytest.raises(ValueError, match="Patch size must be a single integer"):
        model = VisionMambaBackbone(
            patch_size=(16, 8),  # Non-square patch
            embed_dim=192,
            depth=4,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        )
        key_input, key_init = jax.random.split(KEY)
        dummy_input = _get_dummy_input(key_input, BATCH_SIZE, 64, 64, CHANNELS_IN)
        model.init(key_init, dummy_input, training=False)
