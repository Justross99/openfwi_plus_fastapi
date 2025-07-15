# filepath: tests/test_model_defs.py
import pytest
import jax
from typing import Dict, Any

# Import the unified model definition
try:
    from model.full_model_defs.models import UnifiedVAE
except ImportError as e:
    pytest.fail(f"Failed to import UnifiedVAE model: {e}")

# --- Constants for Testing ---
KEY = jax.random.PRNGKey(0)
BATCH_SIZE = 1
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Use square images for simpler test configs
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
LATENT_DIM = 16


# --- Fixtures ---
@pytest.fixture
def dummy_input_batch():
    """Provides a standard (N, H, W, C) input batch."""
    key, _ = jax.random.split(KEY)
    return jax.random.normal(key, (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS))


@pytest.fixture
def common_keys():
    """Provides a dictionary of JAX PRNG keys for tests."""
    init_key, apply_key, reparam_key = jax.random.split(KEY, 3)
    return {"init": init_key, "apply": apply_key, "reparam": reparam_key}


# --- Model Test Configurations ---


def get_unet_config() -> Dict[str, Any]:
    """Configuration for a UnifiedVAE with a UNet encoder."""
    num_down_blocks = 3  # 64 -> 32 -> 16 -> 8. Latent shape: (8, 8)
    return {
        "encoder_name": "unet",
        "encoder_config": {
            "encoder_channels": (16, 32, 64),
            "bottleneck_features": 128,
        },
        "latent_dim": LATENT_DIM,
        "decoder_config": {
            "start_features": 128,
            "upsample_blocks": [
                (64, (16, 16)),
                (32, (32, 32)),
                (16, (64, 64)),
            ],
            "output_channels": OUTPUT_CHANNELS,
            "final_activation": "sigmoid",
        },
    }


def get_resnet_config() -> Dict[str, Any]:
    """Configuration for a UnifiedVAE with a ResNet encoder."""
    output_stride = 8  # 64 -> 8. Latent shape: (8, 8)
    return {
        "encoder_name": "resnet",
        "encoder_config": {
            "block_sizes": [1, 1, 1, 1],  # Minimal layers, FIX: was layers_per_block
            "initial_features": 32,
            "output_stride": output_stride,
        },
        "latent_dim": LATENT_DIM,
        "decoder_config": {
            "start_features": 128,  # Features out of ResNet block
            "upsample_blocks": [
                (64, (16, 16)),
                (32, (32, 32)),
                (16, (64, 64)),
            ],
            "output_channels": OUTPUT_CHANNELS,
            "final_activation": "tanh",
        },
    }


def get_mamba_config() -> Dict[str, Any]:
    """Configuration for a UnifiedVAE with a VisionMamba encoder."""
    patch_size = 8  # 64x64 image -> 8x8 patches. Latent shape: (8, 8)
    return {
        "encoder_name": "mamba",
        "encoder_config": {
            "patch_size": patch_size,
            "embed_dim": 64,
            "depth": 2,
            "bidirectional": True,
        },
        "latent_dim": LATENT_DIM,
        "decoder_config": {
            "start_features": 64,
            "upsample_blocks": [
                (32, (16, 16)),
                (16, (32, 32)),
                (8, (64, 64)),
            ],
            "output_channels": OUTPUT_CHANNELS,
            "final_activation": "identity",
        },
    }


# --- Main Test Class ---


class TestUnifiedVAE:
    @pytest.mark.parametrize(
        "config_fn",
        [
            get_unet_config,
            get_resnet_config,
            get_mamba_config,
        ],
        ids=["UNet-Encoder", "ResNet-Encoder", "Mamba-Encoder"],
    )
    @pytest.mark.parametrize("training", [True, False])
    def test_model_initialization_and_forward_pass(
        self, config_fn, training, dummy_input_batch, common_keys
    ):
        """
        A single, parameterized test to validate all configurations of UnifiedVAE.
        It checks for correct initialization and output shapes in both training
        and evaluation modes.
        """
        config = config_fn()
        model = UnifiedVAE(**config)

        # --- Initialization ---
        variables = model.init(
            {"params": common_keys["init"], "dropout": common_keys["apply"]},
            dummy_input_batch,
            rng_key=common_keys["reparam"],
            training=training,
        )
        assert "params" in variables
        # Check that the correct encoder was created
        assert f"{config['encoder_name']}_encoder" in variables["params"]
        assert "gaussian_latent" in variables["params"]
        assert "resizing_decoder" in variables["params"]

        # --- Forward Pass ---
        if training:
            # When training, batch_stats needs to be mutable
            output_map, variables = model.apply(
                {"params": variables["params"]},
                dummy_input_batch,
                rng_key=common_keys["reparam"],
                training=training,
                rngs={"dropout": common_keys["apply"]},
                mutable=["batch_stats"],
            )
            # The output is a tuple ( (output_map, mu, log_var), variables_dict )
            output_map, mu, log_var = output_map
            assert "batch_stats" in variables  # Check that stats were updated
        else:
            # In eval mode, no need for mutable collections
            output_map, mu, log_var = model.apply(
                {
                    "params": variables["params"],
                    "batch_stats": variables.get("batch_stats", {}),
                },
                dummy_input_batch,
                rng_key=common_keys["reparam"],
                training=training,
                rngs={"dropout": common_keys["apply"]},
            )

        # --- Shape Assertions ---
        # 1. Output map shape
        assert output_map.shape == (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS)

        # 2. Latent variable shapes (mu, log_var)
        if config["encoder_name"] == "mamba":
            patch_size = config["encoder_config"]["patch_size"]
            expected_latent_h = IMG_HEIGHT // patch_size
            expected_latent_w = IMG_WIDTH // patch_size
        elif config["encoder_name"] == "resnet":
            stride = config["encoder_config"]["output_stride"]
            expected_latent_h = IMG_HEIGHT // stride
            expected_latent_w = IMG_WIDTH // stride
        elif config["encoder_name"] == "unet":
            num_downs = len(config["encoder_config"]["encoder_channels"])
            expected_latent_h = IMG_HEIGHT // (2**num_downs)
            expected_latent_w = IMG_WIDTH // (2**num_downs)

        expected_latent_shape = (
            BATCH_SIZE,
            expected_latent_h,
            expected_latent_w,
            LATENT_DIM,
        )
        assert mu.shape == expected_latent_shape
        assert log_var.shape == expected_latent_shape

    def test_invalid_encoder_name_raises_error(self, dummy_input_batch, common_keys):
        """

        Tests that initializing UnifiedVAE with an unknown encoder name
        raises a ValueError.
        """
        invalid_config = {
            "encoder_name": "non_existent_encoder",
            "encoder_config": {},
            "latent_dim": 16,
            "decoder_config": {},
        }
        model = UnifiedVAE(**invalid_config)
        with pytest.raises(ValueError):
            model.init(
                {"params": common_keys["init"], "dropout": common_keys["apply"]},
                dummy_input_batch,
                rng_key=common_keys["reparam"],
                training=True,
            )
