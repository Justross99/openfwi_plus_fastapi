# filepath: /media/ross/TheXFiles/Openfwi_src/tests/test_criterion.py
import pytest
import jax
import jax.numpy as jnp

# Import loss functions from the criterion directory
from criterion.kl_div import kl_divergence_loss
from criterion.structural_similarity_loss import ssim_loss, _apply_window_filter
from criterion.perceptual_loss import perceptual_loss, PerceptualFeatureExtractor
from criterion.combined_loss import create_combined_loss_fn

# --- Constants for testing ---
KEY = jax.random.PRNGKey(42)
BATCH_SIZE = 2
HEIGHT, WIDTH = 64, 64
LATENT_DIM = 8
NUM_GEOPHONES = 10
NUM_TIME_STEPS = 100

# --- Fixtures ---


@pytest.fixture
def dummy_mu_logvar():
    mu = jax.random.normal(KEY, (BATCH_SIZE, LATENT_DIM))
    log_var = jax.random.normal(KEY, (BATCH_SIZE, LATENT_DIM)) * 0.1
    return mu, log_var


@pytest.fixture
def dummy_mu_logvar_spatial():
    key1, key2 = jax.random.split(KEY)
    mu = jax.random.normal(key1, (BATCH_SIZE, HEIGHT // 8, WIDTH // 8, LATENT_DIM))
    log_var = (
        jax.random.normal(key2, (BATCH_SIZE, HEIGHT // 8, WIDTH // 8, LATENT_DIM)) * 0.1
    )
    return mu, log_var


@pytest.fixture
def dummy_velocity_maps():
    key1, key2 = jax.random.split(KEY)
    # Simulate velocity maps (e.g., values between 1500 and 4500)
    # The combined loss function now expects batch and channel dimensions.
    map1 = jax.random.uniform(
        key1, (BATCH_SIZE, HEIGHT, WIDTH, 1), minval=1500.0, maxval=4500.0
    )
    map2 = jax.random.uniform(
        key2, (BATCH_SIZE, HEIGHT, WIDTH, 1), minval=1500.0, maxval=4500.0
    )
    return map1, map2


@pytest.fixture
def perceptual_feature_extractor_and_params():
    model = PerceptualFeatureExtractor()
    dummy_input = jnp.ones((1, HEIGHT, WIDTH, 1))  # Batch, H, W, C
    # The model's __call__ method does not expect a `training` argument for init
    variables = model.init(KEY, dummy_input)
    return model, variables["params"]


@pytest.fixture
def dummy_source_function():
    return jax.random.normal(KEY, (NUM_TIME_STEPS,))


@pytest.fixture
def dummy_geophone_locations():
    # Simple geophone locations, e.g., spread across the top
    locations = jnp.zeros((NUM_GEOPHONES, 2), dtype=jnp.int32)
    locations = locations.at[:, 0].set(
        jnp.linspace(0, WIDTH - 1, NUM_GEOPHONES, dtype=jnp.int32)
    )
    locations = locations.at[:, 1].set(0)  # z_idx = 0
    return locations


@pytest.fixture
def dummy_config():
    """Provides a configuration dictionary for the loss factory."""
    return {
        "loss_config": {
            "max_velocity_val": 4500.0,
            "ssim_window_size": 7,
            "mae_weight": 1.0,
            "ssim_weight": 0.5,
            "percept_weight": 0.1,
            "kld_weight": 0.01,
        }
    }


# --- Test Functions ---


def test_kl_divergence_loss(dummy_mu_logvar, dummy_mu_logvar_spatial):
    mu, log_var = dummy_mu_logvar
    loss = kl_divergence_loss(mu, log_var)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()  # Scalar
    assert loss >= 0

    # Test with spatial dimensions
    mu_s, log_var_s = dummy_mu_logvar_spatial
    loss_s = kl_divergence_loss(mu_s, log_var_s)
    assert isinstance(loss_s, jnp.ndarray)
    assert loss_s.shape == ()
    assert loss_s >= 0


def test_ssim_loss_shapes_and_values(dummy_velocity_maps):
    map1, map2 = dummy_velocity_maps
    max_val = 4500.0
    window_size = 7

    # Test with default window size
    loss = ssim_loss(map1, map2, max_val=max_val, window_size=window_size)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert 0 <= loss <= 1  # SSIM loss is 1 - SSIM, so it's between 0 and 1

    # Test with a different window size
    loss_custom_window = ssim_loss(map1, map2, max_val=max_val, window_size=5)
    assert isinstance(loss_custom_window, jnp.ndarray)
    assert loss_custom_window.shape == ()

    # Test perfect match -> loss should be near zero
    loss_perfect_match = ssim_loss(map1, map1, max_val=max_val, window_size=window_size)
    assert jnp.allclose(loss_perfect_match, 0.0, atol=1e-6)


def test_ssim_window_filter_padding():
    """Checks if the padding in _apply_window_filter is correct."""
    image = jnp.ones((BATCH_SIZE, 16, 16, 1))
    window_size = 7
    # The padding is implicitly 'SAME' via the convolution implementation.
    filtered_same = _apply_window_filter(image, window_size)
    assert filtered_same.shape == image.shape


def test_perceptual_loss(dummy_velocity_maps, perceptual_feature_extractor_and_params):
    map1, map2 = dummy_velocity_maps
    model, params = perceptual_feature_extractor_and_params

    # The model's __call__ method does not expect a `training` argument for apply
    loss = perceptual_loss(map1, map2, params, model)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert loss >= 0

    # Test perfect match -> loss should be zero
    loss_perfect_match = perceptual_loss(map1, map1, params, model)
    assert jnp.allclose(loss_perfect_match, 0.0, atol=1e-6)


def test_perceptual_feature_extractor_forward_pass(
    dummy_velocity_maps, perceptual_feature_extractor_and_params
):
    map1, _ = dummy_velocity_maps
    model, params = perceptual_feature_extractor_and_params

    # The model's __call__ method does not expect a `training` argument
    features = model.apply({"params": params}, map1)
    assert isinstance(features, list)
    # Based on VGG16 architecture, expecting 5 feature maps from the specified layers
    assert len(features) == 5
    # Check shapes (they depend on the VGG architecture)
    expected_shapes = [
        (BATCH_SIZE, 64, 64, 64),
        (BATCH_SIZE, 32, 32, 128),
        (BATCH_SIZE, 16, 16, 256),
        (BATCH_SIZE, 8, 8, 512),
        (BATCH_SIZE, 4, 4, 512),
    ]
    for feat, shape in zip(features, expected_shapes):
        assert feat.shape == shape


def test_combined_loss_factory_creation_and_call(
    dummy_config,
    dummy_velocity_maps,
    dummy_mu_logvar_spatial,
    perceptual_feature_extractor_and_params,
):
    """Test that the factory creates a valid loss function and it runs."""
    loss_fn = create_combined_loss_fn(dummy_config)
    p_model, p_params = perceptual_feature_extractor_and_params

    assert callable(loss_fn)

    # Prepare inputs for the combined loss function
    true_vel, pred_vel = dummy_velocity_maps
    mu, log_var = dummy_mu_logvar_spatial

    # Call the combined loss function
    total_loss, loss_components = loss_fn(
        predicted_velocity_map=pred_vel,
        true_velocity_map=true_vel,
        feature_extractor_params=p_params,
        feature_extractor_model=p_model,
        mu=mu,
        log_var=log_var,
    )

    # --- Assertions ---
    assert isinstance(total_loss, jnp.ndarray)
    assert total_loss.shape == ()
    assert total_loss >= 0

    assert isinstance(loss_components, dict)
    # The keys in the returned dict are now suffixed with "_loss"
    expected_keys = ["mae_loss", "ssim_loss", "percept_loss", "kld_loss"]
    assert all(key in loss_components for key in expected_keys)

    for key, value in loss_components.items():
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()

    # Test with zero weights
    zero_weight_config = dummy_config.copy()
    zero_weight_config["loss_config"] = {
        "max_velocity_val": 4500.0,
        "ssim_window_size": 7,
        "mae_weight": 0.0,
        "ssim_weight": 0.0,
        "percept_weight": 0.0,
        "kld_weight": 0.0,
    }
    loss_fn_zero = create_combined_loss_fn(zero_weight_config)
    total_loss_zero, _ = loss_fn_zero(
        predicted_velocity_map=pred_vel,
        true_velocity_map=true_vel,
        feature_extractor_model=p_model,
        feature_extractor_params=p_params,
        mu=mu,
        log_var=log_var,
    )
    assert jnp.allclose(total_loss_zero, 0.0)


def test_combined_loss_factory_no_perceptual(
    dummy_config, dummy_velocity_maps, dummy_mu_logvar_spatial
):
    """Test factory when perceptual weight is zero."""
    config = dummy_config.copy()
    config["loss_config"]["percept_weight"] = 0.0

    loss_fn = create_combined_loss_fn(config)

    true_vel, pred_vel = dummy_velocity_maps
    mu, log_var = dummy_mu_logvar_spatial

    total_loss, loss_components = loss_fn(
        predicted_velocity_map=pred_vel,
        true_velocity_map=true_vel,
        mu=mu,
        log_var=log_var,
        # Perceptual args are not needed and should be ignored
    )
    assert loss_components["percept_loss"] == 0.0
    assert total_loss >= 0
