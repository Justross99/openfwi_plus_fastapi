import jax.numpy as jnp
from flax import linen as nn


# --- Perceptual Loss ---
class PerceptualFeatureExtractor(nn.Module):
    """
    A VGG-16-like CNN to extract features for perceptual loss.
    This model's parameters will be frozen during training.

    In a full application, this would ideally be a pre-trained backbone
    (e.g., from ImageNet, or a geophysical dataset if available)
    to leverage learned rich feature representations.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # The model now expects a batch of images in NHWC format.
        # Ensure input has a channel dimension if it's missing.
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=-1)  # (N, H, W) -> (N, H, W, 1)

        # Block 1
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", name="conv1_1")(x)
        x = nn.relu(x)
        features1 = x
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", name="conv1_2")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", name="conv2_1")(x)
        x = nn.relu(x)
        features2 = x
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", name="conv2_2")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", name="conv3_1")(x)
        x = nn.relu(x)
        features3 = x
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", name="conv3_2")(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", name="conv3_3")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 4
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv4_1")(x)
        x = nn.relu(x)
        features4 = x
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv4_2")(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv4_3")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 5
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv5_1")(x)
        x = nn.relu(x)
        features5 = x
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv5_2")(x)
        x = nn.relu(x)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", name="conv5_3")(x)
        x = nn.relu(x)

        return [features1, features2, features3, features4, features5]


def perceptual_loss(
    predicted_velocity_map: jnp.ndarray,
    true_velocity_map: jnp.ndarray,
    feature_extractor_params: dict,
    feature_extractor_model: nn.Module,
) -> float:
    """
    Calculates perceptual loss using a fixed feature extractor.

    Args:
        predicted_velocity_map: (N, H, W, C)
        true_velocity_map: (N, H, W, C)
        feature_extractor_params: Parameters for the feature extractor.
        feature_extractor_model: Instance of PerceptualFeatureExtractor.

    Returns:
        float: The perceptual MSE loss.
    """
    # Get feature maps from the extractor
    pred_features = feature_extractor_model.apply(
        {"params": feature_extractor_params}, predicted_velocity_map
    )
    true_features = feature_extractor_model.apply(
        {"params": feature_extractor_params}, true_velocity_map
    )

    # The model returns a list of feature maps, compute loss over all of them
    loss = jnp.mean(
        jnp.array(
            [jnp.mean((p - t) ** 2) for p, t in zip(pred_features, true_features)]
        )
    )
    return loss


# --- 1.1 Initialize Perceptual Feature Extractor (if used by loss) ---
# perceptual_model = PerceptualFeatureExtractor() # Example
# dummy_perceptual_input = jnp.ones(config["perceptual_input_shape"])
# perceptual_params = perceptual_model.init(jax.random.PRNGKey(0), dummy_perceptual_input)["params"]
# perceptual_params = jax.device_put(perceptual_params) # Move to device if not already
# logger.info("Perceptual feature extractor initialized.")
# --- Placeholder for perceptual model ---
# perceptual_model = None
# perceptual_params = None
# --- End Placeholder ---
