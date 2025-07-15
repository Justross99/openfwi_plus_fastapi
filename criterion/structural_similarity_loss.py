import jax
import jax.numpy as jnp

# --- Structural Similarity Index (SSIM) Loss ---


# Helper function to apply a window filter (mean)
def _apply_window_filter(image: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Applies a uniform window filter (average) to a 4D image tensor."""
    # The input is expected to be (N, H, W, C). The filter is applied spatially.
    if image.ndim != 4:
        raise ValueError(
            f"Input image must be 4D (N, H, W, C), but got shape {image.shape}"
        )

    # Create a 2D filter and expand it to match input channels.
    window = jnp.ones((window_size, window_size), dtype=image.dtype) / (window_size**2)
    # Make it (H, W, InC, OutC=1) for depthwise convolution
    kernel = jnp.expand_dims(window, axis=(2, 3))

    # Apply convolution for each channel. We can do this by setting feature_group_count.
    # This is equivalent to a depthwise convolution.
    num_channels = image.shape[3]
    kernel = jnp.tile(kernel, (1, 1, num_channels, 1))  # (H, W, C, 1)

    # Use 'SAME' padding to keep output dimensions the same as input.
    # The convolution will be applied over the spatial dimensions (H, W).
    filtered_image = jax.lax.conv_general_dilated(
        lhs=image,
        rhs=kernel,
        window_strides=(1, 1),
        padding="SAME",
        feature_group_count=num_channels,  # Depthwise convolution
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    return filtered_image


def ssim_loss(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    max_val: float = 1.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Calculates the SSIM loss (1 - SSIM) between two 4D image tensors.
    The input tensors are expected to have shape (N, H, W, C).

    Args:
        img1: (N, H, W, C) - First image tensor.
        img2: (N, H, W, C) - Second image tensor.
        max_val: Maximum possible pixel value of the images.
        window_size: Size of the sliding window for local statistics.
        k1, k2: SSIM constants.

    Returns:
        float: The scalar SSIM loss, averaged over the batch.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, but got {img1.shape} and {img2.shape}"
        )
    if img1.ndim != 4:
        raise ValueError(f"Input images must be 4D tensors, but got shape {img1.shape}")

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    # Compute local means
    mu1 = _apply_window_filter(img1, window_size)
    mu2 = _apply_window_filter(img2, window_size)

    # Compute local variances and covariance
    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu1_mu2 = mu1 * mu2

    # These are computed for x^2, y^2, xy, then filtered, then subtract mean squared
    sigma1_sq = _apply_window_filter(jnp.square(img1), window_size) - mu1_sq
    sigma2_sq = _apply_window_filter(jnp.square(img2), window_size) - mu2_sq
    sigma12 = _apply_window_filter(img1 * img2, window_size) - mu1_mu2

    # --- SSIM Formula ---
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # Avoid division by zero and handle potential negative variances (due to floating point)
    # Clamp variances to be non-negative
    sigma1_sq = jnp.maximum(0.0, sigma1_sq)
    sigma2_sq = jnp.maximum(0.0, sigma2_sq)

    # Add a small epsilon to the denominator to prevent division by zero.
    ssim_map = numerator / (denominator + 1e-12)

    # Take mean over all dimensions to get a single scalar value.
    mean_ssim = jnp.mean(ssim_map)

    # Return 1 - SSIM for loss.
    return 1.0 - mean_ssim
