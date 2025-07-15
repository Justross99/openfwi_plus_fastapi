# filepath: /mnt/E6B6DC5AB6DC2D35/Openfwi_src/model/decoder/decoder_unet.py
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Tuple, Literal


# --- Reusable Convolutional Block ---
class ConvBlock(nn.Module):
    """A simple convolutional block with BatchNorm and ReLU activation."""

    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            kernel_init=nn.initializers.he_normal(),
            bias_init=nn.initializers.zeros,
        )(x)
        # Note: use_running_average=not training ensures correct behavior for train/eval
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


# --- Efficient Resizing Decoder ---
class ResizingDecoder(nn.Module):
    """
    An efficient decoder that reconstructs an image from a latent feature map.

    It avoids a computationally expensive flatten-and-project step by working
    directly on the spatial dimensions of the latent tensor. It progressively
    resizes and refines the feature map using convolutional blocks.
    """

    start_features: int
    upsample_blocks: Sequence[Tuple[int, Tuple[int, int]]]
    output_channels: int = 1
    final_activation: Literal["identity", "sigmoid", "tanh"] = "identity"
    dropout_rate: float = 0.0

    def setup(self):
        if self.final_activation not in ["identity", "sigmoid", "tanh"]:
            raise ValueError(f"Unknown activation function: {self.final_activation}")

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool):
        """
        Forward pass of the ResizingDecoder.

        Args:
            z: The latent tensor, e.g., of shape (batch, latent_h, latent_w, latent_c).
            training: A boolean indicating if the model is in training mode.

        Returns:
            The reconstructed output map.
        """
        # 1. Start directly from the latent tensor 'z'.
        # Apply an initial convolution to adjust the number of features.
        # This replaces the inefficient flatten -> dense -> reshape pattern.
        x = ConvBlock(
            features=self.start_features,
            name="initial_conv_block",
            dropout_rate=self.dropout_rate,
        )(z, training=training)

        # 2. Sequentially upsample the feature map and refine it
        for i, (features, target_shape) in enumerate(self.upsample_blocks):
            # Explicitly resize the image to the new target dimensions
            x = jax.image.resize(
                x,
                (x.shape[0], target_shape[0], target_shape[1], x.shape[3]),
                method="bilinear",
            )

            # Refine the upsampled features with a convolutional block
            x = ConvBlock(
                features=features,
                name=f"upsample_conv_block_{i}",
                dropout_rate=self.dropout_rate,
            )(x, training=training)

        # 3. Final convolution to produce the output map
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(1, 1),
            name="final_output_conv",
        )(x)

        # 4. Apply the final activation function
        if self.final_activation == "sigmoid":
            x = nn.sigmoid(x)
        elif self.final_activation == "tanh":
            x = nn.tanh(x)
        elif self.final_activation != "identity":
            # This case should ideally be caught by the setup method
            raise ValueError(f"Unknown activation function: {self.final_activation}")

        return x


# --- Example Usage ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    params_key, latent_key = jax.random.split(key)

    batch_size = 2
    # Example latent shape from Vision Mamba with 10x10 patching on 1000x70 input
    latent_h, latent_w = 100, 7
    latent_dim = 32

    dummy_z = jax.random.normal(
        latent_key, (batch_size, latent_h, latent_w, latent_dim)
    )
    print(f"Input latent z shape: {dummy_z.shape}")

    # Configure the decoder
    target_shape = (70, 70)
    decoder = ResizingDecoder(
        initial_reshape_target=target_shape,
        initial_features=32,
        upsample_blocks=[(64, 2), (32, 2), (16, 2)],
        output_channels=1,
    )

    # Initialize model and get parameters
    variables = decoder.init(params_key, dummy_z, training=False)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    print(f"Decoder initialized. Target shape: {decoder.output_shape}")

    # Forward pass
    predicted_map, updated_state = decoder.apply(
        {"params": params, "batch_stats": batch_stats},
        dummy_z,
        training=True,
        mutable=["batch_stats"],
    )
    print(f"Predicted map shape: {predicted_map.shape}")

    expected_shape = (batch_size, 1, target_shape[0], target_shape[1])
    print(f"Expected output shape: {expected_shape}")
    assert predicted_map.shape == expected_shape
    print("Output shape is correct!")
