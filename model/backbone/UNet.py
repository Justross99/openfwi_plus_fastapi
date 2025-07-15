from flax import linen as nn
from typing import Sequence


class ConvBlock(nn.Module):
    """A convolutional block with optional BatchNorm and a specified activation."""

    features: int
    kernel_size: int = 3
    norm: bool = True
    act: callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        """Forward pass with explicit training flag for BatchNorm."""
        x = nn.Conv(
            self.features, (self.kernel_size, self.kernel_size), padding="SAME"
        )(x)
        if self.norm:
            # use_running_average=not training is the standard pattern:
            # - In training (training=True), use batch statistics.
            # - In evaluation (training=False), use moving averages.
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    """A downsampling block consisting of two ConvBlocks and a max-pooling layer."""

    features: int

    @nn.compact
    def __call__(self, x, training: bool):
        """Forward pass, propagating the training flag."""
        # Pass the training flag to the convolutional blocks
        x = ConvBlock(self.features)(x, training=training)
        x = ConvBlock(self.features)(x, training=training)
        skip_connection = x
        x = nn.max_pool(x, (2, 2), (2, 2))
        return x, skip_connection


class UNetEncoder(nn.Module):
    """
    A U-Net style encoder that extracts features from an input image.
    It consists of a series of downsampling blocks and a final bottleneck.
    """

    encoder_channels: Sequence[int]
    bottleneck_features: int

    @nn.compact
    def __call__(self, x, training: bool):
        """
        Forward pass of the encoder.

        Args:
            x: The input tensor.
            training: A boolean flag to control BatchNorm behavior.

        Returns:
            The bottleneck feature map and a list of skip connections.
        """
        skips = []
        # --- Encoder Path ---
        for features in self.encoder_channels:
            x, skip = DownBlock(features)(x, training=training)
            skips.append(skip)

        # --- Bottleneck ---
        x = ConvBlock(self.bottleneck_features)(x, training=training)
        x = ConvBlock(self.bottleneck_features)(x, training=training)

        return x, skips


# Example usage (not run at import)
# model = UNetEncoder(encoder_channels=[64, 128, 256], bottleneck_features=512)
# variables = model.init(jax.random.PRNGKey(0), jnp.ones([1, 128, 128, 1]))
# output, skips = model.apply(variables, jnp.ones([1, 128, 128, 1]), training=True)
