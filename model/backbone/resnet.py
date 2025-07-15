from flax import linen as nn
from typing import Sequence, Any, Dict


class BottleneckBlock(nn.Module):
    features: int
    strides: int = 1
    dilation: int = 1
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        residual = x
        y = nn.Conv(self.features, (1, 1), strides=(1, 1), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(
            self.features,
            (3, 3),
            strides=(self.strides, self.strides),
            padding="SAME",
            use_bias=False,
            feature_group_count=1,
            kernel_dilation=(self.dilation, self.dilation),
        )(y)
        y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        y = nn.Conv(self.features * 4, (1, 1), strides=(1, 1), use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not training)(y)
        if (
            self.use_projection
            or self.strides != 1
            or residual.shape[-1] != self.features * 4
        ):
            residual = nn.Conv(
                self.features * 4,
                (1, 1),
                strides=(self.strides, self.strides),
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        y = y + residual
        y = nn.relu(y)
        return y


class ResNetDeepLabBackbone(nn.Module):
    block_sizes: Sequence[int] = (3, 4, 6, 3)  # ResNet-50 default
    features: Sequence[int] = (64, 128, 256, 512)
    output_stride: int = 16  # 8, 16, or 32 for DeepLab
    initial_features: int = 64  # Added to allow configuration

    def setup(self):
        if self.output_stride not in [8, 16, 32]:
            raise ValueError("output_stride must be 8, 16, or 32")

    @nn.compact
    def __call__(self, x, training: bool) -> Dict[str, Any]:
        # Determine strides and dilations based on output_stride
        if self.output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif self.output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif self.output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            # This case is handled by the setup method, but as a fallback
            raise ValueError("output_stride must be 8, 16, or 32")

        # Initial conv and maxpool
        x = nn.Conv(
            self.initial_features,
            (7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        features = {}

        # Stage 1
        for i in range(self.block_sizes[0]):
            x = BottleneckBlock(
                self.features[0],
                strides=strides[0],
                use_projection=(i == 0),
                dilation=dilations[0],
            )(x, training=training)
        features["low"] = x

        # Stage 2
        for i in range(self.block_sizes[1]):
            x = BottleneckBlock(
                self.features[1],
                strides=strides[1] if i == 0 else 1,
                use_projection=(i == 0),
                dilation=dilations[1],
            )(x, training=training)
        features["mid"] = x

        # Stage 3
        for i in range(self.block_sizes[2]):
            x = BottleneckBlock(
                self.features[2],
                strides=strides[2] if i == 0 else 1,
                use_projection=(i == 0),
                dilation=dilations[2],
            )(x, training=training)
        features["high"] = x

        # Stage 4
        for i in range(self.block_sizes[3]):
            x = BottleneckBlock(
                self.features[3],
                strides=strides[3] if i == 0 else 1,
                use_projection=(i == 0),
                dilation=dilations[3],
            )(x, training=training)
        features["out"] = x
        return features


# Example usage (not run at import)
# model = ResNetDeepLabBackbone()
# variables = model.init(jax.random.PRNGKey(0), jnp.ones([1, 224, 224, 3]))
# feats = model.apply(variables, jnp.ones([1, 224, 224, 3]))
# print(feats['low'].shape, feats['mid'].shape, feats['high'].shape, feats['out'].shape)
