# filepath: /mnt/E6B6DC5AB6DC2D35/Openfwi_src/model/full_model_defs/models.py
import flax.linen as nn
import jax
from typing import Any, Dict, Tuple

# Import all potential backbones and other components
from ..backbone.vision_mamba import VisionMambaBackbone
from ..backbone.resnet import ResNetDeepLabBackbone
from ..backbone.UNet import UNetEncoder
from ..latents.gaussian_latents import GaussianLatent
from ..decoder.decoder_unet import ResizingDecoder

# A registry to hold our encoder classes, mapping string names to the classes
ENCODER_REGISTRY = {
    "unet": UNetEncoder,
    "resnet": ResNetDeepLabBackbone,
    "mamba": VisionMambaBackbone,
}


class UnifiedVAE(nn.Module):
    """
    A unified, configurable Variational Autoencoder (VAE) model.

    This model allows swapping different encoders (e.g., UNet, ResNet, Mamba)
    through configuration, without changing the model's core structure. It is
    designed to be flexible and easy to experiment with different architectures.
    """

    encoder_name: str
    encoder_config: Dict[str, Any]
    latent_dim: int
    decoder_config: Dict[str, Any]

    @nn.compact
    def __call__(
        self, x: jax.Array, rng_key: jax.random.PRNGKey, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Defines the forward pass of the VAE.

        Args:
            x: The input batch of data.
            rng_key: JAX random key for stochastic operations (in the latent space).
            training: A boolean flag to control behavior of layers like BatchNorm.

        Returns:
            A tuple containing:
            - output_map: The reconstructed output from the decoder.
            - mu: The mean of the latent distribution.
            - log_var: The log variance of the latent distribution.
        """
        # 1. Instantiate the selected encoder from the registry
        if self.encoder_name not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{self.encoder_name}'. "
                f"Available encoders: {list(ENCODER_REGISTRY.keys())}"
            )
        encoder_cls = ENCODER_REGISTRY[self.encoder_name]
        encoder = encoder_cls(
            **self.encoder_config, name=f"{self.encoder_name}_encoder"
        )

        # 2. Instantiate the latent module (Gaussian in this case)
        latent_module = GaussianLatent(
            latent_dim=self.latent_dim, name="gaussian_latent"
        )

        # 3. Instantiate the decoder
        decoder = ResizingDecoder(**self.decoder_config, name="resizing_decoder")

        # --- Forward Pass ---

        # 4. Get features from the encoder
        encoder_output = encoder(x, training=training)

        # 5. Adapt to the specific output format of the chosen encoder.
        # This logic centralizes the adaptation required for different backbones.
        if self.encoder_name == "unet":
            # UNetEncoder returns (bottleneck_features, skip_connections)
            features_for_latent, _ = encoder_output
        elif self.encoder_name == "resnet":
            # ResNetDeepLabBackbone returns a dictionary {'out': features, ...}
            features_for_latent = encoder_output["out"]
        elif self.encoder_name == "mamba":
            # VisionMambaBackbone also returns a dict
            features_for_latent = encoder_output["out"]
        else:
            # VisionMambaBackbone and others are assumed to return a single tensor
            # Other encoders might return a single tensor
            features_for_latent = encoder_output

        # 6. Pass features through the latent module to get the latent vector `z`
        z, mu, log_var = latent_module(features_for_latent, rng_key, training=training)

        # 7. Decode the latent vector to reconstruct the output.
        output_map = decoder(z, training=training)

        return output_map, mu, log_var
