import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any


class SpectralNorm(nn.Module):
    layer: Any  # The wrapped layer (e.g., nn.Conv)
    power_iterations: int = 1

    @nn.compact
    def __call__(self, x, **kwargs):
        # Get the wrapped layer's kernel
        layer = self.layer
        params = self.param("params", layer.param_dtype, lambda *_: None)
        # If the layer is not initialized, call it to initialize
        if params is None:
            y = layer(x, **kwargs)
            params = self.get_variable("params", "kernel")
        else:
            y = None
        kernel = (
            self.variables["params"]["kernel"]
            if "kernel" in self.variables["params"]
            else params["kernel"]
        )
        w = kernel
        w_shape = w.shape
        w = w.reshape((w_shape[0], -1))
        # Spectral norm estimation
        u = self.variable(
            "spectral_norm",
            "u",
            lambda: jax.random.normal(self.make_rng("params"), (w.shape[0], 1)),
        )
        u_hat = u.value
        for _ in range(self.power_iterations):
            v_hat = jax.lax.stop_gradient(jnp.matmul(w.T, u_hat))
            v_hat = v_hat / (jnp.linalg.norm(v_hat) + 1e-12)
            u_hat = jax.lax.stop_gradient(jnp.matmul(w, v_hat))
            u_hat = u_hat / (jnp.linalg.norm(u_hat) + 1e-12)
        sigma = jnp.dot(u_hat.T, jnp.matmul(w, v_hat))
        w_sn = w / sigma
        w_sn = w_sn.reshape(w_shape)
        # Replace kernel with normalized version
        variables = self.scope.reinterpret({"params": {"kernel": w_sn}})
        return layer.apply(variables, x, **kwargs)
