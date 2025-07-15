import flax.linen as nn
import jax.numpy as jnp


class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    def setup(self):
        if not isinstance(self.patch_size, int):
            raise ValueError("Patch size must be a single integer for square patches.")

    @nn.compact
    def __call__(self, x):
        # The model receives channels-first (B, C, H, W) input.
        # nn.Conv expects channels-last (B, H, W, C) input.
        # We transpose the input to the expected format.
        if x.ndim == 4 and x.shape[1] < x.shape[3]:  # A simple heuristic for NCHW
            x = x.transpose((0, 2, 3, 1))

        # x: (batch_size, height, width, channels)
        batch_size, height, width, _ = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Input height and width must be divisible by patch_size.")

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="patch_conv",
        )(x)
        # x: (batch_size, num_patches_h, num_patches_w, embed_dim)

        num_patches_h, num_patches_w = x.shape[1], x.shape[2]
        x = x.reshape((batch_size, num_patches_h * num_patches_w, self.embed_dim))
        # x: (batch_size, num_patches, embed_dim)
        return x


class MambaBlock(nn.Module):
    dim: int  # Model dimension D
    d_state: int = 16  # SSM state expansion factor N
    d_conv: int = 4  # Local conv kernel size
    expand: int = 2  # Block expansion factor E for d_inner = E * D

    def _ssm_scan_fn(self, carry, scan_inputs_t):
        h_prev, A_log = carry
        x_ssm_t, dt_t, B_t_scan, C_t_scan = scan_inputs_t

        A = -jnp.exp(A_log[None, ...])
        delta_A = dt_t[..., None] * A
        A_bar_t = jnp.exp(delta_A)
        B_bar_t_eff = dt_t[..., None] * B_t_scan
        h_next = A_bar_t * h_prev + B_bar_t_eff * x_ssm_t[..., None]
        y_t = jnp.sum(C_t_scan * h_next, axis=-1)

        return (h_next, A_log), y_t

    def setup(self):
        self.d_inner = self.dim * self.expand  # D_in = E * D

        self.in_proj = nn.Dense(features=self.d_inner * 2, name="in_proj")

        self.conv1d = nn.Conv(
            features=self.d_inner,
            kernel_size=(self.d_conv,),
            padding=[(self.d_conv - 1, 0)],  # Causal padding
            feature_group_count=self.d_inner,  # Depthwise convolution
            name="conv1d",
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

        # SSM parameters
        # A_log: (d_inner, d_state) - learnable log of A matrix components
        self.A_log = self.param(
            "A_log", nn.initializers.normal(stddev=0.02), (self.d_inner, self.d_state)
        )

        # Projection for dt, B, C from input x
        # dt_rank is often d_inner / 16. B and C are projected to d_state.
        dt_rank_eff = max(1, self.d_inner // 16)
        self.x_proj_dt_B_C = nn.Dense(
            features=dt_rank_eff + 2 * self.d_state, name="x_proj_dt_B_C"
        )

        # Projection from dt_rank_eff to d_inner for delta (dt)
        self.dt_proj = nn.Dense(
            features=self.d_inner,
            name="dt_proj",
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(),
        )  # Bias for dt often initialized specially

        # D (skip connection parameter for SSM output)
        self.D_ssm = self.param("D_ssm", nn.initializers.ones, (self.d_inner,))

        self.out_proj = nn.Dense(features=self.dim, name="out_proj")

    @nn.compact
    def __call__(self, x, deterministic: bool):
        batch_size, seq_len, _ = x.shape

        xz_proj = self.in_proj(x)
        x_proj, z = jnp.split(xz_proj, 2, axis=-1)

        x_conv = self.conv1d(x_proj)
        x_conv_act = nn.silu(x_conv)  # This is the main input (x_t) to the SSM part

        # Project x_conv_act to get dt_intermediate, B_continuous, C_continuous
        dt_rank_eff = max(1, self.d_inner // 16)
        # dt_B_C_params_proj: (B, L, dt_rank_eff + 2*d_state)
        dt_B_C_params_proj = self.x_proj_dt_B_C(x_conv_act)

        dt_intermediate, B_continuous, C_continuous = jnp.split(
            dt_B_C_params_proj, [dt_rank_eff, dt_rank_eff + self.d_state], axis=-1
        )  # dt_inter:(B,L,dt_r), B_cont:(B,L,N), C_cont:(B,L,N)

        # Get delta (dt) for SSM: (B, L, d_inner)
        dt = self.dt_proj(dt_intermediate)
        dt = nn.softplus(dt)  # Ensure positivity

        # B_continuous and C_continuous are (B, L, d_state).
        # For the scan, they need to be (B, L, 1, d_state) to be broadcast across d_inner.
        B_for_scan = B_continuous[:, :, None, :]  # (B, L, 1, d_state)
        C_for_scan = C_continuous[:, :, None, :]  # (B, L, 1, d_state)

        initial_h = jnp.zeros((batch_size, self.d_inner, self.d_state), dtype=x.dtype)
        initial_carry = (initial_h, self.A_log)

        scan_inputs_for_fn = (
            x_conv_act.transpose((1, 0, 2)),  # (L, B, d_inner)
            dt.transpose((1, 0, 2)),  # (L, B, d_inner)
            B_for_scan.transpose((1, 0, 2, 3)),  # (L, B, 1, d_state)
            C_for_scan.transpose((1, 0, 2, 3)),  # (L, B, 1, d_state)
        )

        scan = nn.scan(
            MambaBlock._ssm_scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        final_carry, y_ssm_scanned = scan(self, initial_carry, scan_inputs_for_fn)

        y_ssm = y_ssm_scanned.transpose((1, 0, 2))  # (B, L, d_inner)

        y_ssm = y_ssm + x_conv_act * self.D_ssm  # Apply D skip connection

        y_gated = y_ssm * nn.silu(z)

        output = self.out_proj(y_gated)

        return output


class VisionMambaBackbone(nn.Module):
    patch_size: int
    embed_dim: int
    depth: int
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    bidirectional: bool = False
    use_pos_embed: bool = True
    drop_rate: float = 0.0  # Not used in MambaBlock currently, but for pos_embed

    def setup(self):
        if self.depth <= 0:
            raise ValueError("Depth must be a positive integer.")

    @nn.compact
    def __call__(self, x, training: bool):
        deterministic = not training

        # Need to get H, W for reshaping later
        if x.ndim == 4 and x.shape[1] < x.shape[3]:  # NCHW
            _, _, H, W = x.shape
        elif x.ndim == 4:  # NHWC
            _, H, W, _ = x.shape
        else:
            # This case should ideally not be hit if input is always image-like
            raise ValueError("VisionMambaBackbone expects a 4D input.")

        x = PatchEmbedding(
            patch_size=self.patch_size, embed_dim=self.embed_dim, name="patch_embed"
        )(x)

        if self.use_pos_embed:
            num_patches = x.shape[1]
            pos_embed = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02, dtype=x.dtype),
                (1, num_patches, self.embed_dim),
            )
            x = x + pos_embed
            x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)

        for i in range(self.depth):
            block_input = x
            if self.bidirectional:
                x_fwd = MambaBlock(
                    dim=self.embed_dim,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    name=f"mamba_block_fwd_{i}",
                )(block_input, deterministic=deterministic)

                x_rev_input = jnp.flip(block_input, axis=1)
                x_rev_processed = MambaBlock(
                    dim=self.embed_dim,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    name=f"mamba_block_bwd_{i}",  # Distinct params for backward pass
                )(x_rev_input, deterministic=deterministic)
                x_bwd = jnp.flip(x_rev_processed, axis=1)

                x = x_fwd + x_bwd  # Simple summation for bidirectionality
            else:
                x = MambaBlock(
                    dim=self.embed_dim,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    name=f"mamba_block_{i}",
                )(block_input, deterministic=deterministic)

            # Residual connection (common in transformer-like blocks)
            x = x + block_input
            x = nn.LayerNorm(name=f"norm_{i}", dtype=x.dtype)(x)

        x = nn.LayerNorm(name="final_norm", dtype=x.dtype)(x)

        # Reshape back to 4D for convolutional downstream tasks
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        x = x.reshape(x.shape[0], num_patches_h, num_patches_w, -1)

        return {"out": x}
