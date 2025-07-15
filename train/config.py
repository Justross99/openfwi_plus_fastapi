embed_dim = 64  # Example embedding dimension for Mamba encoder
dataset_name = "FlatVel_A"  # Example dataset name
CONFIG = {
    "seed": 42,
    "dataset_name": dataset_name,
    "data_dir": "/mnt/E6B6DC5AB6DC2D35/OPENFWI/"
    + dataset_name
    + "/data",  # E.g., /media/ross/TheXFiles/Openfwi_src/OPENFWI/CurveVel_A/data
    "label_dir": "/mnt/E6B6DC5AB6DC2D35/OPENFWI/"
    + dataset_name
    + "/labels",  # E.g., /media/ross/TheXFiles/Openfwi_src/OPENFWI/CurveVel_A/labels
    "output_dir": "output/training_run_long_plus_warmup_flatvel_a",  # For checkpoints and logs
    "batch_size": 8,  # Effective batch size after accumulation
    "num_train_epochs": 50,
    "gradient_accumulation_steps": 1,  # Accumulate gradients over 1 step
    "learning_rate": 1e-4,
    "lr_warmup_epochs": 2,
    "weight_decay": 0.01,
    "log_every_steps": 50,
    "checkpoint_every_steps": 1000,
    "eval_every_epochs": 1,  # Or steps
    "early_stopping_patience": 10,  # Epochs
    "early_stopping_metric": "val_loss",  # Metric to monitor
    "early_stopping_mode": "min",  # "min" for loss, "max" for accuracy etc.
    "kl_anneal_epochs": 10,  # Number of epochs to anneal KLD weight after warmup
    # --- Unified VAE Model Configuration ---
    "model_name": "MambaDU",
    "model_config": {
        # 1. Specify the encoder architecture
        "encoder_name": "mamba",
        # 2. Configuration for the chosen encoder
        "encoder_config": {
            "patch_size": 10,
            "embed_dim": embed_dim,
            "depth": 4,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
        },
        # 3. Dimension of the latent space
        "latent_dim": 64,
        # 4. Configuration for the new, efficient decoder
        "decoder_config": {
            # Number of features for the first convolutional block.
            "start_features": 128,
            # The decoder's upsampling path to the final output shape.
            "upsample_blocks": [
                (64, (85, 38)),  # Intermediate upsample and refinement
                (32, (70, 70)),  # Final upsample to target shape
            ],
            # The number of output channels for the final map.
            "output_channels": 1,
            # Final activation function for the output map.
            "final_activation": "sigmoid",
        },
    },
    "input_shape": (
        1,
        5,
        1000,
        70,
    ),  # (Batch, C, T, S) - for model init, Batch will be overridden by dataloader
    "output_shape_expected": (1, 1, 70, 70),  # For output map (Batch, C, H, W)
    # Loss specific config (Example for combined_loss)
    "loss_config": {
        "mae_weight": 1.0,
        "ssim_weight": 0.0,
        "percept_weight": 0.0,
        "kld_weight": 0.00,
        "ssim_window_size": 7,
        "max_velocity_val": 4500.0,  # Example
    },
    "perceptual_input_shape": (
        1,
        70,
        70,
    ),  # For initializing PerceptualFeatureExtractor
}
