# configs/ssl_config.py

config = {
    "batch_size": 1024,
    "epochs": 1,
    "learning_rate": 0.01,
    "barlow_lambda": 5e-3,
    "fusion_method": "concat",  # 'sum', 'concat'
    "latent_dim": 128,
    "projector_hidden_dim": 2048,
    "projector_out_dim": 2048,
    "min_valid_timesteps": 20,
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 16,
    "shuffle_tiles": True,
    "log_interval_steps":10,
    "val_interval_steps": 300,
    "eval_method": "linear_probe",
    "warmup_ratio": 0.2,
    "plateau_ratio": 0.2,
    "apply_mixup": False,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    "total_samples": 8500000
}
