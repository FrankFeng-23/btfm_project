# CPU optimized configuration for faster inference

config = {
    # Model architecture
    "latent_dim": 128,
    "fusion_method": "concat",  # or "sum"
    
    # Projection head config
    "projector_hidden_dim": 8192*2,
    "projector_out_dim": 8192*2,
    
    # Sampling config - optimized for CPU
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "repeat_times": 1,
    
    # Data loading config - optimized for CPU
    "batch_size": 256,     # Smaller batch size for CPU
    "num_workers": 1,      # Fewer workers for CPU to avoid overhead
    "min_valid_timesteps": 0,
    
    # Paths (will be overridden by command line args)
    "checkpoint_path": "",
    "output_dir": "",
}