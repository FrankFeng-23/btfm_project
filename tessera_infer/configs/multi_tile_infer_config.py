from copy import deepcopy
from configs.ssl_config import config

# Copy base config to avoid modifying it directly
config = deepcopy(config)

# Update only model-specific configuration
config.update({
    "repeat_times": 1,
    "min_valid_timesteps": 0,
    "fusion_method": "concat",  # Options: 'sum' or 'concat'
    "latent_dim": 128,
    "projector_hidden_dim": 8192*2,
    "projector_out_dim": 8192*2,
    "sample_size_s2": 40,
    "sample_size_s1": 40
})