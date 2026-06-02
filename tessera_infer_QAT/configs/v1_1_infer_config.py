# configs/v1_1_infer_config.py
#
# Tessera v1.1 (QAT) inference config. Follows the same conventions as the v1.0
# `multi_tile_infer_config.py`: model- and runtime-only. Input data paths are NOT
# specified here — they are passed to the inference entry point via CLI
# (`--tile_path <dir>`), where `<dir>` is a standard Tessera preprocessing tile
# layout containing:
#     bands.npy, masks.npy, doys.npy,
#     sar_ascending.npy,  sar_ascending_doy.npy,
#     sar_descending.npy, sar_descending_doy.npy
#
# v1.1 ships TWO checkpoints, one per data source. Each has its OWN per-band
# normalisation stats — pick the matching `data_source` below or you'll feed
# garbage into the model.
#
#   data_source = "mpc"  -> Microsoft Planetary Computer (Sentinel-2 L2A + Sentinel-1 RTC)
#                           checkpoint: best_model_fsdp_20260425_133615.pt
#   data_source = "aws"  -> AWS Open Data (Earth-search S2 L2A + ASF OPERA RTC-S1)
#                           checkpoint: best_model_fsdp_20260425_202039.pt
#
# Stats live in `src/datasets/v1_1_norm_stats.py`.

config = {
    # ---------------- runtime ----------------
    "batch_size": 1024,
    "num_workers": 8,
    "use_bf16": True,
    "apply_amp": True,
    "log_interval_steps": 20,

    # ---------------- data source ----------------
    "data_source": "mpc",   # "mpc" or "aws" — MUST match the checkpoint you load.

    # ---------------- model (v1.1) ----------------
    "fusion_method": "concat",        # v1.1 pretraining uses concat fusion
    "latent_dim": 192,                # encoder width; transformer d_model = latent_dim * 4
    "representation_dim": 192,        # post dim_reducer representation width
    "save_embedding_dim": 128,        # saved embedding width (capped at 128)

    # Transformer settings (shared across the two backbones)
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 2048,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 2048,

    # Modalities — this revision of v1.1 uses a single merged S1 backbone.
    "use_s2": True,
    "use_s1": True,
    "split_s1_modalities": False,       # MUST be False for the new v1.1 ckps
    "disable_s1_ascending": False,

    # Quantisation
    "apply_qat_representation": True,   # save int8 + per-pixel scale
    "qat_representation_bits": 8,

    # ---------------- v1.1 observation selection ----------------
    # v1.1 keeps EVERY valid observation per pixel (no fixed `sample_size_s2/s1`
    # as in v1.0). Observation counts are bucketised to the next entry in
    # `num_obs_checkpoints`; pixels sharing a (s2_bin, s1_bin) bucket are batched
    # together so Transformer inputs within a batch have uniform sequence length.
    # Lower / fewer checkpoints = faster, coarser temporal resolution.
    # Higher / more    checkpoints = slower, richer temporal resolution.
    "num_obs_checkpoints": [8, 16, 24, 32, 40, 48, 56, 64,
                            72, 80, 88, 96, 104, 112, 120, 128,
                            136, 144, 152, 160, 168, 176, 184, 192,
                            200, 208, 216, 224, 232, 240, 248, 256],
}
