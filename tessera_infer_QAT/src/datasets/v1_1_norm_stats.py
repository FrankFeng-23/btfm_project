# src/datasets/v1_1_norm_stats.py
#
# Per-source normalisation statistics used by Tessera v1.1 inference.
# v1.1 ships two checkpoints — one trained on Microsoft Planetary Computer (MPC)
# preprocessing output and one fine-tuned on AWS (Earth-search S2 + ASF OPERA S1)
# preprocessing output. Each checkpoint has its OWN per-band mean/std; mixing them
# silently corrupts the input distribution and degrades embedding quality.
#
# `data_source` is selected from the inference config (`config["data_source"]`).
# Note: although v1.1 uses a single merged S1 backbone (split_s1_modalities=False),
# S1 ascending and S1 descending are normalised with their OWN mean/std before
# being concatenated — this matches the v1.1 training-time preprocessing.

import numpy as np


# NOTE (2026-04-26): the AWS row below was computed from preprocessing output
# that double-applied the PB-04.00 BOA_ADD_OFFSET to AWS / Earth-search
# Sentinel-2 data (the bug fixed in tessera_preprocessing/s2_fast_processor.py).
# These AWS stats are therefore offset ~1000 lower for bands whose true value
# exceeds the offset on post-2022-01-25 acquisitions. They are kept here ONLY
# so the currently-published AWS v1.1 checkpoints continue to receive
# in-distribution inputs — i.e. AWS inference is consistent ONLY when paired
# with the OLD (buggy) preprocessing output. After the AWS checkpoint is
# re-pretrained on corrected preprocessing output, recompute and replace this
# row from the new training data and remove this notice.
NORM_STATS = {
    "mpc": {
        "s2_mean": np.array([2683.4553, 2223.3630, 2432.0950, 3633.1970, 3602.1755,
                             3006.4324, 3400.2710, 3515.6392, 2456.9163, 1983.8783], dtype=np.float32),
        "s2_std":  np.array([2739.5217, 2846.2993, 2690.8250, 2290.0439, 2088.8970,
                             2673.1106, 2381.4521, 2229.5225, 1601.0942, 1495.3545], dtype=np.float32),
        "s1a_mean": np.array([5588.3291, 3025.6270], dtype=np.float32),
        "s1a_std":  np.array([1713.4646, 1693.0471], dtype=np.float32),
        "s1d_mean": np.array([5552.9683, 2955.0520], dtype=np.float32),
        "s1d_std":  np.array([1685.5857, 1677.6414], dtype=np.float32),
    },
    "aws": {
        "s2_mean": np.array([2501.1238, 2113.7524, 2270.7112, 3315.6033, 3289.6584,
                             2757.2700, 3092.8628, 3212.9587, 2170.0745, 1759.2500], dtype=np.float32),
        "s2_std":  np.array([2739.4775, 2843.0742, 2685.2820, 2387.8638, 2194.2751,
                             2733.0715, 2481.0159, 2332.5276, 1673.3186, 1549.3647], dtype=np.float32),
        "s1a_mean": np.array([5664.5439, 2802.9736], dtype=np.float32),
        "s1a_std":  np.array([1678.7821, 1786.0414], dtype=np.float32),
        "s1d_mean": np.array([5710.6992, 2830.1045], dtype=np.float32),
        "s1d_std":  np.array([1616.1969, 1761.8499], dtype=np.float32),
    },
}


def get_stats(data_source: str):
    key = str(data_source).lower()
    if key not in NORM_STATS:
        raise ValueError(
            f"Unknown data_source={data_source!r}. Expected one of {sorted(NORM_STATS)}. "
            "MPC and AWS checkpoints have DIFFERENT normalisation stats — pick the one "
            "that matches your downloaded checkpoint."
        )
    return NORM_STATS[key]
