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
#
# AWS row was refreshed on 2026-05-03 against the AWS pretraining data collected
# AFTER the BOA_ADD_OFFSET double-apply bug was fixed in
# `tessera_preprocessing/s2_fast_processor.py::harmonize_arr` and AFTER the AWS
# v1.1 checkpoint was retrained on that corrected output. The Landsat block under
# MPC is unused by the current v1.1 inference graph (use_landsat=False) and is
# kept for forward compatibility / fine-tuning recipes that turn Landsat on.

import numpy as np


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
        # Landsat first-6-bands; only consumed when use_landsat=True (v1.1 default off).
        "landsat_mean": np.array([16229.6855, 16999.7637, 17590.2109,
                                  21526.4531, 15207.2490, 13286.6963], dtype=np.float32),
        "landsat_std":  np.array([14196.8818, 13033.0557, 13078.4229,
                                  9735.7246,  5547.3047,  4899.0557], dtype=np.float32),
    },
    "aws": {
        "s2_mean": np.array([2793.6589, 2356.7776, 2551.0496, 3741.9229, 3713.7844,
                             3120.1997, 3516.3342, 3637.0342, 2501.0283, 2038.1504], dtype=np.float32),
        "s2_std":  np.array([2810.0093, 2933.8835, 2755.6360, 2344.5027, 2145.7986,
                             2743.9019, 2438.8601, 2286.5977, 1680.7367, 1585.5529], dtype=np.float32),
        "s1a_mean": np.array([5697.0859, 2838.6687], dtype=np.float32),
        "s1a_std":  np.array([1671.3737, 1789.4116], dtype=np.float32),
        "s1d_mean": np.array([5759.1367, 2873.2854], dtype=np.float32),
        "s1d_std":  np.array([1583.2858, 1747.8390], dtype=np.float32),
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
