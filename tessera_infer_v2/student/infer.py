"""All-doy per-pixel inference helpers for Pixel Student V1.1.

Bin-padding semantics used at training time — at each pixel:

  - n_s2 = count of frames where this pixel's S2 mask == 1
  - n_s1 = count of MERGED s1 frames (asc + desc concat) with non-zero bands
  - bin  = smallest of {8, 16, 24, ..., 256} >= n_obs (capped at 256)
  - pad/subsample: linspace if n>=B, else split-and-repeat

Pixels in a (batch / tile) sharing the same (s2_bin, s1_bin) tuple are
batched together (consistent T per batch).

Two entry points:
  - encode_pixels(...): batch of independent per-pixel time series.
  - encode_tile(...):   one (T, H, W, C)-style tile -> (H, W, 128) emb map.

S1 inputs are MERGED (asc + desc concatenated along time) before binning,
matching how the model was trained.
"""
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch

try:  # works both as a package and as a standalone folder
    from .model import (
    PixelStudentV11,
    S1A_BAND_MEAN, S1A_BAND_STD, S1D_BAND_MEAN, S1D_BAND_STD,
    S2_BAND_MEAN, S2_BAND_STD,
    )
except ImportError:  # imported as plain modules from inside this folder
    from model import (
    PixelStudentV11,
    S1A_BAND_MEAN, S1A_BAND_STD, S1D_BAND_MEAN, S1D_BAND_STD,
    S2_BAND_MEAN, S2_BAND_STD,
    )


BIN_EDGES = list(range(8, 257, 8))   # [8, 16, 24, ..., 256]


def get_bin_size(n_obs: int) -> int:
    if n_obs <= 0:
        return 0
    for b in BIN_EDGES:
        if n_obs <= b:
            return b
    return BIN_EDGES[-1]


def _vec_get_bin_size(n_obs: np.ndarray) -> np.ndarray:
    out = np.full_like(n_obs, BIN_EDGES[-1])
    out[n_obs <= 0] = 0
    for b in reversed(BIN_EDGES):
        out = np.where((n_obs > 0) & (n_obs <= b), b, out)
    return out


def _pad_pattern(n: int, B: int) -> np.ndarray:
    """(B,) int64 indices into [0, n) reproducing the training pad_to_bin."""
    if n == 0:
        return np.zeros(B, dtype=np.int64)
    if n >= B:
        return np.linspace(0, n - 1, B, dtype=np.int64)
    remain = B - n
    if remain <= n:
        groups = np.array_split(np.arange(n), remain)
        fill = np.array([gp[len(gp) // 2] for gp in groups], dtype=np.int64)
    else:
        fill = (np.arange(remain) % n).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), fill])


def _build_source_indices(valid_per_pix: np.ndarray, B: int) -> np.ndarray:
    """Vectorized: O(unique_n) loop instead of O(G).

    Within a bin group all pixels have n_obs in (prev_bin, B], so at most
    `bin_step` distinct n values exist (typically <=8). We:
      1) stable-sort `~valid` so valid time-indices come first per row,
      2) group pixels by n_obs and compute the pad pattern once per n-subgroup,
      3) gather sorted_pos[pix, pattern] in one shot per n-subgroup.

    valid_per_pix: (G, T) bool. Returns (G, B) int64.
    """
    G, T = valid_per_pix.shape
    src = np.zeros((G, B), dtype=np.int64)
    if G == 0 or B == 0:
        return src
    n_per = valid_per_pix.sum(axis=1).astype(np.int64)
    sorted_pos = np.argsort(~valid_per_pix, axis=1, kind="stable").astype(np.int64)
    unique_n, inverse = np.unique(n_per, return_inverse=True)
    for ki, n_val in enumerate(unique_n):
        n = int(n_val)
        if n == 0:
            continue
        pix = np.where(inverse == ki)[0]
        pattern = _pad_pattern(n, B)
        src[pix] = sorted_pos[pix][:, pattern]
    return src


def _merge_s1(s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys):
    """Concat asc + desc along the time axis. Either may be None or empty."""
    parts_b, parts_d = [], []
    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        parts_b.append(s1_asc_bands); parts_d.append(s1_asc_doys)
    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        parts_b.append(s1_desc_bands); parts_d.append(s1_desc_doys)
    if not parts_b:
        return np.zeros((0, 0, 0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts_b, axis=0), np.concatenate(parts_d, axis=0)


@torch.no_grad()
def encode_pixels(
    model: PixelStudentV11,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    s2_masks: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cuda"),
    standardize: bool = True,
) -> np.ndarray:
    """Encode B independent pixels' time series into 128-d embeddings.

    Inputs are NumPy arrays in raw units — this function handles
    standardization with the v1.1 stats.

    Args:
        s2_bands     : (B, T_s2, 10)
        s2_doys      : (B, T_s2)        ints (1..365)
        s1_asc_bands : (B, T_s1a, 2)    raw (or None)
        s1_asc_doys  : (B, T_s1a)
        s1_desc_bands: (B, T_s1d, 2)    raw (or None)
        s1_desc_doys : (B, T_s1d)
        s2_masks     : (B, T_s2)        1=valid, 0=cloud (or None -> all valid)
        batch_pixels : max forward batch per (bin-tuple) group
        standardize  : z-score with the v1.1 stats (set False if pre-z-scored)

    Returns: (B, 128) float32.
    """
    B = s2_bands.shape[0]
    out = np.empty((B, model.repr_dim), dtype=np.float32)
    if B == 0:
        return out

    T_s2 = s2_bands.shape[1]

    # Build per-pixel S1 (merged) arrays. asc and desc each have their own
    # mean/std — z-score per-source BEFORE concatenation so the merged S1
    # stream the model consumes is already standardized.
    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        s1a_b = s1_asc_bands.astype(np.float32)
        s1a_valid = np.any(s1a_b != 0, axis=-1)   # raw valid BEFORE z-score (bugfix)
        if standardize:
            s1a_b = (s1a_b - S1A_BAND_MEAN) / (S1A_BAND_STD + 1e-9)
        if s1_asc_doys.ndim == 1:
            s1a_d = np.broadcast_to(s1_asc_doys[None, :], (B, s1_asc_doys.shape[0])).copy()
        else:
            s1a_d = s1_asc_doys
    else:
        s1a_b = np.zeros((B, 0, 2), dtype=np.float32)
        s1a_d = np.zeros((B, 0), dtype=np.float32)
        s1a_valid = np.zeros((B, 0), dtype=bool)

    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        s1d_b = s1_desc_bands.astype(np.float32)
        s1d_valid = np.any(s1d_b != 0, axis=-1)   # raw valid BEFORE z-score (bugfix)
        if standardize:
            s1d_b = (s1d_b - S1D_BAND_MEAN) / (S1D_BAND_STD + 1e-9)
        if s1_desc_doys.ndim == 1:
            s1d_d = np.broadcast_to(s1_desc_doys[None, :], (B, s1_desc_doys.shape[0])).copy()
        else:
            s1d_d = s1_desc_doys
    else:
        s1d_b = np.zeros((B, 0, 2), dtype=np.float32)
        s1d_d = np.zeros((B, 0), dtype=np.float32)
        s1d_valid = np.zeros((B, 0), dtype=bool)

    # Concat along time axis to get merged S1 (already z-scored above).
    if s1a_b.shape[1] + s1d_b.shape[1] > 0:
        s1_b_merged = np.concatenate([s1a_b, s1d_b], axis=1)               # (B, T_s1, 2)
        s1_d_merged = np.concatenate([s1a_d, s1d_d], axis=1)               # (B, T_s1)
    else:
        s1_b_merged = np.zeros((B, 0, 2), dtype=np.float32)
        s1_d_merged = np.zeros((B, 0), dtype=np.float32)
    T_s1 = s1_b_merged.shape[1]

    # Per-pixel valid masks.
    if s2_masks is not None:
        s2_v = s2_masks.astype(bool)                                        # (B, T_s2)
    else:
        s2_v = np.ones((B, T_s2), dtype=bool)
    s1_v = (np.concatenate([s1a_valid, s1d_valid], axis=1)
            if (s1a_valid.shape[1] + s1d_valid.shape[1]) > 0
            else np.zeros((B, 0), dtype=bool))                              # (B, T_s1)

    n_s2 = s2_v.sum(axis=1)
    n_s1 = s1_v.sum(axis=1)
    s2_bin = _vec_get_bin_size(n_s2).astype(np.int32)
    s1_bin = _vec_get_bin_size(n_s1).astype(np.int32)
    keys = s2_bin * 1000 + s1_bin
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    for ki, key in enumerate(unique_keys):
        s2_b_size = int(key // 1000); s1_b_size = int(key % 1000)
        idxs = np.where(inverse == ki)[0]
        if s2_b_size == 0 and s1_b_size == 0:
            continue
        s2_B = max(s2_b_size, 1); s1_B = max(s1_b_size, 1)
        for s in range(0, idxs.size, batch_pixels):
            chunk = idxs[s : s + batch_pixels]
            G = len(chunk)
            s2_in = np.zeros((G, s2_B, 11), dtype=np.float32)
            s1_in = np.zeros((G, s1_B, 3),  dtype=np.float32)

            if s2_b_size > 0:
                src = _build_source_indices(s2_v[chunk], s2_B)
                gathered = np.take_along_axis(
                    s2_bands[chunk], src[:, :, None].repeat(10, axis=2), axis=1
                )
                if standardize:
                    gathered = (gathered - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
                s2_in[:, :, :10] = gathered
                if s2_doys.ndim == 1:
                    s2_doys_pix = np.broadcast_to(s2_doys[None, :], (G, T_s2))
                else:
                    s2_doys_pix = s2_doys[chunk]
                s2_in[:, :, 10] = np.take_along_axis(s2_doys_pix, src, axis=1).astype(np.float32)

            if s1_b_size > 0:
                src = _build_source_indices(s1_v[chunk], s1_B)
                gathered = np.take_along_axis(
                    s1_b_merged[chunk], src[:, :, None].repeat(2, axis=2), axis=1
                )
                # s1_b_merged was already z-scored per-source (S1A/S1D) above.
                s1_in[:, :, :2] = gathered
                s1_in[:, :, 2] = np.take_along_axis(s1_d_merged[chunk], src, axis=1).astype(np.float32)

            s2_t = torch.from_numpy(s2_in).to(device, non_blocking=True)
            s1_t = torch.from_numpy(s1_in).to(device, non_blocking=True)
            emb = model.encode(s2_t, s1_t)
            out[chunk] = emb.float().cpu().numpy()
    return out


@torch.no_grad()
def encode_tile(
    model: PixelStudentV11,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s2_masks: Optional[np.ndarray] = None,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cuda"),
    standardize: bool = True,
) -> np.ndarray:
    """Encode one tile into an (H, W, 128) embedding map.

    Args:
        s2_bands     : (T_s2, H, W, 10)  raw reflectance
        s2_doys      : (T_s2,)            day-of-year per S2 frame
        s2_masks     : (T_s2, H, W) or None (1=valid)
        s1_asc_bands : (T_s1a, H, W, 2) optional
        s1_asc_doys  : (T_s1a,)         optional
        s1_desc_bands: (T_s1d, H, W, 2) optional
        s1_desc_doys : (T_s1d,)         optional

    Returns: (H, W, 128) float32.
    """
    T_s2, H, W, _ = s2_bands.shape
    N = H * W
    s2_flat = s2_bands.transpose(1, 2, 0, 3).reshape(N, T_s2, 10)
    s2_doys_flat = np.broadcast_to(s2_doys[None, :], (N, T_s2)).copy()
    s2_masks_flat = (s2_masks.transpose(1, 2, 0).reshape(N, T_s2)
                     if s2_masks is not None else None)

    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        Ta = s1_asc_bands.shape[0]
        s1a_flat = s1_asc_bands.transpose(1, 2, 0, 3).reshape(N, Ta, 2)
        s1a_doys_flat = np.broadcast_to(s1_asc_doys[None, :], (N, Ta)).copy()
    else:
        s1a_flat = None; s1a_doys_flat = None

    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        Td = s1_desc_bands.shape[0]
        s1d_flat = s1_desc_bands.transpose(1, 2, 0, 3).reshape(N, Td, 2)
        s1d_doys_flat = np.broadcast_to(s1_desc_doys[None, :], (N, Td)).copy()
    else:
        s1d_flat = None; s1d_doys_flat = None

    out = encode_pixels(
        model, s2_flat, s2_doys_flat,
        s1_asc_bands=s1a_flat, s1_asc_doys=s1a_doys_flat,
        s1_desc_bands=s1d_flat, s1_desc_doys=s1d_doys_flat,
        s2_masks=s2_masks_flat,
        batch_pixels=batch_pixels, device=device, standardize=standardize,
    )
    return out.reshape(H, W, model.repr_dim)
