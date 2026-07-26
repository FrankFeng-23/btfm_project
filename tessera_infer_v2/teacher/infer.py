"""Inference helpers for the TESSERA v2 2B teacher.

Per pixel, every valid observation is used. The observation count is bucketized
to the nearest bin in {8, 16, 24, ..., 256} and the series is padded or
subsampled to that bin size, matching the training-time procedure. Pixels
sharing the same (s2_bin, s1_bin) pair are batched together so every forward
pass sees a consistent T.

Sentinel-1 ascending and descending are concatenated along time in RAW units
and the merged stream is z-scored with the pooled S1 statistics.

Two entry points:
  - encode_pixels(...) : a batch of independent per-pixel time series.
  - encode_tile(...)   : one tile -> an (H, W, 1024) embedding map.
"""
from typing import Optional

import numpy as np
import torch

try:  # works both as a package and as a standalone folder
    from .model import S1_BAND_MEAN, S1_BAND_STD, S2_BAND_MEAN, S2_BAND_STD
except ImportError:
    from model import S1_BAND_MEAN, S1_BAND_STD, S2_BAND_MEAN, S2_BAND_STD


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
    """(B,) int64 indices into [0, n) reproducing the training-time padding."""
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
    """valid_per_pix: (G, T) bool -> (G, B) int64 gather indices."""
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
        src[pix] = sorted_pos[pix][:, _pad_pattern(n, B)]
    return src


@torch.no_grad()
def encode_pixels(
    model,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    s2_masks: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cuda"),
    autocast_dtype: Optional[torch.dtype] = None,
    standardize: bool = True,
) -> np.ndarray:
    """Encode B independent pixels' time series into 1024-d representations.

    Args:
        s2_bands     : (B, T_s2, 10)  raw reflectance
        s2_doys      : (B, T_s2) or (T_s2,)   integer day-of-year
        s1_asc_bands : (B, T_s1a, 2)  raw (or None)
        s1_asc_doys  : (B, T_s1a) or (T_s1a,)
        s1_desc_bands: (B, T_s1d, 2)  raw (or None)
        s1_desc_doys : (B, T_s1d) or (T_s1d,)
        s2_masks     : (B, T_s2)      1=valid, 0=cloud (or None -> all valid)
        autocast_dtype: e.g. torch.bfloat16 to halve activation memory on GPU
        standardize  : z-score inputs (set False if already standardized)

    Returns: (B, 1024) float32.
    """
    B = s2_bands.shape[0]
    out = np.empty((B, model.repr_dim), dtype=np.float32)
    if B == 0:
        return out
    T_s2 = s2_bands.shape[1]

    def _bcast(doys, n):
        if doys is None:
            return np.zeros((B, 0), dtype=np.float32)
        if doys.ndim == 1:
            return np.broadcast_to(doys[None, :], (B, n)).copy()
        return doys

    # Merge S1 in RAW units, then z-score the merged stream with pooled stats.
    parts_b, parts_d = [], []
    if s1_asc_bands is not None and s1_asc_bands.size > 0:
        parts_b.append(s1_asc_bands.astype(np.float32))
        parts_d.append(_bcast(s1_asc_doys, s1_asc_bands.shape[1]))
    if s1_desc_bands is not None and s1_desc_bands.size > 0:
        parts_b.append(s1_desc_bands.astype(np.float32))
        parts_d.append(_bcast(s1_desc_doys, s1_desc_bands.shape[1]))

    if parts_b:
        s1_raw = np.concatenate(parts_b, axis=1)                    # (B, T_s1, 2)
        s1_doy = np.concatenate(parts_d, axis=1)                    # (B, T_s1)
        s1_valid = np.any(s1_raw != 0, axis=-1)   # validity from RAW, pre-z-score
        s1_b = ((s1_raw - S1_BAND_MEAN) / (S1_BAND_STD + 1e-9)
                if standardize else s1_raw)
    else:
        s1_b = np.zeros((B, 0, 2), dtype=np.float32)
        s1_doy = np.zeros((B, 0), dtype=np.float32)
        s1_valid = np.zeros((B, 0), dtype=bool)

    s2_v = (s2_masks.astype(bool) if s2_masks is not None
            else np.ones((B, T_s2), dtype=bool))
    s2_doys_pix = _bcast(s2_doys, T_s2)

    s2_bin = _vec_get_bin_size(s2_v.sum(axis=1)).astype(np.int32)
    s1_bin = _vec_get_bin_size(s1_valid.sum(axis=1)).astype(np.int32)
    keys = s2_bin * 1000 + s1_bin
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    for ki, key in enumerate(unique_keys):
        s2_b_size, s1_b_size = int(key // 1000), int(key % 1000)
        idxs = np.where(inverse == ki)[0]
        if s2_b_size == 0 and s1_b_size == 0:
            out[idxs] = 0.0
            continue
        s2_B, s1_B = max(s2_b_size, 1), max(s1_b_size, 1)

        for s in range(0, idxs.size, batch_pixels):
            chunk = idxs[s: s + batch_pixels]
            G = len(chunk)
            s2_in = np.zeros((G, s2_B, 11), dtype=np.float32)
            s1_in = np.zeros((G, s1_B, 3), dtype=np.float32)

            if s2_b_size > 0:
                src = _build_source_indices(s2_v[chunk], s2_B)
                gathered = np.take_along_axis(
                    s2_bands[chunk], src[:, :, None].repeat(10, axis=2), axis=1)
                if standardize:
                    gathered = (gathered - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
                s2_in[:, :, :10] = gathered
                s2_in[:, :, 10] = np.take_along_axis(
                    s2_doys_pix[chunk], src, axis=1).astype(np.float32)

            if s1_b_size > 0:
                src = _build_source_indices(s1_valid[chunk], s1_B)
                s1_in[:, :, :2] = np.take_along_axis(
                    s1_b[chunk], src[:, :, None].repeat(2, axis=2), axis=1)
                s1_in[:, :, 2] = np.take_along_axis(
                    s1_doy[chunk], src, axis=1).astype(np.float32)

            s2_t = torch.from_numpy(s2_in).to(device, non_blocking=True)
            s1_t = torch.from_numpy(s1_in).to(device, non_blocking=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    emb = model.encode(s2_t, s1_t)
            else:
                emb = model.encode(s2_t, s1_t)
            out[chunk] = emb.float().cpu().numpy()
    return out


@torch.no_grad()
def encode_tile(
    model,
    s2_bands: np.ndarray,
    s2_doys: np.ndarray,
    s2_masks: Optional[np.ndarray] = None,
    s1_asc_bands: Optional[np.ndarray] = None,
    s1_asc_doys: Optional[np.ndarray] = None,
    s1_desc_bands: Optional[np.ndarray] = None,
    s1_desc_doys: Optional[np.ndarray] = None,
    batch_pixels: int = 1024,
    device: torch.device = torch.device("cuda"),
    autocast_dtype: Optional[torch.dtype] = None,
    standardize: bool = True,
) -> np.ndarray:
    """Encode one tile into an (H, W, 1024) representation map.

    Args:
        s2_bands     : (T_s2, H, W, 10)  raw reflectance
        s2_doys      : (T_s2,)           day-of-year per S2 frame
        s2_masks     : (T_s2, H, W)      1=valid (or None)
        s1_asc_bands : (T_s1a, H, W, 2)  optional
        s1_desc_bands: (T_s1d, H, W, 2)  optional
    """
    T_s2, H, W, _ = s2_bands.shape
    N = H * W

    def _flat(arr, doys):
        if arr is None or arr.size == 0:
            return None, None
        T = arr.shape[0]
        return (arr.transpose(1, 2, 0, 3).reshape(N, T, arr.shape[-1]),
                np.broadcast_to(doys[None, :], (N, T)).copy())

    s2_flat = s2_bands.transpose(1, 2, 0, 3).reshape(N, T_s2, 10)
    s2_doys_flat = np.broadcast_to(s2_doys[None, :], (N, T_s2)).copy()
    s2_masks_flat = (s2_masks.transpose(1, 2, 0).reshape(N, T_s2)
                     if s2_masks is not None else None)
    s1a_flat, s1a_doys_flat = _flat(s1_asc_bands, s1_asc_doys)
    s1d_flat, s1d_doys_flat = _flat(s1_desc_bands, s1_desc_doys)

    out = encode_pixels(
        model, s2_flat, s2_doys_flat,
        s1_asc_bands=s1a_flat, s1_asc_doys=s1a_doys_flat,
        s1_desc_bands=s1d_flat, s1_desc_doys=s1d_doys_flat,
        s2_masks=s2_masks_flat, batch_pixels=batch_pixels, device=device,
        autocast_dtype=autocast_dtype, standardize=standardize,
    )
    return out.reshape(H, W, model.repr_dim)
