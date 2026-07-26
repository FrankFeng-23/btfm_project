"""Per-pixel symmetric LINEAR int8 quantization for 128-d student embeddings.

Format produced by inference (per pixel):
    emb_int8 : (..., 128) int8    — round(emb / scale), clipped to [-127, 127]
    scale    : (...,)     float32  — per-pixel  max(|emb|) / 127

Storage: 128 B (int8) + 4 B (scale) = 132 B / pixel  (vs 512 B for f32 → ~3.9x).

Why linear (and not a non-linear codebook): after per-pixel max-abs
normalization these distilled embeddings quantize near-losslessly — plain
symmetric int8 preserves cosine to ~4 decimals (cos ≈ 0.99996 vs f32) on an
86k-pixel test, and a min-MSE non-linear (Lloyd-Max) codebook only edged it by a
negligible margin. Linear needs no per-model codebook and dequantizes with a
single multiply, so it is the default. (Non-linear helpers are kept below for
the rare case of a much smaller / heavier-tailed embedding.)

Dequantization is just:  emb_f32 = emb_int8.astype(float32) * scale[..., None]
"""
from typing import Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # torch optional for pure-numpy downstream use
    _HAS_TORCH = False


QMAX = 127  # symmetric int8 range [-127, 127]


# --------------------------------------------------------------------------- #
#  Linear quantize / dequantize  (numpy) — THE DEFAULT                          #
# --------------------------------------------------------------------------- #

def quantize(emb: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel symmetric linear int8 quantization.

    Args:
        emb : (..., 128) float — raw student embedding.
    Returns:
        code  : (..., 128) int8     — round(emb / scale), in [-127, 127].
        scale : (...,)     float32  — per-pixel max(|emb|) / 127.
    """
    emb = np.asarray(emb, dtype=np.float32)
    absmax = np.maximum(np.abs(emb).max(axis=-1), eps).astype(np.float32)   # (...,)
    scale = (absmax / QMAX).astype(np.float32)
    code = np.clip(np.round(emb / scale[..., None]), -QMAX, QMAX).astype(np.int8)
    return code, scale


def dequantize(code: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Invert `quantize`. The only thing a downstream consumer needs:

        emb_f32 = emb_int8.astype(float32) * scale

    Args:
        code  : (..., 128) int8
        scale : (...,)     float32
    Returns:
        emb : (..., 128) float32 reconstruction.
    """
    return (code.astype(np.float32) * scale[..., None]).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Linear quantize / dequantize  (torch, for on-GPU inference)                  #
# --------------------------------------------------------------------------- #

if _HAS_TORCH:

    @torch.no_grad()
    def quantize_torch(emb: "torch.Tensor", eps: float = 1e-12):
        """Torch version of `quantize`. emb (...,128) float → (code int8, scale f32)."""
        absmax = emb.abs().amax(dim=-1).clamp_min(eps)
        scale = (absmax / QMAX).to(torch.float32)
        code = torch.clamp((emb / scale.unsqueeze(-1)).round(), -QMAX, QMAX).to(torch.int8)
        return code, scale

    @torch.no_grad()
    def dequantize_torch(code: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
        """Torch version of `dequantize`."""
        return code.to(torch.float32) * scale.unsqueeze(-1)


# --------------------------------------------------------------------------- #
#  OPTIONAL non-linear (Lloyd-Max) codebook quantizer                           #
#  Not used by the default inference path. Kept for embeddings whose per-pixel  #
#  normalized distribution is peaked/heavy-tailed enough that linear wastes      #
#  resolution. Fit once with `fit_codebook`, then pass the codebook explicitly.  #
# --------------------------------------------------------------------------- #

N_LEVELS = 256


def _lloyd_max(v_sorted: np.ndarray, c_init: np.ndarray, iters: int) -> np.ndarray:
    c = c_init.astype(np.float64).copy()
    n = c.size
    for _ in range(iters):
        bounds = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(bounds, v_sorted)
        sums = np.bincount(idx, weights=v_sorted, minlength=n)
        cnts = np.bincount(idx, minlength=n)
        ne = cnts > 0
        c_new = c.copy()
        c_new[ne] = sums[ne] / cnts[ne]
        if np.max(np.abs(c_new - c)) < 1e-8:
            c = c_new
            break
        c = c_new
    return c


def _mse(v_sorted: np.ndarray, c: np.ndarray) -> float:
    bounds = 0.5 * (c[:-1] + c[1:])
    idx = np.searchsorted(bounds, v_sorted)
    return float(np.mean((v_sorted - c[idx]) ** 2))


def fit_codebook(values: np.ndarray, n_levels: int = N_LEVELS, iters: int = 60) -> np.ndarray:
    """MSE-optimal non-linear 1-D codebook (Lloyd, seeded from uniform so it is
    never worse than linear int8). values: normalized scalars in [-1, 1]."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(-1.0, 1.0, n_levels, dtype=np.float32)
    v.sort()
    lo, hi = float(v[0]), float(v[-1])
    c_uniform = np.linspace(lo, hi, n_levels)
    cands = [c_uniform, _lloyd_max(v, c_uniform, iters)]
    q = (np.arange(n_levels) + 0.5) / n_levels
    cands.append(_lloyd_max(v, np.quantile(v, q), iters))
    best = min(cands, key=lambda c: _mse(v, c))
    return np.sort(best).astype(np.float32)


def quantize_nonlinear(emb: np.ndarray, codebook: np.ndarray,
                        eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Non-linear int8 via a fitted codebook. Returns (code int8, scale f32)."""
    emb = np.asarray(emb, dtype=np.float32)
    scale = np.maximum(np.abs(emb).max(axis=-1), eps).astype(np.float32)
    v = emb / scale[..., None]
    bounds = 0.5 * (codebook[:-1] + codebook[1:])
    idx = np.searchsorted(bounds, v).astype(np.int16)
    return (idx - 128).astype(np.int8), scale


def dequantize_nonlinear(code: np.ndarray, scale: np.ndarray,
                          codebook: np.ndarray) -> np.ndarray:
    """Invert `quantize_nonlinear`: codebook[code + 128] * scale."""
    idx = code.astype(np.int32) + 128
    return (codebook[idx] * scale[..., None]).astype(np.float32)
