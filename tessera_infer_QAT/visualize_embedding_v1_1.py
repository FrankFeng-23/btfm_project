#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise Tessera v1.1 QAT embeddings.

Loads the int8 embedding + per-pixel scales saved by src/infer_v1_1.py, dequantises
to fp32, and saves:
  - <prefix>_first3_rgb.png — first 3 embedding dimensions as an RGB image
                              (each channel min/max normalised independently)
  - <prefix>_pca3_rgb.png   — PCA to 3 components (for comparison)

Usage:
    python visualize_embedding_v1_1.py \
        --emb_int8  path/to/<prefix>_emb128_int8.npy \
        --emb_scales path/to/<prefix>_emb128_scales.npy \
        --out_dir   path/to/out_dir \
        --prefix    v1_1_austrian_crop
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def dequantize(emb_int8_path, scales_path):
    emb = np.load(emb_int8_path)          # (H, W, C) int8
    scales = np.load(scales_path)         # (H, W) float32
    if emb.ndim != 3:
        raise ValueError(f"Expected emb shape (H,W,C), got {emb.shape}")
    if scales.ndim == 3 and scales.shape[-1] == 1:
        scales = scales[..., 0]
    f32 = emb.astype(np.float32) * scales[..., None]
    return f32


def minmax_norm(img):
    out = img.astype(np.float32)
    for c in range(out.shape[-1]):
        lo = np.min(out[:, :, c])
        hi = np.max(out[:, :, c])
        if hi > lo:
            out[:, :, c] = (out[:, :, c] - lo) / (hi - lo)
        else:
            out[:, :, c] = 0.0
    return np.clip(out, 0.0, 1.0)


def save_rgb(img, path, title):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_int8", required=True)
    ap.add_argument("--emb_scales", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="v1_1")
    ap.add_argument("--skip_pca", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = dequantize(args.emb_int8, args.emb_scales)
    print(f"Dequantised shape: {data.shape}, dtype={data.dtype}")

    # First 3 dims as RGB
    first3 = minmax_norm(data[:, :, :3].copy())
    first3_path = os.path.join(args.out_dir, f"{args.prefix}_first3_rgb.png")
    save_rgb(first3, first3_path, "v1.1 embedding — first 3 dims (min-max per channel)")
    print("Saved:", first3_path)

    if args.skip_pca:
        return
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not available; skipping PCA.")
        return

    flat = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat)
    pca_img = proj.reshape(data.shape[0], data.shape[1], 3)
    pca_img = minmax_norm(pca_img)
    pca_path = os.path.join(args.out_dir, f"{args.prefix}_pca3_rgb.png")
    save_rgb(pca_img, pca_path, f"v1.1 embedding — PCA3 (explained={pca.explained_variance_ratio_.round(3)})")
    print("Saved:", pca_path)


if __name__ == "__main__":
    main()
