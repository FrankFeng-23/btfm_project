# src/datasets/ssl_dataset_v1_1.py
#
# Tessera v1.1 inference dataset with bucketised "all observations" resampling.
#
# v1.1 architecture (this revision):
#   - Two backbones: S2 + S1-merged (split_s1_modalities = False, matching v1.0).
#   - S1 ascending / descending are concatenated time-wise into a single S1 stream,
#     but each is normalised with its OWN mean/std (asc -> S1A stats, desc -> S1D
#     stats) BEFORE concatenation. This matches v1.1 training preprocessing.
#
# v1.1 inference uses every valid observation per pixel; counts are bucketised to
# the next entry in `num_obs_checkpoints`. Pixels sharing a (s2_bin, s1_bin) key
# are batched together so attention sequences are rectangular.

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.v1_1_norm_stats import get_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_resample_indices(valid_len: int, target_size: int) -> np.ndarray:
    """Deterministic index vector that resamples `valid_len` observations to `target_size`.

    - target_size == valid_len: identity
    - target_size <  valid_len: evenly-spaced subsample (median element of each chunk)
    - target_size >  valid_len: every original index is kept; extras are chosen at
      evenly-spaced positions to fill the remainder (duplicates allowed).
    """
    valid_len = int(valid_len)
    target_size = int(target_size)
    if valid_len <= 0:
        return np.array([], dtype=np.int64)
    if target_size == valid_len:
        return np.arange(valid_len, dtype=np.int64)
    if target_size < valid_len:
        chunks = np.array_split(np.arange(valid_len), target_size)
        return np.array([c[len(c) // 2] for c in chunks if len(c) > 0], dtype=np.int64)
    extra = target_size - valid_len
    anchors = np.linspace(0, valid_len - 1, num=extra + 2, dtype=np.float64)[1:-1]
    extras = np.rint(anchors).astype(np.int64)
    extras = np.clip(extras, 0, valid_len - 1)
    return np.concatenate([np.arange(valid_len, dtype=np.int64), extras], axis=0)


def _resolve_paths(tile_path):
    """Standard Tessera preprocessing tile layout."""
    return (
        os.path.join(tile_path, "bands.npy"),
        os.path.join(tile_path, "masks.npy"),
        os.path.join(tile_path, "doys.npy"),
        os.path.join(tile_path, "sar_ascending.npy"),
        os.path.join(tile_path, "sar_ascending_doy.npy"),
        os.path.join(tile_path, "sar_descending.npy"),
        os.path.join(tile_path, "sar_descending_doy.npy"),
    )


class SingleTileInferenceDatasetV1_1(Dataset):
    """
    Per-pixel iteration with v1.1 bucketised all-observation resampling.

    __getitem__ returns a dict:
        s2:        (s2_target, 11)   float32 (standardised S2 bands + doy)
        s1:        (s1_target, 3)    float32 (S1-asc + S1-desc concatenated, each
                                              normalised with its own per-modality stats)
        i, j:      pixel coordinates
        global_idx: flat pixel index (row-major H*W)
        bin_key:   (s2_target, s1_target) for the custom batch sampler
    """

    def __init__(self, tile_path, config):
        super().__init__()
        self.config = config
        self.tile_path = tile_path

        self.use_s2 = bool(config.get("use_s2", True))
        self.use_s1 = bool(config.get("use_s1", True))
        self.disable_s1_asc = bool(config.get("disable_s1_ascending", False))

        # v1.1 (this revision) requires merged S1 — guard against accidentally
        # loading an older split-s1 ckp.
        if bool(config.get("split_s1_modalities", False)):
            raise ValueError(
                "v1.1 inference (this revision) expects split_s1_modalities=False. "
                "If you have an older split-S1 v1.1 checkpoint you need an older "
                "snapshot of this code."
            )

        ckps = sorted({int(v) for v in config.get("num_obs_checkpoints", [8, 16, 32, 64, 128]) if int(v) > 0})
        if not ckps:
            raise ValueError("num_obs_checkpoints must contain positive integers.")
        self.ckps = ckps
        self.max_ckp = ckps[-1]

        # Per-source normalisation stats (MPC vs AWS). Pick MUST match the ckp.
        stats = get_stats(config.get("data_source", "mpc"))
        self.s2_mean = stats["s2_mean"]
        self.s2_std  = stats["s2_std"]
        self.s1a_mean = stats["s1a_mean"]
        self.s1a_std  = stats["s1a_std"]
        self.s1d_mean = stats["s1d_mean"]
        self.s1d_std  = stats["s1d_std"]

        s2_b, s2_m, s2_d, s1a_b, s1a_d, s1d_b, s1d_d = _resolve_paths(tile_path)

        # Memory-map so we don't blow up RAM on a large (T, H, W, B) volume.
        self.s2_bands = np.load(s2_b, mmap_mode="r") if self.use_s2 else None
        self.s2_masks = np.load(s2_m, mmap_mode="r") if self.use_s2 else None
        self.s2_doys  = np.load(s2_d, mmap_mode="r") if self.use_s2 else None

        if self.use_s1 and not self.disable_s1_asc:
            self.s1a_bands = np.load(s1a_b, mmap_mode="r")
            self.s1a_doys  = np.load(s1a_d, mmap_mode="r")
        else:
            self.s1a_bands = None
            self.s1a_doys = None

        if self.use_s1:
            self.s1d_bands = np.load(s1d_b, mmap_mode="r")
            self.s1d_doys  = np.load(s1d_d, mmap_mode="r")
        else:
            self.s1d_bands = None
            self.s1d_doys = None

        # Image shape from whichever modality is available.
        self.H, self.W = self._infer_hw()
        self.num_pixels = self.H * self.W
        coords = np.indices((self.H, self.W)).reshape(2, -1).T
        self.pixel_coords = coords.astype(np.int64, copy=False)

        # Precompute valid-count per pixel per modality → (s2_bin, s1_bin) per pixel.
        self.pixel_bin_keys, self.bins_to_indices = self._build_bin_keys()

        logging.info(
            f"[SingleTileInferenceDatasetV1_1] H={self.H}, W={self.W}, "
            f"pixels={self.num_pixels}, buckets={len(self.bins_to_indices)}, "
            f"data_source={config.get('data_source', 'mpc')}"
        )

    def _infer_hw(self):
        for arr in (self.s2_bands, self.s1d_bands, self.s1a_bands):
            if arr is not None:
                return int(arr.shape[1]), int(arr.shape[2])
        raise ValueError("At least one modality must be enabled for inference.")

    def _to_bin(self, n: int) -> int:
        n = int(n)
        for c in self.ckps:
            if n <= c:
                return c
        return self.max_ckp

    def _build_bin_keys(self):
        s2_valid = np.zeros(self.num_pixels, dtype=np.int32)
        s1_valid = np.zeros(self.num_pixels, dtype=np.int32)

        if self.s2_masks is not None:
            s2_valid = self.s2_masks.reshape(self.s2_masks.shape[0], -1).sum(axis=0).astype(np.int32, copy=False)

        if self.s1a_bands is not None and self.s1a_bands.shape[0] > 0:
            flat_a = np.any(self.s1a_bands != 0, axis=-1).reshape(self.s1a_bands.shape[0], -1)
            s1_valid = s1_valid + flat_a.sum(axis=0).astype(np.int32, copy=False)
        if self.s1d_bands is not None and self.s1d_bands.shape[0] > 0:
            flat_d = np.any(self.s1d_bands != 0, axis=-1).reshape(self.s1d_bands.shape[0], -1)
            s1_valid = s1_valid + flat_d.sum(axis=0).astype(np.int32, copy=False)

        keys = np.empty(self.num_pixels, dtype=np.dtype([("s2", np.int32), ("s1", np.int32)]))
        bins = {}
        for p in range(self.num_pixels):
            k = (self._to_bin(s2_valid[p]), self._to_bin(s1_valid[p]))
            keys[p] = k
            bins.setdefault(k, []).append(p)
        return keys, bins

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_pixels

    def _sample_s2(self, i, j, target):
        bands = self.s2_bands[:, i, j, :]
        masks = self.s2_masks[:, i, j]
        valid = np.nonzero(masks)[0]
        if len(valid) == 0:
            return np.zeros((target, bands.shape[1] + 1), dtype=np.float32)
        idx_local = build_resample_indices(len(valid), target)
        real = valid[idx_local]
        sub_b = bands[real].astype(np.float32, copy=False)
        sub_d = np.asarray(self.s2_doys)[real]
        sub_b = (sub_b - self.s2_mean) / (self.s2_std + 1e-9)
        return np.hstack([sub_b, sub_d.reshape(-1, 1)]).astype(np.float32, copy=False)

    def _sample_s1_merged(self, i, j, target):
        """Concatenate asc+desc for pixel (i,j), each normalised with its own stats.

        We collect valid asc rows, valid desc rows, normalise each block, then
        concatenate (asc first, desc second) and resample to `target` length.
        """
        asc_b, asc_d = None, None
        if self.s1a_bands is not None and self.s1a_bands.shape[0] > 0:
            stream = self.s1a_bands[:, i, j, :]
            valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
            if len(valid) > 0:
                asc_b = stream[valid].astype(np.float32, copy=False)
                asc_b = (asc_b - self.s1a_mean) / (self.s1a_std + 1e-9)
                asc_d = np.asarray(self.s1a_doys)[valid].astype(np.float32, copy=False)

        desc_b, desc_d = None, None
        if self.s1d_bands is not None and self.s1d_bands.shape[0] > 0:
            stream = self.s1d_bands[:, i, j, :]
            valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
            if len(valid) > 0:
                desc_b = stream[valid].astype(np.float32, copy=False)
                desc_b = (desc_b - self.s1d_mean) / (self.s1d_std + 1e-9)
                desc_d = np.asarray(self.s1d_doys)[valid].astype(np.float32, copy=False)

        parts_b = [b for b in (asc_b, desc_b) if b is not None]
        parts_d = [d for d in (asc_d, desc_d) if d is not None]
        if not parts_b:
            return np.zeros((target, 3), dtype=np.float32)

        all_b = np.concatenate(parts_b, axis=0)
        all_d = np.concatenate(parts_d, axis=0)

        idx_local = build_resample_indices(len(all_b), target)
        sub_b = all_b[idx_local]
        sub_d = all_d[idx_local]
        return np.hstack([sub_b, sub_d.reshape(-1, 1)]).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        i, j = self.pixel_coords[idx]
        bin_key = self.pixel_bin_keys[idx]
        s2_t, s1_t = int(bin_key["s2"]), int(bin_key["s1"])

        s2 = self._sample_s2(i, j, s2_t) if self.use_s2 else np.zeros((0, 11), dtype=np.float32)
        s1 = self._sample_s1_merged(i, j, s1_t) if self.use_s1 else np.zeros((0, 3), dtype=np.float32)

        return {
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1),
            "i": int(i),
            "j": int(j),
            "global_idx": int(i) * self.W + int(j),
            "bin_key": (s2_t, s1_t),
        }
