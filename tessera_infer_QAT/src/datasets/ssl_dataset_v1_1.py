# src/datasets/ssl_dataset_v1_1.py
#
# Tessera v1.1 inference dataset with bucketised "all observations" resampling.
#
# Unlike v1.0 (which draws a fixed sample_size_s2/s1 random subset per pixel), v1.1
# preserves every valid observation. To keep batches rectangular the observation
# count is bucketised: for each pixel we pick the smallest `num_obs_checkpoints`
# entry >= actual_valid_count, capped at the largest checkpoint. Pixels sharing the
# same (s2_bin, s1a_bin, s1d_bin) key are batched together by the custom batch
# sampler (see bucket_sampler.py).

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.ssl_dataset import S1_BAND_MEAN, S1_BAND_STD, S2_BAND_MEAN, S2_BAND_STD

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
        s2_bands:  (s2_target, 11)   float32 (standardised S2 bands + doy)
        s1a_bands: (s1a_target, 3)   float32 (standardised S1-asc bands + doy)
        s1d_bands: (s1d_target, 3)   float32 (standardised S1-desc bands + doy)
        i, j:      pixel coordinates
        global_idx: flat pixel index (H*W convention, row-major)
        bin_key:   (s2_target, s1a_target, s1d_target) for the custom batch sampler
    """

    def __init__(self, tile_path, config):
        super().__init__()
        self.config = config
        self.tile_path = tile_path

        self.use_s2 = bool(config.get("use_s2", True))
        self.use_s1 = bool(config.get("use_s1", True))
        self.split_s1 = bool(config.get("split_s1_modalities", True))
        self.disable_s1_asc = bool(config.get("disable_s1_ascending", False))

        ckps = sorted({int(v) for v in config.get("num_obs_checkpoints", [8, 16, 32, 64, 128]) if int(v) > 0})
        if not ckps:
            raise ValueError("num_obs_checkpoints must contain positive integers.")
        self.ckps = ckps
        self.max_ckp = ckps[-1]

        s2_b, s2_m, s2_d, s1a_b, s1a_d, s1d_b, s1d_d = _resolve_paths(tile_path)

        # Memory-map the big arrays so we don't blow up RAM on a 459x518x94x10 volume.
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

        # Precompute valid-count per pixel per modality → bucket key per pixel.
        self.pixel_bin_keys, self.bins_to_indices = self._build_bin_keys()

        logging.info(
            f"[SingleTileInferenceDatasetV1_1] H={self.H}, W={self.W}, "
            f"pixels={self.num_pixels}, buckets={len(self.bins_to_indices)}"
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
        s1a_valid = np.zeros(self.num_pixels, dtype=np.int32)
        s1d_valid = np.zeros(self.num_pixels, dtype=np.int32)

        if self.s2_masks is not None:
            s2_valid = self.s2_masks.reshape(self.s2_masks.shape[0], -1).sum(axis=0).astype(np.int32, copy=False)

        if self.s1a_bands is not None and self.s1a_bands.shape[0] > 0:
            flat = np.any(self.s1a_bands != 0, axis=-1).reshape(self.s1a_bands.shape[0], -1)
            s1a_valid = flat.sum(axis=0).astype(np.int32, copy=False)
        if self.s1d_bands is not None and self.s1d_bands.shape[0] > 0:
            flat = np.any(self.s1d_bands != 0, axis=-1).reshape(self.s1d_bands.shape[0], -1)
            s1d_valid = flat.sum(axis=0).astype(np.int32, copy=False)

        keys = np.empty(self.num_pixels, dtype=np.dtype([("s2", np.int32), ("s1a", np.int32), ("s1d", np.int32)]))
        bins = {}
        for p in range(self.num_pixels):
            k = (self._to_bin(s2_valid[p]), self._to_bin(s1a_valid[p]), self._to_bin(s1d_valid[p]))
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
        sub_b = (sub_b - S2_BAND_MEAN) / (S2_BAND_STD + 1e-9)
        return np.hstack([sub_b, sub_d.reshape(-1, 1)]).astype(np.float32, copy=False)

    def _sample_s1_stream(self, bands_arr, doys_arr, i, j, target):
        if bands_arr is None or bands_arr.shape[0] == 0:
            return np.zeros((target, 3), dtype=np.float32)
        stream = bands_arr[:, i, j, :]
        valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
        if len(valid) == 0:
            return np.zeros((target, 3), dtype=np.float32)
        idx_local = build_resample_indices(len(valid), target)
        real = valid[idx_local]
        sub_b = stream[real].astype(np.float32, copy=False)
        sub_d = np.asarray(doys_arr)[real]
        sub_b = (sub_b - S1_BAND_MEAN) / (S1_BAND_STD + 1e-9)
        return np.hstack([sub_b, sub_d.reshape(-1, 1)]).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        i, j = self.pixel_coords[idx]
        bin_key = self.pixel_bin_keys[idx]
        s2_t, s1a_t, s1d_t = int(bin_key["s2"]), int(bin_key["s1a"]), int(bin_key["s1d"])

        s2 = self._sample_s2(i, j, s2_t) if self.use_s2 else np.zeros((0, 11), dtype=np.float32)
        s1a = self._sample_s1_stream(self.s1a_bands, self.s1a_doys, i, j, s1a_t) if self.use_s1 else np.zeros((0, 3), dtype=np.float32)
        s1d = self._sample_s1_stream(self.s1d_bands, self.s1d_doys, i, j, s1d_t) if self.use_s1 else np.zeros((0, 3), dtype=np.float32)

        return {
            "s2": torch.from_numpy(s2),
            "s1a": torch.from_numpy(s1a),
            "s1d": torch.from_numpy(s1d),
            "i": int(i),
            "j": int(j),
            "global_idx": int(i) * self.W + int(j),
            "bin_key": (s2_t, s1a_t, s1d_t),
        }
