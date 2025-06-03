# src/datasets/ssl_dataset.py

import os
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset
import logging

import multiprocessing
import os
import pickle
from tqdm import tqdm
import threading
from joblib import Parallel, delayed
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mean and variance
S2_BAND_MEAN = np.array([1711.0938,1308.8511,1546.4543,3010.1293,3106.5083,
                        2068.3044,2685.0845,2931.5889,2514.6928,1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026,1862.9751,1803.1792,1741.7837,1677.4543,
                        1888.7862,1736.3090,1715.8104,1514.5199,1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

class SingleTileInferenceDataset(Dataset):
    """
    Dataset for single tile inference, only returns pixel (i, j) and corresponding s2/s1 data.
    No random sampling in __getitem__ because we need to repeat random sampling multiple times (10 times) and then average.
    """
    def __init__(self,
                 tile_path,
                 min_valid_timesteps=10,
                 standardize=True):
        super().__init__()
        self.tile_path = tile_path
        self.min_valid_timesteps = min_valid_timesteps
        self.standardize = standardize

        # Load S2
        s2_bands_path = os.path.join(tile_path, "bands.npy")    # (t_s2, H, W, 10)
        s2_masks_path = os.path.join(tile_path, "masks.npy")    # (t_s2, H, W)
        s2_doy_path   = os.path.join(tile_path, "doys.npy")     # (t_s2,)

        # >>> Fix UInt16 type: Convert to int32 or other general types <
        self.s2_bands = np.load(s2_bands_path).astype(np.float32)
        self.s2_masks = np.load(s2_masks_path).astype(np.int32)
        self.s2_doys  = np.load(s2_doy_path).astype(np.int32)   # Avoid uint16, convert to int32

        # Load S1 asc
        s1_asc_bands_path = os.path.join(tile_path, "sar_ascending.npy")      # (t_s1a, H, W, 2)
        s1_asc_doy_path   = os.path.join(tile_path, "sar_ascending_doy.npy")  # (t_s1a,)

        self.s1_asc_bands = np.load(s1_asc_bands_path).astype(np.float32)
        self.s1_asc_doys  = np.load(s1_asc_doy_path).astype(np.int32)

        # Load S1 desc
        s1_desc_bands_path = os.path.join(tile_path, "sar_descending.npy")      # (t_s1d, H, W, 2)
        s1_desc_doy_path   = os.path.join(tile_path, "sar_descending_doy.npy")  # (t_s1d,)

        self.s1_desc_bands = np.load(s1_desc_bands_path).astype(np.float32)
        self.s1_desc_doys  = np.load(s1_desc_doy_path).astype(np.int32)

        # Shape
        self.t_s2, self.H, self.W, _ = self.s2_bands.shape

        self.s2_band_mean = S2_BAND_MEAN
        self.s2_band_std = S2_BAND_STD
        self.s1_band_mean = S1_BAND_MEAN
        self.s1_band_std = S1_BAND_STD

        # Filter pixels
        self.valid_pixels = []
        ij_coords = np.indices((self.H, self.W)).reshape(2, -1).T
        for idx, (i, j) in enumerate(ij_coords):
            # s2 valid frames count
            s2_mask_ij = self.s2_masks[:, i, j]
            s2_valid = s2_mask_ij.sum()

            # s1 asc
            s1_asc_ij = self.s1_asc_bands[:, i, j, :]  # (t_s1a, 2)
            s1_asc_valid = np.any(s1_asc_ij != 0, axis=-1).sum()

            # s1 desc
            s1_desc_ij = self.s1_desc_bands[:, i, j, :]  # (t_s1d, 2)
            s1_desc_valid = np.any(s1_desc_ij != 0, axis=-1).sum()

            s1_total_valid = s1_asc_valid + s1_desc_valid

            # Added: Check if all values in S2 bands are 0
            s2_bands_ij = self.s2_bands[:, i, j, :]  # (t_s2, 10)
            s2_nonzero = np.any(s2_bands_ij != 0)  # Check if there are any non-zero values

            # Only keep the pixel if S2 bands have non-zero values and other conditions are met
            if s2_nonzero and (s2_valid >= self.min_valid_timesteps) and (s1_total_valid >= self.min_valid_timesteps):
                # Keep this pixel
                self.valid_pixels.append((idx, i, j))

        logging.info(f"[SingleTileInferenceDataset] tile={tile_path}, total_valid_pixels={len(self.valid_pixels)}")

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, index):
        global_idx, i, j = self.valid_pixels[index]

        # Get all s2, s1 data for the entire pixel
        s2_bands_ij = self.s2_bands[:, i, j, :]  # (t_s2, 10)
        s2_masks_ij = self.s2_masks[:, i, j]     # (t_s2,)
        s2_doys_ij  = self.s2_doys              # (t_s2,)

        s1_asc_bands_ij = self.s1_asc_bands[:, i, j, :]  # (t_s1a, 2)
        s1_asc_doys_ij  = self.s1_asc_doys               # (t_s1a,)

        s1_desc_bands_ij = self.s1_desc_bands[:, i, j, :]  # (t_s1d, 2)
        s1_desc_doys_ij  = self.s1_desc_doys               # (t_s1d,)

        sample = {
            "global_idx": global_idx,
            "i": i,
            "j": j,
            "s2_bands": s2_bands_ij,
            "s2_masks": s2_masks_ij,
            "s2_doys": s2_doys_ij,

            "s1_asc_bands": s1_asc_bands_ij,
            "s1_asc_doys": s1_asc_doys_ij,
            "s1_desc_bands": s1_desc_bands_ij,
            "s1_desc_doys": s1_desc_doys_ij,
        }
        return sample
