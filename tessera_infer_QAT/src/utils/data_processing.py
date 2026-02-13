#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                    band_mean, band_std, sample_size_s2, standardize=True, profiler=None):
    """
    Process S2 batch data with random sampling.
      s2_bands_batch.shape = (B, T_s2, 10)
      s2_masks_batch.shape = (B, T_s2)
      s2_doys_batch.shape  = (B, T_s2)
    Returns: np.array, shape=(B, sample_size_s2, 11), dtype float32
    """
    if profiler:
        profiler.start('sample_s2_batch')
        
    B = s2_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        valid_idx = np.nonzero(s2_masks_batch[b])[0]
        
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s2_bands_batch.shape[1])
        
        if len(valid_idx) < sample_size_s2:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s2_bands_batch[b, idx_chosen, :]  # (sample_size_s2, 10)
        sub_doys  = s2_doys_batch[b, idx_chosen]      # (sample_size_s2,)
        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s2, 11)
        
        out_list.append(out_arr.astype(np.float32))

    result = np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s2, 11)
    
    if profiler:
        profiler.end('sample_s2_batch')
        
    return result


def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                    s1_desc_bands_batch, s1_desc_doys_batch,
                    band_mean, band_std, sample_size_s1, standardize=True, profiler=None):
    """
    Process S1 batch data with random sampling.
      s1_asc_bands_batch.shape = (B, t_s1a, 2)
      s1_asc_doys_batch.shape  = (B, t_s1a)
      s1_desc_bands_batch.shape= (B, t_s1d, 2)
      s1_desc_doys_batch.shape = (B, t_s1d)
    Returns: np.array, shape=(B, sample_size_s1, 3), dtype float32
    """
    if profiler:
        profiler.start('sample_s1_batch')
        
    B = s1_asc_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)  # shape (t_s1a+t_s1d, 2)
        s1_doys_all  = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s1_bands_all.shape[0])
        if len(valid_idx) < sample_size_s1:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s1_bands_all[idx_chosen, :]  # (sample_size_s1, 2)
        sub_doys  = s1_doys_all[idx_chosen]

        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s1, 3)
        
        out_list.append(out_arr.astype(np.float32))

    result = np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s1, 3)
    
    if profiler:
        profiler.end('sample_s1_batch')
        
    return result