#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tessera v1.1 QAT inference entry point (single-tile, single-process, GPU or CPU).
#
# Data layout (same as v1.0 `tessera_infer` / `tessera_infer_QAT` pipeline): point
# `--tile_path` at a directory produced by `tessera_preprocessing`, containing
#     bands.npy, masks.npy, doys.npy,
#     sar_ascending.npy,  sar_ascending_doy.npy,
#     sar_descending.npy, sar_descending_doy.npy
#
# Usage (GPU):
#     python src/infer_v1_1.py \
#         --config          configs/v1_1_infer_config.py \
#         --checkpoint_path checkpoints/best_model_fsdp_20260408_154724.pt \
#         --tile_path       /path/to/retiled_d_pixel/0_3500_500_4000 \
#         --output_dir      /path/to/representation_retiled_v1_1 \
#         --output_prefix   0_3500_500_4000
#
# Output (in --output_dir):
#     <prefix>_emb128_int8.npy   (H, W, 128) int8
#     <prefix>_emb128_scales.npy (H, W)      float32 per-pixel scale
#
#     To reconstruct fp32: emb_int8.astype(float32) * scales[..., None]

import argparse
import importlib.util
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION", "1")

# Make `src/` imports work regardless of how the script is launched.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.ssl_dataset_v1_1 import SingleTileInferenceDatasetV1_1
from models.quantization import quantize_tensor_symmetric_per_row  # added below
from models.ssl_model_v1_1 import build_v1_1_inference_model, load_v1_1_checkpoint


# ---------------------------------------------------------------------------
# Bucketised batch sampler: each batch contains pixels sharing the same bin_key
# so Transformer inputs within a batch have identical sequence length.
# ---------------------------------------------------------------------------

class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset: SingleTileInferenceDatasetV1_1, batch_size: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self._batches = []
        for key in sorted(dataset.bins_to_indices.keys()):
            idxs = dataset.bins_to_indices[key]
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start:start + self.batch_size]
                if chunk:
                    self._batches.append(chunk)

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)


def _collate(batch_list):
    """Pixels in one batch share a bin_key → all tensors already have identical
    seq length; we just stack."""
    return {
        "s2": torch.stack([b["s2"] for b in batch_list], dim=0),
        "s1": torch.stack([b["s1"] for b in batch_list], dim=0),
        "i": torch.tensor([b["i"] for b in batch_list], dtype=torch.long),
        "j": torch.tensor([b["j"] for b in batch_list], dtype=torch.long),
        "global_idx": torch.tensor([b["global_idx"] for b in batch_list], dtype=torch.long),
    }


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tessera v1.1 QAT single-tile inference")
    p.add_argument("--config", required=True, help="Path to Python config file (exports `config = {...}`)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--tile_path", required=True,
                   help="Directory with bands.npy, masks.npy, doys.npy, sar_ascending{,_doy}.npy, sar_descending{,_doy}.npy")
    p.add_argument("--output_dir", required=True, help="Directory to write <prefix>_emb128_int8.npy / _scales.npy")
    p.add_argument("--output_prefix", default=None,
                   help="Output file prefix. Defaults to basename of --tile_path (matches v1.0 pipeline).")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "xpu"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--data_source", default=None, choices=["mpc", "aws"],
                   help="Override config['data_source']. MUST match the loaded checkpoint.")
    return p.parse_args()


def load_config(path):
    spec = importlib.util.spec_from_file_location("v1_1_config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def pick_device(arg_device):
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if arg_device == "xpu":
        import intel_extension_for_pytorch  # noqa: F401
        if not torch.xpu.is_available():
            raise RuntimeError("XPU requested but not available.")
        return torch.device("xpu")
    # Auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.data_source is not None:
        config["data_source"] = args.data_source

    tile_path = os.path.abspath(args.tile_path.rstrip("/"))
    output_dir = os.path.abspath(args.output_dir)
    output_prefix = args.output_prefix or os.path.basename(tile_path)

    device = pick_device(args.device)
    logging.info("Device: %s", device)
    logging.info("Checkpoint: %s", args.checkpoint_path)
    logging.info("Tile: %s", tile_path)
    logging.info("Output: %s  (prefix=%s)", output_dir, output_prefix)
    logging.info("data_source: %s", config.get("data_source", "mpc"))
    logging.info("num_obs_checkpoints: %s", config["num_obs_checkpoints"])

    # Build and load model
    model = build_v1_1_inference_model(config, device)
    missing, unexpected, ckpt_cfg = load_v1_1_checkpoint(model, args.checkpoint_path)
    logging.info("Loaded checkpoint. missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        logging.info("Missing (first 10): %s", missing[:10])
    if unexpected:
        logging.info("Unexpected (first 10): %s", unexpected[:10])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build dataset + bucketised loader
    dataset = SingleTileInferenceDatasetV1_1(tile_path=tile_path, config=config)
    sampler = BucketBatchSampler(dataset, batch_size=int(config.get("batch_size", 1024)))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(int(config.get("num_workers", 8)) > 0),
        collate_fn=_collate,
    )

    # Output buffers (allocated on host)
    H, W = dataset.H, dataset.W
    save_dim = int(min(int(config.get("save_embedding_dim", 128)), int(config.get("representation_dim", 192))))
    emb_int8 = np.zeros((H, W, save_dim), dtype=np.int8)
    scales = np.zeros((H, W), dtype=np.float32)

    amp_ctx = nullcontext()
    if bool(config.get("apply_amp", True)) and device.type == "cuda":
        amp_dtype = torch.bfloat16 if bool(config.get("use_bf16", True)) else torch.float16
        amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)

    total = len(loader)
    log_every = int(config.get("log_interval_steps", 20))
    start = time.time()
    written = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            s2 = batch["s2"].to(device, non_blocking=True)
            s1 = batch["s1"].to(device, non_blocking=True)

            with amp_ctx:
                emb = model(s2, s1)  # (B, repr_dim)

            emb = emb[:, :save_dim].to(torch.float32)
            q_int8, per_row_scale = quantize_tensor_symmetric_per_row(emb, bits=int(config.get("qat_representation_bits", 8)))

            q_np = q_int8.cpu().numpy()
            s_np = per_row_scale.cpu().numpy().astype(np.float32).reshape(-1)
            ii = batch["i"].numpy()
            jj = batch["j"].numpy()
            emb_int8[ii, jj, :] = q_np
            scales[ii, jj] = s_np
            written += q_np.shape[0]

            if step % log_every == 0 or step == total:
                elapsed = time.time() - start
                rate = written / elapsed if elapsed > 0 else 0.0
                logging.info("Progress %d/%d batches, %d pixels, %.1f px/s", step, total, written, rate)

    os.makedirs(output_dir, exist_ok=True)
    out_int8 = os.path.join(output_dir, f"{output_prefix}_emb{save_dim}_int8.npy")
    out_scl = os.path.join(output_dir, f"{output_prefix}_emb{save_dim}_scales.npy")
    np.save(out_int8, emb_int8)
    np.save(out_scl, scales)
    logging.info("Saved: %s  (shape=%s, dtype=%s)", out_int8, emb_int8.shape, emb_int8.dtype)
    logging.info("Saved: %s  (shape=%s, dtype=%s)", out_scl, scales.shape, scales.dtype)
    logging.info("Done. total_pixels=%d elapsed=%.1fs", written, time.time() - start)


if __name__ == "__main__":
    main()
