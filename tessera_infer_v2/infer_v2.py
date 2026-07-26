#!/usr/bin/env python3
"""TESSERA v2 inference over preprocessed tiles.

Consumes the tile directories produced by `tessera_preprocessing`
(`my_data/retiled_d_pixel/<tile>/`), each containing:

    bands.npy               (T_s2, H, W, 10)  raw Sentinel-2 L2A reflectance
    masks.npy               (T_s2, H, W)      1 = clear, 0 = cloud
    doys.npy                (T_s2,)           day-of-year per S2 frame
    sar_ascending.npy       (T_a, H, W, 2)    raw Sentinel-1 RTC, VV/VH
    sar_ascending_doy.npy   (T_a,)
    sar_descending.npy      (T_d, H, W, 2)
    sar_descending_doy.npy  (T_d,)

and writes one embedding array per tile.

Examples
--------
    # default student (medium), fp32 128-d output
    python infer_v2.py --model medium \
        --data-root my_data/retiled_d_pixel --out-dir my_data/embeddings_v2

    # 16-d Matryoshka prefix, int8-quantized on disk
    python infer_v2.py --model medium --dim 16 --int8 \
        --data-root my_data/retiled_d_pixel --out-dir my_data/embeddings_v2_d16

    # the 2B teacher (1024-d). Expensive — read the README first.
    python infer_v2.py --model teacher --bf16 --batch-pixels 512 \
        --tile my_data/retiled_d_pixel/0_3500_500_4000 --out-dir out
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

STUDENT_CKPT = {
    "nano":   "student_nano.pt",
    "small":  "student_small.pt",
    "medium": "student_medium.pt",
    "large":  "student_large.pt",
}
TEACHER_CKPT = "tessera_v2_2B_teacher.pt"


def _load_bundle(kind: str):
    """Import model.py / infer.py from student/ or teacher/.

    A run uses exactly one of the two bundles, so putting its directory first
    on sys.path is enough — this is also how the model cards document loading
    a bundle downloaded from the Hub.
    """
    sys.path.insert(0, os.path.join(HERE, kind))
    import model as model_mod  # noqa: E402
    import infer as infer_mod  # noqa: E402
    return model_mod, infer_mod


def _safe_load(path: str, shape, dtype):
    try:
        return np.load(path)
    except (FileNotFoundError, ValueError, OSError):
        return np.zeros(shape, dtype=dtype)


def load_tile(tile_dir: str):
    bands = np.load(os.path.join(tile_dir, "bands.npy")).astype(np.float32)
    T, H, W = bands.shape[0], bands.shape[1], bands.shape[2]
    masks = _safe_load(os.path.join(tile_dir, "masks.npy"), (T, H, W), np.uint8)
    doys = _safe_load(os.path.join(tile_dir, "doys.npy"), (T,), np.int32)
    s1a = _safe_load(os.path.join(tile_dir, "sar_ascending.npy"), (0, H, W, 2), np.float32)
    s1ad = _safe_load(os.path.join(tile_dir, "sar_ascending_doy.npy"), (0,), np.int32)
    s1d = _safe_load(os.path.join(tile_dir, "sar_descending.npy"), (0, H, W, 2), np.float32)
    s1dd = _safe_load(os.path.join(tile_dir, "sar_descending_doy.npy"), (0,), np.int32)
    return dict(
        s2_bands=bands, s2_masks=masks.astype(np.uint8), s2_doys=doys.astype(np.int64),
        s1_asc_bands=s1a.astype(np.float32), s1_asc_doys=s1ad.astype(np.int64),
        s1_desc_bands=s1d.astype(np.float32), s1_desc_doys=s1dd.astype(np.int64),
    )


def collect_tiles(data_root: str):
    return sorted(
        os.path.join(data_root, n) for n in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, n))
        and os.path.exists(os.path.join(data_root, n, "bands.npy"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium",
                    choices=list(STUDENT_CKPT) + ["teacher"])
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (default: <student|teacher>/checkpoints/<name>)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root", help="directory of tile subdirectories")
    src.add_argument("--tile", help="a single tile directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-pixels", type=int, default=4096,
                    help="pixels per forward pass (lower this if you run out of memory)")
    ap.add_argument("--dim", type=int, default=None,
                    help="students only: truncate to a Matryoshka prefix (16/32/64/128)")
    ap.add_argument("--int8", action="store_true",
                    help="students only: also write an int8 array + float32 scale map")
    ap.add_argument("--bf16", action="store_true",
                    help="run the forward pass under bfloat16 autocast (GPU)")
    args = ap.parse_args()

    is_teacher = args.model == "teacher"
    kind = "teacher" if is_teacher else "student"
    model_mod, infer_mod = _load_bundle(kind)

    ckpt = args.ckpt or os.path.join(
        HERE, kind, "checkpoints", TEACHER_CKPT if is_teacher else STUDENT_CKPT[args.model])
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}\n"
                 f"Fetch it first:  python download_weights.py --model {args.model}")

    if args.dim is not None:
        if is_teacher:
            sys.exit("--dim applies to the students only; the teacher's 1024-d output "
                     "is not Matryoshka-ordered and must not be truncated.")
        if args.dim not in (16, 32, 64, 128):
            sys.exit("--dim must be one of 16, 32, 64, 128")
    if args.int8 and is_teacher:
        sys.exit("--int8 applies to the students only.")

    device = torch.device(args.device)
    print(f"[init] model={args.model}  device={device}  ckpt={ckpt}")
    model = model_mod.load_model(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] parameters={n_params:,}  output_dim={model.repr_dim}")

    tiles = [args.tile] if args.tile else collect_tiles(args.data_root)
    if not tiles:
        sys.exit("no tiles found (expected subdirectories containing bands.npy)")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[init] {len(tiles)} tile(s) -> {args.out_dir}")

    kw = {}
    if args.bf16:
        if is_teacher:
            kw["autocast_dtype"] = torch.bfloat16
        else:
            print("[warn] --bf16 is only wired for the teacher; students run in fp32")

    for i, tile_dir in enumerate(tiles, 1):
        name = os.path.basename(tile_dir.rstrip("/"))
        t0 = time.time()
        d = load_tile(tile_dir)
        emb = infer_mod.encode_tile(
            model, d["s2_bands"], d["s2_doys"], s2_masks=d["s2_masks"],
            s1_asc_bands=d["s1_asc_bands"], s1_asc_doys=d["s1_asc_doys"],
            s1_desc_bands=d["s1_desc_bands"], s1_desc_doys=d["s1_desc_doys"],
            batch_pixels=args.batch_pixels, device=device, **kw)

        if args.dim is not None:
            emb = emb[..., :args.dim]

        out = os.path.join(args.out_dir, f"{name}.npy")
        if args.int8:
            import quantize as qmod  # student bundle is already on sys.path
            code, scales = qmod.quantize(emb)
            np.save(out, code)
            np.save(os.path.join(args.out_dir, f"{name}_scales.npy"), scales)
        else:
            np.save(out, emb.astype(np.float32))

        print(f"[{i}/{len(tiles)}] {name}: {emb.shape} in {time.time()-t0:.1f}s -> {out}")

    print("[done]")


if __name__ == "__main__":
    main()
