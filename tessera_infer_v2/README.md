# TESSERA v2 — inference

Inference code for the **TESSERA v2** family: four compact pixel students and
the 2B teacher they were distilled from. See the preprint,
[*TESSERA v2: Scaling Pixel-wise Earth Foundation Models*](https://arxiv.org/abs/2607.03949).

This directory is independent of `tessera_infer/` (v1.0) and
`tessera_infer_QAT/` (v1.1) — v2 checkpoints cannot be loaded by the older
code, and vice versa.

## The models

| Model | `--model` | Parameters | Output | Use for |
|---|---|---|---|---|
| Nano | `nano` | 1.07 M | 128-d Matryoshka | edge / on-device |
| Small | `small` | 7.11 M | 128-d Matryoshka | low-resource |
| **Medium** | `medium` | 21.03 M | 128-d Matryoshka | **balanced default** |
| Large | `large` | 43.83 M | 128-d Matryoshka | provider-side global inference |
| 2B teacher | `teacher` | 2,064,266,242 | 1024-d | distillation / research only |

The students produce **Matryoshka** embeddings: the first K dimensions are
independently usable for K ∈ {16, 32, 64, 128}, so you can store 16, 32 or 64
dimensions instead of 128 without retraining or a second checkpoint.

> **The teacher is not a deployment model.** It evaluates 2.06 billion
> parameters per pixel, which makes tile-scale — let alone global-scale —
> embedding generation impractical on ordinary hardware. It is published so the
> distillation is reproducible and so researchers can build on it. For real
> work, use a student.

## Weights

Checkpoints are **not** stored in this repository. They are hosted on the
Hugging Face Hub under [`geotessera`](https://huggingface.co/geotessera):

| Model | Hugging Face repository | Download size |
|---|---|---|
| Nano | [`geotessera/TESSERA-V-2.0-2B-N`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-N) | 4 MB |
| Small | [`geotessera/TESSERA-V-2.0-2B-S`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-S) | 28 MB |
| Medium | [`geotessera/TESSERA-V-2.0-2B-M`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-M) | 84 MB |
| Large | [`geotessera/TESSERA-V-2.0-2B-L`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L) | 175 MB |
| 2B teacher | [`geotessera/TESSERA-V-2.0-2B-Teacher`](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-Teacher) | 8.26 GB |

The `2B` in each name records the teacher the model came from.

### Fetch them

```bash
pip install -r requirements.txt

python download_weights.py --model medium        # the recommended default
python download_weights.py --model all-students  # nano + small + medium + large
python download_weights.py --model teacher       # 8.26 GB
```

Checkpoints land in `student/checkpoints/` and `teacher/checkpoints/`, which is
where `infer_v2.py` looks by default. To place them yourself:

```
tessera_infer_v2
 ┣ student
 ┃   ┗ checkpoints
 ┃       ┣ student_nano.pt
 ┃       ┣ student_small.pt
 ┃       ┣ student_medium.pt
 ┃       ┗ student_large.pt
 ┗ teacher
     ┗ checkpoints
         ┗ tessera_v2_2B_teacher.pt
```

You can also pull a checkpoint directly with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("geotessera/TESSERA-V-2.0-2B-M", "ckpt/student_medium.pt")
```

## Run inference on preprocessed tiles

`infer_v2.py` consumes the tile directories produced by
`tessera_preprocessing` — the same `my_data/retiled_d_pixel/<tile>/` layout used
by v1.0 and v1.1, with `bands.npy`, `masks.npy`, `doys.npy`,
`sar_ascending{,_doy}.npy` and `sar_descending{,_doy}.npy`.

```bash
# default student, fp32 128-d output, one .npy per tile
python infer_v2.py --model medium \
    --data-root my_data/retiled_d_pixel \
    --out-dir   my_data/embeddings_v2

# 16-d Matryoshka prefix, stored as int8 + a float32 scale map
python infer_v2.py --model medium --dim 16 --int8 \
    --data-root my_data/retiled_d_pixel \
    --out-dir   my_data/embeddings_v2_d16

# the 2B teacher on a single tile (GPU strongly recommended)
python infer_v2.py --model teacher --bf16 --batch-pixels 512 \
    --tile    my_data/retiled_d_pixel/0_3500_500_4000 \
    --out-dir my_data/embeddings_v2_teacher
```

Outputs, one file per tile named after its directory:

- default: `<tile>.npy`, float32 `(H, W, 128)` — or `(H, W, D)` with `--dim D`,
  and `(H, W, 1024)` for the teacher
- with `--int8`: `<tile>.npy` int8 `(H, W, D)` plus `<tile>_scales.npy` float32
  `(H, W)`. Reconstruct with `code.astype("float32") * scales[..., None]`

Useful flags: `--batch-pixels` (lower it if you run out of memory),
`--device cpu`, `--bf16` (teacher; halves activation memory), `--ckpt` to point
at a checkpoint somewhere else.

## Use the models directly

```python
import sys, torch
sys.path.insert(0, "student")          # or "teacher"

from model import load_model
from infer import encode_tile, encode_pixels
from quantize import quantize as quantize_int8    # students only

model = load_model("student/checkpoints/student_medium.pt", torch.device("cuda"))

emb = encode_tile(
    model, s2_bands, s2_doys, s2_masks=s2_masks,
    s1_asc_bands=s1_asc, s1_asc_doys=s1_asc_doys,
    s1_desc_bands=s1_desc, s1_desc_doys=s1_desc_doys,
    batch_pixels=4096, device=torch.device("cuda"),
)                                       # -> (H, W, 128) float32

emb16 = emb[..., :16]                   # Matryoshka truncation, 1/8 the storage
code, scales = quantize_int8(emb)       # int8 + per-pixel float32 scale
```

`encode_pixels()` takes `(N, T, C)` arrays for per-pixel time series that are
not arranged as a grid, and returns `(N, 128)`.

## Input conventions

Both `encode_tile` and `encode_pixels` standardize their inputs internally
(`standardize=True`, the default), so **pass raw values**. Two conventions are
easy to get wrong, and both fail silently — you get plausible-looking but
meaningless embeddings rather than an error.

**1. The Sentinel-2 channel order is not ascending wavelength.** `bands.npy`
from `tessera_preprocessing` is already in the right order; if you build inputs
yourself, it must be exactly:

```
B04  B02  B03  B08  B8A  B05  B06  B07  B11  B12
```

**2. The students and the teacher normalize Sentinel-1 differently.**

- *Students*: ascending and descending are z-scored with **their own per-source
  statistics**, then concatenated along time.
- *Teacher*: ascending and descending are concatenated in **raw** units and the
  merged stream is z-scored with a **single pooled** set of statistics.

Do not carry one normalization across to the other model. The constants live in
each bundle's `model.py`.

Other expectations:

- Day-of-year is a **raw integer 1–365**, not normalized.
- Sentinel-2 mask: `1` = clear, `0` = cloud.
- Sentinel-1 timesteps that are all zero are treated as missing.
- Per pixel, the valid-observation count is bucketized to the nearest bin in
  `{8, 16, 24, ..., 256}` and the series padded or subsampled to that size,
  matching the training-time procedure.

Every embedding is passed through a final non-affine LayerNorm, so each pixel's
output vector has mean 0 and standard deviation 1 across its dimensions.

## Files

```
tessera_infer_v2
 ┣ infer_v2.py          tile-level entry point for both students and teacher
 ┣ download_weights.py  fetch checkpoints from the Hugging Face Hub
 ┣ requirements.txt
 ┣ student
 ┃   ┣ model.py         PixelStudent + load_model()
 ┃   ┣ infer.py         encode_pixels() / encode_tile()
 ┃   ┣ quantize.py      linear int8 quantize() / dequantize()
 ┃   ┗ checkpoints/
 ┗ teacher
     ┣ model.py         TesseraTeacher2B + load_model()
     ┣ infer.py         encode_pixels() / encode_tile()
     ┗ checkpoints/
```

Both bundles depend only on `torch` and `numpy`. `nn.RMSNorm` requires
PyTorch ≥ 2.4.
