# src/models/ssl_model_v1_1.py
#
# Inference-only assembly of the Tessera v1.1 representation model (this revision:
# split_s1_modalities=False, matching v1.0's two-backbone topology).
#
#   S2 backbone   ┐
#                 ├─ concat (latent_dim*4 * 2) → MLP dim_reducer → 192-D embedding
#   S1 backbone   ┘   (S1 backbone consumes the concatenated S1-asc + S1-desc stream,
#                      each per-modality-normalised by the dataset before merging.)
#
# The projector used during pretraining is intentionally NOT included.

import torch
import torch.nn as nn

from .modules import TransformerEncoder


class MultimodalV1_1InferenceModel(nn.Module):
    """Two backbones (S2 + merged S1) with an MLP `dim_reducer`.

    The dim_reducer matches v1.1 training exactly:
        Linear(in, in*2) -> LayerNorm(in*2) -> ReLU -> Dropout(0.2) -> Linear(in*2, repr_dim)
    where in = latent_dim * 4 * num_active_backbones (concat fusion).
    """

    def __init__(self, s2_backbone, s1_backbone, dim_reducer, fusion_method="concat"):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method

    def forward(self, s2_x, s1_x):
        reprs = []
        if self.s2_backbone is not None:
            reprs.append(self.s2_backbone(s2_x))
        if self.s1_backbone is not None:
            reprs.append(self.s1_backbone(s1_x))

        if self.fusion_method == "concat":
            fused = torch.cat(reprs, dim=-1)
        elif self.fusion_method == "sum":
            fused = sum(reprs)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        return self.dim_reducer(fused)


def build_v1_1_inference_model(config, device):
    """Construct the v1.1 inference model from config.

    Required config keys: latent_dim, representation_dim, fusion_method,
    s2_num_heads/num_layers/dim_feedforward, s1_num_heads/num_layers/dim_feedforward,
    num_obs_checkpoints (max determines max_seq_len passed to the encoders).
    """
    if bool(config.get("split_s1_modalities", False)):
        raise NotImplementedError(
            "v1.1 (this revision) requires split_s1_modalities=False — S1-asc and "
            "S1-desc are concatenated before the single merged S1 backbone."
        )

    latent_dim = int(config.get("latent_dim", 192))
    repr_dim = int(config.get("representation_dim", latent_dim))
    fusion_method = config.get("fusion_method", "concat")

    max_seq_len = max(int(v) for v in config.get("num_obs_checkpoints", [128]) if int(v) > 0)

    s2_enc = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=int(config["s2_num_heads"]),
        num_encoder_layers=int(config["s2_num_layers"]),
        dim_feedforward=int(config["s2_dim_feedforward"]),
        dropout=0.1,
        max_seq_len=max_seq_len,
    ).to(device)

    s1_enc = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=int(config["s1_num_heads"]),
        num_encoder_layers=int(config["s1_num_layers"]),
        dim_feedforward=int(config["s1_dim_feedforward"]),
        dropout=0.1,
        max_seq_len=max_seq_len,
    ).to(device)

    active = 2 if fusion_method == "concat" else 1
    reducer_in = latent_dim * 4 * active
    dim_reducer = nn.Sequential(
        nn.Linear(reducer_in, reducer_in * 2),
        nn.LayerNorm(reducer_in * 2),
        nn.ReLU(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(reducer_in * 2, repr_dim),
    ).to(device)

    return MultimodalV1_1InferenceModel(
        s2_backbone=s2_enc,
        s1_backbone=s1_enc,
        dim_reducer=dim_reducer,
        fusion_method=fusion_method,
    )


def load_v1_1_checkpoint(model: MultimodalV1_1InferenceModel, checkpoint_path: str):
    """Load an FSDP v1.1 checkpoint into the inference model.

    Strips the `_orig_mod.` prefix added by torch.compile-wrapped FSDP and skips
    the projector / segmented-matryoshka-projector, which are not part of the
    inference graph.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    raw = ckpt[state_key]

    cleaned = {}
    for k, v in raw.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("projector.") or k.startswith("segmented_matryoshka_projector."):
            continue
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected, ckpt.get("config", {})
