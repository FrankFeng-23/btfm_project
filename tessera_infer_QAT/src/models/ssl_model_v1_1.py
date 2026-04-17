# src/models/ssl_model_v1_1.py
#
# Inference-only assembly of the Tessera v1.1 representation model.
# Three backbones (S2, S1-ascending, S1-descending) -> concat fusion -> MLP dim_reducer.
# The projector is intentionally NOT included — v1.1 pretraining's projector is
# discarded at inference time, matching v2 training's inference path.

import torch
import torch.nn as nn

from .modules import TransformerEncoder


class MultimodalV1_1InferenceModel(nn.Module):
    """Assembles the three v1.1 backbones and the MLP dim_reducer.

    The dim_reducer matches v2 training exactly:
        Linear(in, in*2) -> LayerNorm(in*2) -> ReLU -> Dropout(0.2) -> Linear(in*2, repr_dim)
    where in = latent_dim * 4 * num_active_backbones (concat fusion).
    """

    def __init__(self, s2_backbone, s1a_backbone, s1d_backbone, dim_reducer, fusion_method="concat"):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1a_backbone = s1a_backbone
        self.s1d_backbone = s1d_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method

    def forward(self, s2_x, s1a_x, s1d_x):
        reprs = []
        if self.s2_backbone is not None:
            reprs.append(self.s2_backbone(s2_x))
        if self.s1a_backbone is not None:
            reprs.append(self.s1a_backbone(s1a_x))
        if self.s1d_backbone is not None:
            reprs.append(self.s1d_backbone(s1d_x))

        if self.fusion_method == "concat":
            fused = torch.cat(reprs, dim=-1)
        elif self.fusion_method == "sum":
            fused = sum(reprs)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        return self.dim_reducer(fused)


def build_v1_1_inference_model(config, device):
    """Construct the v1.1 inference model from config.

    Input config must specify at minimum: latent_dim, representation_dim,
    s2/s1 heads/layers/dim_feedforward, split_s1_modalities=True, fusion_method,
    num_obs_checkpoints (max determines max_seq_len in the encoders).
    """
    latent_dim = int(config.get("latent_dim", 192))
    repr_dim = int(config.get("representation_dim", latent_dim))
    fusion_method = config.get("fusion_method", "concat")
    split_s1 = bool(config.get("split_s1_modalities", True))
    if not split_s1:
        raise NotImplementedError("Tessera v1.1 expects split_s1_modalities=True.")

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

    s1a_enc = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=int(config["s1_num_heads"]),
        num_encoder_layers=int(config["s1_num_layers"]),
        dim_feedforward=int(config["s1_dim_feedforward"]),
        dropout=0.1,
        max_seq_len=max_seq_len,
    ).to(device)

    s1d_enc = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=int(config["s1_num_heads"]),
        num_encoder_layers=int(config["s1_num_layers"]),
        dim_feedforward=int(config["s1_dim_feedforward"]),
        dropout=0.1,
        max_seq_len=max_seq_len,
    ).to(device)

    active = 3 if fusion_method == "concat" else 1
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
        s1a_backbone=s1a_enc,
        s1d_backbone=s1d_enc,
        dim_reducer=dim_reducer,
        fusion_method=fusion_method,
    )


def load_v1_1_checkpoint(model: MultimodalV1_1InferenceModel, checkpoint_path: str):
    """Load an FSDP v1.1 checkpoint into the inference model.

    Strips the `_orig_mod.` prefix added by torch.compile-wrapped FSDP and skips
    projector weights (which are not part of the inference graph).
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
