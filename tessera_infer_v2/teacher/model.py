"""TESSERA v2 2B teacher — pixel-wise encoder.

Self-contained: only torch + numpy are required. Two per-modality backbones
(Sentinel-2 and merged Sentinel-1), a small transformer that fuses the two
modality tokens, and an MLP dim_reducer that emits a 1024-d per-pixel
representation.

This module contains the encoder and nothing else — no training objective, no
projection head, no quantization.
"""
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Standardization stats (must match training preprocessing) -------------
S2_BAND_MEAN = np.array(
    [1633.0042, 1341.1090, 1539.5536, 3054.8269, 3117.4658,
     2004.1648, 2694.7275, 2945.1504, 2266.6079, 1657.3094],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [1999.4603, 2014.7549, 1929.2201, 1754.2493, 1649.9807,
     1936.8988, 1748.6041, 1708.6991, 1207.5250, 1108.6046],
    dtype=np.float32,
)
# Sentinel-1 uses a SINGLE merged stream: ascending and descending are
# concatenated along time in RAW units and the merged stream is z-scored with
# these pooled statistics. Do NOT z-score ascending and descending separately.
S1_BAND_MEAN = np.array([5862.7652, 3341.3949], dtype=np.float32)
S1_BAND_STD = np.array([1531.8051, 1540.2014], dtype=np.float32)

# S2 input channel order (NOT ascending wavelength order):
S2_BAND_ORDER = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
# S1 input channel order:
S1_BAND_ORDER = ["VV", "VH"]


# ----- Building blocks -------------------------------------------------------


class AttentionPooling(nn.Module):
    """Single-head softmax attention pool over the time axis."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D)
        w = torch.softmax(self.query(x), dim=1)
        return (w * x).sum(dim=1)


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding computed from the raw integer DOY."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class QKNormEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with ReLU FFN and QK-norm.

    Per-head RMSNorm is applied to Q and K after their linear projection and
    before the attention dot product, which bounds the attention logits
    independently of how large the Q/K projection weights grow.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} % nhead {nhead} != 0"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self._attn_dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim)
        # Per-head RMSNorm in fp32 for a stable norm under reduced precision.
        q = self.q_norm(q.float()).to(v.dtype).transpose(1, 2)
        k = self.k_norm(k.float()).to(v.dtype).transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self._attn_dropout_p if self.training else 0.0)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self._attention(self.norm1(x)))
        h = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        return x + self.dropout2(h)


class TransformerEncoderStack(nn.Module):
    """Sequential stack of independently initialized encoder layers."""

    def __init__(self, layer_factory, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoder(nn.Module):
    """Per-modality backbone: band embedding + DOY encoding + transformer + pool."""

    def __init__(self, band_num: int, latent_dim: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        d_model = latent_dim * 4
        self.embedding = nn.Sequential(
            nn.Linear(band_num, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=d_model)
        self.transformer_encoder = TransformerEncoderStack(
            lambda: QKNormEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers,
        )
        self.attn_pool = AttentionPooling(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, band_num + 1) — last channel is the raw integer DOY
        bands, doy = x[:, :, :-1], x[:, :, -1]
        x = self.embedding(bands) + self.temporal_encoder(doy)
        return self.attn_pool(self.transformer_encoder(x))


class TesseraTeacher2B(nn.Module):
    """TESSERA v2 2B pixel-wise teacher encoder.

    Inputs (per pixel):
        s2_x : (B, T_s2, 11)   10 bands + 1 DOY (raw integer 1..365)
        s1_x : (B, T_s1,  3)    2 bands + 1 DOY (ascending + descending MERGED)

    encode() returns (B, 1024).
    """

    def __init__(self, repr_dim: int = 1024, latent_dim: int = 1024,
                 num_layers: int = 4, nhead: int = 4,
                 dim_feedforward: int = 16384, dropout: float = 0.0,
                 final_layernorm: bool = True) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        d_model = latent_dim * 4

        def make_backbone(band_num: int) -> TransformerEncoder:
            return TransformerEncoder(
                band_num=band_num, latent_dim=latent_dim, nhead=nhead,
                num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
            )

        self.s2_backbone = make_backbone(10)
        self.s1_backbone = make_backbone(2)

        # Modality fusion: each backbone output is one token of a length-2
        # sequence; a learnable modality embedding tells the transformer which
        # token came from which modality.
        self.fusion_modality_embed = nn.Parameter(torch.zeros(2, d_model))
        self.fusion_transformer = TransformerEncoderStack(
            lambda: QKNormEncoderLayer(d_model, 4, d_model * 2, dropout), 2)

        fused_in = 2 * d_model
        layers = [
            nn.Linear(fused_in, fused_in * 2),
            nn.LayerNorm(fused_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(fused_in * 2, repr_dim),
        ]
        if final_layernorm:
            # Non-affine LayerNorm: locks the output scale to per-pixel
            # mean 0 / std 1. No learnable parameters.
            layers.append(nn.LayerNorm(repr_dim, elementwise_affine=False))
        self.dim_reducer = nn.Sequential(*layers)

    def encode(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        seq = torch.stack([self.s2_backbone(s2_x), self.s1_backbone(s1_x)], dim=1)
        seq = seq + self.fusion_modality_embed.unsqueeze(0)
        seq = self.fusion_transformer(seq)
        return self.dim_reducer(seq.flatten(start_dim=1))

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        return self.encode(s2_x, s1_x)


def load_model(ckpt_path: str, device: torch.device = torch.device("cpu")):
    """Load the teacher encoder from a .pt file."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = payload.get("arch", {}) or {}
    model = TesseraTeacher2B(
        repr_dim=int(arch.get("repr_dim", 1024)),
        latent_dim=int(arch.get("latent_dim", 1024)),
        num_layers=int(arch.get("num_layers", 4)),
        nhead=int(arch.get("nhead", 4)),
        dim_feedforward=int(arch.get("dim_feedforward", 16384)),
        dropout=0.0,
        final_layernorm=bool(arch.get("final_layernorm", True)),
    )
    model.load_state_dict(payload["encoder_state_dict"], strict=True)
    return model.to(device).eval()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
