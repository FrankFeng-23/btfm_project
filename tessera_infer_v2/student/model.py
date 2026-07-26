"""Standalone TESSERA Pixel Student — 128-d Matryoshka pixel encoder.

Self-contained: only torch + numpy required. Two backbones (S2 + merged S1),
each a small per-modality TransformerEncoder with attention pooling, then
concat fusion + dim_reducer. Native output 128-d, Matryoshka-ordered: the
first K dims are independently usable for K in {16, 32, 64, 128}.

Distilled from a TESSERA teacher. This is the encoder only — everything used
solely for training has been removed.
"""
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Standardization stats (must match training preprocessing) -----
# S1 ascending and descending use independent stats — applied per-source
# BEFORE merging into the single S1 stream the model consumes.
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
S1A_BAND_MEAN = np.array([5909.3921, 3405.0322], dtype=np.float32)
S1A_BAND_STD  = np.array([1507.1750, 1531.2615], dtype=np.float32)
S1D_BAND_MEAN = np.array([5816.1382, 3277.7576], dtype=np.float32)
S1D_BAND_STD  = np.array([1554.6475, 1546.4733], dtype=np.float32)

# S2 input channel order:
S2_BAND_ORDER = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
# S1 input channel order:
S1_BAND_ORDER = ["VV", "VH"]


# ===== Vendored building blocks (no qk_norm, ReLU, attention pooling) =====


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding using the (raw integer) DOY value."""

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


class AttentionPooling(nn.Module):
    """Plain single-head softmax attention pool over T.

    A single `Linear(D, 1) -> softmax -> weighted sum`. No GRU, no
    LayerNorm.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D)
        B, T, D = x.shape
        if T == 0:
            return torch.zeros(B, D, device=x.device, dtype=x.dtype)
        if T == 1:
            return x.squeeze(1)
        w = torch.softmax(self.query(x), dim=1)
        return (w * x).sum(dim=1)


class QKNormEncoderLayer(nn.Module):
    """Transformer encoder layer identical to `nn.TransformerEncoderLayer`
    (post-LN, ReLU FFN, batch_first) EXCEPT it applies per-head RMSNorm to Q and
    K before scaled-dot-product attention (QK-norm). QK-norm is the ONLY
    difference, so a model built from these is directly comparable to one built
    from the standard layer (≈same param count; q/k/v as 3 separate Linears
    matches the standard layer's combined in_proj, plus two tiny RMSNorm gains)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} % nhead {nhead} != 0"
        self.nhead = nhead
        self.head_dim = d_model // nhead
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
        self._attn_p = float(dropout)

    def _sa(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, T, H, hd)
        k = self.k_proj(x).view(B, T, H, hd)
        v = self.v_proj(x).view(B, T, H, hd)
        q = self.q_norm(q); k = self.k_norm(k)                    # per-head RMSNorm (QK-norm)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # (B,H,T,hd)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self._attn_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self._sa(x)))            # post-LN
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x))))))
        return x


class TransformerEncoder(nn.Module):
    """Per-pixel band embedding + DOY positional + Transformer + attention pooling.

    `enable_qk_norm=False` (default): standard `nn.TransformerEncoderLayer`
    (ReLU, no qk_norm). `True`: `QKNormEncoderLayer` stack (same structure +
    per-head RMSNorm on Q/K).
    """

    def __init__(self, band_num: int, latent_dim: int, nhead: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 256,
                 enable_qk_norm: bool = False) -> None:
        super().__init__()
        input_dim = band_num
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        self.enable_qk_norm = bool(enable_qk_norm)
        if self.enable_qk_norm:
            self.transformer_encoder = nn.ModuleList([
                QKNormEncoderLayer(latent_dim * 4, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim * 4, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation="relu", batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attn_pool = AttentionPooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, band_num + 1) — last channel is the (raw integer) DOY value
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        bands_embedded = self.embedding(bands)
        temporal_encoding = self.temporal_encoder(doy)
        x = bands_embedded + temporal_encoding
        if self.enable_qk_norm:
            for layer in self.transformer_encoder:
                x = layer(x)
        else:
            x = self.transformer_encoder(x)
        return self.attn_pool(x)


class PixelStudent(nn.Module):
    """2-backbone pixel encoder. emb_dim = 128 (Matryoshka-ordered).

    Inputs (per pixel):
        s2_x : (B, T_s2, 11)   10 bands + 1 DOY (raw integer 1..365)
        s1_x : (B, T_s1,  3)    2 bands + 1 DOY (raw integer; s1a + s1d MERGED)

    encode() returns (B, 128). The first K dims are usable on their own for
    K in {16, 32, 64, 128}.
    """

    def __init__(
        self,
        repr_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 64, 128),
        enable_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.enable_qk_norm = bool(enable_qk_norm)
        self.matryoshka_dims = tuple(int(d) for d in matryoshka_dims)
        for d in self.matryoshka_dims:
            if d > repr_dim:
                raise ValueError(f"matryoshka cut {d} > repr_dim {repr_dim}")

        def make_enc(band_num: int) -> TransformerEncoder:
            return TransformerEncoder(
                band_num=band_num, latent_dim=latent_dim,
                nhead=nhead, num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                max_seq_len=max_seq_len, enable_qk_norm=enable_qk_norm,
            )

        self.s2_backbone = make_enc(10)
        self.s1_backbone = make_enc(2)

        backbone_out = latent_dim * 4
        fused_in = 2 * backbone_out
        self.dim_reducer = nn.Sequential(
            nn.Linear(fused_in, fused_in * 2),
            nn.LayerNorm(fused_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(fused_in * 2, repr_dim),
            # Final non-affine LayerNorm: normalizes the 128-d output to
            # per-pixel mean 0 / std 1 (well-conditioned for downstream use).
            # No learnable params -> state_dict + param count unchanged.
            nn.LayerNorm(repr_dim, elementwise_affine=False),
        )

    def encode(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        s2 = self.s2_backbone(s2_x)
        s1 = self.s1_backbone(s1_x)
        fused = torch.cat([s2, s1], dim=-1)
        return self.dim_reducer(fused)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        return self.encode(s2_x, s1_x)


# Backwards-compatible alias.
PixelStudentV11 = PixelStudent


def load_model(ckpt_path: str, device: torch.device = torch.device("cpu")):
    """Load a pretrained pixel student (encoder) from a .pt file."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload.get("args", {}) or {}
    matryoshka_dims = tuple(
        int(d) for d in str(cfg.get("matryoshka_dims", "16,32,64,128")).split(",")
    )
    model = PixelStudent(
        repr_dim=int(cfg.get("repr_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        num_layers=int(cfg.get("num_layers", 4)),
        nhead=int(cfg.get("nhead", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=0.0,
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        matryoshka_dims=matryoshka_dims,
        enable_qk_norm=bool(cfg.get("enable_qk_norm", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
