# src/models/ssl_model.py

import torch
import torch.nn as nn
from .modules import *

class MultimodalBTModel(nn.Module):
    def __init__(self, s2_backbone, s1_backbone, projector, fusion_method='concat', return_repr=False, latent_dim=128):
        """
        fusion_method: 'sum', 'concat' or 'transformer'
        If 'transformer' is used, latent_dim must be provided
        """
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.projector = projector
        self.fusion_method = fusion_method
        self.return_repr = return_repr

        if fusion_method == 'concat':
            in_dim = 8 * latent_dim
        elif fusion_method == 'sum':
            in_dim = 4 * latent_dim

        self.dim_reducer = nn.Sequential(nn.Linear(in_dim, latent_dim))

    def forward(self, s2_x, s1_x):
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)
        if self.fusion_method == 'concat':
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        elif self.fusion_method == 'sum':
            fused = s2_repr + s1_repr
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        # Reduce dimension to 128
        fused = self.dim_reducer(fused)
        feats = self.projector(fused)
        if self.return_repr:
            return feats, fused
        return feats

class MultimodalBTInferenceModel(torch.nn.Module):
    """
    Model for the inference phase, containing only two Transformer encoders (S2 + S1),
    without the projection head.
    """
    def __init__(self, s2_backbone, s1_backbone, fusion_method, dim_reducer):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.fusion_method = fusion_method
        self.dim_reducer = dim_reducer

    def forward(self, s2_x, s1_x):
        """
        s2_x.shape = (batch, seq_len_s2, band_num_s2)
        s1_x.shape = (batch, seq_len_s1, band_num_s1)
        Output: (batch, latent_dim) or (batch, 2*latent_dim) if fusion=concat
        """
        s2_repr = self.s2_backbone(s2_x)  # (batch, latent_dim)
        s1_repr = self.s1_backbone(s1_x)  # (batch, latent_dim)

        if self.fusion_method == "sum":
            fused = s2_repr + s1_repr
        elif self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        fused = self.dim_reducer(fused)
        return fused