"""
Continuous global instance relation field for SSR3DLLM.

For each instance i, we build pairwise geometry features
Δ^{geo}_{ij} = [Δx, Δy, Δz, d_xy, sin φ, cos φ, sin θ, cos θ], project them
to d_model, then aggregate neighbors with a multi-head spatial module to
produce one relation-field vector f_i^{glob} per instance.
"""

from __future__ import annotations

import os, sys
from pathlib import Path

# Add repo root to sys.path so local SSR3DLLM packages resolve first.
repo_root = Path(__file__).resolve().parents[1]  # release repo root
sys.path.insert(0, str(repo_root))               # Add repo root first so local packages take precedence.
sys.path.insert(0, str(repo_root / "src"))       # Also add src/ if the project keeps source code there.

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rel3DSpatialNetMultiHead(nn.Module):
    """
    Multi-head spatial aggregator for relation fields.

    Given a sequence of per-neighbour geometric embeddings
    ``spatial_features[b, j]`` for a fixed anchor instance ``i``, each head
    predicts attention weights over j and aggregates neighbour values into
    a single d_model-dimensional vector.
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_hidden: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head

        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.score_layers = nn.ModuleList(
            [nn.Linear(self.d_head, 1) for _ in range(n_head)]
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, spatial_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_features: Tensor of shape [B*, N, D]
                For each anchor (flattened into the batch dimension), we
                have N neighbour embeddings of dimension D.

        Returns:
            features: Tensor of shape [B*, D]
                Aggregated relation‑field vector per anchor.
            score: Tensor of shape [B*, N]
                Mean attention weights over heads for inspection.
        """
        bv, n, _ = spatial_features.shape
        k = self.k_proj(spatial_features).view(bv, n, self.n_head, self.d_head)
        v = self.v_proj(spatial_features).view(bv, n, self.n_head, self.d_head)

        multihead_features = []
        multihead_scores = []
        for i in range(self.n_head):
            k_head = k[:, :, i, :]  # [B*, N, d_head]
            v_head = v[:, :, i, :]  # [B*, N, d_head]
            score = self.score_layers[i](k_head).squeeze(-1)  # [B*, N]
            score = F.softmax(score, dim=-1).unsqueeze(-1)    # [B*, N, 1]
            feature = (score * v_head).sum(dim=1)             # [B*, d_head]
            multihead_features.append(feature)
            multihead_scores.append(score)

        feature = torch.cat(multihead_features, dim=-1)        # [B*, D]
        score = torch.cat(multihead_scores, dim=-1).mean(dim=-1)  # [B*, N]
        features = self.ffn(feature)                           # [B*, D]
        return features, score


@dataclass
class RelationFieldConfig:
    """
    Configuration for the global relation field.

    Attributes:
        d_model:     Output feature dimension per instance.
        n_head:      Number of spatial attention heads.
        d_hidden:    Hidden size inside the FFN.
        dropout:     Dropout rate used in the FFN.
        norm_xy:     Whether to normalise planar offsets by d_xy.
        norm_z:      Whether to map vertical offsets to [0,1].
        norm_d:      Whether to map d_xy to [0,1].
    """

    d_model: int = 128
    n_head: int = 8
    d_hidden: int = 2048
    dropout: float = 0.1
    norm_xy: bool = True
    norm_z: bool = True
    norm_d: bool = True


class InstanceRelationField(nn.Module):
    """
    Global instance relation field module.

    Given per-instance centres ``coords[b, i] ∈ R^3``, this module builds
    continuous pairwise features Δ^{geo}_{ij} and uses a multi-head spatial
    aggregator to produce a relation‑field vector f^{glob}_i for every
    instance i in the scene.
    """

    def __init__(self, cfg: RelationFieldConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 8D geometry → d_model
        self.spatial_enc = nn.Sequential(
            nn.Linear(8, cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.d_model),
        )
        self.spatial_agg = Rel3DSpatialNetMultiHead(
            n_head=cfg.n_head,
            d_model=cfg.d_model,
            d_hidden=cfg.d_hidden,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _scale_to_unit_range(x: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalisation to [0,1] along the last dimension.
        """
        max_x = torch.max(x, dim=-1, keepdim=True).values
        min_x = torch.min(x, dim=-1, keepdim=True).values
        return x / (max_x - min_x + 1e-9)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: Tensor of shape [B, N, 3]
                Instance centres in a common 3D coordinate frame.

        Returns:
            field: Tensor of shape [B, N, d_model]
                Global relation-field vector f^{glob}_i for each instance.
            weights: Tensor of shape [B, N, N]
                Aggregated attention weights over neighbours j for each
                instance i (for inspection / visualisation).
        """
        if coords.dim() != 3 or coords.size(-1) != 3:
            raise ValueError(f"coords must have shape [B,N,3], got {tuple(coords.shape)}")

        bsz, n_inst, _ = coords.shape
        device = coords.device

        # Δ_{ij} = c_j - c_i
        rel = coords[:, None, :, :] - coords[:, :, None, :]  # [B, N, N, 3]
        xy = rel[..., :2].norm(dim=-1, keepdim=True) + 1e-9  # [B, N, N, 1]

        r = xy.squeeze(-1)                                   # [B, N, N]
        phi = torch.atan2(rel[..., 1], rel[..., 0])          # azimuth
        theta = torch.atan2(r, rel[..., 2])                  # elevation

        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)

        # Assemble 8D features: [Δx,Δy,Δz,d_xy,sinφ,cosφ,sinθ,cosθ]
        rel_pos = torch.cat(
            [
                rel,                      # Δx, Δy, Δz
                xy,                       # d_xy
                sin_phi.unsqueeze(-1),
                cos_phi.unsqueeze(-1),
                sin_theta.unsqueeze(-1),
                cos_theta.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, N, N, 8]

        # Optional per-dimension normalization.
        if self.cfg.norm_xy:
            rel_pos[..., :2] = rel_pos[..., :2] / xy
        if self.cfg.norm_z:
            rel_pos[..., 2] = self._scale_to_unit_range(rel_pos[..., 2])
        if self.cfg.norm_d:
            rel_pos[..., 3] = self._scale_to_unit_range(rel_pos[..., 3])

        # 8D → d_model, still per pair (i,j).
        rel_pos = self.spatial_enc(rel_pos)                  # [B, N, N, d_model]
        rel_pos = rel_pos.view(bsz * n_inst, n_inst, self.cfg.d_model)

        # For each anchor i (flattened into B*N), aggregate over neighbours j.
        agg, score = self.spatial_agg(rel_pos)               # [B*N, d_model], [B*N, N]
        field = agg.view(bsz, n_inst, self.cfg.d_model)      # [B, N, d_model]
        weights = score.view(bsz, n_inst, n_inst)            # [B, N, N]

        return field, weights
