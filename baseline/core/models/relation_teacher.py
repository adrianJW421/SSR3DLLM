import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RelationTeacher(nn.Module):
    """
    A lightweight geometric–text alignment teacher T_φ.

    - Geometric tower: takes precomputed relation features (e.g., target/anchor field
      vectors or concatenated delta coords) and maps to a d-dimensional embedding.
    - Text tower: encodes relation phrases with a frozen/finetuned encoder (default BERT),
      then projects to the same d-dimensional space.

    This module only runs forward passes and projections; contrastive loss is computed in the external trainer.
    """

    def __init__(
        self,
        geom_dim: int,
        embed_dim: int,
        text_model_name: str = "bert-base-uncased",
        proj_dropout: float = 0.1,
        finetune_text: bool = False,
    ):
        super().__init__()
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # Whether to update text encoder weights is controlled by finetune_text.
        if not finetune_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def encode_geom(self, geom_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom_feat: [B, geom_dim] relation geometry features.
        Returns:
            geom_emb: [B, embed_dim] L2-normalized embedding.
        """
        geom_emb = self.geom_proj(geom_feat)
        return self._l2_normalize(geom_emb)

    def encode_text(self, texts, device=None) -> torch.Tensor:
        """
        Tokenize and encode a list of relation phrases.
        Args:
            texts: list[str] relation phrases.
            device: torch device; defaults to model device.
        Returns:
            text_emb: [B, embed_dim] L2-normalized embedding.
        """
        if device is None:
            device = next(self.parameters()).device
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(**batch)
        # Use the [CLS] representation for projection.
        cls = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
        text_emb = self.text_proj(cls)
        return self._l2_normalize(text_emb)

    def forward(self, geom_feat: torch.Tensor, texts) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            geom_feat: [B, geom_dim] geometric relation features.
            texts: list[str] relation phrases.
        Returns:
            geom_emb, text_emb: both [B, embed_dim], L2-normalized.
        """
        geom_emb = self.encode_geom(geom_feat)
        text_emb = self.encode_text(texts, device=geom_feat.device)
        return geom_emb, text_emb


def load_teacher(ckpt_path: str, map_location="cpu") -> RelationTeacher:
    """
    Utility to load a saved RelationTeacher checkpoint.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    teacher = RelationTeacher(
        geom_dim=state["hyper_params"]["geom_dim"],
        embed_dim=state["hyper_params"]["embed_dim"],
        text_model_name=state["hyper_params"]["text_model_name"],
        proj_dropout=state["hyper_params"]["proj_dropout"],
        finetune_text=state["hyper_params"]["finetune_text"],
    )
    teacher.load_state_dict(state["state_dict"], strict=True)
    return teacher
