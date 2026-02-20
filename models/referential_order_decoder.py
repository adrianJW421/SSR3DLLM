"""
Causal referential order decoder for SSR3DLLM.

This module follows the high-level design in our ReF3D-LLM method section
and CoT3DRef: given geometry-enhanced instance tokens and text tokens, it
predicts a sequence of anchor/target indices via a Transformer-based
pointer network.

Key properties:
- Uses a stack of TransformerDecoderLayer blocks with a strict causal mask
  on the target side (auto-regressive in the step dimension).
- Cross-attends to a memory composed of instance tokens (and optionally
  text tokens), so that each step can read both geometry and language.
- Produces pointer logits over N+1 positions at each step, where N is the
  number of instances and the extra slot corresponds to a learned STOP
  token.

The module itself is agnostic to how order labels are constructed. On
Sr3D minimal data we can start with a simple 2-step chain:
  step 0 -> anchor_idx
  step 1 -> target_idx
and optionally extend to longer chains when CoT3DRef/Vigor-style labels
are integrated.
"""

from __future__ import annotations

import os, sys
from pathlib import Path

# Add repo root to sys.path so local SSR3DLLM packages resolve first.
repo_root = Path(__file__).resolve().parents[1]  # release repo root
sys.path.insert(0, str(repo_root))               # Add repo root first so local packages take precedence.
sys.path.insert(0, str(repo_root / "src"))       # Also add src/ if the project keeps source code there.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ReferentialDecoderConfig:
    """
    Configuration for the referential order decoder.

    Attributes:
        d_model:         Hidden dimension for all tokens.
        nhead:           Number of attention heads.
        num_layers:      Number of Transformer decoder layers.
        dim_feedforward: FFN hidden size inside each layer.
        dropout:         Dropout rate.
        max_steps:       Maximum decoding steps (length of the chain).
    """

    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_steps: int = 4


class ReferentialOrderDecoder(nn.Module):
    """
    Transformer-based causal referential order decoder.

    Forward usage (training):
        logits = decoder(
            obj_tokens=obj_tokens,
            text_tokens=text_tokens,
            order_labels=order_labels,
        )
    where:
        obj_tokens:   [B, N, D] geometry-enhanced instance tokens.
        text_tokens:  [B, L, D] text encoder outputs (optional; may be None).
        order_labels: [B, T]    integers in [0, N] where N denotes STOP.

    The decoder returns:
        pointer_logits: [B, T, N+1]
    which can be trained with cross-entropy against order_labels.
    """

    def __init__(self, cfg: ReferentialDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.pos_embedding = nn.Embedding(cfg.max_steps, cfg.d_model)
        self.start_vector = nn.Parameter(torch.zeros(cfg.d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)

        # Pointer projections: h_t → scores over instances; plus a STOP logit.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.stop_vector = nn.Parameter(torch.zeros(cfg.d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:  # pragma: no cover - standard init
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.start_vector, mean=0.0, std=0.02)
        nn.init.normal_(self.stop_vector, mean=0.0, std=0.02)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Build an upper-triangular mask for auto-regressive decoding.
        Shape: [T, T], True indicates positions that should be masked.
        """
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        obj_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        order_labels: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        text_init: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        obj_padding_mask: Optional[torch.Tensor] = None,
        use_prev_obj_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            obj_tokens: [B, N, D] instance tokens.
            text_tokens: [B, L, D] text tokens (optional; if None, memory
                consists only of obj_tokens).
            order_labels: [B, T] chain of indices in [0, N] where N is STOP.
                If provided, T is the decoding length and we use teacher
                forcing in the time dimension.
            max_steps: maximum number of decoding steps when order_labels
                is None; defaults to cfg.max_steps.
            text_init: [B, D] optional summary vector from text; if provided,
                we add it to the first target embedding to bias decoding.
            tgt_key_padding_mask: [B, T] optional mask where True marks padding
                positions that should be ignored in self-attention.

        Returns:
            pointer_logits: [B, T, N+1] pointer scores per step.
        """
        if obj_tokens.dim() != 3:
            raise ValueError(f"obj_tokens must be [B,N,D], got {tuple(obj_tokens.shape)}")
        bsz, n_inst, d_model = obj_tokens.shape
        device = obj_tokens.device

        if d_model != self.cfg.d_model:
            raise ValueError(
                f"obj_tokens dim={d_model} does not match decoder d_model={self.cfg.d_model}"
            )

        if order_labels is not None:
            if order_labels.dim() != 2 or order_labels.size(0) != bsz:
                raise ValueError(
                    f"order_labels must be [B,T], got {tuple(order_labels.shape)}"
                )
            T = order_labels.size(1)
        else:
            T = max_steps if max_steps is not None else self.cfg.max_steps

        if T > self.cfg.max_steps:
            raise ValueError(
                f"Requested T={T} exceeds cfg.max_steps={self.cfg.max_steps}"
            )

        # Build target-side input embeddings.
        # If `order_labels` is provided (teacher forcing), we make the decoder truly
        # causal by feeding the embedding of the *previously selected* object token:
        #   step0 input = <START>
        #   stept input = embedding(order_labels[t-1])  (object token or STOP)
        # This is important for chain reasoning because the next prediction should
        # depend on what has been selected so far.
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
        tgt = self.pos_embedding(pos_ids)  # [B, T, D]

        if order_labels is not None and use_prev_obj_tokens:
            prev_ids = torch.full_like(order_labels, fill_value=n_inst)  # previous=STOP by default
            if T > 1:
                prev_ids[:, 1:] = order_labels[:, :-1]
            stop_tok = self.stop_vector.view(1, 1, d_model).expand(bsz, 1, d_model)
            cand = torch.cat([obj_tokens, stop_tok], dim=1)  # [B, N+1, D]
            prev_emb = cand.gather(
                1, prev_ids.clamp(0, n_inst).unsqueeze(-1).expand(bsz, T, d_model)
            )
            prev_emb = prev_emb.clone()
            prev_emb[:, 0, :] = self.start_vector.view(1, d_model).expand(bsz, d_model)
            tgt = tgt + prev_emb

        if text_init is not None:
            if text_init.shape != (bsz, d_model):
                raise ValueError(
                    f"text_init must be [B,{d_model}], got {tuple(text_init.shape)}"
                )
            tgt = tgt.clone()
            tgt[:, 0, :] = tgt[:, 0, :] + text_init

        # Memory: concatenate instance tokens and optional text tokens.
        if text_tokens is not None:
            if text_tokens.dim() != 3 or text_tokens.size(0) != bsz:
                raise ValueError(
                    f"text_tokens must be [B,L,D], got {tuple(text_tokens.shape)}"
                )
            if text_tokens.size(-1) != d_model:
                raise ValueError(
                    f"text_tokens dim={text_tokens.size(-1)} "
                    f"does not match decoder d_model={d_model}"
                )
            memory = torch.cat([obj_tokens, text_tokens], dim=1)  # [B, N+L, D]
        else:
            memory = obj_tokens  # [B, N, D]

        # Memory key padding mask (True = ignore).
        memory_key_padding_mask = None
        if obj_padding_mask is not None:
            m = obj_padding_mask
            if m.dim() == 3 and m.size(-1) == 1:
                m = m.squeeze(-1)
            if m.dim() != 2 or m.size(0) != bsz or m.size(1) != n_inst:
                raise ValueError(
                    f"obj_padding_mask must be [B,N] (or [B,N,1]), got {tuple(obj_padding_mask.shape)}"
                )
            if text_tokens is not None:
                pad_text = torch.zeros((bsz, text_tokens.size(1)), device=device, dtype=torch.bool)
                memory_key_padding_mask = torch.cat([m.to(torch.bool), pad_text], dim=1)
            else:
                memory_key_padding_mask = m.to(torch.bool)

        # Causal mask for target sequence.
        tgt_mask = self._build_causal_mask(T, device=device)

        # Run Transformer decoder (batch_first=True).
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, D]

        # Pointer scores over instances + STOP.
        pointer_logits = self._compute_pointer_logits(
            dec_out, obj_tokens, obj_padding_mask=obj_padding_mask
        )
        # pointer_logits: [B, T, N+1]
        return pointer_logits

    def _compute_pointer_logits(
        self,
        dec_out: torch.Tensor,
        obj_tokens: torch.Tensor,
        obj_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pointer logits over instances and STOP from decoder outputs.

        Args:
            dec_out: [B, T, D] decoder hidden states.
            obj_tokens: [B, N, D] instance tokens.
            obj_padding_mask: [B, N] boolean mask where True marks padding
                positions that should not be selected.

        Returns:
            logits: [B, T, N+1] scores for each instance and STOP.
        """
        bsz, T, d_model = dec_out.shape
        _, n_inst, _ = obj_tokens.shape

        q = self.q_proj(dec_out)          # [B, T, D]
        k = self.k_proj(obj_tokens)       # [B, N, D]

        # Instance scores via scaled dot-product.
        # logits_inst[b,t,i] = <q_{b,t}, k_{b,i}>
        scale = float(d_model) ** 0.5
        logits_inst = torch.bmm(q, k.transpose(1, 2)) / scale  # [B, T, N]
        if obj_padding_mask is not None:
            if obj_padding_mask.shape != (bsz, n_inst):
                raise ValueError(
                    f"obj_padding_mask must be [B,N], got {tuple(obj_padding_mask.shape)}"
                )
            # Mask out padded slots with large negative logits
            mask = obj_padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B,T,N]
            logits_inst = logits_inst.masked_fill(mask, float("-inf"))

        # STOP score: <h_t, w_stop>.
        stop_vec = self.stop_vector.view(1, 1, d_model)        # [1,1,D]
        logits_stop = (dec_out * stop_vec).sum(dim=-1, keepdim=True) / scale  # [B,T,1]

        logits = torch.cat([logits_inst, logits_stop], dim=-1)  # [B,T,N+1]
        return logits


def compute_order_loss(
    pointer_logits: torch.Tensor,
    order_labels: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Utility to compute cross-entropy loss over pointer logits.

    Args:
        pointer_logits: [B, T, N+1]
        order_labels:   [B, T] with values in [0, N] or ignore_index.
        ignore_index:   label to ignore in loss (for padding steps).

    Returns:
        loss: scalar tensor.
    """
    if pointer_logits.dim() != 3 or order_labels.dim() != 2:
        raise ValueError(
            f"Expect logits [B,T,N+1] and labels [B,T], got "
            f"{tuple(pointer_logits.shape)}, {tuple(order_labels.shape)}"
        )
    bsz, T, _ = pointer_logits.shape
    if order_labels.size(0) != bsz or order_labels.size(1) != T:
        raise ValueError(
            f"Label shape {tuple(order_labels.shape)} incompatible with "
            f"logits shape {tuple(pointer_logits.shape)}"
        )

    # Flatten over batch and time for CE.
    logits_flat = pointer_logits.reshape(bsz * T, -1)     # [(B*T), N+1]
    labels_flat = order_labels.reshape(bsz * T)           # [(B*T)]

    loss = nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
    )
    return loss
