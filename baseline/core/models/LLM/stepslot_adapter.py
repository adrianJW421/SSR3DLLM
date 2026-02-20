from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class StepSlotAdapterExport:
    order_len: int
    mem_tokens: int
    hidden_size: int
    step_tokens: list[str]
    global_token: str
    token_embeds: Dict[str, torch.Tensor]
    proj_step: Dict[str, torch.Tensor]
    proj_global: Dict[str, torch.Tensor]
    mem_token_embeds: Optional[torch.Tensor] = None
    proj_mem: Optional[Dict[str, torch.Tensor]] = None


class SoftStepSlotAdapter(nn.Module):
    """
    Small trainable adapter extracted from `mask3d-vigor-llama-step`:
    - step token embedding rows live in the LLM embedding table (loaded separately)
    - optional soft memory tokens (learnable embeddings in hidden space)
    - projection layers to Vigor's inner_dim (typically 768)
    """

    def __init__(self, hidden_size: int, out_dim: int = 768, mem_tokens: int = 0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.out_dim = int(out_dim)
        self.mem_tokens = int(mem_tokens)

        self.proj_step = nn.Sequential(nn.Linear(self.hidden_size, self.out_dim), nn.LayerNorm(self.out_dim))
        self.proj_global = nn.Sequential(nn.Linear(self.hidden_size, self.out_dim), nn.LayerNorm(self.out_dim))

        self.mem_token_embeds = None
        self.proj_mem = None
        if self.mem_tokens > 0:
            self.mem_token_embeds = nn.Parameter(torch.empty(self.mem_tokens, self.hidden_size))
            nn.init.normal_(self.mem_token_embeds, mean=0.0, std=0.02)
            self.proj_mem = nn.Sequential(nn.Linear(self.hidden_size, self.out_dim), nn.LayerNorm(self.out_dim))

    @staticmethod
    def _load_export(path: str | Path) -> StepSlotAdapterExport:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        obj = torch.load(str(p), map_location="cpu")
        if not isinstance(obj, dict) or str(obj.get("format", "")) != "llama_stepslot_adapter":
            raise RuntimeError(f"Not a llama_stepslot_adapter export: {p}")

        token_embeds = obj.get("token_embeds", {})
        if not isinstance(token_embeds, dict):
            token_embeds = {}
        token_embeds = {str(k): (v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in token_embeds.items()}

        proj_step = obj.get("proj_step", {})
        proj_global = obj.get("proj_global", {})
        proj_mem = obj.get("proj_mem", None)
        mem_tok = obj.get("mem_token_embeds", None)

        def _as_sd(x):
            if not isinstance(x, dict):
                return {}
            out = {}
            for k, v in x.items():
                if torch.is_tensor(v):
                    out[str(k)] = v.detach().cpu()
            return out

        export = StepSlotAdapterExport(
            order_len=int(obj.get("order_len", 4)),
            mem_tokens=int(obj.get("mem_tokens", 0)),
            hidden_size=int(obj.get("hidden_size", 0)),
            step_tokens=[str(x) for x in obj.get("step_tokens", [])],
            global_token=str(obj.get("global_token", "<cls>")),
            token_embeds=token_embeds,
            proj_step=_as_sd(proj_step),
            proj_global=_as_sd(proj_global),
            mem_token_embeds=mem_tok.detach().cpu() if torch.is_tensor(mem_tok) else None,
            proj_mem=_as_sd(proj_mem) if isinstance(proj_mem, dict) else None,
        )
        if export.hidden_size <= 0:
            raise RuntimeError("export.hidden_size is missing/invalid")
        return export

    @classmethod
    def from_export(cls, path: str | Path, out_dim: int = 768) -> tuple["SoftStepSlotAdapter", StepSlotAdapterExport]:
        export = cls._load_export(path)
        adapter = cls(hidden_size=export.hidden_size, out_dim=int(out_dim), mem_tokens=int(export.mem_tokens))

        # Load projection weights.
        if export.proj_step:
            adapter.proj_step.load_state_dict(export.proj_step, strict=True)
        if export.proj_global:
            adapter.proj_global.load_state_dict(export.proj_global, strict=True)
        if adapter.mem_tokens > 0:
            if adapter.mem_token_embeds is not None and export.mem_token_embeds is not None:
                if tuple(adapter.mem_token_embeds.shape) != tuple(export.mem_token_embeds.shape):
                    raise RuntimeError(
                        f"mem_token_embeds shape mismatch: export={tuple(export.mem_token_embeds.shape)} "
                        f"model={tuple(adapter.mem_token_embeds.shape)}"
                    )
                with torch.no_grad():
                    adapter.mem_token_embeds.copy_(export.mem_token_embeds)
            if adapter.proj_mem is not None and export.proj_mem:
                adapter.proj_mem.load_state_dict(export.proj_mem, strict=True)
        return adapter, export

