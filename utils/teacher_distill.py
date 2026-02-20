from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, Optional, Tuple, Union, List, Any

import torch


_WS_RE = re.compile(r"\s+")


def normalize_teacher_text(text: str) -> str:
    """
    Normalize text used to build teacher/student matching keys.

    We intentionally keep this lightweight (no heavy tokenization) so it can
    be used consistently in:
      - rel3d JSON -> lang_info_data conversion
      - teacher logit export scripts
      - Step-3 training-time lookup
    """
    s = str(text or "").strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def make_teacher_key(
    *,
    teacher_name: str,
    scene_id: str,
    target_gt_id: int,
    text: str,
) -> str:
    """
    Build a deterministic key to join teacher logits with rel3dref samples.

    We hash the payload to keep exported files small and avoid very long dict keys.
    """
    payload = f"{teacher_name}|{scene_id}|{int(target_gt_id)}|{normalize_teacher_text(text)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def scatter_context_logits_to_queries(
    *,
    context_logits: torch.Tensor,  # [N_ctx]
    context_instance_ids: torch.Tensor,  # [N_ctx]
    gt_to_query_map: Dict[int, int],
    num_queries: int,
    fill_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Map teacher logits over ReferIt3D context slots to Mask3D query logits.
    """
    if context_logits.dim() != 1 or context_instance_ids.dim() != 1:
        raise ValueError("context_logits and context_instance_ids must be 1D tensors.")
    if context_logits.numel() != context_instance_ids.numel():
        raise ValueError("context_logits and context_instance_ids must have same length.")

    out = torch.full((num_queries,), fill_value, dtype=context_logits.dtype)
    for j in range(int(context_logits.numel())):
        inst_id = int(context_instance_ids[j].item())
        if inst_id < 0:
            continue
        qidx = gt_to_query_map.get(inst_id, None)
        if qidx is None:
            continue
        if 0 <= int(qidx) < int(num_queries):
            val = float(context_logits[j].item())
            # If multiple context slots map to the same query, keep the max logit.
            if out[int(qidx)].isfinite():
                out[int(qidx)] = max(float(out[int(qidx)].item()), val)
            else:
                out[int(qidx)] = val
    return out


class TeacherLogitsDB:
    """
    Simple in-memory lookup for teacher logits exported by tools/export_*_teacher_logits.py.
    """

    def __init__(self, path: Union[str, List[str]]) -> None:
        # Accept a single path or a list of paths (comma-separated env vars).
        if isinstance(path, list):
            self.paths = [str(p) for p in path if str(p)]
        else:
            self.paths = [str(path)] if str(path) else []
        # Values can be either a single [Q] tensor (legacy) or a dict containing:
        #   - "final": [Q] tensor (required)
        #   - "steps": [T, Q] tensor (optional; per-step logits for distillation)
        #   - "ori_len": scalar tensor or int (optional; valid step count)
        self._table: Dict[str, Union[torch.Tensor, Dict[str, Any]]] = {}

    def load(self) -> None:
        for p in self.paths:
            self._load_one(p)

    def _load_one(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            # Accept {teacher_key: query_logits} (legacy) OR {teacher_key: {"final": ..., "steps": ...}}.
            for k, v in data.items():
                if not isinstance(k, str):
                    continue
                if torch.is_tensor(v):
                    self._table[k] = v.detach().cpu()
                    continue
                if isinstance(v, dict):
                    final = v.get("final", None)
                    if final is None:
                        final = v.get("query_logits", None)
                    steps = v.get("steps", None)
                    if not torch.is_tensor(final):
                        continue
                    entry: Dict[str, Any] = {"final": final.detach().cpu()}
                    if torch.is_tensor(steps):
                        entry["steps"] = steps.detach().cpu()
                    ori_len = v.get("ori_len", None)
                    if torch.is_tensor(ori_len):
                        entry["ori_len"] = ori_len.detach().cpu()
                    elif isinstance(ori_len, int):
                        entry["ori_len"] = int(ori_len)
                    # Keep any other tensor payloads (e.g. feature-level distillation targets).
                    # Example keys:
                    #   - rf_target: [D] teacher relation-field feature for target object
                    #   - rf_anchor: [D] teacher relation-field feature for anchor object
                    #   - rf_*_qidx: scalar tensor/int, query index bookkeeping
                    for kk, vv in v.items():
                        if kk in {"final", "query_logits", "steps", "ori_len"}:
                            continue
                        if torch.is_tensor(vv):
                            entry[kk] = vv.detach().cpu()
                        elif isinstance(vv, (int, float, str)):
                            entry[kk] = vv
                    self._table[k] = entry
            return

        if not isinstance(data, list):
            raise ValueError(f"Unsupported teacher logits format: {type(data)}")

        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("teacher_key", None)
            qlogits = item.get("query_logits", None)
            if isinstance(key, str) and torch.is_tensor(qlogits):
                self._table[key] = qlogits.detach().cpu()

    def add_path(self, path: str) -> None:
        p = str(path or "")
        if not p:
            return
        self.paths.append(p)
        self._load_one(p)

    def get(self, teacher_key: str) -> Optional[torch.Tensor]:
        """
        Legacy accessor: returns the final [Q] logits.
        """
        return self.get_final(teacher_key)

    def get_final(self, teacher_key: str) -> Optional[torch.Tensor]:
        v = self._table.get(teacher_key, None)
        if torch.is_tensor(v):
            return v
        if isinstance(v, dict):
            final = v.get("final", None)
            return final if torch.is_tensor(final) else None
        return None

    def get_steps(self, teacher_key: str) -> Optional[torch.Tensor]:
        v = self._table.get(teacher_key, None)
        if isinstance(v, dict):
            steps = v.get("steps", None)
            return steps if torch.is_tensor(steps) else None
        return None

    def get_ori_len(self, teacher_key: str) -> Optional[int]:
        v = self._table.get(teacher_key, None)
        if isinstance(v, dict):
            ori_len = v.get("ori_len", None)
            if torch.is_tensor(ori_len) and ori_len.numel() == 1:
                try:
                    return int(ori_len.item())
                except Exception:
                    return None
            if isinstance(ori_len, int):
                return int(ori_len)
        return None

    def get_tensor(self, teacher_key: str, name: str) -> Optional[torch.Tensor]:
        """
        Fetch an extra tensor payload from a dict-style entry.
        """
        v = self._table.get(teacher_key, None)
        if isinstance(v, dict):
            t = v.get(name, None)
            return t if torch.is_tensor(t) else None
        return None

    def get_value(self, teacher_key: str, name: str) -> Optional[Any]:
        """
        Fetch an extra payload (tensor/int/float/str) from a dict-style entry.
        """
        v = self._table.get(teacher_key, None)
        if isinstance(v, dict):
            return v.get(name, None)
        return None
