"""
Spatial-state geometric head for Grounded 3D-LLM.

This module has two responsibilities:

1) Geometry-only branch
   Given per-instance query features and 3D centres for a single scene,
   run InstanceRelationField over the centres and project the resulting
   relation field to the query feature dimension, returning a delta
   tensor that can be added to queries before sending them into the LLM.

2) Auxiliary rel3dref supervision branch
   For rel3dref:* language samples (from lang_info_data), build fused
   geometry+text tokens and compute auxiliary losses:
     - referential CE over target query index,
     - anchor multi-label BCE over query positions,
     - relation-type classification from text (between/left/right/...).
"""

from __future__ import annotations

import os, sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# Ensure release repo root and src/ are importable when this file is executed directly.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from models.relation_field import InstanceRelationField, RelationFieldConfig  # type: ignore
from models.referential_order_decoder import (  # type: ignore
    ReferentialOrderDecoder,
    ReferentialDecoderConfig,
    compute_order_loss,
)
from utils.teacher_distill import TeacherLogitsDB, make_teacher_key  # type: ignore

try:
    from models.referit3d_listener_runtime import ReferIt3DListenerRuntime as VigorRuntimeListener
except Exception:  # pragma: no cover
    VigorRuntimeListener = None  # type: ignore


class SSR3DLLMGeomHeadForLLM(nn.Module):
    """
    Geometry head used as an add-on before feeding instance queries to the LLM.

    Args:
        hidden_dim:    Query feature dimension D (e.g. Mask3DLang.mask_dim).
        d_model:       Internal relation-field dimension. By default we keep it
                       equal to hidden_dim to avoid extra projections.
        bert_model:    Path/name for BERT encoder used in the supervisory head.
        max_rel_types: Max number of distinct relation types (rel3dref:between,
                       rel3dref:left, etc.) to support in the language classifier.
    """

    def __init__(
        self,
        hidden_dim: int,
        d_model: Optional[int] = None,
        bert_model: str = "pretrained/bert-base-uncased",
        max_rel_types: int = 16,
    ) -> None:
        super().__init__()
        # `hidden_dim` is the Mask3D/Mask3DLang query feature dimension (e.g. 128).
        # Internally, we may use a higher-capacity student feature space (e.g. 768)
        # to better match listener capacity during geometry-guided supervision.
        # If `d_model` is not provided, allow an env-var override.
        if d_model is None:
            d_model_env = os.environ.get("SSR3DLLM_GEOM_STUDENT_DIM", "").strip()
            try:
                d_model = int(d_model_env) if d_model_env else int(hidden_dim)
            except Exception:
                d_model = int(hidden_dim)
        self.query_dim = int(hidden_dim)
        self.student_dim = int(d_model)

        cfg = RelationFieldConfig(d_model=self.student_dim, n_head=8, d_hidden=1024, dropout=0.1)
        self.relation_field = InstanceRelationField(cfg)

        # Project student-space relation-field features back to query-space deltas.
        if self.student_dim == self.query_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(self.student_dim, self.query_dim)

        # Project query features up to student-space for decoding/losses.
        if self.student_dim == self.query_dim:
            self.query_up = nn.Identity()
        else:
            self.query_up = nn.Linear(self.query_dim, self.student_dim)

        # Text encoder + projection for rel3dref supervision.
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.text_encoder = BertModel.from_pretrained(bert_model)
        bert_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(bert_hidden, self.student_dim)

        # Heads for auxiliary losses.
        self.ref_head = nn.Linear(self.student_dim, 1)      # per-query referential logit
        self.anchor_head = nn.Linear(self.student_dim, 1)   # per-query anchor logit
        self.rel_cls_head = nn.Linear(self.student_dim, max_rel_types)  # relation-type cls

        # Referential order decoder (pointer network over instance tokens).
        dec_cfg = ReferentialDecoderConfig(d_model=self.student_dim, max_steps=4)
        self.decoder = ReferentialOrderDecoder(dec_cfg)

        # A0 (optional): single-pass step-slot classifier on top of LLM step-token embeddings.
        # We keep it in query_dim space so gradients can flow into `<stepK>` token rows via
        # `LLama3d.hidden_state2query` without requiring extra LLM forwards.
        self._stepslot_num_classes = 201  # 200 ScanNet200 + 1 STOP
        self.stepslot_head = nn.Linear(self.query_dim, self._stepslot_num_classes)

        self.max_rel_types = max_rel_types
        self.rel_type_to_id: Dict[str, int] = {}

        # Optional offline teacher-logits DBs for distillation (loaded lazily).
        self._teacher_dbs: Dict[str, TeacherLogitsDB] = {}
        self._teacher_db_warned: set[str] = set()
        # Lazy cache for Mask3D feature files (used to align Vigor runtime inputs).
        # Keyed by (root_dir, scene_id).
        # NOTE: unbounded caching can OOM/kill long eval runs (hundreds of scenes).
        # We therefore keep an LRU cache with a configurable max size.
        self._mask3d_feat_cache: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
        self._mask3d_feat_cache_warned: set[tuple[str, str]] = set()
        # Optional ScanNet scan objects cache (for GT box_info like Vigor training).
        self._vigor_scans_cache = None
        self._vigor_scans_cache_warned = False
        # Lazy projection: LLM step embeddings (mask_dim/query_dim) -> Vigor inner_dim.
        self._vigor_step_proj: Optional[nn.Module] = None

        # IMPORTANT:
        # When we fine-tune the Vigor listener (SSR3DLLM_VIGOR_FINETUNE=1), its weights must be
        # saved/restored as part of the Lightning checkpoint. PyTorch/Lightning can only load
        # submodule weights if the submodule exists at `load_state_dict` time, so we optionally
        # pre-create the runtime here when the geometry backend is set to "vigor".
        #
        # NOTE (DDP safety):
        # If we pre-create the Vigor listener while keeping it frozen (FINETUNE=0), its parameters
        # still default to `requires_grad=True` until `_get_vigor_runtime()` is called (which happens
        # after DDP wraps the module). This can lead to *per-rank trainable parameter mismatches*
        # (e.g. if import/init fails on a subset of ranks), causing:
        #   "DDP expects same model across all ranks ..."
        # Therefore:
        # - Only pre-create when FINETUNE=1 (we truly need it in the checkpoint state).
        # - Otherwise keep it lazy; `_get_vigor_runtime()` will create it later and immediately
        #   freeze it (FINETUNE=0), so it will not participate in DDP buckets.
        self._vigor_runtime: Optional["VigorRuntimeListener"] = None
        try:
            finetune_listener = self._env_flag("SSR3DLLM_VIGOR_FINETUNE", "0")
            if finetune_listener and VigorRuntimeListener is not None and self._get_geom_backend() == "vigor":
                # Init on CPU; Lightning will later move parameters to the right device.
                self._vigor_runtime = VigorRuntimeListener(device=torch.device("cpu"))
        except Exception:
            # Keep it lazy if anything goes wrong; runtime will be created on-demand.
            self._vigor_runtime = None

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        v = os.environ.get(name, default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _get_mask3d_feat_root(self) -> str:
        # Prefer the same env names used by Vigor tooling/scripts.
        for k in (
            "MASK3D_FEATS_TRAIN",
            "MASK3D_FEATS_TEST",
            "MASK3D_FEATS_VAL",
            "MASK3D_FEAT_VAL",
            "MASK3D_FEATURES_TEST",
            "MASK3D_FEATURES_VAL",
        ):
            v = os.environ.get(k, "").strip()
            if v:
                return v
        return ""

    def compute_stepslot_loss_for_batch(
        self,
        *,
        batch_lang_infos: List[object],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional A0 loss: supervise step-slot predictions (ScanNet200 + STOP) from LLM `<stepK>` embeddings.

        Enabled by:
          - SSR3DLLM_ORDER_MODE=slots
          - SSR3DLLM_ORDER_LOSS_WEIGHT>0

        Label sources:
          - ScanRefer/M3DRef: `lang_info.ssr3dllm_target_class_id200` (preferred)
          - Rel3DRef/ReferIt3D steps: `lang_info.rel_referential_order` mapped to ScanNet200 (best-effort)

        This loss is designed to be non-fatal: samples with missing/unmappable labels are ignored.
        """
        order_mode = str(os.environ.get("SSR3DLLM_ORDER_MODE", "")).strip().lower()
        if order_mode != "slots":
            return {}
        try:
            w = float(str(os.environ.get("SSR3DLLM_ORDER_LOSS_WEIGHT", "0")).strip() or "0")
        except Exception:
            w = 0.0
        if w <= 0.0:
            return {}
        if not batch_lang_infos:
            return {}

        try:
            from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_200 as _CLS200  # type: ignore
        except Exception:
            _CLS200 = []
        labels_set = {str(x).strip().lower().replace("_", " ") for x in _CLS200} if _CLS200 else set()
        name_to_id = {str(x).strip().lower().replace("_", " "): int(i) for i, x in enumerate(_CLS200)} if _CLS200 else {}
        stop_id = int(self._stepslot_num_classes) - 1
        order_len = int(max(1, int(str(os.environ.get("SSR3DLLM_ORDER_MAX_LEN", "4")).strip() or "4")))

        def _normalize_phrase(s: str) -> str:
            import re as _re
            s = str(s).strip().lower().replace("_", " ")
            s = _re.sub(r"[\t\r\n]+", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            for prefix in ("a ", "an ", "the "):
                if s.startswith(prefix):
                    s = s[len(prefix) :].strip()
                    break
            s = s.replace("&", " ")
            s = _re.sub(r"[^a-z0-9 ]+", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            return s

        def _singularize_last_word(phrase: str) -> str:
            w0 = phrase.split()
            if not w0:
                return phrase
            last = w0[-1]
            cand = [last]
            if last.endswith("ies") and len(last) > 3:
                cand.append(last[:-3] + "y")
            if last.endswith("es") and len(last) > 2:
                cand.append(last[:-2])
            if last.endswith("s") and len(last) > 1 and not last.endswith("ss"):
                cand.append(last[:-1])
            for c in cand[1:]:
                if c and c != last:
                    return " ".join(w0[:-1] + [c])
            return phrase

        synonyms = {
            "television": "tv",
            "tv monitor": "tv",
            "trashcan": "trash can",
            "trash bin": "trash can",
            "garbage can": "trash can",
            "sofa": "couch",
            "bookcase": "bookshelf",
            "tub": "bathtub",
            "computer monitor": "monitor",
            "countertop": "counter",
            "counter top": "counter",
            "kitchen table": "table",
            "kitchen sink": "sink",
            "bathroom sink": "sink",
            "wardrobe closet": "closet",
        }

        def _map_to_id200(phrase: str) -> Optional[int]:
            if not name_to_id:
                return None
            p = _normalize_phrase(phrase)
            if not p:
                return None
            if p in name_to_id:
                return int(name_to_id[p])
            p2 = _singularize_last_word(p)
            if p2 in name_to_id:
                return int(name_to_id[p2])
            if p in synonyms:
                s = synonyms[p]
                if s in name_to_id:
                    return int(name_to_id[s])
            if p2 in synonyms:
                s = synonyms[p2]
                if s in name_to_id:
                    return int(name_to_id[s])
            last = p.split()[-1] if p.split() else ""
            if last in name_to_id:
                return int(name_to_id[last])
            last2 = _singularize_last_word(last)
            if last2 in name_to_id:
                return int(name_to_id[last2])
            return None

        # Collect supervised (step_embeds, labels) pairs.
        embeds_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        for li in batch_lang_infos:
            lt = getattr(li, "lang_type", "")
            if not isinstance(lt, str) or not lt:
                continue
            prefix = lt.split(":")[0]
            if prefix not in {"scanrefer", "m3dref", "rel3dref"}:
                continue

            step_emb = getattr(li, "llm_step_embeds", None)
            if not torch.is_tensor(step_emb) or step_emb.dim() != 2:
                continue
            if int(step_emb.size(-1)) != int(self.query_dim):
                # In slots mode we expect query-dim step embeds (from main LLM forward).
                continue

            # Build label sequence up to order_len with STOP padding.
            ys: List[int] = []
            # Prefer explicit target class id when present (ScanRefer/M3DRef single-step).
            tgt_id200 = getattr(li, "ssr3dllm_target_class_id200", None)
            if prefix in {"scanrefer", "m3dref"} and isinstance(tgt_id200, int) and 0 <= int(tgt_id200) < 200:
                ys = [int(tgt_id200)]
            else:
                order = getattr(li, "rel_referential_order", None)
                if isinstance(order, list):
                    for x in order:
                        if not isinstance(x, str) or not str(x).strip():
                            continue
                        mid = _map_to_id200(str(x))
                        if mid is None:
                            continue
                        ys.append(int(mid))
            if not ys:
                continue
            ys = ys[: int(order_len)]
            # STOP padding after the last valid step.
            while len(ys) < int(order_len):
                ys.append(int(stop_id))

            y_t = torch.as_tensor(ys, device=device, dtype=torch.long)
            e_t = step_emb[: int(order_len)].to(device=device, dtype=torch.float32)
            if int(e_t.size(0)) < int(order_len):
                pad = e_t[-1:].repeat(int(order_len - int(e_t.size(0))), 1)
                e_t = torch.cat([e_t, pad], dim=0)
            embeds_list.append(e_t)
            labels_list.append(y_t)

        if not embeds_list:
            return {}

        embeds = torch.stack(embeds_list, dim=0)  # [B,O,D]
        labels = torch.stack(labels_list, dim=0)  # [B,O]
        logits = self.stepslot_head(embeds)  # [B,O,C]
        loss = F.cross_entropy(
            logits.reshape(-1, int(logits.size(-1))),
            labels.reshape(-1),
            reduction="mean",
        )
        return {"loss_ssr3dllm_stepslot": loss * float(w)}

    def _get_scannet_pkl_path(self) -> str:
        for k in ("SSR3DLLM_VIGOR_SCANNET_PKL", "SCANNET_PKL"):
            v = os.environ.get(k, "").strip()
            if v:
                return v
        return ""

    def _mask3d_feat_cache_max(self) -> int:
        """
        Max number of cached Mask3D feature dicts.

        Env:
          - SSR3DLLM_MASK3D_FEAT_CACHE_MAX (preferred)
          - SSR3DLLM_MASK3D_FEAT_CACHE_MAX_SCENES (backward-compat)

        Semantics:
          - max < 0: unlimited (not recommended for full eval)
          - max = 0: disable caching
          - max > 0: keep at most `max` entries (LRU)
        """
        for k in ("SSR3DLLM_MASK3D_FEAT_CACHE_MAX", "SSR3DLLM_MASK3D_FEAT_CACHE_MAX_SCENES"):
            v = os.environ.get(k, "").strip()
            if v:
                try:
                    return int(v)
                except Exception:
                    pass
        # Safe default for long eval runs.
        return 32

    def _get_vigor_scans(self) -> Optional[dict]:
        if self._vigor_scans_cache is not None:
            return self._vigor_scans_cache
        if not self._env_flag("SSR3DLLM_VIGOR_USE_SCANNET_BOXES", "0"):
            self._vigor_scans_cache = None
            return None
        p = self._get_scannet_pkl_path()
        if not p:
            self._vigor_scans_cache = None
            return None
        try:
            from pathlib import Path

            pp = Path(p)
            if not pp.exists():
                self._vigor_scans_cache = None
                return None
            # Heavy load (1513 scans); do it once per process.
            from third_party.Vigor.referit3d.in_out.neural_net_oriented import (
                load_scan_related_data,
            )

            all_scans, _, _ = load_scan_related_data(str(pp), verbose=(not self._vigor_scans_cache_warned), add_pad=False)
            self._vigor_scans_cache_warned = True
            self._vigor_scans_cache = all_scans if isinstance(all_scans, dict) else None
            return self._vigor_scans_cache
        except Exception:
            self._vigor_scans_cache = None
            return None

    def _load_mask3d_feat(self, scene_id: str) -> Optional[dict]:
        if not scene_id:
            return None

        # Try multiple roots to cover train/val splits without requiring callers to
        # know which split a scene belongs to.
        roots = []
        for k in (
            "MASK3D_FEATS_TRAIN",
            "MASK3D_FEATS_VAL",
            "MASK3D_FEATS_TEST",
            "MASK3D_FEAT_VAL",
            "MASK3D_FEATURES_VAL",
            "MASK3D_FEATURES_TEST",
        ):
            v = os.environ.get(k, "").strip()
            if v:
                roots.append(v)
        # Fallback to the default single-root resolver.
        r0 = self._get_mask3d_feat_root()
        if r0:
            roots.append(r0)
        # De-dup while preserving order.
        seen = set()
        roots = [r for r in roots if not (r in seen or seen.add(r))]

        if not roots:
            return None

        try:
            from pathlib import Path

            for root in roots:
                key = (root, str(scene_id))
                if key in self._mask3d_feat_cache:
                    feat = self._mask3d_feat_cache[key]
                    try:
                        self._mask3d_feat_cache.move_to_end(key)
                    except Exception:
                        pass
                    return feat
                p = Path(root) / f"{scene_id}.pt"
                if not p.exists():
                    continue
                feat = torch.load(str(p), map_location="cpu")
                if not isinstance(feat, dict):
                    continue

                if self._env_flag("SSR3DLLM_DEBUG_VIGOR_NAMES", "0") and key not in self._mask3d_feat_cache_warned:
                    self._mask3d_feat_cache_warned.add(key)
                    try:
                        oq = feat.get("object_queries", None)
                        oq_shape = tuple(oq.shape) if torch.is_tensor(oq) else None
                    except Exception:
                        oq_shape = None
                    try:
                        pcn = feat.get("pred_class_names", None)
                        pcn_len = len(pcn) if isinstance(pcn, list) else None
                    except Exception:
                        pcn_len = None
                    try:
                        pc = feat.get("pred_classes", None)
                        pc_shape = tuple(pc.shape) if torch.is_tensor(pc) else None
                        if pc_shape is None and isinstance(pc, (list, tuple)):
                            pc_shape = (len(pc),)
                    except Exception:
                        pc_shape = None
                    try:
                        gmap = feat.get("gt_to_query_map", None)
                        gmap_len = len(gmap) if isinstance(gmap, dict) else None
                    except Exception:
                        gmap_len = None
                    try:
                        gcls = feat.get("gt_instance_classes", None)
                        gcls_len = len(gcls) if isinstance(gcls, dict) else None
                    except Exception:
                        gcls_len = None
                    try:
                        keys_preview = list(feat.keys())[:30]
                    except Exception:
                        keys_preview = []
                    print(
                        "[SSR3DLLMGeomHead][vigor_names] "
                        f"root={root} scene={scene_id} "
                        f"object_queries={oq_shape} pred_class_names_len={pcn_len} pred_classes={pc_shape} "
                        f"gt_to_query_map_len={gmap_len} gt_instance_classes_len={gcls_len} "
                        f"keys={keys_preview}",
                        flush=True,
                    )

                cache_max = int(self._mask3d_feat_cache_max())
                if cache_max != 0:
                    self._mask3d_feat_cache[key] = feat
                    try:
                        self._mask3d_feat_cache.move_to_end(key)
                    except Exception:
                        pass
                    if cache_max > 0:
                        while len(self._mask3d_feat_cache) > cache_max:
                            try:
                                evicted_key, _ = self._mask3d_feat_cache.popitem(last=False)
                                if self._env_flag("SSR3DLLM_DEBUG_VIGOR_NAMES", "0"):
                                    print(
                                        f"[SSR3DLLMGeomHead][mask3d_cache] evict={evicted_key} size={len(self._mask3d_feat_cache)}/{cache_max}",
                                        flush=True,
                                    )
                            except Exception:
                                break
                return feat

            # Warn once if none of the roots contained the file.
            if self._env_flag("SSR3DLLM_DEBUG_VIGOR_NAMES", "0"):
                root0 = roots[0]
                key0 = (root0, str(scene_id))
                if key0 not in self._mask3d_feat_cache_warned:
                    self._mask3d_feat_cache_warned.add(key0)
                    print(
                        f"[SSR3DLLMGeomHead][vigor_names][warn] feat file missing for scene={scene_id} roots={roots}",
                        flush=True,
                    )
            return None
        except Exception:
            return None

    def _get_pred_class_names_from_feat(self, scene_id: str, q_expected: int) -> Optional[List[str]]:
        feat = self._load_mask3d_feat(scene_id)
        if not isinstance(feat, dict):
            return None
        names = feat.get("pred_class_names", None)
        if isinstance(names, list) and len(names) == int(q_expected):
            out = [str(x) if x is not None else "unknown" for x in names]
            # Match Vigor's `_get_mask3d_pred_name` fallback behaviour:
            # if a predicted class name is missing/unknown, fall back to the GT instance class
            # of the ScanNet object mapped to this query (when available).
            try:
                gt_map = feat.get("gt_to_query_map", None)
                inst_cls = feat.get("gt_instance_classes", None)
                if isinstance(gt_map, dict) and isinstance(inst_cls, dict):
                    from baseline.dataset.datasets.scannet200.scannet200_constants import (
                        CLASS_LABELS_200,
                        VALID_CLASS_IDS_200,
                    )

                    id_to_name = {int(cid): str(CLASS_LABELS_200[i]) for i, cid in enumerate(VALID_CLASS_IDS_200)}
                    # Invert mapping: qidx -> inst_id (best-effort first match).
                    q2inst: Dict[int, int] = {}
                    for inst_id, qidx in gt_map.items():
                        try:
                            qi = int(qidx) if qidx is not None else None
                        except Exception:
                            qi = None
                        if qi is None or not (0 <= qi < int(q_expected)):
                            continue
                        if int(qi) not in q2inst:
                            try:
                                q2inst[int(qi)] = int(inst_id)
                            except Exception:
                                continue
                    for qi in range(int(q_expected)):
                        name = str(out[qi]).strip().lower()
                        if name and name != "unknown":
                            continue
                        inst_id = q2inst.get(int(qi), None)
                        if inst_id is None:
                            continue
                        cid = inst_cls.get(inst_id, None)
                        try:
                            cid_int = int(cid) if cid is not None else None
                        except Exception:
                            cid_int = None
                        if cid_int is None:
                            continue
                        out[qi] = id_to_name.get(int(cid_int), "unknown")
            except Exception:
                pass
            return out
        # Fallback: map pred_classes -> ScanNet200 names if available.
        pred_classes = feat.get("pred_classes", None)
        if torch.is_tensor(pred_classes):
            try:
                pc = pred_classes.detach().cpu().tolist()
            except Exception:
                pc = None
        elif isinstance(pred_classes, (list, tuple)):
            pc = list(pred_classes)
        else:
            pc = None
        if isinstance(pc, list) and len(pc) == int(q_expected):
            try:
                from baseline.dataset.datasets.scannet200.scannet200_constants import (
                    CLASS_LABELS_200,
                    VALID_CLASS_IDS_200,
                )

                # ScanNet200 ids are sparse; map id->name via VALID_CLASS_IDS_200.
                id_to_name = {int(cid): str(CLASS_LABELS_200[i]) for i, cid in enumerate(VALID_CLASS_IDS_200)}
                out = []
                for c in pc:
                    try:
                        ci = int(c)
                    except Exception:
                        ci = -1
                    out.append(id_to_name.get(int(ci), "unknown") if ci >= 0 else "unknown")
                return out
            except Exception:
                return ["unknown"] * int(q_expected)
        # Fallback: build per-query *GT* class names via gt_to_query_map + gt_instance_classes.
        # This mirrors Vigor's training-time behaviour where it can fall back to GT instance labels
        # when predicted class names are missing.
        try:
            gt_map = feat.get("gt_to_query_map", None)
            inst_cls = feat.get("gt_instance_classes", None)
            if isinstance(gt_map, dict) and isinstance(inst_cls, dict):
                from baseline.dataset.datasets.scannet200.scannet200_constants import (
                    CLASS_LABELS_200,
                    VALID_CLASS_IDS_200,
                )

                id_to_name = {int(cid): str(CLASS_LABELS_200[i]) for i, cid in enumerate(VALID_CLASS_IDS_200)}
                out = ["unknown"] * int(q_expected)
                for inst_id, qidx in gt_map.items():
                    try:
                        q = int(qidx) if qidx is not None else None
                    except Exception:
                        q = None
                    if q is None or not (0 <= int(q) < int(q_expected)):
                        continue
                    cid = inst_cls.get(inst_id, None)
                    try:
                        cid_int = int(cid) if cid is not None else None
                    except Exception:
                        cid_int = None
                    if cid_int is None:
                        continue
                    out[int(q)] = id_to_name.get(int(cid_int), "unknown")
                return out
        except Exception:
            pass
        return None

    def _get_gt_to_query_map_from_feat(self, scene_id: str) -> Optional[Dict[int, int]]:
        feat = self._load_mask3d_feat(scene_id)
        if not isinstance(feat, dict):
            return None
        m = feat.get("gt_to_query_map", None)
        if not isinstance(m, dict):
            return None
        out: Dict[int, int] = {}
        for k, v in m.items():
            try:
                kk = int(k)
                vv = int(v) if v is not None else None
            except Exception:
                continue
            if vv is None:
                continue
            out[kk] = vv
        return out

    def _get_box_info_from_scannet(
        self, scene_id: str, gt_to_query_map: Dict[int, int], q_expected: int
    ) -> Optional[torch.Tensor]:
        scans = self._get_vigor_scans()
        if not isinstance(scans, dict) or scene_id not in scans:
            return None
        try:
            scan = scans[scene_id]
            # Default to zeros; fill only mapped queries.
            box = torch.zeros((int(q_expected), 4), dtype=torch.float32)
            # Invert mapping: qidx -> inst_id (first one wins).
            q2inst: Dict[int, int] = {}
            for inst_id, qidx in gt_to_query_map.items():
                try:
                    qi = int(qidx)
                    ii = int(inst_id)
                except Exception:
                    continue
                if 0 <= qi < int(q_expected) and qi not in q2inst:
                    q2inst[qi] = ii
            for qi, ii in q2inst.items():
                try:
                    obj = scan.three_d_objects[int(ii)]
                    bb = obj.get_bbox()
                    box[int(qi), 0] = float(bb.cx)
                    box[int(qi), 1] = float(bb.cy)
                    box[int(qi), 2] = float(bb.cz)
                    box[int(qi), 3] = float(bb.volume())
                except Exception:
                    continue
            return box
        except Exception:
            return None

    def _get_teacher_db(self, teacher_name: str) -> Optional[TeacherLogitsDB]:
        """
        Lazy-load an offline teacher logits DB from env vars.

        Expected env var (paths can be relative to repo root):
          - SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH
        """
        name = str(teacher_name or "").strip().lower()
        if not name:
            return None
        if name in self._teacher_dbs:
            return self._teacher_dbs[name]

        env_var = None
        if name == "vigor":
            env_var = "SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH"
        else:
            return None

        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return None
        # Support comma-separated paths so train/test teacher logits can be merged:
        #   SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH=.../vigor_train_kv.pt,.../vigor_test_kv.pt
        raw_paths = [p.strip() for p in raw.split(",") if p.strip()]
        paths: List[str] = []
        missing: List[str] = []
        for p in raw_paths:
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            if path.exists():
                paths.append(str(path))
            else:
                missing.append(str(path))

        if not paths:
            if name not in self._teacher_db_warned:
                if missing:
                    print(f"[SSR3DLLMGeomHead][distill] teacher logits not found: {missing[0]}")
                else:
                    print(f"[SSR3DLLMGeomHead][distill] teacher logits not found (empty paths): {env_var}")
                self._teacher_db_warned.add(name)
            return None

        # Convenience: if the user only passed a train KV, automatically pick up a sibling
        # test KV (same folder) if present, so eval-time teacher_top1 won't be None simply
        # due to missing split exports.
        if len(paths) == 1:
            p0 = Path(paths[0])
            cand: List[Path] = []
            # Common naming patterns we use in this repo.
            s = p0.name
            if "train_kv" in s:
                cand.append(p0.with_name(s.replace("train_kv", "test_kv")))
            if "train" in s and "kv" in s and "test" not in s:
                cand.append(p0.with_name(s.replace("train", "test")))
            for cp in cand:
                if cp.exists():
                    paths.append(str(cp))

        db = TeacherLogitsDB(paths)
        db.load()
        self._teacher_dbs[name] = db
        if len(paths) == 1:
            print(f"[SSR3DLLMGeomHead][distill] loaded teacher='{name}' logits from {paths[0]}")
        else:
            print(f"[SSR3DLLMGeomHead][distill] loaded teacher='{name}' logits from {len(paths)} files")
        return db

    def _get_geom_backend(self) -> str:
        """
        Select geometry backend for inference-time grounding.

        - "decoder" (default): use SSR3DLLM relation_field + pointer decoder.
        - "vigor": use a pretrained Vigor listener on Mask3D queries.
        """
        v = os.environ.get("SSR3DLLM_GEOM_BACKEND", "decoder")
        v = str(v).strip().lower()
        return v or "decoder"

    def _get_vigor_runtime(self, device: torch.device) -> Optional["VigorRuntimeListener"]:
        if VigorRuntimeListener is None:
            return None
        if self._vigor_runtime is None:
            self._vigor_runtime = VigorRuntimeListener(device=device)
        # Ensure runtime is on the requested device.
        try:
            self._vigor_runtime.to(device)
            self._vigor_runtime.device = device
        except Exception:
            pass
        # Control whether/how we fine-tune the Vigor listener weights inside SSR3DLLM.
        #
        # Env:
        #   - SSR3DLLM_VIGOR_FINETUNE=0/1
        #   - SSR3DLLM_VIGOR_FINETUNE_MODE=none|full|partial (default: full when FINETUNE=1)
        #   - SSR3DLLM_VIGOR_TRAINABLE_PATTERNS="substr1,substr2,..." (only for partial)
        #
        # Rationale:
        # - stage1/2 typically keep listener frozen to preserve cross-class behavior.
        # - stage3 can optionally unfreeze only a tiny subset to avoid "training collapse".
        finetune = self._env_flag("SSR3DLLM_VIGOR_FINETUNE", "0")
        mode = str(os.environ.get("SSR3DLLM_VIGOR_FINETUNE_MODE", "full" if finetune else "none")).strip().lower()
        if mode in {"0", "false", "off", "no", "n"}:
            mode = "none"
        if mode in {"1", "true", "on", "yes", "y", "all"}:
            mode = "full"
        patterns_s = str(os.environ.get("SSR3DLLM_VIGOR_TRAINABLE_PATTERNS", "")).strip()
        patterns = [p.strip() for p in patterns_s.split(",") if p.strip()]

        try:
            if (not finetune) or mode == "none":
                for p in self._vigor_runtime.parameters():
                    p.requires_grad = False
            elif mode == "full":
                for p in self._vigor_runtime.parameters():
                    p.requires_grad = True
            elif mode == "partial":
                for p in self._vigor_runtime.parameters():
                    p.requires_grad = False
                if not patterns:
                    warned = getattr(self, "_vigor_partial_finetune_no_patterns_warned", False)
                    if not warned:
                        setattr(self, "_vigor_partial_finetune_no_patterns_warned", True)
                        print(
                            "[SSR3DLLM][vigor_finetune] FINETUNE_MODE=partial but SSR3DLLM_VIGOR_TRAINABLE_PATTERNS is empty; keeping listener fully frozen.",
                            flush=True,
                        )
                else:
                    for name, p in self._vigor_runtime.named_parameters():
                        if any(sub in name for sub in patterns):
                            p.requires_grad = True
            else:
                # Unknown mode -> safest default.
                for p in self._vigor_runtime.parameters():
                    p.requires_grad = False
        except Exception:
            pass
        # Log finetune status once (rank0 only) to avoid silent "pattern mismatch" failures.
        try:
            if os.environ.get("LOCAL_RANK", "0") == "0" and not getattr(self, "_vigor_finetune_logged", False):
                trainable = [(n, p) for n, p in self._vigor_runtime.named_parameters() if getattr(p, "requires_grad", False)]
                total_tensors = sum(1 for _ in self._vigor_runtime.parameters())
                trainable_tensors = len(trainable)
                total_params = sum(int(p.numel()) for p in self._vigor_runtime.parameters())
                trainable_params = sum(int(p.numel()) for _, p in trainable)
                preview = [n for n, _ in trainable[:12]]
                print(
                    f"[SSR3DLLM][vigor_finetune] finetune={int(bool(finetune))} mode={mode} "
                    f"patterns={patterns if patterns else '[]'} "
                    f"trainable_tensors={trainable_tensors}/{total_tensors} "
                    f"trainable_params={trainable_params}/{total_params}",
                    flush=True,
                )
                if preview:
                    print(f"[SSR3DLLM][vigor_finetune] trainable[:{len(preview)}]={preview}", flush=True)
                setattr(self, "_vigor_finetune_logged", True)
        except Exception:
            pass
        return self._vigor_runtime

    def _get_vigor_step_proj(self, inner_dim: int, device: torch.device) -> nn.Module:
        """
        Project per-step embeddings (query_dim / mask_dim) to Vigor inner_dim.
        Lazily constructed to avoid hard-coding inner_dim in __init__.
        """
        need_new = True
        if self._vigor_step_proj is not None:
            try:
                if isinstance(self._vigor_step_proj, nn.Linear):
                    need_new = not (int(self._vigor_step_proj.in_features) == int(self.query_dim) and int(self._vigor_step_proj.out_features) == int(inner_dim))
                else:
                    need_new = False
            except Exception:
                need_new = True
        if need_new:
            self._vigor_step_proj = nn.Linear(self.query_dim, int(inner_dim), bias=True).to(device=device)
        # By default we keep this projection frozen, since training the Vigor runtime/listener
        # should be opt-in. Stage3 can enable training of ONLY this projection to better align
        # SSR3DLLM query features with the (frozen) Vigor listener input space.
        train_step_proj = self._env_flag("SSR3DLLM_VIGOR_TRAIN_STEP_PROJ", "0")
        try:
            for p in self._vigor_step_proj.parameters():
                p.requires_grad = bool(train_step_proj)
        except Exception:
            pass
        return self._vigor_step_proj

    @staticmethod
    def _strip_geom_token(text: str) -> str:
        return str(text).replace("<geom>", " ").replace("  ", " ").strip()

    def _compute_vigor_chain_loss_for_batch(
        self,
        *,
        batch_lang_infos: List[object],
        sampled_coords: torch.Tensor,
        device: torch.device,
        w_chain: float,
    ) -> Dict[str, torch.Tensor]:
        runtime = self._get_vigor_runtime(device)
        if runtime is None:
            return {}

        # Align with Vigor step-slot training:
        # - Build a sampled context (target + distractors) capped by max_context_size.
        # - Map each context GT instance id -> Mask3D query idx (gt_to_query_map) and use the
        #   CLASP-aligned q_hidden[qidx] as the slot feature.
        # - Use predicted predbox info pred_box_info[qidx] as box_info (cx,cy,cz,volume).
        # - Supervise by target_pos (slot index), not global query index.
        texts: List[str] = []
        slot_feats: List[torch.Tensor] = []     # [N_ctx, query_dim]
        slot_boxes: List[torch.Tensor] = []     # [N_ctx, 4]
        slot_scannet_labels: List[torch.Tensor] = []  # [N_ctx] (contiguous ScanNet200 ids, -1 ignore)
        order_embeds: List[torch.Tensor] = []   # [O,1,inner_dim]
        order_valid_masks: List[torch.Tensor] = []  # [O] float, 1=valid
        target_pos: List[int] = []
        slot_pred_names: List[List[str]] = []   # [N_ctx] per sample (for pred_class_mask)
        slot_orders: List[List[str]] = []       # referential_order strings per sample
        # Optional: LLM-provided token-level memory to bypass BERT lang encoder in Vigor.
        llm_lang_embeds_list: List[torch.Tensor] = []
        # Optional: offline teacher distillation (teacher logits are stored in query space).
        # We keep, for each sample, the query index of every sampled context slot so we can
        # map teacher query logits -> context-slot logits.
        ctx_qidx_list: List[torch.Tensor] = []  # list of [N] long tensors
        teacher_keys: List[Optional[str]] = []  # aligned with `texts`

        require_steps = self._env_flag("SSR3DLLM_VIGOR_REQUIRE_STEP_EMBEDS", "1")
        use_basic_loss = self._env_flag("SSR3DLLM_VIGOR_TRAIN_USE_BASIC_LOSS", "1")
        require_geom_token = self._env_flag("SSR3DLLM_VIGOR_TRAIN_REQUIRE_GEOM_TOKEN", "1")
        prefixes_raw = str(os.environ.get("SSR3DLLM_VIGOR_TRAIN_PREFIXES", "rel3dref")).strip()
        train_prefixes = {p.strip() for p in prefixes_raw.split(",") if p.strip()}
        order_len = int(getattr(runtime, "order_len", 4))
        inner_dim = int(getattr(getattr(runtime, "model", None), "inner_dim", 768))
        enable_text_loss = self._env_flag("SSR3DLLM_VIGOR_TRAIN_ENABLE_TEXT_LOSS", "0")
        use_llm_lang = self._env_flag("SSR3DLLM_VIGOR_USE_LLM_LANG_EMBEDS", "0")
        max_distractors = int(str(os.environ.get("SSR3DLLM_VIGOR_TRAIN_MAX_DISTRACTORS", "51")).strip() or "51")
        if max_distractors < 0:
            max_distractors = 51
        max_context_size = int(max_distractors) + 1
        # Whether to use Mask3D predicted class ids (pred_classes) to construct the Vigor-style
        # context (same-class distractors). In many exports pred_classes can be missing / all -1,
        # so default to GT semantic ids for curriculum/debug. If you want strictly predicted
        # sampling, set SSR3DLLM_VIGOR_CONTEXT_USE_PRED_CLASS=1 and ensure pred_classes are valid.
        context_use_pred_class = self._env_flag("SSR3DLLM_VIGOR_CONTEXT_USE_PRED_CLASS", "0")
        predmask_mode = os.environ.get("SSR3DLLM_VIGOR_PRED_CLASS_MASK_MODE", "all_ones").strip().lower()
        use_predmask_train = self._env_flag("SSR3DLLM_VIGOR_USE_PREDMASK_TRAIN", "0")
        cascading = self._env_flag("SSR3DLLM_VIGOR_CASCADING", "1")
        # Offline distillation knobs (optional; default off). These are meant to preserve
        # cross-class suppression by matching a strong teacher distribution in query space.
        try:
            w_distill_vigor = float(str(os.environ.get("SSR3DLLM_DISTILL_VIGOR_WEIGHT", "0")).strip() or "0")
        except Exception:
            w_distill_vigor = 0.0
        try:
            distill_temperature = float(str(os.environ.get("SSR3DLLM_DISTILL_TEMPERATURE", "1.0")).strip() or "1.0")
        except Exception:
            distill_temperature = 1.0
        teacher_db = self._get_teacher_db("vigor") if (use_basic_loss and w_distill_vigor > 0.0) else None
        if use_basic_loss and w_distill_vigor > 0.0 and teacher_db is None:
            warned = getattr(self, "_vigor_distill_teacher_missing_warned", False)
            if not warned:
                setattr(self, "_vigor_distill_teacher_missing_warned", True)
                print(
                    "[SSR3DLLM][vigor_distill] SSR3DLLM_DISTILL_VIGOR_WEIGHT>0 but teacher logits DB is not loaded. "
                    "Set `SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH=/abs/path/*.pt[,more.pt]` to enable distillation.",
                    flush=True,
                )

        def _norm_name(name: str) -> str:
            return str(name).strip().lower().replace("_", " ")

        def _get_qidx(gt_map: dict, inst_id: int) -> Optional[int]:
            qid = gt_map.get(int(inst_id), None)
            if qid is None and (int(inst_id) + 1) in gt_map and 0 not in gt_map and 1 in gt_map:
                qid = gt_map.get(int(inst_id) + 1, None)
            try:
                return int(qid) if qid is not None else None
            except Exception:
                return None

        def _first_inst_id(li: object) -> Optional[int]:
            inst_ids_answer = getattr(li, "inst_ids_answer", None)
            if not isinstance(inst_ids_answer, list) or not inst_ids_answer:
                return None
            first = inst_ids_answer[0]
            if isinstance(first, list) and first:
                try:
                    return int(first[0])
                except Exception:
                    return None
            if isinstance(first, (int, np.integer)):
                try:
                    return int(first)
                except Exception:
                    return None
            return None

        for li in batch_lang_infos:
            lang_type = getattr(li, "lang_type", "")
            if not isinstance(lang_type, str):
                continue
            prefix = lang_type.split(":")[0]
            if train_prefixes and prefix not in train_prefixes:
                continue
            question = getattr(li, "question", "") or ""
            use_geom_trigger = getattr(li, "use_geom_trigger", False)
            if require_geom_token and ("<geom>" not in question) and (not use_geom_trigger):
                continue

            # Need CLASP-aligned query features (Mask3DLang output).
            q_hidden = getattr(li, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor) or q_hidden.dim() != 2:
                continue
            if int(q_hidden.size(-1)) != int(self.query_dim):
                continue
            Q = int(q_hidden.size(0))
            if Q <= 0:
                continue

            # Need teacher-forced step embeddings (LLM <stepK> states).
            step_emb = getattr(li, "llm_step_embeds", None)
            if step_emb is None or (not isinstance(step_emb, torch.Tensor)) or step_emb.dim() not in (2, 3):
                if require_steps:
                    continue
                step_emb = torch.zeros((order_len, self.query_dim), device=q_hidden.device, dtype=q_hidden.dtype)
            # Accept either query_dim embeddings (need proj) or inner_dim embeddings (ready for Vigor).
            if step_emb.dim() == 2:
                if int(step_emb.size(-1)) not in (int(self.query_dim), int(inner_dim)):
                    if require_steps:
                        continue
                    step_emb = torch.zeros((order_len, self.query_dim), device=q_hidden.device, dtype=q_hidden.dtype)
                if int(step_emb.size(0)) < int(order_len):
                    pad = (
                        step_emb[-1:].repeat(int(order_len - int(step_emb.size(0))), 1)
                        if int(step_emb.size(0)) > 0
                        else torch.zeros((int(order_len), int(step_emb.size(-1))), device=step_emb.device, dtype=step_emb.dtype)
                    )
                    step_emb = torch.cat([step_emb, pad], dim=0)
                elif int(step_emb.size(0)) > int(order_len):
                    step_emb = step_emb[: int(order_len)]
            else:
                # [O,L,D] path (rare): keep only first L token to match downstream shape.
                if int(step_emb.size(-1)) != int(inner_dim):
                    if require_steps:
                        continue
                    step_emb = torch.zeros((order_len, 1, inner_dim), device=q_hidden.device, dtype=q_hidden.dtype)
                if int(step_emb.size(0)) < int(order_len):
                    pad = (
                        step_emb[-1:].repeat(int(order_len - int(step_emb.size(0))), 1, 1)
                        if int(step_emb.size(0)) > 0
                        else torch.zeros((int(order_len), int(step_emb.size(1)), int(step_emb.size(2))), device=step_emb.device, dtype=step_emb.dtype)
                    )
                    step_emb = torch.cat([step_emb, pad], dim=0)
                elif int(step_emb.size(0)) > int(order_len):
                    step_emb = step_emb[: int(order_len)]

            scene_id = None
            target_gt_id = None
            if prefix == "rel3dref":
                scene_id = getattr(li, "rel_scene_id", None)
                target_gt_id = getattr(li, "rel_target_object_gt_id", None)
            else:
                scene_id = getattr(li, "scene_id", None)
                # IMPORTANT: for ScanRefer/M3DRef we must use the dataset-defined ScanNet instance id
                # (pre remap_inst_ids()). This is attached upstream as `lang_info.target_gt_id`
                # in `trainer/trainer.py` and is required to keep grounding_steps lookup + chain
                # supervision consistent across ranks. `inst_ids_answer` can be remapped / not
                # the true target id, so only use it as a fallback.
                target_gt_id = getattr(li, "target_gt_id", None)
                try:
                    target_gt_id = int(target_gt_id) if target_gt_id is not None else None
                except Exception:
                    target_gt_id = None
                if target_gt_id is None:
                    target_gt_id = _first_inst_id(li)
            if not isinstance(scene_id, str) or not scene_id:
                continue
            if not isinstance(target_gt_id, int):
                continue

            feat = self._load_mask3d_feat(scene_id)
            if not isinstance(feat, dict):
                continue
            gt_map = feat.get("gt_to_query_map", None) or {}
            inst_classes = feat.get("gt_instance_classes", None) or {}
            pred_classes = feat.get("pred_classes", None)
            pred_box_info = feat.get("pred_box_info", None)
            pred_class_names = feat.get("pred_class_names", None)

            qidx_t = _get_qidx(gt_map, int(target_gt_id))
            if qidx_t is None or not (0 <= int(qidx_t) < int(Q)):
                continue

            # Candidate pool = all GT instance ids we can enumerate.
            all_ids: List[int] = []
            if isinstance(inst_classes, dict) and inst_classes:
                try:
                    all_ids = [int(x) for x in inst_classes.keys()]
                except Exception:
                    all_ids = []
            if not all_ids and isinstance(gt_map, dict) and gt_map:
                try:
                    all_ids = [int(x) for x in gt_map.keys()]
                except Exception:
                    all_ids = []
            if not all_ids:
                continue
            # Keep only instances that can be mapped to a valid Mask3D query slot.
            # This mirrors Vigor's context construction: candidates must have features.
            try:
                valid_ids: List[int] = []
                for _iid in all_ids:
                    try:
                        iid = int(_iid)
                    except Exception:
                        continue
                    qid = _get_qidx(gt_map, iid)
                    if qid is None:
                        continue
                    if 0 <= int(qid) < int(Q):
                        valid_ids.append(iid)
                all_ids = valid_ids
            except Exception:
                pass
            if int(target_gt_id) not in set(all_ids):
                all_ids.append(int(target_gt_id))
            if not all_ids:
                continue

            pred_classes_t = None
            try:
                if pred_classes is not None:
                    pred_classes_t = torch.as_tensor(pred_classes).detach().cpu()
            except Exception:
                pred_classes_t = None

            # IMPORTANT (no silent fallback):
            # If you explicitly request pred-class based sampling, pred_classes must be present
            # and not degenerate (all -1). Otherwise, raise early to expose export issues.
            if context_use_pred_class:
                if pred_classes_t is None or (not isinstance(pred_classes_t, torch.Tensor)) or pred_classes_t.numel() != int(Q):
                    raise RuntimeError(
                        f"[SSR3DLLM][vigor_train] pred_classes missing or wrong shape for scene={scene_id}: "
                        f"expected [{Q}], got {None if pred_classes_t is None else tuple(pred_classes_t.shape)}. "
                        "This usually means the exported Mask3D feature file does not contain valid class predictions."
                    )
                if not bool((pred_classes_t >= 0).any().item()):
                    # Expose the problem early instead of silently degrading sampling.
                    raise RuntimeError(
                        f"[SSR3DLLM][vigor_train] pred_classes are all -1 for scene={scene_id} (Q={Q}). "
                        "This indicates class predictions were not exported correctly."
                    )

            def _inst_cls_for_sampling(inst_id: int) -> int:
                qid = _get_qidx(gt_map, inst_id)
                if context_use_pred_class and pred_classes_t is not None and qid is not None and 0 <= int(qid) < int(pred_classes_t.numel()):
                    return int(pred_classes_t[int(qid)].item())
                # If context_use_pred_class is off, fall back to GT semantic ids.
                try:
                    v = inst_classes.get(int(inst_id), -1)
                    return int(v) if v is not None else -1
                except Exception:
                    return -1

            target_cls = _inst_cls_for_sampling(int(target_gt_id))
            same_cls = [i for i in all_ids if int(i) != int(target_gt_id) and _inst_cls_for_sampling(int(i)) == int(target_cls)]
            clutter = [i for i in all_ids if _inst_cls_for_sampling(int(i)) != int(target_cls)]
            try:
                np.random.shuffle(clutter)
            except Exception:
                pass
            distractors = list(same_cls) + list(clutter)
            distractors = distractors[: int(max_distractors)]
            try:
                np.random.shuffle(distractors)
            except Exception:
                pass
            try:
                tp = int(np.random.randint(len(distractors) + 1))
            except Exception:
                tp = 0
            context_ids = list(distractors)
            context_ids.insert(tp, int(target_gt_id))
            context_ids = context_ids[: int(max_context_size)]
            if len(context_ids) < int(max_context_size):
                context_ids = context_ids + ([-1] * (int(max_context_size) - len(context_ids)))

            # Slot features / boxes from Mask3D mapping.
            slots = torch.zeros((int(max_context_size), int(self.query_dim)), device=device, dtype=torch.float32)
            boxes = torch.zeros((int(max_context_size), 4), device=device, dtype=torch.float32)
            names: List[str] = ["unknown"] * int(max_context_size)
            # Optional ScanNet200 labels for Vigor's auxiliary object/text losses.
            sc_labels = torch.full((int(max_context_size),), -1, device=device, dtype=torch.long)
            sc_map = getattr(getattr(runtime, "model", None), "scannet_id_to_contig", None)
            sc_num = getattr(getattr(runtime, "model", None), "scannet_num_classes", None)
            enable_obj_loss = False
            if use_basic_loss:
                enable_obj_loss = self._env_flag("SSR3DLLM_VIGOR_TRAIN_ENABLE_OBJ_LOSS", "1")
                if enable_obj_loss and (not isinstance(sc_map, dict) or not sc_map or sc_num is None):
                    raise RuntimeError(
                        "[SSR3DLLM][vigor_train] requested Vigor BASIC_LOSS but ScanNet200 label mapping "
                        "is unavailable (scannet_id_to_contig/scannet_num_classes missing)."
                    )
                if enable_text_loss and (not bool(getattr(runtime.model, "use_scannet200_text_cls", False))):
                    raise RuntimeError(
                        "[SSR3DLLM][vigor_train] text-clf loss is enabled but Vigor text head is not in "
                        "ScanNet200 mode. Set `VIGOR_TEXT_CLS_SCANNET200=1` (and rebuild the runtime) "
                        "so `use_scannet200_text_cls=1`."
                    )
            # Strict alignment: Vigor predbox training expects a per-query [cx,cy,cz,volume].
            # Do NOT silently fall back to zeros; that hides export/config bugs.
            if pred_box_info is None:
                raise RuntimeError(
                    f"[SSR3DLLM][vigor_train] pred_box_info missing for scene={scene_id}. "
                    "This run requires Mask3D predbox exports (pred_box_info: [Q,4])."
                )
            try:
                pbi = torch.as_tensor(pred_box_info, dtype=torch.float32).detach().cpu()
            except Exception as e:
                raise RuntimeError(
                    f"[SSR3DLLM][vigor_train] failed to parse pred_box_info for scene={scene_id}: {e}"
                ) from e
            if (not torch.is_tensor(pbi)) or pbi.dim() != 2 or int(pbi.size(1)) != 4 or int(pbi.size(0)) != int(Q):
                raise RuntimeError(
                    f"[SSR3DLLM][vigor_train] pred_box_info has wrong shape for scene={scene_id}: "
                    f"expected [{Q},4], got {tuple(pbi.shape) if torch.is_tensor(pbi) else type(pbi)}"
                )
            pc_names = pred_class_names if isinstance(pred_class_names, list) else None
            ctx_qidx = torch.full((int(max_context_size),), -1, device=device, dtype=torch.long)
            for j, inst_id in enumerate(context_ids):
                if int(inst_id) < 0:
                    continue
                qid = _get_qidx(gt_map, int(inst_id))
                if qid is None or not (0 <= int(qid) < int(Q)):
                    continue
                ctx_qidx[int(j)] = int(qid)
                slots[j] = q_hidden[int(qid)].to(device=device, dtype=torch.float32)
                boxes[j] = pbi[int(qid)].to(device=device, dtype=torch.float32)
                if pc_names is not None and 0 <= int(qid) < len(pc_names):
                    names[j] = _norm_name(pc_names[int(qid)])
                if use_basic_loss and isinstance(inst_classes, dict) and inst_classes and isinstance(sc_map, dict) and sc_num is not None:
                    # Match Vigor _encode_with_mask3d: map per-instance GT semantic id -> ScanNet200 contiguous id.
                    lookup_id = int(inst_id)
                    cls_id = inst_classes.get(lookup_id, None)
                    if cls_id is None and (lookup_id + 1) in inst_classes and 0 not in inst_classes and 1 in inst_classes:
                        cls_id = inst_classes.get(lookup_id + 1, None)
                    try:
                        cls_id_int = int(cls_id) if cls_id is not None else None
                    except Exception:
                        cls_id_int = None
                    if cls_id_int is not None:
                        mapped = sc_map.get(cls_id_int, -1)
                        try:
                            mapped_int = int(mapped)
                        except Exception:
                            mapped_int = -1
                        if 0 <= mapped_int < int(sc_num):
                            sc_labels[j] = int(mapped_int)

            if use_basic_loss and enable_obj_loss and (not bool((sc_labels >= 0).any().item())):
                raise RuntimeError(
                    f"[SSR3DLLM][vigor_train] no valid ScanNet200 labels found for scene={scene_id} "
                    f"(max_context_size={int(max_context_size)}). "
                    "This likely indicates `gt_instance_classes` are not in ScanNet200 id space or the "
                    "label_database.yaml mapping is not aligned."
                )

            # Store for batch forward.
            if use_llm_lang:
                m = getattr(li, "llm_lang_embeds", None)
                if m is None or (not torch.is_tensor(m)) or m.dim() != 2 or int(m.size(-1)) != int(inner_dim):
                    llm_lang_strict = self._env_flag("SSR3DLLM_VIGOR_LLM_LANG_STRICT", "1")
                    llm_lang_skip = self._env_flag("SSR3DLLM_VIGOR_LLM_LANG_SKIP_MISSING", "0")
                    if llm_lang_skip:
                        # Drop this sample as bad data (avoid fabricating lang embeds).
                        total = int(getattr(self, "_vigor_llm_lang_missing_total", 0)) + 1
                        setattr(self, "_vigor_llm_lang_missing_total", total)
                        printed = int(getattr(self, "_vigor_llm_lang_missing_printed", 0))
                        if printed < 5:
                            try:
                                q_prev = str(question).replace("\n", " ").strip()
                                if len(q_prev) > 180:
                                    q_prev = q_prev[:180] + "..."
                            except Exception:
                                q_prev = "<unavailable>"
                            try:
                                print(
                                    "[SSR3DLLM][vigor_train] missing/invalid llm_lang_embeds; skip sample "
                                    f"(strict={int(llm_lang_strict)}, skip_missing=1) "
                                    f"scene={scene_id} prefix={prefix} question='{q_prev}'",
                                    flush=True,
                                )
                            except Exception:
                                pass
                            setattr(self, "_vigor_llm_lang_missing_printed", printed + 1)
                        continue
                    if llm_lang_strict:
                        raise RuntimeError(
                            "[SSR3DLLM][vigor_train] SSR3DLLM_VIGOR_USE_LLM_LANG_EMBEDS=1 but missing/invalid "
                            f"llm_lang_embeds (expected [M,{inner_dim}] tensor) for scene={scene_id}."
                        )
                    continue
                llm_lang_embeds_list.append(m)

            texts.append(self._strip_geom_token(question))
            slot_feats.append(slots)
            slot_boxes.append(boxes)
            slot_scannet_labels.append(sc_labels)
            ctx_qidx_list.append(ctx_qidx)
            if step_emb.dim() == 2 and int(step_emb.size(-1)) == int(inner_dim):
                order_embeds.append(step_emb.to(device=device, dtype=torch.float32).unsqueeze(1))
            elif step_emb.dim() == 3:
                # Already [O,L,D] -> normalise to [O,1,D] for stable stacking.
                oe = step_emb.to(device=device, dtype=torch.float32)
                if int(oe.size(1)) != 1:
                    oe = oe.mean(dim=1, keepdim=True)
                order_embeds.append(oe)
            else:
                proj = self._get_vigor_step_proj(inner_dim=inner_dim, device=device)
                order_embeds.append(proj(step_emb.to(device=device, dtype=torch.float32)).unsqueeze(1))
            target_pos.append(int(tp))
            slot_pred_names.append(names)

            # For optional pred_class_mask: use Vigor-style referential_order strings if present.
            # - rel3dref: comes from rel3d JSON (obj.rel_referential_order)
            # - scanrefer/m3dref: optionally synthesized (SSR3DLLM_GROUNDING_BUILD_STEP_ORDER=1)
            order_raw = getattr(li, "rel_referential_order", None)
            if isinstance(order_raw, list):
                o = [_norm_name(x) for x in order_raw if str(x).strip()]
                o = [x for x in o if x]
            else:
                o = []
            slot_orders.append(o)

            # Oracle chain length for VarLen-STOP masking:
            # - ScanRefer/M3DRef: always 1 step (single-step grounding)
            # - Rel3DRef/Vigor: prefer provided `ori_order_len` (or length of referential_order list)
            # - Fallback: treat all steps as valid (L=order_len)
            try:
                eff_len = getattr(li, "ori_order_len", None)
                eff_len_int = int(eff_len) if eff_len is not None else None
            except Exception:
                eff_len_int = None
            if prefix in {"scanrefer", "m3dref"}:
                eff_len_int = 1
            if eff_len_int is None:
                eff_len_int = int(len(o)) if o else int(order_len)
            eff_len_int = max(1, min(int(order_len), int(eff_len_int)))
            ovm = torch.zeros((int(order_len),), device=device, dtype=torch.float32)
            ovm[: int(eff_len_int)] = 1.0
            order_valid_masks.append(ovm)

            # Offline teacher distillation: build a stable key (if teacher DB is configured).
            teacher_key = getattr(li, "teacher_key", None)
            if not isinstance(teacher_key, str) or not teacher_key:
                rel_src = getattr(li, "relation_source", None)
                rel_src = rel_src.strip().lower() if isinstance(rel_src, str) else ""
                if teacher_db is not None and rel_src == "vigor":
                    sid = getattr(li, "rel_scene_id", None)
                    tid = getattr(li, "rel_target_object_gt_id", None)
                    dtext = getattr(li, "rel_distill_text", None)
                    if dtext is None:
                        dtext = self._strip_geom_token(question)
                    if isinstance(sid, str) and isinstance(tid, int):
                        try:
                            teacher_key = make_teacher_key(
                                teacher_name="vigor",
                                scene_id=sid,
                                target_gt_id=int(tid),
                                text=str(dtext),
                            )
                        except Exception:
                            teacher_key = None
                    else:
                        teacher_key = None
                else:
                    teacher_key = None
            teacher_keys.append(teacher_key if isinstance(teacher_key, str) and teacher_key else None)

        if not texts:
            return {}

        B = len(texts)
        N = int(max_context_size)
        lang_tokens = None
        lang_embeds = None
        if use_llm_lang:
            if len(llm_lang_embeds_list) != int(B):
                raise RuntimeError(
                    f"[SSR3DLLM][vigor_train] lang_embeds count mismatch: got {len(llm_lang_embeds_list)} expected {int(B)}. "
                    "This usually indicates the batch ordering changed without updating the gather logic."
                )
            lang_embeds = torch.stack([x.to(device=device, dtype=torch.float32) for x in llm_lang_embeds_list], dim=0)
        else:
            # HF tokenizers return a `BatchEncoding` (Mapping) which is not always an actual `dict`.
            # `VigorRuntime.forward_train_with_order_embeds()` expects a real dict for strict checks.
            enc = runtime.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            try:
                lang_tokens = dict(enc)
            except Exception:
                lang_tokens = {k: v for k, v in enc.items()}
        q = torch.stack(slot_feats, dim=0)   # [B,N,query_dim]
        boxes = torch.stack(slot_boxes, dim=0)  # [B,N,4]
        orders = torch.stack(order_embeds, dim=0)  # [B,O,1,D]
        scannet_labels = torch.stack(slot_scannet_labels, dim=0) if slot_scannet_labels else None
        if enable_text_loss and scannet_labels is not None:
            # Ensure at least one sample has a valid target label; otherwise text loss is all ignored.
            tp = torch.as_tensor(target_pos, device=scannet_labels.device, dtype=torch.long)
            idx = torch.arange(int(B), device=scannet_labels.device)
            tgt = scannet_labels[idx, tp].detach()
            if not bool((tgt >= 0).any().item()):
                raise RuntimeError(
                    "[SSR3DLLM][vigor_train] text-clf loss enabled but all target labels are -1 "
                    f"(B={B}, max_context_size={int(max_context_size)}). Check gt_instance_classes mapping."
                )
            # Optional periodic debug: verify that text-clf has a meaningful supervision signal.
            # Enable with: export SSR3DLLM_VIGOR_DEBUG_TEXTLABELS=1
            # Optional: export SSR3DLLM_VIGOR_DEBUG_TEXTLABELS_EVERY=50
            if self._env_flag("SSR3DLLM_VIGOR_DEBUG_TEXTLABELS", "0"):
                try:
                    every = int(str(os.environ.get("SSR3DLLM_VIGOR_DEBUG_TEXTLABELS_EVERY", "50")).strip() or "50")
                except Exception:
                    every = 50
                if every <= 0:
                    every = 1
                n_seen = int(getattr(self, "_vigor_debug_textlabels_seen", 0))
                setattr(self, "_vigor_debug_textlabels_seen", n_seen + 1)
                if (n_seen % every) == 0:
                    try:
                        valid = (tgt >= 0)
                        n_valid = int(valid.sum().item())
                        uniq = torch.unique(tgt[valid]).detach().cpu() if n_valid > 0 else torch.empty((0,), dtype=torch.long)
                        print(
                            "[SSR3DLLM][vigor_textclf_dbg] "
                            f"batch={n_seen} B={int(B)} valid_targets={n_valid}/{int(B)} "
                            f"valid_frac={float(n_valid)/float(max(int(B),1)):.4f} "
                            f"uniq_labels={int(uniq.numel())} "
                            f"label_min={int(uniq.min().item()) if uniq.numel() else -1} "
                            f"label_max={int(uniq.max().item()) if uniq.numel() else -1}",
                            flush=True,
                        )
                        # Print a couple of examples to spot systematic label/text mismatch.
                        for i in range(min(2, int(B))):
                            ti = int(tgt[i].item())
                            txt = texts[i]
                            if len(txt) > 140:
                                txt = txt[:140] + "..."
                            print(
                                f"[SSR3DLLM][vigor_textclf_dbg] ex{i} tgt_label={ti} text='{txt}'",
                                flush=True,
                            )
                    except Exception:
                        pass

        # One-time debug print to validate alignment without waiting for epoch-end eval.
        # Enable with: export SSR3DLLM_VIGOR_DEBUG_CHAIN_BATCH=1
        if self._env_flag("SSR3DLLM_VIGOR_DEBUG_CHAIN_BATCH", "0") and (not hasattr(self, "_vigor_debug_chain_printed")):
            self._vigor_debug_chain_printed = True
            try:
                b0 = 0
                box0 = boxes[b0].detach().float().cpu()
                q0 = q[b0].detach().float().cpu()
                print(
                    "[SSR3DLLM][vigor_train_dbg] "
                    f"B={B} N={N} order_len={int(order_len)} "
                    f"slot_feat_norm(mean,max)=({float(q0.norm(dim=-1).mean()):.4f},{float(q0.norm(dim=-1).max()):.4f}) "
                    f"box(min,max,mean)=({float(box0.min()):.4f},{float(box0.max()):.4f},{float(box0.mean()):.4f}) "
                    f"target_pos0={int(target_pos[b0])} predmask_mode={predmask_mode} use_predmask_train={int(use_predmask_train)}",
                    flush=True,
                )
            except Exception:
                pass

        # pred_class_mask: by default all ones (matches all_ones ablation).
        pred_class_mask = torch.ones((B, int(order_len), int(N)), device=device, dtype=torch.float32)
        if use_predmask_train and predmask_mode != "all_ones":
            for bi in range(B):
                order = list(slot_orders[bi]) if bi < len(slot_orders) else []
                if not order:
                    continue
                while len(order) > int(order_len):
                    order.pop(0)
                if len(order) == 1:
                    order = order * int(order_len)
                elif len(order) < int(order_len):
                    order = order + [order[-1]] * (int(order_len) - len(order))
                names = slot_pred_names[bi]
                for si in range(int(order_len)):
                    if cascading:
                        allowed = {_norm_name(x) for x in order[si:] if x}
                    else:
                        allowed = {_norm_name(order[si])} if order[si] else set()
                    for j in range(int(N)):
                        pred_class_mask[bi, si, j] = 1.0 if (_norm_name(names[j]) in allowed) else 0.0

        tgt = torch.as_tensor(target_pos, device=device, dtype=torch.long)
        order_valid_mask = None
        if order_valid_masks and len(order_valid_masks) == int(B):
            order_valid_mask = torch.stack(order_valid_masks, dim=0)  # [B,O]

        if use_basic_loss:
            out = runtime.forward_train_with_order_embeds(
                lang_tokens=lang_tokens,
                lang_embeds=lang_embeds,
                order_embeds=orders,
                order_valid_mask=order_valid_mask,
                mask3d_queries=q,
                box_info=boxes,
                pred_class_mask=pred_class_mask,
                target_pos=tgt,
                scannet_class_labels=scannet_labels,
            )
            loss_total = out["loss_total"]
            # Optional: offline distillation against a strong teacher distribution in query space.
            # NOTE: this is only meaningful when teacher logits are provided via:
            #   SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH=.../*.pt
            # and enabled by:
            #   SSR3DLLM_DISTILL_VIGOR_WEIGHT>0
            loss_distill = None
            if (
                teacher_db is not None
                and w_distill_vigor > 0.0
                and isinstance(out.get("logits", None), torch.Tensor)
                and len(ctx_qidx_list) == int(B)
                and len(teacher_keys) == int(B)
            ):
                try:
                    student_logits = out["logits"]  # [B,N]
                    distill_losses: List[torch.Tensor] = []
                    hit = 0
                    total = 0
                    for bi in range(int(B)):
                        tkey = teacher_keys[int(bi)]
                        if not isinstance(tkey, str) or not tkey:
                            continue
                        t_qlogits = teacher_db.get_final(tkey)
                        total += 1
                        if not isinstance(t_qlogits, torch.Tensor):
                            continue
                        tq = t_qlogits.to(device=device, dtype=student_logits.dtype)
                        qidx = ctx_qidx_list[int(bi)].to(device=device)
                        t_slot = torch.full((int(N),), float("-inf"), device=device, dtype=student_logits.dtype)
                        valid = (qidx >= 0) & (qidx < int(tq.numel()))
                        if bool(valid.any().item()):
                            t_slot[valid] = tq[qidx[valid]]
                        distill_losses.append(
                            self._kl_distill_1d(
                                student_logits[int(bi)].view(-1),
                                t_slot.view(-1),
                                temperature=float(distill_temperature),
                            )
                        )
                        hit += 1
                    if distill_losses:
                        loss_distill = torch.stack(distill_losses, dim=0).mean()
                        if self._env_flag("SSR3DLLM_VIGOR_DEBUG_DISTILL", "0"):
                            printed = int(getattr(self, "_vigor_distill_dbg_printed", 0))
                            if printed < 5:
                                try:
                                    print(
                                        f"[SSR3DLLM][vigor_distill] matched {hit}/{max(total,1)} "
                                        f"(w={w_distill_vigor:.3f} T={distill_temperature:.2f})",
                                        flush=True,
                                    )
                                except Exception:
                                    pass
                                setattr(self, "_vigor_distill_dbg_printed", printed + 1)
                except Exception:
                    loss_distill = None
            if self._env_flag("SSR3DLLM_VIGOR_DEBUG_BASIC_LOSS", "0") and (not hasattr(self, "_vigor_debug_basic_loss_printed")):
                self._vigor_debug_basic_loss_printed = True
                try:
                    valid = None
                    if scannet_labels is not None:
                        valid = float((scannet_labels >= 0).float().mean().item())
                    print(
                        "[SSR3DLLM][vigor_basic_loss_dbg] "
                        f"loss_total={float(loss_total.detach().cpu().item()):.4f} "
                        f"ref_ce={float(out.get('loss_ref_ce', torch.tensor(0.0)).detach().cpu().item()):.4f} "
                        f"obj_ce={float(out.get('loss_obj_ce', torch.tensor(0.0)).detach().cpu().item()):.4f} "
                        f"lang_ce={float(out.get('loss_lang_ce', torch.tensor(0.0)).detach().cpu().item()):.4f} "
                        + (f"scannet_valid_frac={valid:.4f}" if valid is not None else "scannet_valid_frac=NA"),
                        flush=True,
                    )
                except Exception:
                    pass
            # NOTE: Return per-component losses for logging only. These must NOT be added into
            # total_loss again (trainer filters them out) to avoid double-counting.
            w = float(w_chain)
            out_losses = {"loss_ssr3d_chain": loss_total * w}
            if "loss_ref_ce" in out:
                out_losses["loss_ssr3d_chain_ref_ce"] = out["loss_ref_ce"].detach() * w
            if "loss_obj_ce" in out:
                out_losses["loss_ssr3d_chain_obj_ce"] = out["loss_obj_ce"].detach() * w
            if "loss_lang_ce" in out:
                out_losses["loss_ssr3d_chain_lang_ce"] = out["loss_lang_ce"].detach() * w
            if loss_distill is not None:
                out_losses["loss_ssr3d_distill_vigor"] = loss_distill * float(w_distill_vigor)
            # Optional: "anchor" regularization in weight space to prevent catastrophic drift
            # when fine-tuning a tiny subset of Vigor listener parameters (stage3).
            #
            # This is intentionally lightweight (L2-SP) and only applies to params that are
            # currently trainable (requires_grad=True), so it remains cheap under partial finetune.
            #
            # Env:
            #   SSR3DLLM_VIGOR_ANCHOR_WEIGHT: float, default 0 (disabled)
            try:
                w_anchor = float(str(os.environ.get("SSR3DLLM_VIGOR_ANCHOR_WEIGHT", "0")).strip() or "0")
            except Exception:
                w_anchor = 0.0
            if w_anchor > 0.0:
                try:
                    terms: List[torch.Tensor] = []

                    # (A) Anchor trainable Vigor-runtime params (original behavior).
                    trainable_runtime: List[Tuple[str, torch.nn.Parameter]] = [
                        (n, p) for n, p in runtime.named_parameters() if getattr(p, "requires_grad", False)
                    ]
                    if trainable_runtime:
                        init_runtime: Optional[Dict[str, torch.Tensor]] = getattr(self, "_vigor_anchor_init", None)
                        init_runtime_id: Optional[int] = getattr(self, "_vigor_anchor_init_id", None)
                        if init_runtime is None or init_runtime_id != id(runtime):
                            init_runtime = {n: p.detach().to(dtype=torch.float32).clone() for n, p in trainable_runtime}
                            setattr(self, "_vigor_anchor_init", init_runtime)
                            setattr(self, "_vigor_anchor_init_id", id(runtime))
                        for n, p in trainable_runtime:
                            p0 = init_runtime.get(n) if isinstance(init_runtime, dict) else None
                            if not isinstance(p0, torch.Tensor) or tuple(p0.shape) != tuple(p.shape):
                                continue
                            terms.append((p.to(dtype=torch.float32) - p0.to(device=p.device)).pow(2).mean())

                    # (B) Anchor trainable step-projection params (stage3 "step_proj only" mode).
                    if getattr(self, "_vigor_step_proj", None) is not None:
                        trainable_step_proj: List[Tuple[str, torch.nn.Parameter]] = [
                            (n, p) for n, p in self._vigor_step_proj.named_parameters() if getattr(p, "requires_grad", False)
                        ]
                        if trainable_step_proj:
                            init_step: Optional[Dict[str, torch.Tensor]] = getattr(self, "_vigor_step_proj_anchor_init", None)
                            init_step_id: Optional[int] = getattr(self, "_vigor_step_proj_anchor_init_id", None)
                            if init_step is None or init_step_id != id(self._vigor_step_proj):
                                init_step = {
                                    n: p.detach().to(dtype=torch.float32).clone()
                                    for n, p in trainable_step_proj
                                }
                                setattr(self, "_vigor_step_proj_anchor_init", init_step)
                                setattr(self, "_vigor_step_proj_anchor_init_id", id(self._vigor_step_proj))
                            for n, p in trainable_step_proj:
                                p0 = init_step.get(n) if isinstance(init_step, dict) else None
                                if not isinstance(p0, torch.Tensor) or tuple(p0.shape) != tuple(p.shape):
                                    continue
                                terms.append((p.to(dtype=torch.float32) - p0.to(device=p.device)).pow(2).mean())

                    if terms:
                        out_losses["loss_ssr3d_vigor_anchor"] = torch.stack(terms, dim=0).mean() * w_anchor
                except Exception:
                    pass
            return out_losses

        logits = runtime.forward_logits_with_order_embeds(
            lang_tokens=lang_tokens,
            lang_embeds=lang_embeds,
            order_embeds=orders,
            order_valid_mask=order_valid_mask,
            mask3d_queries=q,
            box_info=boxes,
            pred_class_mask=pred_class_mask,
        )  # [B,N]
        loss = F.cross_entropy(logits, tgt)
        return {"loss_ssr3d_chain": loss * float(w_chain)}

    @staticmethod
    def _kl_distill_1d(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        KL(teacher || student) over a 1D categorical distribution.
        Both inputs are unnormalized logits of shape [K].
        """
        T = float(max(temperature, 1e-6))
        # Use only finite teacher positions (teacher is defined on a subset of queries).
        valid = torch.isfinite(teacher_logits)
        if valid.sum() <= 1:
            return torch.zeros((), device=student_logits.device)
        t = teacher_logits[valid] / T
        s = student_logits[valid] / T
        t_probs = F.softmax(t, dim=-1)
        s_log_probs = F.log_softmax(s, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return kl * (T * T)

    def forward(self, queries: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Tensor of shape [Q, D]  — instance query features for one scene.
            coords:  Tensor of shape [Q, 3] or [1, Q, 3] — instance centres.

        Returns:
            geom_delta: Tensor of shape [Q, D], to be *added* to `queries`.
        """
        if queries.dim() != 2:
            raise ValueError(f"queries must be [Q,D], got {tuple(queries.shape)}")

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # [1,Q,3]
        if coords.dim() != 3 or coords.size(-1) != 3:
            raise ValueError(f"coords must be [Q,3] or [1,Q,3], got {tuple(coords.shape)}")

        # Ensure batch dimension is 1 to match queries.
        if coords.size(0) != 1:
            raise ValueError(
                f"SSR3DLLMGeomHeadForLLM expects coords for a single scene (B=1), "
                f"got batch size {coords.size(0)}"
            )

        device = queries.device
        coords = coords.to(device=device, dtype=torch.float32)

        # Run relation field over instance centers.
        field, _ = self.relation_field(coords)  # [1,Q,d_model]
        field = field.squeeze(0)                # [Q,d_model]

        if field.size(0) != queries.size(0):
            raise ValueError(
                f"coords and queries disagree on Q: field={field.size(0)}, queries={queries.size(0)}"
            )

        geom_delta = self.proj(field)  # [Q,D]
        return geom_delta

    # ------------------------------------------------------------------
    #   Auxiliary supervision for rel3dref:* language samples
    # ------------------------------------------------------------------

    def _encode_text(
        self, texts: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode a list of texts with BERT and project CLS to student_dim.

        Returns:
            text_feats: [B,student_dim]
            token_feats_list: list of [Li,student_dim] token features (unpadded)
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.text_encoder(**enc)
        last = out.last_hidden_state  # [B, L, H]
        cls = last[:, 0, :]  # [B, H]
        cls_feat = self.text_proj(cls)  # [B, student_dim]

        # Token-level features for decoder cross-attention.
        tok = self.text_proj(last)  # [B, L, student_dim]
        attn = enc.get("attention_mask", None)
        token_feats_list: List[torch.Tensor] = []
        if isinstance(attn, torch.Tensor) and attn.dim() == 2 and attn.size(0) == tok.size(0):
            for i in range(tok.size(0)):
                li = int(attn[i].sum().item())
                li = max(1, min(li, int(tok.size(1))))
                token_feats_list.append(tok[i, :li, :])
        else:
            for i in range(tok.size(0)):
                token_feats_list.append(tok[i])

        return cls_feat, token_feats_list

    def _get_rel_type_id(self, lang_type: str) -> int:
        """
        Map a lang_type like 'rel3dref:between:with_grounding' to
        a compact integer id in [0, max_rel_types).
        """
        tokens = lang_type.split(":")
        rel = tokens[1] if len(tokens) >= 2 else lang_type
        if rel not in self.rel_type_to_id:
            if len(self.rel_type_to_id) >= self.max_rel_types:
                # Clamp new types to the last id to avoid out-of-range.
                return self.max_rel_types - 1
            self.rel_type_to_id[rel] = len(self.rel_type_to_id)
        return self.rel_type_to_id[rel]

    def compute_rel3dref_losses_for_batch(
        self,
        batch_lang_infos: List[object],
        sampled_coords: torch.Tensor,
        device: torch.device,
        w_ref: float = 1.0,
        w_anchor: float = 1.0,
        w_relcls: float = 1.0,
        w_chain: float = 1.0,
        w_distill_vigor: float = 0.0,
        distill_temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for all rel3dref:* samples in a batch.

        Args:
            batch_lang_infos: list of `lang_info_data` objects from `prepare_llm`.
            sampled_coords: coordinates for all scenes, shape [B, Q, 3].
            device: torch device for computation.

        Returns:
            Dict with averaged losses (keys: loss_ssr3d_ref, loss_ssr3d_anchor,
            loss_ssr3d_relcls), or an empty dict when no valid sample exists.
        """
        backend = self._get_geom_backend()
        if backend == "vigor":
            return self._compute_vigor_chain_loss_for_batch(
                batch_lang_infos=batch_lang_infos,
                sampled_coords=sampled_coords,
                device=device,
                w_chain=w_chain,
            )
        if not batch_lang_infos:
            return {}
        if not isinstance(sampled_coords, torch.Tensor):
            raise ValueError("sampled_coords must be a torch.Tensor.")

        ref_losses: List[torch.Tensor] = []
        anchor_losses: List[torch.Tensor] = []
        relcls_losses: List[torch.Tensor] = []
        chain_losses: List[torch.Tensor] = []
        distill_vigor_losses: List[torch.Tensor] = []

        texts: List[str] = []
        meta: List[Tuple[int, List[int], List[int], object]] = []
        rel_type_ids: List[int] = []

        allowed_sources_raw = os.environ.get("SSR3DLLM_GEOM_REL_SOURCES", "").strip()
        allowed_sources = set()
        if allowed_sources_raw:
            allowed_sources = {
                s.strip().lower() for s in allowed_sources_raw.split(",") if s.strip()
            }
        require_anchor = os.environ.get("SSR3DLLM_GEOM_REQUIRE_ANCHOR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        for lang_info in batch_lang_infos:
            lang_type = getattr(lang_info, "lang_type", "")
            if not isinstance(lang_type, str) or not lang_type.startswith("rel3dref"):
                continue

            if allowed_sources:
                src = getattr(lang_info, "relation_source", None)
                if not isinstance(src, str) or src.lower() not in allowed_sources:
                    continue

            question_text = getattr(lang_info, "question", "") or ""
            use_geom_trigger = getattr(lang_info, "use_geom_trigger", False)
            if ("<geom>" not in question_text) and (not use_geom_trigger):
                continue

            q_hidden = getattr(lang_info, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                continue

            Q, _ = q_hidden.shape
            batch_idx = getattr(lang_info, "batch_idx", None)
            if batch_idx is None or batch_idx >= sampled_coords.size(0):
                continue

            # Anchor / target query indices.
            anchor_ids: List[int] = []
            q_ids_q = getattr(lang_info, "query_ids_question", None)
            if q_ids_q:
                for ids in q_ids_q:
                    anchor_ids.extend(ids)
            anchor_ids = [i for i in anchor_ids if 0 <= i < Q]

            target_ids: List[int] = []
            q_ids_a = getattr(lang_info, "query_ids_answer", None)
            if q_ids_a:
                for ids in q_ids_a:
                    target_ids.extend(ids)
            target_ids = [i for i in target_ids if 0 <= i < Q]

            if not target_ids:
                continue
            if require_anchor and (not anchor_ids):
                continue

            text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
            if not isinstance(text, str) or not text:
                continue

            texts.append(text)
            meta.append((batch_idx, anchor_ids, target_ids, lang_info))
            rel_type_ids.append(self._get_rel_type_id(lang_type))

        if not texts:
            return {}

        # Encode all texts in one go.
        text_feats, bert_token_feats = self._encode_text(texts, device=device)  # [M,student_dim]
        rel_type_ids_tensor = torch.tensor(rel_type_ids, device=device, dtype=torch.long)

        used_llm_init = 0
        fallback_init = 0
        # Teacher distillation stats
        distill_vigor_hit = 0
        distill_vigor_total = 0

        for idx, (batch_idx, anchor_ids, target_ids, lang_info) in enumerate(meta):
            text_feat = text_feats[idx]  # [student_dim]
            bert_tokens_i = bert_token_feats[idx]

            coords_bid = sampled_coords[batch_idx].to(device=device, dtype=torch.float32)  # [Q,3]
            q_hidden = getattr(lang_info, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                continue
            q_hidden = q_hidden.to(device=device)

            Q, D = q_hidden.shape
            if coords_bid.dim() != 2 or coords_bid.size(0) != Q:
                continue
            if D != int(getattr(self, "query_dim", D)):
                continue

            # Relation field over centres.
            field_s, _ = self.relation_field(coords_bid.unsqueeze(0))  # [1,Q,student_dim]
            field_s = field_s.squeeze(0)  # [Q,student_dim]

            # Fuse Mask3D query semantics + geometry + text.
            #
            # NOTE: `lang_info.query_hidden_feature` is the same tensor that will be fed
            # into the LLM. In our Step-3 pipeline, it already includes a *detached*
            # geometry delta (added in trainer) so the forward values contain geometry,
            # but the relation_field params would not get gradients unless we connect
            # them here. To avoid double-counting geometry while still training the
            # relation_field, we add a "zero-forward / non-zero-backward" term:
            #   (field - field.detach()).
            inject_to_llm = self._env_flag("SSR3DLLM_GEOM_INJECT_TO_LLM", "0")
            q_hidden_q = q_hidden.to(device=device, dtype=torch.float32).detach()  # [Q,D] (STAMP: no grad to queries)
            q_hidden_s = self.query_up(q_hidden_q)  # [Q,student_dim]
            text_feat_s = text_feat.to(dtype=q_hidden_s.dtype)  # [student_dim]
            if inject_to_llm:
                # LLM already consumes query-space delta (proj(field_s)). Keep forward values
                # unchanged here (avoid double-counting), but backprop into relation_field.
                obj_tokens = q_hidden_s + (field_s.to(dtype=q_hidden_s.dtype) - field_s.detach().to(dtype=q_hidden_s.dtype))
            else:
                # Geometry-only path: use full student-space relation field directly.
                obj_tokens = q_hidden_s + field_s.to(dtype=q_hidden_s.dtype)
            fused = obj_tokens + text_feat_s.unsqueeze(0)  # [Q,student_dim]

            # ---- referential CE: pick first target as pointer label ----
            target_pos = int(target_ids[0])
            if not (0 <= target_pos < Q):
                continue
            ref_logits = self.ref_head(fused).squeeze(-1)             # [Q]
            ref_loss = F.cross_entropy(
                ref_logits.view(1, Q),
                torch.tensor([target_pos], device=device),
            )

            # ---- anchor multi-label BCE over queries ----
            if anchor_ids:
                anchor_logits = self.anchor_head(fused).squeeze(-1)   # [Q]
                anchor_gt = torch.zeros(Q, device=device)
                anchor_gt[anchor_ids] = 1.0
                anchor_loss = F.binary_cross_entropy_with_logits(anchor_logits, anchor_gt)
            else:
                anchor_loss = torch.zeros((), device=device)

            # ---- relation-type classification from text CLS ----
            rel_logits = self.rel_cls_head(text_feat.unsqueeze(0))    # [1,R]
            rel_id = rel_type_ids_tensor[idx : idx + 1]               # [1]
            relcls_loss = F.cross_entropy(rel_logits, rel_id)

            ref_losses.append(ref_loss)
            anchor_losses.append(anchor_loss)
            relcls_losses.append(relcls_loss)

            # Choose initial state for referential order decoder:
            # - If lang_info.llm_text_init is available (set by the LLM core),
            #   we use it as a detached context vector so that geometric losses
            #   do not backpropagate into the LLM.
            # - Otherwise, fall back to the BERT-based text_feat.
            llm_text_init = getattr(lang_info, "llm_text_init", None)
            if isinstance(llm_text_init, torch.Tensor):
                init_vec = llm_text_init.to(device=device)
                if init_vec.dim() == 2:
                    init_vec = init_vec[0]
                if init_vec.shape[0] == D:
                    text_init_vec = self.query_up(init_vec.detach().to(dtype=q_hidden_q.dtype)).to(
                        dtype=q_hidden_s.dtype
                    )
                    used_llm_init += 1
                else:
                    text_init_vec = text_feat
                    fallback_init += 1
            else:
                text_init_vec = text_feat
                fallback_init += 1

            # Token-level text tokens for decoder cross-attention.
            # Prefer LLM-provided projected tokens (mask_dim space), then fall back to BERT tokens.
            text_tokens = None
            llm_text_tokens = getattr(lang_info, "llm_text_tokens", None)
            if isinstance(llm_text_tokens, torch.Tensor) and llm_text_tokens.dim() == 2:
                if int(llm_text_tokens.size(-1)) == int(getattr(self, "query_dim", D)):
                    tt = llm_text_tokens.to(device=device, dtype=q_hidden_q.dtype).detach()
                    text_tokens = self.query_up(tt).to(dtype=obj_tokens.dtype).unsqueeze(0)  # [1,L,student_dim]
            if text_tokens is None and isinstance(bert_tokens_i, torch.Tensor) and bert_tokens_i.dim() == 2:
                text_tokens = bert_tokens_i.to(device=device, dtype=obj_tokens.dtype).unsqueeze(0)  # [1,L,student_dim]

            # -------------------------
            # Offline teacher distillation (Vigor)
            # -------------------------
            relation_source = getattr(lang_info, "relation_source", None)
            rel_src = relation_source.strip().lower() if isinstance(relation_source, str) else ""

            distill_teachers: List[str] = []
            if rel_src == "vigor" and w_distill_vigor > 0.0:
                distill_teachers.append("vigor")

            for teacher_name in distill_teachers:
                is_vigor = teacher_name == "vigor"
                if teacher_name == "vigor":
                    distill_vigor_total += 1

                teacher_key = getattr(lang_info, "teacher_key", None)
                if (not isinstance(teacher_key, str)) or (not teacher_key) or (rel_src != teacher_name):
                    # Build key on the fly from stored rel_* metadata.
                    sid = getattr(lang_info, "rel_scene_id", None)
                    tid = getattr(lang_info, "rel_target_object_gt_id", None)
                    dtext = getattr(lang_info, "rel_distill_text", None)
                    if isinstance(sid, str) and isinstance(tid, int) and dtext is not None:
                        try:
                            teacher_key = make_teacher_key(
                                teacher_name=teacher_name,
                                scene_id=sid,
                                target_gt_id=int(tid),
                                text=str(dtext),
                            )
                        except Exception:
                            teacher_key = None
                    else:
                        teacher_key = None

                db = self._get_teacher_db(teacher_name)
                if db is not None and isinstance(teacher_key, str) and teacher_key:
                    t_qlogits = db.get_final(teacher_key)
                    t_step_qlogits = db.get_steps(teacher_key)
                    t_ori_len = db.get_ori_len(teacher_key)
                else:
                    t_qlogits = None
                    t_step_qlogits = None
                    t_ori_len = None

                # Always initialize these locals for safety; some teachers/splits may miss entries.
                student_ptr = None
                order_labels_for_distill = None
                anchor_seq: List[int] = []

                if isinstance(t_qlogits, torch.Tensor):
                    # Build a teacher-forced chain for student decoding.
                    # Use *effective-length* supervision: chain_len + 1 (STOP).
                    #   - with anchor: [anchor, target, STOP]
                    #   - no anchor:   [target, STOP]
                    stop_idx = Q
                    if anchor_ids and target_ids:
                        max_steps = int(getattr(getattr(self.decoder, "cfg", None), "max_steps", 4))
                        max_anchor_steps = max(0, max_steps - 2)
                        anchor_seq: List[int] = []
                        seen = set()
                        for a in anchor_ids:
                            try:
                                aa = int(a)
                            except Exception:
                                continue
                            if 0 <= aa < Q and aa not in seen:
                                anchor_seq.append(aa)
                                seen.add(aa)
                            if len(anchor_seq) >= max_anchor_steps:
                                break
                        target_idx = int(target_ids[0])
                        if any((i < 0 or i >= Q) for i in anchor_seq) or not (0 <= target_idx < Q):
                            student_ptr = None
                            order_labels_for_distill = None
                        else:
                            order_labels_for_distill = torch.tensor(
                                [[*anchor_seq, target_idx, stop_idx]],
                                device=device,
                                dtype=torch.long,
                            )  # [1,T]
                            student_ptr = self.decoder(
                                obj_tokens=obj_tokens.unsqueeze(0),
                                text_tokens=text_tokens,
                                order_labels=order_labels_for_distill,
                                text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0),
                                obj_padding_mask=None,
                            )  # [1,T,Q+1]
                    else:
                        # No anchor: single-step target then STOP.
                        if not target_ids:
                            student_ptr = None
                            order_labels_for_distill = None
                        else:
                            target_idx = int(target_ids[0])
                            if not (0 <= target_idx < Q):
                                student_ptr = None
                                order_labels_for_distill = None
                            else:
                                order_labels_for_distill = torch.tensor(
                                    [[target_idx, stop_idx]],
                                    device=device,
                                    dtype=torch.long,
                                )  # [1,2]
                                student_ptr = self.decoder(
                                    obj_tokens=obj_tokens.unsqueeze(0),
                                    text_tokens=text_tokens,
                                    order_labels=order_labels_for_distill,
                                    text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0),
                                    obj_padding_mask=None,
                                )  # [1,T,Q+1]

                if isinstance(student_ptr, torch.Tensor):
                        # Per-step distillation (preferred when teacher provides step logits).
                        per_step_loss = None
                        if isinstance(t_step_qlogits, torch.Tensor) and t_step_qlogits.dim() == 2:
                            # Distill only the "object steps" (exclude STOP/pad).
                            #
                            # IMPORTANT: Vigor uses a fixed order_len=4 decoder with padding-by-duplication
                            # (common case ori_len=2 => referential_order padded like [A, A, T, T]).
                            # Our student uses a 2-step chain [A, T] (plus STOP), so we must collapse
                            # Vigor's per-step logits into (anchor_step, target_step) before distillation.
                            teacher_steps_for_student: List[torch.Tensor] = []
                            if is_vigor:
                                if anchor_seq:
                                    # Vigor per-step logits are exported for a fixed small number of steps
                                    # (often 4 with duplication). If our sample has multiple anchors
                                    # (e.g. "between A and B"), step alignment is ambiguous; fall back
                                    # to final-step distillation only.
                                    if len(anchor_seq) != 1:
                                        teacher_steps_for_student = []
                                    # Prefer the canonical Vigor padding scheme if we have 4 steps.
                                    # Prefer the canonical Vigor padding scheme if we have 4 steps.
                                    elif int(t_step_qlogits.size(0)) >= 4:
                                        teacher_steps_for_student = [
                                            t_step_qlogits[0:2, :Q].mean(dim=0),
                                            t_step_qlogits[2:4, :Q].mean(dim=0),
                                        ]
                                    elif int(t_step_qlogits.size(0)) >= 2:
                                        teacher_steps_for_student = [
                                            t_step_qlogits[0, :Q],
                                            t_step_qlogits[1, :Q],
                                        ]
                                else:
                                    # Target-only: collapse all provided steps.
                                    teacher_steps_for_student = [t_step_qlogits[:, :Q].mean(dim=0)]
                            else:
                                # Non-Vigor: treat ori_len as the number of valid object steps.
                                if isinstance(t_ori_len, int) and t_ori_len > 0:
                                    valid_steps = int(t_ori_len)
                                else:
                                    # Our student uses "effective-length" chain:
                                    #   - with K anchors: K anchor steps + 1 target step
                                    #   - no anchor: 1 target step
                                    valid_steps = (len(anchor_seq) + 1) if anchor_seq else 1
                                valid_steps = min(valid_steps, int(t_step_qlogits.size(0)), 4)
                                if valid_steps > 0:
                                    teacher_steps_for_student = [t_step_qlogits[t, :Q] for t in range(valid_steps)]

                            if teacher_steps_for_student:
                                step_losses = []
                                for t, tt0 in enumerate(teacher_steps_for_student):
                                    if t >= int(student_ptr.size(1)):
                                        break
                                    st = student_ptr[0, t, :Q]
                                    tt = tt0.to(device=device, dtype=st.dtype)
                                    step_losses.append(
                                        self._kl_distill_1d(
                                            student_logits=st,
                                            teacher_logits=tt,
                                            temperature=float(distill_temperature),
                                        )
                                    )
                                per_step_loss = torch.stack(step_losses).mean()

                        # Fallback: final-step distillation on the target step.
                        final_loss = None
                        if isinstance(t_qlogits, torch.Tensor):
                            # Target is predicted at step = (#anchor_steps) for effective-length chains.
                            if isinstance(order_labels_for_distill, torch.Tensor) and int(order_labels_for_distill.size(1)) >= 2:
                                target_step = int(order_labels_for_distill.size(1)) - 2
                            else:
                                target_step = 0
                            if target_step >= int(student_ptr.size(1)):
                                target_step = int(student_ptr.size(1)) - 1
                            st = student_ptr[0, target_step, :Q]
                            tt = t_qlogits[:Q].to(device=device, dtype=st.dtype)
                            final_loss = self._kl_distill_1d(
                                student_logits=st,
                                teacher_logits=tt,
                                temperature=float(distill_temperature),
                            )

                        distill_loss = per_step_loss if per_step_loss is not None else final_loss
                        if distill_loss is not None:
                            if teacher_name == "vigor":
                                distill_vigor_losses.append(distill_loss)
                                distill_vigor_hit += 1

            # ---- referential order decoder ----
            # Effective-length chain supervision (chain_len + 1 STOP):
            #   - with anchor: [anchor, target, STOP]
            #   - no anchor:   [target, STOP]
            if anchor_ids and target_ids:
                target_idx = target_ids[0]
                # Build a compact anchor sequence (preserve order, dedupe, truncate).
                max_steps = int(getattr(getattr(self.decoder, "cfg", None), "max_steps", 4))
                max_anchor_steps = max(0, max_steps - 2)
                anchor_seq: List[int] = []
                seen = set()
                for a in anchor_ids:
                    try:
                        aa = int(a)
                    except Exception:
                        continue
                    if 0 <= aa < Q and aa not in seen:
                        anchor_seq.append(aa)
                        seen.add(aa)
                    if len(anchor_seq) >= max_anchor_steps:
                        break

                if anchor_seq and 0 <= target_idx < Q:
                    order_labels = torch.tensor(
                        [[*anchor_seq, int(target_idx), Q]],
                        dtype=torch.long,
                        device=device,
                    )
                    pointer_logits = self.decoder(
                        obj_tokens=obj_tokens.unsqueeze(0),   # [1,Q,D]
                        text_tokens=text_tokens,
                        order_labels=order_labels,
                        text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0),
                        obj_padding_mask=None,
                    )
                    chain_loss = compute_order_loss(pointer_logits, order_labels)
                else:
                    chain_loss = torch.zeros((), device=device)
            else:
                # Target-only chain when anchor is absent.
                if target_ids:
                    target_idx = target_ids[0]
                    if 0 <= target_idx < Q:
                        order_labels = torch.tensor(
                            [[int(target_idx), Q]],
                            dtype=torch.long,
                            device=device,
                        )
                        pointer_logits = self.decoder(
                            obj_tokens=obj_tokens.unsqueeze(0),   # [1,Q,D]
                            text_tokens=text_tokens,
                            order_labels=order_labels,
                            text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0),
                            obj_padding_mask=None,
                        )
                        chain_loss = compute_order_loss(pointer_logits, order_labels)
                    else:
                        chain_loss = torch.zeros((), device=device)
                else:
                    chain_loss = torch.zeros((), device=device)
            chain_losses.append(chain_loss)

        if not ref_losses:
            return {}

        def _mean(xs: List[torch.Tensor]) -> torch.Tensor:
            if not xs:
                return torch.zeros((), device=device)
            return sum(xs) / float(len(xs))

        out: Dict[str, torch.Tensor] = {}
        out["loss_ssr3d_ref"] = w_ref * _mean(ref_losses)
        out["loss_ssr3d_anchor"] = w_anchor * _mean(anchor_losses)
        out["loss_ssr3d_relcls"] = w_relcls * _mean(relcls_losses)
        out["loss_ssr3d_chain"] = w_chain * _mean(chain_losses)
        if w_distill_vigor > 0.0:
            out["loss_ssr3d_distill_vigor"] = float(w_distill_vigor) * _mean(distill_vigor_losses)

        try:
            total_init = used_llm_init + fallback_init
            if total_init > 0:
                print(
                    f"[SSR3DLLMGeomHead] llm_text_init_used={used_llm_init} "
                    f"fallback_bert_init={fallback_init}"
                )
        except Exception:
            pass

        # Distillation debug: show how many rel3dref samples matched teacher logits.
        try:
            if (w_distill_vigor > 0.0) and (distill_vigor_total > 0):
                print(
                    f"[SSR3DLLMGeomHead][distill] vigor matched {distill_vigor_hit}/{distill_vigor_total} "
                    f"(T={float(distill_temperature):.2f})"
                )
        except Exception:
            pass

        return out

    # ------------------------------------------------------------------
    # Evaluation helper for rel3dref:* language samples
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_rel3dref_for_batch(
        self,
        batch_lang_infos: List[object],
        sampled_coords: torch.Tensor,
        device: torch.device,
        *,
        box_info_by_bid: Optional[torch.Tensor] = None,
        pred_class_names_by_bid: Optional[List[List[str]]] = None,
        valid_queries_by_bid: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate referential target / chain accuracy for rel3dref:* samples in a batch.

        This mirrors the geometry path used in compute_rel3dref_losses_for_batch
        but only produces metrics:
          - num_rel:         number of valid rel3dref samples
          - num_target_hit:  count of samples where predicted target index matches GT
          - num_chain_hit:   count of samples where [anchor, target, STOP] chain is
                             exactly correct
        """
        if not batch_lang_infos or not isinstance(sampled_coords, torch.Tensor):
            return {"num_rel": 0.0, "num_target_hit": 0.0, "num_chain_hit": 0.0}

        geom_backend = self._get_geom_backend()
        use_vigor = geom_backend == "vigor"
        vigor = self._get_vigor_runtime(device) if use_vigor else None

        num_rel = 0
        num_target_hit = 0
        num_chain_hit = 0
        num_no_anchor = 0
        num_has_anchor = 0
        num_used_llm_init = 0
        num_fallback_bert_init = 0
        debug_eval = os.environ.get("SSR3DLLM_DEBUG_REL3D_EVAL", "0").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }
        debug_max_raw = os.environ.get("SSR3DLLM_DEBUG_REL3D_EVAL_MAX", "").strip()
        try:
            debug_max = int(debug_max_raw) if debug_max_raw else 5
        except Exception:
            debug_max = 5
        debug_lines: List[str] = []
        debug_teacher = os.environ.get("SSR3DLLM_DEBUG_REL3D_EVAL_TEACHER", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        debug_topk_raw = os.environ.get("SSR3DLLM_DEBUG_REL3D_EVAL_TOPK", "").strip()
        try:
            debug_topk = int(debug_topk_raw) if debug_topk_raw else 5
        except Exception:
            debug_topk = 5

        # Optional: limit evaluation to certain relation sources (e.g. vigor).
        allowed_sources_raw = os.environ.get("SSR3DLLM_GEOM_REL_SOURCES", "").strip()
        allowed_sources = set()
        if allowed_sources_raw:
            allowed_sources = {
                s.strip().lower() for s in allowed_sources_raw.split(",") if s.strip()
            }

        # Whether to require explicit anchor supervision in evaluation.
        # Keep consistent with training-time filtering to avoid mixing "no_anchor"
        # samples when we care about chain reasoning.
        require_anchor = os.environ.get("SSR3DLLM_GEOM_REQUIRE_ANCHOR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        def _norm_name(name: str) -> str:
            return str(name).strip().lower().replace("_", " ")

        def _pad_order(order: List[str], order_len: int) -> List[str]:
            # Mirror Vigor ListeningDataset padding for order_len in {1..6} (we mainly use 4).
            # Also strip '*' markers used by some LLM prompts (matches Vigor ListeningDataset).
            order = [str(x).strip().strip("*").strip() for x in order]
            order = [x for x in order if x]
            if not order:
                return []
            # Truncate from the front if too long (Vigor deletes from head).
            while len(order) > int(order_len):
                del order[0]
            if int(order_len) == 5:
                if len(order) == 1:
                    order *= int(order_len)
                elif len(order) == 2:
                    order = [order[0]] * 2 + [order[1]] * 3
                elif len(order) == 3:
                    order = [order[0]] * 1 + [order[1]] * 1 + [order[2]] * 3
                elif len(order) == 4:
                    order.append(order[-1])
            if int(order_len) == 6:
                if len(order) == 1:
                    order *= int(order_len)
                elif len(order) == 2:
                    order = [order[0]] * 3 + [order[1]] * 3
                elif len(order) == 3:
                    order = [order[0]] * 2 + [order[1]] * 2 + [order[2]] * 2
                elif len(order) == 4:
                    order = [order[0]] * 1 + [order[1]] * 1 + [order[2]] * 1 + [order[3]] * 3
                elif len(order) == 5:
                    order.append(order[-1])
            if int(order_len) == 4:
                if len(order) == 1:
                    order *= int(order_len)
                elif len(order) == 2:
                    order = [order[0]] * 2 + [order[1]] * 2
                elif len(order) == 3:
                    order.append(order[-1])
            elif int(order_len) == 3:
                if len(order) == 1:
                    order *= int(order_len)
                elif len(order) == 2:
                    order = [order[0]] * 1 + [order[1]] * 2
            elif int(order_len) == 2:
                if len(order) == 1:
                    order *= int(order_len)
            elif int(order_len) == 1:
                pass
            return order

        def _build_cascaded_order(order: List[str]) -> List[str]:
            out = []
            for i in range(len(order)):
                sub = list(dict.fromkeys(order[i:]))  # preserve order, unique
                out.append(", ".join([str(x) for x in sub if str(x)]))
            return out

        def _flatten_ints(x: Any) -> List[int]:
            if x is None:
                return []
            if isinstance(x, (int, np.integer)):
                return [int(x)]
            if torch.is_tensor(x):
                if x.numel() == 1:
                    try:
                        return [int(x.item())]
                    except Exception:
                        return []
                return []
            if isinstance(x, (list, tuple)):
                out: List[int] = []
                for it in x:
                    out.extend(_flatten_ints(it))
                return out
            return []

        # ------------------------------------------------------------------
        # Backend: pretrained mask3d-vigor listener
        # ------------------------------------------------------------------
        if vigor is not None:
            def _is_rank0() -> bool:
                for k in ("RANK", "LOCAL_RANK"):
                    v = os.environ.get(k, None)
                    if v is None:
                        continue
                    try:
                        return int(str(v)) == 0
                    except Exception:
                        continue
                return True

            is_rank0 = _is_rank0()
            cascading = os.environ.get("SSR3DLLM_VIGOR_CASCADING", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            order_len = int(getattr(vigor, "order_len", 4))
            use_mask3d_feats = os.environ.get("SSR3DLLM_VIGOR_USE_MASK3D_FEATS", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            use_pred_box_info = os.environ.get("SSR3DLLM_VIGOR_USE_PRED_BOX_INFO", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            skipped: Dict[str, int] = {}

            def _as_int(x: Any) -> Optional[int]:
                if x is None:
                    return None
                if isinstance(x, (int, np.integer)):
                    return int(x)
                if torch.is_tensor(x) and x.numel() == 1:
                    try:
                        return int(x.item())
                    except Exception:
                        return None
                try:
                    return int(str(x))
                except Exception:
                    return None

            for lang_info in batch_lang_infos:
                lang_type = getattr(lang_info, "lang_type", "")
                if not isinstance(lang_type, str) or not lang_type.startswith("rel3dref"):
                    continue

                src = getattr(lang_info, "relation_source", None)
                if allowed_sources and isinstance(src, str) and str(src).strip().lower() not in allowed_sources:
                    continue

                batch_idx = getattr(lang_info, "batch_idx", None)
                if batch_idx is None or not (0 <= int(batch_idx) < int(sampled_coords.size(0))):
                    continue

                # Text: use distill_text (teacher-key aligned) when available.
                text = getattr(lang_info, "rel_distill_text", None) or getattr(lang_info, "question", None) or ""
                text = str(text).replace("<geom>", "").strip()

                # Prefer feature-file driven evaluation (exactly mirrors mask3d-vigor):
                # use {scene}.pt object_queries + gt_to_query_map to define BOTH
                # (a) candidate set (real objects) and (b) GT query id.
                scene_id = getattr(lang_info, "rel_scene_id", None)
                scene_id = str(scene_id).strip() if scene_id is not None else ""
                tgt_gt_id = _as_int(getattr(lang_info, "rel_target_object_gt_id", None))

                cand_qidxs: Optional[List[int]] = None

                feat = self._load_mask3d_feat(scene_id) if (use_mask3d_feats and scene_id) else None
                gt_map = self._get_gt_to_query_map_from_feat(scene_id) if (use_mask3d_feats and scene_id) else None
                oq = None
                if isinstance(feat, dict):
                    oq = feat.get("object_queries", None)
                    if isinstance(oq, np.ndarray):
                        oq = torch.from_numpy(oq)
                if use_mask3d_feats and torch.is_tensor(oq) and oq.dim() == 2 and isinstance(gt_map, dict) and gt_map:
                    Q_full, D_full = oq.shape
                    # Candidate set: only mapped queries (real objects).
                    cand_qidxs = sorted({int(v) for v in gt_map.values() if isinstance(v, int) and 0 <= int(v) < int(Q_full)})
                    if not cand_qidxs:
                        skipped["empty_candidates"] = int(skipped.get("empty_candidates", 0)) + 1
                        continue
                    # GT target query: from gt_to_query_map.
                    gt_q_feat = gt_map.get(int(tgt_gt_id), None) if tgt_gt_id is not None else None
                    if gt_q_feat is None or not (0 <= int(gt_q_feat) < int(Q_full)):
                        skipped["no_gt_query"] = int(skipped.get("no_gt_query", 0)) + 1
                        continue
                    if int(gt_q_feat) not in set(cand_qidxs):
                        skipped["gt_not_in_candidates"] = int(skipped.get("gt_not_in_candidates", 0)) + 1
                        continue
                    # Remap into candidate index space.
                    gt_target_set = {int(cand_qidxs.index(int(gt_q_feat)))}
                    # Use Mask3D feature queries as Vigor input.
                    q_hidden = oq[cand_qidxs].to(device=device, dtype=torch.float32)
                    Q = int(q_hidden.size(0))
                    # Build box_info in full Q space then subset.
                    box_info_full = None
                    if isinstance(gt_map, dict) and scene_id:
                        box_info_full = self._get_box_info_from_scannet(scene_id, gt_map, int(Q_full))
                    # Optional: predicted AABB-derived box_info from Mask3D full-res masks (no GT bboxes).
                    if box_info_full is None and use_pred_box_info and isinstance(feat, dict):
                        pred_box = feat.get("pred_box_info", None)
                        pred_aabb = feat.get("pred_aabb", None)
                        if pred_box is not None:
                            try:
                                pb = torch.as_tensor(pred_box, dtype=torch.float32)
                                if pb.ndim == 2 and int(pb.size(0)) == int(Q_full) and int(pb.size(1)) == 4:
                                    box_info_full = pb.detach().cpu()
                            except Exception:
                                box_info_full = None
                        if box_info_full is None and pred_aabb is not None:
                            try:
                                aabb = torch.as_tensor(pred_aabb, dtype=torch.float32)
                                if aabb.ndim == 2 and int(aabb.size(0)) == int(Q_full) and int(aabb.size(1)) == 6:
                                    mn = aabb[:, 0:3]
                                    mx = aabb[:, 3:6]
                                    center = (mn + mx) * 0.5
                                    size = (mx - mn).clamp(min=0.0)
                                    vol = size[:, 0] * size[:, 1] * size[:, 2]
                                    box_info_full = torch.zeros((int(Q_full), 4), dtype=torch.float32)
                                    box_info_full[:, 0:3] = center
                                    box_info_full[:, 3] = vol
                            except Exception:
                                box_info_full = None
                    if box_info_full is None and isinstance(feat, dict):
                        coords = feat.get("sampled_coords", None)
                        try:
                            coords_t = torch.as_tensor(coords, dtype=torch.float32) if coords is not None else None
                        except Exception:
                            coords_t = None
                        if torch.is_tensor(coords_t) and coords_t.dim() == 2 and coords_t.size(0) == int(Q_full) and coords_t.size(1) == 3:
                            box_info_full = torch.zeros((int(Q_full), 4), dtype=torch.float32)
                            box_info_full[:, :3] = coords_t
                            box_info_full[:, 3] = 1.0
                    if box_info_full is None:
                        # Last fallback: use sampled_coords from the current forward (online).
                        coords_bid = sampled_coords[int(batch_idx)].to(device="cpu", dtype=torch.float32)
                        box_info_full = torch.zeros((int(Q_full), 4), dtype=torch.float32)
                        if coords_bid.dim() == 2 and coords_bid.size(0) == int(Q_full) and coords_bid.size(1) == 3:
                            box_info_full[:, :3] = coords_bid
                        box_info_full[:, 3] = 1.0
                    box_info = box_info_full[cand_qidxs].to(device=device, dtype=torch.float32)
                else:
                    # Fallback to online query space (less aligned; keep for robustness).
                    q_hidden = getattr(lang_info, "query_hidden_feature", None)
                    if q_hidden is None or not isinstance(q_hidden, torch.Tensor) or q_hidden.dim() != 2:
                        skipped["no_online_queries"] = int(skipped.get("no_online_queries", 0)) + 1
                        continue
                    Q, D = q_hidden.shape
                    if D != int(getattr(self, "query_dim", D)):
                        skipped["bad_query_dim"] = int(skipped.get("bad_query_dim", 0)) + 1
                        continue
                    gt_target_ids = _flatten_ints(getattr(lang_info, "query_ids_answer", None))
                    gt_target_set = {int(t) for t in gt_target_ids if 0 <= int(t) < int(Q)}
                    if not gt_target_set:
                        skipped["no_gt_target_ids"] = int(skipped.get("no_gt_target_ids", 0)) + 1
                        continue
                    box_info = None
                    if (
                        isinstance(box_info_by_bid, torch.Tensor)
                        and box_info_by_bid.dim() == 3
                        and int(box_info_by_bid.size(0)) > int(batch_idx)
                        and int(box_info_by_bid.size(1)) == int(Q)
                        and int(box_info_by_bid.size(2)) == 4
                    ):
                        box_info = box_info_by_bid[int(batch_idx)].to(device=device, dtype=torch.float32)
                    else:
                        coords_bid = sampled_coords[int(batch_idx)].to(device=device, dtype=torch.float32)
                        box_info = torch.zeros((Q, 4), device=device, dtype=torch.float32)
                        box_info[:, :3] = coords_bid
                        box_info[:, 3] = 1.0

                # Optional: require anchor supervision (same behaviour as decoder path).
                if require_anchor:
                    anchor_ids = _flatten_ints(getattr(lang_info, "query_ids_question", None))
                    anchor_set = {int(a) for a in anchor_ids if 0 <= int(a) < int(Q)}
                    if not anchor_set:
                        continue

                num_rel += 1

                # Build referential-order tokens + pred_class_mask (Vigor step-pointer).
                order_raw = getattr(lang_info, "rel_referential_order", None)
                order_list = []
                if isinstance(order_raw, list):
                    order_list = [str(x).strip().strip("*").strip() for x in order_raw]
                    order_list = [x for x in order_list if x]
                order_padded = _pad_order(order_list, order_len)
                order_texts = _build_cascaded_order(order_padded) if (cascading and order_padded) else order_padded
                if not order_texts:
                    order_texts = [text] * int(order_len)

                # pred_class_mask based on predicted class names (best effort).
                # Prefer explicit per-query names from Mask3D feature files (Vigor-aligned),
                # because Mask3DLang's `pred_logits` is often token-conditioned and cannot
                # be interpreted as semantic classes.
                pred_names: Optional[List[str]] = None
                pred_names_src = "none"
                vq_feat: Optional[torch.Tensor] = None
                vq_src = "none"
                # If we are using feature-driven candidates, treat all Q as valid by construction.
                # Otherwise, build validity mask from gt_to_query_map when possible.
                if use_mask3d_feats and scene_id and isinstance(gt_map, dict) and gt_map and torch.is_tensor(oq):
                    vq_feat = torch.ones((int(Q),), dtype=torch.bool, device=device)
                    vq_src = "candidates"
                    pred_names = self._get_pred_class_names_from_feat(scene_id, int(Q_full)) if scene_id else None
                    if isinstance(pred_names, list) and len(pred_names) == int(Q_full) and isinstance(cand_qidxs, list):
                        pred_names = [pred_names[int(q)] if 0 <= int(q) < int(Q_full) else "unknown" for q in cand_qidxs]
                        pred_names_src = "mask3d_feats"
                if (
                    isinstance(pred_class_names_by_bid, list)
                    and 0 <= int(batch_idx) < len(pred_class_names_by_bid)
                    and isinstance(pred_class_names_by_bid[int(batch_idx)], list)
                    and len(pred_class_names_by_bid[int(batch_idx)]) == int(Q)
                ):
                    if pred_names is None:
                        pred_names = [str(x) if x is not None else "unknown" for x in pred_class_names_by_bid[int(batch_idx)]]
                        pred_names_src = "trainer"

                pred_mask = torch.ones((int(order_len), int(Q)), device=device, dtype=torch.float32)
                # If we know which queries correspond to any predicted points, mask out "empty"
                # queries to better match Vigor's padded-context behavior.
                try:
                    if (
                        isinstance(valid_queries_by_bid, torch.Tensor)
                        and valid_queries_by_bid.dim() == 2
                        and 0 <= int(batch_idx) < int(valid_queries_by_bid.size(0))
                        and int(valid_queries_by_bid.size(1)) == int(Q)
                    ):
                        vq = valid_queries_by_bid[int(batch_idx)].to(device=device)
                        if vq.dtype != torch.bool:
                            vq = vq > 0
                        pred_mask = pred_mask * vq.to(dtype=torch.float32).unsqueeze(0)
                except Exception:
                    pass
                # If we can infer "real object" queries from feature mapping, apply it as the strongest
                # validity mask (overrides the weaker "non-empty mask" heuristic).
                if torch.is_tensor(vq_feat) and vq_feat.numel() == int(Q):
                    pred_mask = pred_mask * vq_feat.to(dtype=torch.float32).unsqueeze(0)
                if pred_names is not None and order_padded:
                    mask_rows = []
                    for i in range(int(order_len)):
                        if i >= len(order_padded):
                            # padded order_texts already repeated; keep mask as all-ones
                            mask_rows.append(torch.ones((Q,), device=device, dtype=torch.float32))
                            continue
                        if cascading:
                            all_obj = {_norm_name(x) for x in order_padded[i:] if str(x)}
                        else:
                            all_obj = {_norm_name(order_padded[i])}
                        row = torch.tensor(
                            [1.0 if _norm_name(pred_names[j]) in all_obj else 0.0 for j in range(int(Q))],
                            device=device,
                            dtype=torch.float32,
                        )
                        # Also apply the per-query validity mask if available.
                        try:
                            if (
                                isinstance(valid_queries_by_bid, torch.Tensor)
                                and valid_queries_by_bid.dim() == 2
                                and 0 <= int(batch_idx) < int(valid_queries_by_bid.size(0))
                                and int(valid_queries_by_bid.size(1)) == int(Q)
                            ):
                                vq = valid_queries_by_bid[int(batch_idx)].to(device=device)
                                if vq.dtype != torch.bool:
                                    vq = vq > 0
                                row = row * vq.to(dtype=torch.float32)
                        except Exception:
                            pass
                        # Apply feature-derived validity as well (hard constraint).
                        if torch.is_tensor(vq_feat) and vq_feat.numel() == int(Q):
                            row = row * vq_feat.to(dtype=torch.float32)
                        mask_rows.append(row)
                    if mask_rows:
                        pred_mask = torch.stack(mask_rows, dim=0)

                try:
                    logits = vigor.predict_logits(
                        text=text,
                        mask3d_queries=q_hidden.detach(),
                        box_info=box_info,
                        order_texts=order_texts,
                        pred_class_mask=pred_mask,
                    )
                except Exception as exc:
                    if debug_eval and is_rank0 and len(debug_lines) < debug_max:
                        debug_lines.append(
                            "[SSR3DLLMGeomHead][eval_dbg][vigor] "
                            f"scene={getattr(lang_info, 'rel_scene_id', None)} "
                            f"src={getattr(lang_info, 'relation_source', None)} "
                            f"err={type(exc).__name__}: {str(exc)[:200]}"
                        )
                    continue

                # Hard-mask candidates: in online mode, exclude invalid queries;
                # in feature-driven mode, candidates are already restricted.
                if not (use_mask3d_feats and isinstance(feat, dict) and torch.is_tensor(oq) and isinstance(gt_map, dict) and gt_map):
                    try:
                        if torch.is_tensor(logits) and logits.numel() == int(Q):
                            cand = torch.ones((int(Q),), dtype=torch.bool)
                            if torch.is_tensor(vq_feat) and vq_feat.numel() == int(Q):
                                cand = cand & vq_feat.detach().to("cpu").to(dtype=torch.bool)
                            if (
                                isinstance(valid_queries_by_bid, torch.Tensor)
                                and valid_queries_by_bid.dim() == 2
                                and 0 <= int(batch_idx) < int(valid_queries_by_bid.size(0))
                                and int(valid_queries_by_bid.size(1)) == int(Q)
                            ):
                                vq2 = valid_queries_by_bid[int(batch_idx)].detach().to("cpu")
                                if vq2.dtype != torch.bool:
                                    vq2 = vq2 > 0
                                cand = cand & vq2.to(dtype=torch.bool)
                            if bool(cand.any()):
                                logits = logits.clone()
                                logits[~cand] = float("-inf")
                    except Exception:
                        pass

                pred = int(torch.argmax(logits).item()) if torch.is_tensor(logits) and logits.numel() > 0 else -1
                if pred in gt_target_set:
                    num_target_hit += 1
                    # Vigor does not expose an explicit STOP token; treat target-hit as chain-hit for logging.
                    num_chain_hit += 1
                if debug_eval and is_rank0 and len(debug_lines) < debug_max:
                    # Best-effort self-check: verify that pred/gt indices live in the same
                    # query space by printing their predicted class names (if available).
                    pred_name = None
                    gt_names: List[str] = []
                    try:
                        if pred_names is not None and len(pred_names) == int(Q):
                            if 0 <= int(pred) < int(Q):
                                pred_name = str(pred_names[int(pred)])
                            for g in sorted(list(gt_target_set))[:8]:
                                if 0 <= int(g) < int(Q):
                                    gt_names.append(str(pred_names[int(g)]))
                    except Exception:
                        pred_name = None
                        gt_names = []
                    # Extra sanity: print feature-derived GT query id if available.
                    gt_q_feat_dbg = None
                    try:
                        if use_mask3d_feats and isinstance(gt_map, dict) and tgt_gt_id is not None:
                            gt_q_feat_dbg = gt_map.get(int(tgt_gt_id), None)
                    except Exception:
                        gt_q_feat_dbg = None
                    try:
                        kk = int(min(max(debug_topk, 1), int(logits.numel())))
                        vals, idx = torch.topk(logits.view(-1), kk)
                        topk = ",".join(
                            [f"{int(idx[i].item())}:{float(vals[i].item()):.3f}" for i in range(int(kk))]
                        )
                    except Exception:
                        topk = "[]"
                    try:
                        hit = 1 if pred in gt_target_set else 0
                        pred_q_full = None
                        gt_q_full = None
                        try:
                            if isinstance(cand_qidxs, list) and 0 <= int(pred) < len(cand_qidxs):
                                pred_q_full = int(cand_qidxs[int(pred)])
                            if isinstance(cand_qidxs, list) and len(gt_target_set) == 1:
                                g0 = int(next(iter(gt_target_set)))
                                if 0 <= g0 < len(cand_qidxs):
                                    gt_q_full = int(cand_qidxs[g0])
                        except Exception:
                            pred_q_full = None
                            gt_q_full = None
                        debug_lines.append(
                            "[SSR3DLLMGeomHead][eval_dbg][vigor] "
                            f"scene={getattr(lang_info, 'rel_scene_id', None)} "
                            f"src={getattr(lang_info, 'relation_source', None)} "
                            f"Q={int(Q)} pred={pred} pred_name={repr(pred_name)} hit={hit} "
                            f"pred_q={repr(pred_q_full)} gt_q={repr(gt_q_full)} "
                            f"gt={sorted(list(gt_target_set))} gt_names={repr(gt_names)} "
                            f"gt_q_feat={repr(gt_q_feat_dbg)} "
                            f"names_src={pred_names_src} "
                            f"vq_src={vq_src} vq_n={(int(vq_feat.sum().item()) if torch.is_tensor(vq_feat) else 'na')} "
                            f"order={order_texts} topk=[{topk}] "
                            f"tkey={getattr(lang_info, 'teacher_key', None)} "
                            f"dtext={repr(str(text)[:120])}"
                        )
                    except Exception:
                        pass

            if debug_eval and is_rank0 and debug_lines:
                for ln in debug_lines[:debug_max]:
                    print(ln, flush=True)
                print(
                    "[SSR3DLLMGeomHead][eval_dbg][vigor_summary] "
                    f"num_rel={num_rel} target_hit={num_target_hit} chain_hit={num_chain_hit}",
                    flush=True,
                )
                if skipped:
                    print(f"[SSR3DLLMGeomHead][eval_dbg][vigor_skipped] {skipped}", flush=True)

            return {
                "num_rel": float(num_rel),
                "num_target_hit": float(num_target_hit),
                "num_chain_hit": float(num_chain_hit),
                "num_has_anchor": float(num_has_anchor),
                "num_no_anchor": float(num_no_anchor),
                "num_used_llm_init": float(num_used_llm_init),
                "num_fallback_bert_init": float(num_fallback_bert_init),
            }

        def _topk_str(logits_1d: torch.Tensor, k: int) -> str:
            if not isinstance(logits_1d, torch.Tensor) or logits_1d.dim() != 1 or logits_1d.numel() == 0:
                return "[]"
            kk = int(min(max(k, 1), int(logits_1d.numel())))
            vals, idx = torch.topk(logits_1d, kk)
            pairs = []
            for i in range(int(kk)):
                pairs.append(f"{int(idx[i].item())}:{float(vals[i].item()):.3f}")
            return "[" + ",".join(pairs) + "]"

        def _margin_str(logits_1d: torch.Tensor) -> str:
            if not isinstance(logits_1d, torch.Tensor) or logits_1d.dim() != 1 or logits_1d.numel() < 2:
                return "nan"
            vals, _ = torch.topk(logits_1d, 2)
            return f"{float(vals[0].item() - vals[1].item()):.3f}"

        for lang_info in batch_lang_infos:
            lang_type = getattr(lang_info, "lang_type", "")
            if not isinstance(lang_type, str) or not lang_type.startswith("rel3dref"):
                continue

            if allowed_sources:
                src = getattr(lang_info, "relation_source", None)
                if not isinstance(src, str) or src.lower() not in allowed_sources:
                    continue

            question_text = getattr(lang_info, "question", "") or ""
            use_geom_trigger = getattr(lang_info, "use_geom_trigger", False)
            if ("<geom>" not in question_text) and (not use_geom_trigger):
                continue

            q_hidden = getattr(lang_info, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                continue

            Q, D = q_hidden.shape
            if D != int(getattr(self, "query_dim", D)):
                continue
            batch_idx = getattr(lang_info, "batch_idx", None)
            if batch_idx is None or batch_idx >= sampled_coords.size(0):
                continue

            # Anchor / target query indices.
            anchor_ids: List[int] = []
            q_ids_q = getattr(lang_info, "query_ids_question", None)
            if q_ids_q:
                for ids in q_ids_q:
                    anchor_ids.extend(ids)
            anchor_ids = [i for i in anchor_ids if 0 <= i < Q]

            target_ids: List[int] = []
            q_ids_a = getattr(lang_info, "query_ids_answer", None)
            if q_ids_a:
                for ids in q_ids_a:
                    target_ids.extend(ids)
            target_ids = [i for i in target_ids if 0 <= i < Q]

            if not target_ids:
                continue
            if require_anchor and (not anchor_ids):
                continue
            if anchor_ids:
                num_has_anchor += 1
            else:
                num_no_anchor += 1

            coords_bid = sampled_coords[batch_idx].to(device=device, dtype=torch.float32)
            q_hidden = q_hidden.to(device=device)
            if coords_bid.dim() != 2 or coords_bid.size(0) != Q:
                continue

            # Relation field over centres.
            field_s, _ = self.relation_field(coords_bid.unsqueeze(0))  # [1,Q,student_dim]
            field_s = field_s.squeeze(0)  # [Q,student_dim]
            inject_to_llm = self._env_flag("SSR3DLLM_GEOM_INJECT_TO_LLM", "0")
            q_hidden_q = q_hidden.to(device=device, dtype=torch.float32).detach()  # [Q,D]
            q_hidden_s = self.query_up(q_hidden_q)  # [Q,student_dim]
            if inject_to_llm:
                obj_tokens = q_hidden_s + (field_s.to(dtype=q_hidden_s.dtype) - field_s.detach().to(dtype=q_hidden_s.dtype))
            else:
                obj_tokens = q_hidden_s + field_s.to(dtype=q_hidden_s.dtype)

            llm_text_init = getattr(lang_info, "llm_text_init", None)
            if isinstance(llm_text_init, torch.Tensor):
                init_vec = llm_text_init.to(device=device)
                if init_vec.dim() == 2:
                    init_vec = init_vec[0]
                if init_vec.shape[0] == D:
                    text_init_vec = self.query_up(init_vec.detach().to(dtype=q_hidden_q.dtype)).to(
                        dtype=obj_tokens.dtype
                    )
                    num_used_llm_init += 1
                else:
                    text_init_vec = None
            else:
                text_init_vec = None

            if text_init_vec is None:
                # BERT fallback (CLS) — same behavior as training-time fallback.
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        text_init_vec = self._encode_text([text], device=device)[0][0]
                        num_fallback_bert_init += 1
                    except Exception:
                        text_init_vec = None

            # Token-level text tokens for decoder cross-attention (eval/predict path).
            text_tokens = None
            llm_text_tokens = getattr(lang_info, "llm_text_tokens", None)
            if isinstance(llm_text_tokens, torch.Tensor) and llm_text_tokens.dim() == 2:
                if int(llm_text_tokens.size(-1)) == int(getattr(self, "query_dim", D)):
                    tt = llm_text_tokens.to(device=device, dtype=q_hidden_q.dtype).detach()
                    text_tokens = self.query_up(tt).to(dtype=obj_tokens.dtype).unsqueeze(0)  # [1,L,student_dim]
            if text_tokens is None:
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        _, tok_list = self._encode_text([text], device=device)
                        if tok_list and isinstance(tok_list[0], torch.Tensor):
                            text_tokens = tok_list[0].to(device=device, dtype=obj_tokens.dtype).unsqueeze(0)
                    except Exception:
                        text_tokens = None

            # Note: query_ids_* can contain multiple valid query indices mapping to the
            # same GT instance. For evaluation we treat prediction as a hit if it
            # matches ANY valid query id, rather than only the first one.
            gt_target_set = set(int(i) for i in target_ids)
            gt_anchor_set = set(int(i) for i in anchor_ids)

            stop_idx = Q  # STOP index is Q (N+1-th slot)
            src = getattr(lang_info, "relation_source", None)
            if anchor_ids:
                max_steps = int(getattr(getattr(self.decoder, "cfg", None), "max_steps", 4))
                max_anchor_steps = max(0, max_steps - 2)
                anchor_seq: List[int] = []
                seen = set()
                for a in anchor_ids:
                    try:
                        aa = int(a)
                    except Exception:
                        continue
                    if 0 <= aa < Q and aa not in seen:
                        anchor_seq.append(aa)
                        seen.add(aa)
                    if len(anchor_seq) >= max_anchor_steps:
                        break
                target_idx0 = int(target_ids[0])
                if any((i < 0 or i >= Q) for i in anchor_seq) or not (0 <= target_idx0 < Q):
                    continue
                num_rel += 1
                # Strict effective-length chain: [a1, a2, ..., target, STOP]
                order_labels = torch.tensor(
                    [[*anchor_seq, target_idx0, stop_idx]],
                    device=device,
                    dtype=torch.long,
                )  # [1,T]
                pointer_logits = self.decoder(
                    obj_tokens=obj_tokens.unsqueeze(0),
                    text_tokens=text_tokens,
                    order_labels=order_labels,
                    text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0) if text_init_vec is not None else None,
                    obj_padding_mask=None,
                )  # [1,T,Q+1]
                pred_chain = pointer_logits.argmax(dim=-1).view(-1).tolist()  # [T]
                pred_target_step = len(anchor_seq)
                if int(pred_chain[pred_target_step]) in gt_target_set:
                    num_target_hit += 1
                gt_chain = [*anchor_seq, target_idx0, stop_idx]
                if len(pred_chain) == len(gt_chain) and all(int(p) == int(g) for p, g in zip(pred_chain, gt_chain)):
                    num_chain_hit += 1

                if debug_eval and len(debug_lines) < debug_max:
                    sid = getattr(lang_info, "rel_scene_id", None) or getattr(lang_info, "scene_id", None)
                    tname = str(src or "")
                    teacher_top1 = None
                    teacher_status = None
                    teacher_key = getattr(lang_info, "teacher_key", None)
                    dtext_dbg = getattr(lang_info, "rel_distill_text", None)
                    if isinstance(dtext_dbg, str):
                        dtext_dbg = dtext_dbg[:80].replace("\n", " ")
                    else:
                        dtext_dbg = None

                    # Student-side quick sanity: show top-k logits for anchor step and target step.
                    # This helps diagnose "collapse to same pred" quickly.
                    try:
                        st_anchor_logits = pointer_logits[0, 0, : (Q + 1)].detach()
                        st_target_logits = pointer_logits[0, pred_target_step, : (Q + 1)].detach()
                        st_anchor_topk = _topk_str(st_anchor_logits, debug_topk)
                        st_target_topk = _topk_str(st_target_logits, debug_topk)
                        st_anchor_margin = _margin_str(st_anchor_logits)
                        st_target_margin = _margin_str(st_target_logits)
                    except Exception:
                        st_anchor_topk = "[]"
                        st_target_topk = "[]"
                        st_anchor_margin = "nan"
                        st_target_margin = "nan"

                    teacher_dbg = ""
                    if debug_teacher and isinstance(src, str):
                        if not isinstance(teacher_key, str) or not teacher_key:
                            sid2 = getattr(lang_info, "rel_scene_id", None)
                            tid2 = getattr(lang_info, "rel_target_object_gt_id", None)
                            dtext2 = getattr(lang_info, "rel_distill_text", None)
                            if isinstance(sid2, str) and isinstance(tid2, int) and dtext2 is not None:
                                try:
                                    teacher_key = make_teacher_key(
                                        teacher_name=str(src),
                                        scene_id=sid2,
                                        target_gt_id=int(tid2),
                                        text=str(dtext2),
                                    )
                                except Exception:
                                    teacher_key = None
                        if isinstance(teacher_key, str) and teacher_key:
                            db = self._get_teacher_db(src)
                            if db is not None:
                                t = db.get_final(teacher_key)
                                t_steps = db.get_steps(teacher_key)
                                if isinstance(t, torch.Tensor) and t.numel() > 0:
                                    teacher_top1 = int(torch.argmax(t[:Q]).item())
                                    teacher_status = "hit"
                                    try:
                                        t_final_logits = t[: (Q + 1)]
                                        teacher_final_topk = _topk_str(t_final_logits, debug_topk)
                                        teacher_final_margin = _margin_str(t_final_logits)
                                    except Exception:
                                        teacher_final_topk = "[]"
                                        teacher_final_margin = "nan"

                                    # If teacher provides per-step logits (e.g. Vigor export with steps),
                                    # also print step0/step_last top-k to see whether teacher itself collapses.
                                    if isinstance(t_steps, torch.Tensor) and t_steps.dim() == 2 and t_steps.numel() > 0:
                                        try:
                                            t0 = t_steps[0, : (Q + 1)]
                                            tlast = t_steps[-1, : (Q + 1)]
                                            teacher_step0_topk = _topk_str(t0, debug_topk)
                                            teacher_last_topk = _topk_str(tlast, debug_topk)
                                            teacher_step0_margin = _margin_str(t0)
                                            teacher_last_margin = _margin_str(tlast)
                                        except Exception:
                                            teacher_step0_topk = "[]"
                                            teacher_last_topk = "[]"
                                            teacher_step0_margin = "nan"
                                            teacher_last_margin = "nan"
                                    else:
                                        teacher_step0_topk = None
                                        teacher_last_topk = None
                                        teacher_step0_margin = None
                                        teacher_last_margin = None

                                    teacher_dbg = (
                                        f" teacher_top1={teacher_top1} teacher=hit "
                                        f"t_final_topk={teacher_final_topk} t_final_margin={teacher_final_margin}"
                                    )
                                    if teacher_step0_topk is not None:
                                        teacher_dbg += (
                                            f" t_step0_topk={teacher_step0_topk} t_step0_margin={teacher_step0_margin}"
                                        )
                                    if teacher_last_topk is not None:
                                        teacher_dbg += (
                                            f" t_last_topk={teacher_last_topk} t_last_margin={teacher_last_margin}"
                                        )
                                else:
                                    teacher_status = "miss"
                                    teacher_dbg = " teacher_top1=None teacher=miss"
                    debug_lines.append(
                        f"[SSR3DLLMGeomHead][eval_dbg] scene={sid} src={tname} "
                        f"anchor_qs={sorted(gt_anchor_set)[:5]} target_qs={sorted(gt_target_set)[:5]} "
                        f"gt_chain={gt_chain} "
                        f"pred_chain={pred_chain}"
                        f" st_anchor_topk={st_anchor_topk} st_anchor_margin={st_anchor_margin}"
                        f" st_target_topk={st_target_topk} st_target_margin={st_target_margin}"
                        + (f" tkey={str(teacher_key)[:8] if isinstance(teacher_key, str) else None} dtext='{dtext_dbg}'" if debug_teacher else "")
                        + (teacher_dbg if debug_teacher else "")
                    )
            else:
                # No anchor: strict effective-length chain [target, STOP]
                num_rel += 1
                target_idx0 = int(target_ids[0])
                if not (0 <= target_idx0 < Q):
                    continue
                order_labels = torch.tensor(
                    [[target_idx0, stop_idx]], device=device, dtype=torch.long
                )  # [1,2]
                pointer_logits = self.decoder(
                    obj_tokens=obj_tokens.unsqueeze(0),
                    text_tokens=text_tokens,
                    order_labels=order_labels,
                    text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0) if text_init_vec is not None else None,
                    obj_padding_mask=None,
                )  # [1,2,Q+1]
                pred = int(pointer_logits.argmax(dim=-1).view(-1)[0].item())
                if pred in gt_target_set:
                    num_target_hit += 1
                    num_chain_hit += 1  # for 1-step, "chain" == "target"

                if debug_eval and len(debug_lines) < debug_max:
                    sid = getattr(lang_info, "rel_scene_id", None) or getattr(lang_info, "scene_id", None)
                    src = getattr(lang_info, "relation_source", None)
                    try:
                        st_logits = pointer_logits[0, 0, : (Q + 1)].detach()
                        st_topk = _topk_str(st_logits, debug_topk)
                        st_margin = _margin_str(st_logits)
                    except Exception:
                        st_topk = "[]"
                        st_margin = "nan"
                    debug_lines.append(
                        f"[SSR3DLLMGeomHead][eval_dbg] scene={sid} src={src} "
                        f"(no_anchor) target_qs={sorted(gt_target_set)[:5]} "
                        f"gt_chain={[sorted(gt_target_set)[0] if gt_target_set else None, stop_idx]} pred={pred} "
                        f"st_topk={st_topk} st_margin={st_margin}"
                    )

        if debug_eval and debug_lines:
            rank = 0
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    rank = int(torch.distributed.get_rank())
            except Exception:
                rank = 0
            if rank == 0:
                for line in debug_lines:
                    print(line)
                print(
                    f"[SSR3DLLMGeomHead][eval_dbg] num_rel={num_rel} target_hit={num_target_hit} "
                    f"chain_hit={num_chain_hit} has_anchor={num_has_anchor} no_anchor={num_no_anchor} "
                    f"llm_init={num_used_llm_init} bert_fallback={num_fallback_bert_init}"
                )

        return {
            "num_rel": float(num_rel),
            "num_target_hit": float(num_target_hit),
            "num_chain_hit": float(num_chain_hit),
            "num_has_anchor": float(num_has_anchor),
            "num_no_anchor": float(num_no_anchor),
            "num_used_llm_init": float(num_used_llm_init),
            "num_fallback_bert_init": float(num_fallback_bert_init),
        }

    def predict_rel3dref_for_batch(
        self,
        batch_lang_infos: List[object],
        sampled_coords: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        """
        Return per-sample predictions for rel3dref:* samples, for downstream
        lightweight eval (e.g. bbox IoU@0.25 on predicted target query).

        Each item in the returned list contains:
          - batch_idx:      scene index within the batch
          - pred_target_q:  predicted target query index (or STOP=Q)
          - gt_target_q:    GT target query index
          - gt_inst_id:     GT instance index (remapped) when available
          - relation_source/lang_type: for filtering/debug
        """
        results: List[Dict[str, Any]] = []
        if not batch_lang_infos or not isinstance(sampled_coords, torch.Tensor):
            return results

        allowed_sources_raw = os.environ.get("SSR3DLLM_GEOM_REL_SOURCES", "").strip()
        allowed_sources = set()
        if allowed_sources_raw:
            allowed_sources = {
                s.strip().lower() for s in allowed_sources_raw.split(",") if s.strip()
            }

        for lang_info in batch_lang_infos:
            lang_type = getattr(lang_info, "lang_type", "")
            if not isinstance(lang_type, str) or not lang_type.startswith("rel3dref"):
                continue

            src = getattr(lang_info, "relation_source", None)
            if allowed_sources:
                if not isinstance(src, str) or src.lower() not in allowed_sources:
                    continue

            question_text = getattr(lang_info, "question", "") or ""
            use_geom_trigger = getattr(lang_info, "use_geom_trigger", False)
            if ("<geom>" not in question_text) and (not use_geom_trigger):
                continue

            q_hidden = getattr(lang_info, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                continue

            Q, D = q_hidden.shape
            if D != int(getattr(self, "query_dim", D)):
                continue
            batch_idx = getattr(lang_info, "batch_idx", None)
            if batch_idx is None or batch_idx >= sampled_coords.size(0):
                continue

            anchor_ids: List[int] = []
            q_ids_q = getattr(lang_info, "query_ids_question", None)
            if q_ids_q:
                for ids in q_ids_q:
                    anchor_ids.extend(ids)
            anchor_ids = [i for i in anchor_ids if 0 <= i < Q]

            target_ids: List[int] = []
            q_ids_a = getattr(lang_info, "query_ids_answer", None)
            if q_ids_a:
                for ids in q_ids_a:
                    target_ids.extend(ids)
            target_ids = [i for i in target_ids if 0 <= i < Q]
            if not target_ids:
                continue

            coords_bid = sampled_coords[batch_idx].to(device=device, dtype=torch.float32)
            q_hidden = q_hidden.to(device=device)
            if coords_bid.dim() != 2 or coords_bid.size(0) != Q:
                continue

            field_s, _ = self.relation_field(coords_bid.unsqueeze(0))  # [1,Q,student_dim]
            field_s = field_s.squeeze(0)  # [Q,student_dim]
            inject_to_llm = self._env_flag("SSR3DLLM_GEOM_INJECT_TO_LLM", "0")
            q_hidden_q = q_hidden.to(device=device, dtype=torch.float32).detach()  # [Q,D]
            q_hidden_s = self.query_up(q_hidden_q)  # [Q,student_dim]
            if inject_to_llm:
                obj_tokens = q_hidden_s + (field_s.to(dtype=q_hidden_s.dtype) - field_s.detach().to(dtype=q_hidden_s.dtype))
            else:
                obj_tokens = q_hidden_s + field_s.to(dtype=q_hidden_s.dtype)

            llm_text_init = getattr(lang_info, "llm_text_init", None)
            if isinstance(llm_text_init, torch.Tensor):
                init_vec = llm_text_init.to(device=device)
                if init_vec.dim() == 2:
                    init_vec = init_vec[0]
                text_init_vec = (
                    self.query_up(init_vec.detach().to(dtype=q_hidden_q.dtype)).to(dtype=obj_tokens.dtype)
                    if init_vec.shape[0] == D
                    else None
                )
            else:
                text_init_vec = None

            if text_init_vec is None:
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        text_init_vec = self._encode_text([text], device=device)[0][0]
                    except Exception:
                        text_init_vec = None

            # Token-level text tokens for decoder cross-attention.
            text_tokens = None
            llm_text_tokens = getattr(lang_info, "llm_text_tokens", None)
            if isinstance(llm_text_tokens, torch.Tensor) and llm_text_tokens.dim() == 2:
                if int(llm_text_tokens.size(-1)) == int(getattr(self, "query_dim", D)):
                    tt = llm_text_tokens.to(device=device, dtype=q_hidden_q.dtype).detach()
                    text_tokens = self.query_up(tt).to(dtype=obj_tokens.dtype).unsqueeze(0)
            if text_tokens is None:
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        _, tok_list = self._encode_text([text], device=device)
                        if tok_list and isinstance(tok_list[0], torch.Tensor):
                            text_tokens = tok_list[0].to(device=device, dtype=obj_tokens.dtype).unsqueeze(0)
                    except Exception:
                        text_tokens = None

            anchor_idx = anchor_ids[0] if anchor_ids else None
            target_idx = target_ids[0]
            stop_idx = Q
            # Effective-length chain: K anchors + target + STOP, padded to max_steps for stable tracing.
            max_steps = int(getattr(getattr(self.decoder, "cfg", None), "max_steps", 4))
            max_anchor_steps = max(0, max_steps - 2)
            anchor_seq: List[int] = []
            seen = set()
            for a in anchor_ids:
                try:
                    aa = int(a)
                except Exception:
                    continue
                if 0 <= aa < Q and aa not in seen:
                    anchor_seq.append(aa)
                    seen.add(aa)
                if len(anchor_seq) >= max_anchor_steps:
                    break

            if anchor_seq and (0 <= int(target_idx) < Q):
                pad_len = max(0, max_steps - (len(anchor_seq) + 2))
                order_labels = torch.tensor(
                    [[*anchor_seq, int(target_idx), stop_idx, *([stop_idx] * pad_len)]],
                    device=device,
                    dtype=torch.long,
                )
                pred_target_step = len(anchor_seq)
                pointer_logits = self.decoder(
                    obj_tokens=obj_tokens.unsqueeze(0),
                    text_tokens=text_tokens,
                    order_labels=order_labels,
                    text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0) if text_init_vec is not None else None,
                    obj_padding_mask=None,
                )  # [1,T,Q+1]
                pred_chain = pointer_logits.argmax(dim=-1).view(-1).tolist()
                pred_target_q = int(pred_chain[pred_target_step]) if len(pred_chain) > pred_target_step else -1
            else:
                # No anchor: [target, STOP] padded to max_steps.
                if not (0 <= int(target_idx) < Q):
                    continue
                pad_len = max(0, max_steps - 2)
                order_labels = torch.tensor(
                    [[int(target_idx), stop_idx, *([stop_idx] * pad_len)]], device=device, dtype=torch.long
                )
                pointer_logits = self.decoder(
                    obj_tokens=obj_tokens.unsqueeze(0),
                    text_tokens=text_tokens,
                    order_labels=order_labels,
                    text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0) if text_init_vec is not None else None,
                    obj_padding_mask=None,
                )  # [1,T,Q+1]
                pred_target_q = int(pointer_logits.argmax(dim=-1).view(-1)[0].item())

            gt_inst_id = None
            inst_ids_answer = getattr(lang_info, "inst_ids_answer", None)
            if isinstance(inst_ids_answer, list) and inst_ids_answer:
                first = inst_ids_answer[0]
                if isinstance(first, list) and first:
                    try:
                        gt_inst_id = int(first[0])
                    except Exception:
                        gt_inst_id = None

            results.append(
                {
                    "batch_idx": int(batch_idx),
                    "pred_target_q": pred_target_q,
                    "gt_target_q": int(target_idx),
                    "gt_target_qs": [int(i) for i in target_ids],
                    "gt_inst_id": gt_inst_id,
                    "relation_source": src,
                    "lang_type": lang_type,
                }
            )

        return results

    @torch.no_grad()
    def predict_geom_target_for_batch(
        self,
        batch_lang_infos: List[object],
        sampled_coords: torch.Tensor,
        device: torch.device,
        *,
        lang_prefixes: Tuple[str, ...] = ("scanrefer", "m3dref"),
        require_geom_trigger: bool = True,
        box_info_by_bid: Optional[torch.Tensor] = None,
        pred_class_names_by_bid: Optional[List[List[str]]] = None,
        valid_queries_by_bid: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict a single target query index for generic grounding tasks (e.g. scanrefer/m3dref),
        using the same geometry+decoder path as rel3dref, but in an anchor-free setting.

        This is intended for evaluation-time probing: "use geometry-chain output as grounding
        prediction", by selecting a single Mask3D query.

        Returns items with:
          - batch_idx:       scene index within the batch
          - pred_target_q:   predicted target query index (or STOP=Q)
          - gt_target_qs:    list of GT target query indices (may be empty)
          - gt_inst_ids:     list of GT instance ids (remapped) when available
          - lang_type:       original lang_type
        """
        results: List[Dict[str, Any]] = []
        if not batch_lang_infos or not isinstance(sampled_coords, torch.Tensor):
            return results

        geom_backend = self._get_geom_backend()
        use_vigor = geom_backend == "vigor"
        vigor = self._get_vigor_runtime(device) if use_vigor else None

        geom_chain_mode = str(os.environ.get("SSR3DLLM_GEOM_CHAIN_MODE", "bypass")).strip().lower()
        geom_chain_pad = str(os.environ.get("SSR3DLLM_GEOM_CHAIN_PAD", "repeat_last")).strip().lower()
        geom_chain_end_token = str(os.environ.get("SSR3DLLM_GEOM_CHAIN_END_TOKEN", "<END>")).strip()
        try:
            order_len = int(os.environ.get("SSR3DLLM_STEP_ORDER_LEN", "4"))
        except Exception:
            order_len = 4
        order_len = max(1, int(order_len))
        vigor_cascading = str(os.environ.get("SSR3DLLM_VIGOR_CASCADING", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        order_mode = str(os.environ.get("SSR3DLLM_ORDER_MODE", "")).strip().lower()

        def _norm_name(name: str) -> str:
            return str(name).strip().lower().replace("_", " ")

        def _format_chain(phrases: List[str]) -> str:
            phrases = [str(p).strip() for p in phrases if isinstance(p, str) and str(p).strip()]
            if not phrases:
                return ""
            if len(phrases) < order_len:
                # Keep the target phrase at the final step to preserve the (common) convention:
                # step4 == target. Fill earlier missing steps either by repeating the last phrase
                # (backward compatible) or inserting an explicit END token.
                if geom_chain_pad in {"end", "end_token", "eos"}:
                    out = list(phrases)
                    # Insert END tokens before the last phrase (target) until reaching order_len.
                    while len(out) < order_len:
                        if len(out) == 1:
                            out.insert(0, geom_chain_end_token)
                        else:
                            out.insert(-1, geom_chain_end_token)
                    phrases = out
                else:
                    phrases = phrases + [phrases[-1]] * (order_len - len(phrases))
            phrases = phrases[:order_len]
            # Training format: "a <step1> b <step2> c <step3> d <step4>"
            chunks = []
            for i, p in enumerate(phrases, start=1):
                chunks.append(f"{p} <step{i}>")
            return " ".join(chunks).strip()

        def _pad_phrases_for_order_len(phrases: List[str]) -> List[str]:
            phrases = [str(p).strip() for p in phrases if isinstance(p, str) and str(p).strip()]
            if not phrases:
                return []
            # Truncate from the head to keep the tail (target) when too long (matches Vigor padding).
            while len(phrases) > int(order_len):
                del phrases[0]
            if len(phrases) < int(order_len):
                # Keep the target phrase at the final step (common convention: step4 == target).
                if geom_chain_pad in {"end", "end_token", "eos"}:
                    out = list(phrases)
                    while len(out) < int(order_len):
                        if len(out) == 1:
                            out.insert(0, geom_chain_end_token)
                        else:
                            out.insert(-1, geom_chain_end_token)
                    phrases = out
                else:
                    phrases = phrases + [phrases[-1]] * (int(order_len) - len(phrases))
            return phrases[: int(order_len)]

        def _build_cascaded_order(order: List[str]) -> List[str]:
            out = []
            for i in range(len(order)):
                sub = list(dict.fromkeys(order[i:]))  # preserve order, unique
                out.append(", ".join([str(x) for x in sub if str(x)]))
            return out

        def _normalize_phrase(s: str) -> str:
            import re as _re
            s = str(s).strip().lower()
            s = _re.sub(r"[\t\r\n]+", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            for prefix in ("a ", "an ", "the "):
                if s.startswith(prefix):
                    s = s[len(prefix) :].strip()
                    break
            s = s.replace("&", " ")
            s = _re.sub(r"[^a-z0-9 ]+", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            return s

        def _singularize_last_word(phrase: str) -> str:
            w = phrase.split()
            if not w:
                return phrase
            last = w[-1]
            cand = [last]
            if last.endswith("ies") and len(last) > 3:
                cand.append(last[:-3] + "y")
            if last.endswith("es") and len(last) > 2:
                cand.append(last[:-2])
            if last.endswith("s") and len(last) > 1 and not last.endswith("ss"):
                cand.append(last[:-1])
            for c in cand[1:]:
                if c and c != last:
                    return " ".join(w[:-1] + [c])
            return phrase

        def _map_to_scannet200_label(phrase: str) -> Optional[str]:
            # Best-effort canonicalization: map a noisy step phrase to a ScanNet200 class label.
            try:
                from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_200

                labels = {str(x).strip().lower() for x in CLASS_LABELS_200}
            except Exception:
                return None
            p = _normalize_phrase(phrase)
            if not p:
                return None
            if p in labels:
                return p
            p2 = _singularize_last_word(p)
            if p2 in labels:
                return p2
            synonyms = {
                "television": "tv",
                "tv monitor": "tv",
                "trashcan": "trash can",
                "trash bin": "trash can",
                "garbage can": "trash can",
                "sofa": "couch",
                "bookcase": "bookshelf",
                "tub": "bathtub",
                "computer monitor": "monitor",
                "countertop": "counter",
                "counter top": "counter",
                "kitchen table": "table",
                "kitchen sink": "sink",
                "bathroom sink": "sink",
                "wardrobe closet": "closet",
            }
            if p in synonyms and synonyms[p] in labels:
                return synonyms[p]
            if p2 in synonyms and synonyms[p2] in labels:
                return synonyms[p2]
            # Last resort: use last token if it is a valid label.
            last = p.split()[-1] if p.split() else ""
            if last in labels:
                return last
            last2 = _singularize_last_word(last)
            if last2 in labels:
                return last2
            return None

        for lang_info in batch_lang_infos:
            lang_type = getattr(lang_info, "lang_type", "")
            if not isinstance(lang_type, str):
                continue
            prefix = lang_type.split(":")[0]
            if prefix not in set(lang_prefixes):
                continue

            if require_geom_trigger:
                question_text = getattr(lang_info, "question", "") or ""
                use_geom_trigger = getattr(lang_info, "use_geom_trigger", False)
                if ("<geom>" not in question_text) and (not use_geom_trigger):
                    continue

            q_hidden = getattr(lang_info, "query_hidden_feature", None)
            if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                continue

            Q, D = q_hidden.shape
            if D != int(getattr(self, "query_dim", D)):
                continue
            batch_idx = getattr(lang_info, "batch_idx", None)
            if batch_idx is None or batch_idx >= sampled_coords.size(0):
                continue

            coords_bid = sampled_coords[batch_idx].to(device=device, dtype=torch.float32)
            q_hidden = q_hidden.to(device=device)
            if coords_bid.dim() != 2 or coords_bid.size(0) != Q:
                continue

            # Backend: Vigor listener on Mask3D queries (direct inference, no STOP token).
            if vigor is not None:
                # A0: `SSR3DLLM_ORDER_MODE=slots` bypasses textual chain prompts and uses
                # LLM-derived `<stepK>` embeddings directly as Vigor order embeddings.
                if order_mode == "slots":
                    step_emb = getattr(lang_info, "llm_step_embeds", None)
                    if not isinstance(step_emb, torch.Tensor) or step_emb.dim() != 2:
                        continue
                    # Normalize to [O, D]
                    if int(step_emb.size(0)) < int(order_len):
                        pad = (
                            step_emb[-1:].repeat(int(order_len - int(step_emb.size(0))), 1)
                            if int(step_emb.size(0)) > 0
                            else torch.zeros((int(order_len), int(step_emb.size(-1))), device=step_emb.device, dtype=step_emb.dtype)
                        )
                        step_emb = torch.cat([step_emb, pad], dim=0)
                    elif int(step_emb.size(0)) > int(order_len):
                        step_emb = step_emb[: int(order_len)]

                    inner_dim = int(getattr(getattr(vigor, "model", None), "inner_dim", 768))
                    if int(step_emb.size(-1)) == int(self.query_dim):
                        proj = self._get_vigor_step_proj(inner_dim=inner_dim, device=device)
                        order_embeds = proj(step_emb.to(device=device, dtype=torch.float32)).unsqueeze(0)  # [1,O,D]
                    elif int(step_emb.size(-1)) == int(inner_dim):
                        order_embeds = step_emb.to(device=device, dtype=torch.float32).unsqueeze(0)
                    else:
                        continue

                    raw = getattr(lang_info, "question", None) or getattr(lang_info, "answer", None) or ""
                    if not isinstance(raw, str):
                        raw = str(raw)
                    text_utt = raw.replace("<geom>", "").strip()

                    try:
                        box_info = None
                        if (
                            isinstance(box_info_by_bid, torch.Tensor)
                            and box_info_by_bid.dim() == 3
                            and int(box_info_by_bid.size(0)) > int(batch_idx)
                            and int(box_info_by_bid.size(1)) == int(Q)
                            and int(box_info_by_bid.size(2)) == 4
                        ):
                            box_info = box_info_by_bid[int(batch_idx)].to(device=device, dtype=torch.float32)
                        else:
                            box_info = torch.zeros((Q, 4), device=device, dtype=torch.float32)
                            box_info[:, :3] = coords_bid.to(device=device, dtype=torch.float32)
                            box_info[:, 3] = 1.0

                        vq = None
                        if (
                            isinstance(valid_queries_by_bid, torch.Tensor)
                            and valid_queries_by_bid.dim() == 2
                            and 0 <= int(batch_idx) < int(valid_queries_by_bid.size(0))
                            and int(valid_queries_by_bid.size(1)) == int(Q)
                        ):
                            vq = valid_queries_by_bid[int(batch_idx)].to(device=device)
                            if vq.dtype != torch.bool:
                                vq = vq > 0

                        pred_class_mask = None
                        if torch.is_tensor(vq) and vq.numel() == int(Q):
                            pred_class_mask = vq.to(dtype=torch.float32).unsqueeze(0).repeat(int(order_len), 1)
                        else:
                            pred_class_mask = torch.ones((int(order_len), int(Q)), device=device, dtype=torch.float32)
                        pred_class_mask = pred_class_mask.unsqueeze(0)  # [1,O,Q]

                        lang_tokens = vigor.tokenizer([text_utt], return_tensors="pt", padding=True, truncation=True)
                        lang_tokens = {k: v.to(device=device) for k, v in lang_tokens.items()}

                        # Oracle chain length for VarLen-STOP masking in inference.
                        # - ScanRefer/M3DRef: force L=1
                        # - Otherwise: prefer `lang_info.ori_order_len`, fall back to full length.
                        order_valid_mask = None
                        try:
                            lt = getattr(lang_info, "lang_type", "") or ""
                            pfx = str(lt).split(":")[0] if isinstance(lt, str) else ""
                            eff = getattr(lang_info, "ori_order_len", None)
                            eff_i = int(eff) if eff is not None else None
                            if pfx in {"scanrefer", "m3dref"}:
                                eff_i = 1
                            if eff_i is None:
                                eff_i = int(order_len)
                            eff_i = max(1, min(int(order_len), int(eff_i)))
                            ovm = torch.zeros((1, int(order_len)), device=device, dtype=torch.float32)
                            ovm[0, : int(eff_i)] = 1.0
                            order_valid_mask = ovm
                        except Exception:
                            order_valid_mask = None

                        logits = vigor.forward_logits_with_order_embeds(
                            lang_tokens=lang_tokens,
                            order_embeds=order_embeds,
                            mask3d_queries=q_hidden.detach(),
                            box_info=box_info,
                            pred_class_mask=pred_class_mask,
                            order_valid_mask=order_valid_mask,
                        )
                        logits = logits.squeeze(0) if isinstance(logits, torch.Tensor) else None

                        if torch.is_tensor(vq) and vq.numel() == int(Q) and torch.is_tensor(logits) and logits.numel() == int(Q):
                            mask_cpu = vq.detach().to("cpu").to(dtype=torch.bool)
                            logits_cpu = logits.detach().to("cpu")
                            logits_cpu[~mask_cpu] = float("-inf")
                            pred_target_q = int(torch.argmax(logits_cpu).item()) if bool(mask_cpu.any().item()) else -1
                        else:
                            pred_target_q = int(torch.argmax(logits).item()) if torch.is_tensor(logits) and logits.numel() > 0 else -1
                    except Exception:
                        continue

                    def _collect_inst_ids(obj: Any) -> List[int]:
                        if obj is None:
                            return []
                        if isinstance(obj, (int, np.integer)):
                            return [int(obj)]
                        if torch.is_tensor(obj):
                            if obj.numel() == 1:
                                try:
                                    return [int(obj.item())]
                                except Exception:
                                    return []
                            return []
                        if isinstance(obj, (list, tuple)):
                            out0: List[int] = []
                            for it in obj:
                                out0.extend(_collect_inst_ids(it))
                            return out0
                        return []

                    def _dedup_keep_order(xs: List[int]) -> List[int]:
                        seen0 = set()
                        out0 = []
                        for x in xs:
                            if x in seen0:
                                continue
                            seen0.add(x)
                            out0.append(int(x))
                        return out0

                    inst_ids_answer = getattr(lang_info, "inst_ids_answer", None)
                    gt_inst_ids = _dedup_keep_order(_collect_inst_ids(inst_ids_answer))
                    if not gt_inst_ids:
                        inst_ids_question = getattr(lang_info, "inst_ids_question", None)
                        gt_inst_ids = _dedup_keep_order(_collect_inst_ids(inst_ids_question))

                    results.append(
                        {
                            "batch_idx": int(batch_idx),
                            "pred_target_q": int(pred_target_q),
                            "gt_target_qs": [],
                            "gt_inst_ids": gt_inst_ids,
                            "lang_type": getattr(lang_info, "lang_type", ""),
                            "geom_chain_mode": "slots",
                            "geom_text": "<slots>",
                        }
                    )
                    continue

                # NOTE: Vigor listener expects a *step-chain style* prompt in many training setups.
                # We support probing modes to distinguish "chain execution" vs "bypass" baselines.
                #
                # Modes (env `SSR3DLLM_GEOM_CHAIN_MODE`):
                # - bypass (default): use raw question text (no chain).
                # - normal: use `lang_info.rel_referential_order` as a chain prompt.
                # - oracle_sameclass: chain prompt where every step is the GT target class.
                # - oracle_gtanchors: chain prompt where steps are canonicalized to ScanNet200 labels
                #   and step4 is forced to GT target class.
                override = getattr(lang_info, "ssr3dllm_geom_text_override", None)
                raw = getattr(lang_info, "question", None) or getattr(lang_info, "answer", None) or ""
                if not isinstance(raw, str):
                    raw = str(raw)
                raw = raw.replace("<geom>", "").strip()
                text_utt = override.strip() if isinstance(override, str) and override.strip() else raw

                order = getattr(lang_info, "rel_referential_order", None)
                if geom_chain_mode in {"normal", "oracle_sameclass", "oracle_gtanchors"}:
                    if not isinstance(order, list) or not order:
                        # No order => cannot execute a chain prompt in these modes.
                        continue

                order_phrases: List[str] = []
                if geom_chain_mode == "normal":
                    order_phrases = [str(x) for x in order if isinstance(x, str) and str(x).strip()]
                elif geom_chain_mode == "oracle_sameclass":
                    tgt = getattr(lang_info, "ssr3dllm_target_class_name", None)
                    if not isinstance(tgt, str) or not tgt.strip():
                        tgt = str(order[-1]).strip() if isinstance(order, list) and order else ""
                    tgt = _map_to_scannet200_label(str(tgt)) or _normalize_phrase(str(tgt))
                    if not tgt:
                        continue
                    order_phrases = [tgt]
                elif geom_chain_mode == "oracle_gtanchors":
                    tgt = getattr(lang_info, "ssr3dllm_target_class_name", None)
                    tgt = tgt if isinstance(tgt, str) else ""
                    tgt_mapped = _map_to_scannet200_label(tgt) if tgt else None
                    mapped: List[str] = []
                    for x in order:
                        if not isinstance(x, str):
                            continue
                        m = _map_to_scannet200_label(x)
                        if m is None:
                            mapped = []
                            break
                        mapped.append(m)
                    if not mapped:
                        continue
                    if tgt_mapped:
                        mapped[-1] = tgt_mapped
                    order_phrases = mapped

                order_padded = _pad_phrases_for_order_len(order_phrases) if order_phrases else []
                order_texts = _build_cascaded_order(order_padded) if (order_padded and vigor_cascading) else (order_padded if order_padded else None)
                chain_str = _format_chain(order_padded) if order_padded else ""

                # In non-bypass chain modes, we require a non-empty order prompt (no heuristics).
                if geom_chain_mode in {"normal", "oracle_sameclass", "oracle_gtanchors"} and not order_texts:
                    continue

                try:
                    box_info = None
                    if (
                        isinstance(box_info_by_bid, torch.Tensor)
                        and box_info_by_bid.dim() == 3
                        and int(box_info_by_bid.size(0)) > int(batch_idx)
                        and int(box_info_by_bid.size(1)) == int(Q)
                        and int(box_info_by_bid.size(2)) == 4
                    ):
                        box_info = box_info_by_bid[int(batch_idx)].to(device=device, dtype=torch.float32)
                    else:
                        # Fallback: only centers are available, volume is a constant placeholder.
                        box_info = torch.zeros((Q, 4), device=device, dtype=torch.float32)
                        box_info[:, :3] = coords_bid.to(device=device, dtype=torch.float32)
                        box_info[:, 3] = 1.0

                    vq = None
                    if (
                        isinstance(valid_queries_by_bid, torch.Tensor)
                        and valid_queries_by_bid.dim() == 2
                        and 0 <= int(batch_idx) < int(valid_queries_by_bid.size(0))
                        and int(valid_queries_by_bid.size(1)) == int(Q)
                    ):
                        vq = valid_queries_by_bid[int(batch_idx)].to(device=device)
                        if vq.dtype != torch.bool:
                            vq = vq > 0

                    pred_names = None
                    if (
                        isinstance(pred_class_names_by_bid, list)
                        and 0 <= int(batch_idx) < len(pred_class_names_by_bid)
                        and isinstance(pred_class_names_by_bid[int(batch_idx)], list)
                        and len(pred_class_names_by_bid[int(batch_idx)]) == int(Q)
                    ):
                        pred_names = [str(x) if x is not None else "unknown" for x in pred_class_names_by_bid[int(batch_idx)]]

                    pred_class_mask = None
                    if order_padded and pred_names is not None:
                        mask_rows = []
                        for i in range(int(order_len)):
                            if vigor_cascading:
                                allowed = {_norm_name(x) for x in order_padded[i:] if str(x)}
                            else:
                                allowed = {_norm_name(order_padded[i])}
                            row = torch.tensor(
                                [1.0 if _norm_name(pred_names[j]) in allowed else 0.0 for j in range(int(Q))],
                                device=device,
                                dtype=torch.float32,
                            )
                            if torch.is_tensor(vq) and vq.numel() == int(Q):
                                row = row * vq.to(dtype=torch.float32)
                            mask_rows.append(row)
                        pred_class_mask = torch.stack(mask_rows, dim=0) if mask_rows else None
                    elif torch.is_tensor(vq) and vq.numel() == int(Q):
                        pred_class_mask = vq.to(dtype=torch.float32).unsqueeze(0).repeat(int(order_len), 1)

                    logits = vigor.predict_logits(
                        text=text_utt,
                        mask3d_queries=q_hidden.detach(),
                        box_info=box_info,
                        order_texts=order_texts,
                        pred_class_mask=pred_class_mask,
                    )

                    # Final hard-mask on CPU to guarantee we never pick an empty query.
                    if torch.is_tensor(vq) and vq.numel() == int(Q) and torch.is_tensor(logits) and logits.numel() == int(Q):
                        mask_cpu = vq.detach().to("cpu").to(dtype=torch.bool)
                        logits = logits.detach().to("cpu")
                        logits[~mask_cpu] = float("-inf")
                        if not bool(mask_cpu.any().item()):
                            pred_target_q = -1
                        else:
                            pred_target_q = int(torch.argmax(logits).item())
                    else:
                        pred_target_q = int(torch.argmax(logits).item()) if torch.is_tensor(logits) and logits.numel() > 0 else -1
                except Exception:
                    continue

                def _collect_inst_ids(obj: Any) -> List[int]:
                    if obj is None:
                        return []
                    if isinstance(obj, (int, np.integer)):
                        return [int(obj)]
                    if torch.is_tensor(obj):
                        if obj.numel() == 1:
                            try:
                                return [int(obj.item())]
                            except Exception:
                                return []
                        return []
                    if isinstance(obj, (list, tuple)):
                        out: List[int] = []
                        for it in obj:
                            out.extend(_collect_inst_ids(it))
                        return out
                    return []

                def _dedup_keep_order(xs: List[int]) -> List[int]:
                    seen = set()
                    out = []
                    for x in xs:
                        if x in seen:
                            continue
                        seen.add(x)
                        out.append(int(x))
                    return out

                # NOTE: In this codebase, `inst_ids_*` are usually a list of ints
                # (target-space instance indices), not a nested list.
                inst_ids_answer = getattr(lang_info, "inst_ids_answer", None)
                gt_inst_ids = _dedup_keep_order(_collect_inst_ids(inst_ids_answer))

                # Fallback: some datasets store GT ids on the question side.
                if not gt_inst_ids:
                    inst_ids_question = getattr(lang_info, "inst_ids_question", None)
                    gt_inst_ids = _dedup_keep_order(_collect_inst_ids(inst_ids_question))

                results.append(
                    {
                        "batch_idx": int(batch_idx),
                        "pred_target_q": pred_target_q,
                        "gt_target_qs": [],
                        "gt_inst_ids": gt_inst_ids,
                        "lang_type": getattr(lang_info, "lang_type", ""),
                        "geom_chain_mode": geom_chain_mode,
                        "geom_text": chain_str if chain_str else (text_utt if isinstance(text_utt, str) else ""),
                    }
                )
                continue

            # Build object tokens (STAMP-style: do not backprop into instance queries).
            # NOTE: `query_hidden_feature` already contains the relation-field delta used
            # in the LLM input; we keep this path consistent with rel3dref.
            field_s, _ = self.relation_field(coords_bid.unsqueeze(0))  # [1,Q,student_dim]
            field_s = field_s.squeeze(0)  # [Q,student_dim]
            inject_to_llm = self._env_flag("SSR3DLLM_GEOM_INJECT_TO_LLM", "0")
            q_hidden_q = q_hidden.to(device=device, dtype=torch.float32).detach()  # [Q,D]
            q_hidden_s = self.query_up(q_hidden_q)  # [Q,student_dim]
            if inject_to_llm:
                obj_tokens = q_hidden_s + (field_s.to(dtype=q_hidden_s.dtype) - field_s.detach().to(dtype=q_hidden_s.dtype))
            else:
                obj_tokens = q_hidden_s + field_s.to(dtype=q_hidden_s.dtype)

            # Text init: prefer LLM-derived init; fallback to BERT.
            llm_text_init = getattr(lang_info, "llm_text_init", None)
            if isinstance(llm_text_init, torch.Tensor):
                init_vec = llm_text_init.to(device=device)
                if init_vec.dim() == 2:
                    init_vec = init_vec[0]
                text_init_vec = (
                    self.query_up(init_vec.detach().to(dtype=q_hidden_q.dtype)).to(dtype=obj_tokens.dtype)
                    if init_vec.shape[0] == D
                    else None
                )
            else:
                text_init_vec = None
            if text_init_vec is None:
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        text_init_vec = self._encode_text([text], device=device)[0][0]
                    except Exception:
                        text_init_vec = None

            # Token-level text tokens for decoder cross-attention.
            text_tokens = None
            llm_text_tokens = getattr(lang_info, "llm_text_tokens", None)
            if isinstance(llm_text_tokens, torch.Tensor) and llm_text_tokens.dim() == 2:
                if int(llm_text_tokens.size(-1)) == int(getattr(self, "query_dim", D)):
                    tt = llm_text_tokens.to(device=device, dtype=q_hidden_q.dtype).detach()
                    text_tokens = self.query_up(tt).to(dtype=obj_tokens.dtype).unsqueeze(0)
            if text_tokens is None:
                text = getattr(lang_info, "answer", None) or getattr(lang_info, "question", None)
                if isinstance(text, str) and text:
                    try:
                        _, tok_list = self._encode_text([text], device=device)
                        if tok_list and isinstance(tok_list[0], torch.Tensor):
                            text_tokens = tok_list[0].to(device=device, dtype=obj_tokens.dtype).unsqueeze(0)
                    except Exception:
                        text_tokens = None

            stop_idx = Q
            order_labels = torch.tensor([[stop_idx]], device=device, dtype=torch.long)  # [1,1]
            pointer_logits = self.decoder(
                obj_tokens=obj_tokens.unsqueeze(0),
                text_tokens=text_tokens,
                order_labels=order_labels,
                text_init=text_init_vec.to(dtype=obj_tokens.dtype).unsqueeze(0) if text_init_vec is not None else None,
                obj_padding_mask=None,
            )  # [1,1,Q+1]
            pred_target_q = int(pointer_logits.argmax(dim=-1).view(-1)[0].item())

            gt_target_qs: List[int] = []
            q_ids_a = getattr(lang_info, "query_ids_answer", None)
            if q_ids_a:
                for ids in q_ids_a:
                    gt_target_qs.extend([int(i) for i in ids if isinstance(i, int)])
            gt_target_qs = [i for i in gt_target_qs if 0 <= i < Q]

            gt_inst_ids: List[int] = []
            inst_ids_answer = getattr(lang_info, "inst_ids_answer", None)
            if isinstance(inst_ids_answer, list) and inst_ids_answer:
                for ids in inst_ids_answer:
                    if isinstance(ids, list):
                        for inst in ids:
                            if isinstance(inst, (int, np.integer)):
                                gt_inst_ids.append(int(inst))

            results.append(
                {
                    "batch_idx": int(batch_idx),
                    "pred_target_q": pred_target_q,
                    "gt_target_qs": gt_target_qs,
                    "gt_inst_ids": gt_inst_ids,
                    "lang_type": lang_type,
                }
            )

        return results
