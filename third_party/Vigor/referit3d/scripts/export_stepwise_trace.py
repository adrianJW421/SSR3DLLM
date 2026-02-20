#!/usr/bin/env python3
"""
Export step-wise listener traces for interpretability figures.

This script mirrors the initialization path of `train_referit3d_llama_stepslot.py`,
but runs inference on a small, reproducible subset and dumps per-step top-k, ranks,
and varlen/STOP "freeze" diagnostics.

Usage pattern (recommended):
  - Put all config in a shell script (see `final_scripts/export_stepwise_trace.sh`)
  - Call this script in `--mode evaluate`
  - Control selection/output via env vars:
      VIGOR_TRACE_STIMULUS_IDS_JSON=/path/to/interp_cases.json
      VIGOR_TRACE_OUT_JSON=/path/to/out.json
      VIGOR_TRACE_TOPK=5
      VIGOR_TRACE_MAX_SAMPLES=0
      VIGOR_TRACE_SEED=1
      # Optional sharding for large exports:
      VIGOR_TRACE_SHARD_ID=0
      VIGOR_TRACE_NUM_SHARDS=8
"""

from __future__ import annotations

import json
import os
import random
import sys
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer


_vigor_root = Path(__file__).resolve().parents[2]  # .../Vigor
sys.path.insert(0, str(_vigor_root))

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import (  # type: ignore
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
)
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders  # type: ignore
from referit3d.models.referit3d_net import ReferIt3DNet_transformer  # type: ignore
from referit3d.models.llama_stepslot import (  # type: ignore
    LlamaStepSlotConfig,
    LlamaStepSlotOrderEncoder,
    ReferIt3DNetTransformerLlamaStepSlot,
)
from referit3d.models.utils import load_state_dicts  # type: ignore
from referit3d.utils import seed_training_code, set_gpu_to_zero_position  # type: ignore


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def _entropy(probs: torch.Tensor) -> float:
    p = probs.clamp(min=1e-12)
    return float((-p * torch.log(p)).sum().item())


def _rank_1based(probs: torch.Tensor, gt_idx: int) -> int:
    if gt_idx < 0 or gt_idx >= int(probs.numel()):
        return -1
    gt = float(probs[gt_idx].item())
    higher = int((probs > gt).sum().item())
    return int(higher + 1)


def _read_cases_json(path: Path) -> List[str]:
    """
    Accepts:
      - {"cases":[{"stimulus_id":"..."}]} or {"cases":[...ids...]}
      - {"success":[...], "failure":[...]} (flatten values)
      - a plain list ["id1","id2",...]
    """
    obj = json.loads(path.read_text())
    out: List[str] = []
    if isinstance(obj, list):
        out = [str(x) for x in obj if str(x).strip()]
        return out
    if isinstance(obj, dict):
        if "cases" in obj:
            v = obj["cases"]
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict) and "stimulus_id" in x:
                        out.append(str(x["stimulus_id"]))
                    else:
                        out.append(str(x))
                return [s for s in out if s.strip()]
        # flatten top-level lists
        for _, v in obj.items():
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict) and "stimulus_id" in x:
                        out.append(str(x["stimulus_id"]))
                    else:
                        out.append(str(x))
        return [s for s in out if s.strip()]
    return []

def _maybe_infer_lora_from_resume_ckpt(resume_ckpt: str) -> None:
    """
    Auto-enable LoRA adapters to match a resume checkpoint.

    Enable via:
      - VIGOR_LLM_LORA_AUTO=1   (recommended)
        or
      - VIGOR_LLM_LORA=auto

    If the checkpoint contains LoRA keys, infer and set:
      - VIGOR_LLM_LORA=1
      - VIGOR_LLM_LORA_R (from lora_A shape)
      - VIGOR_LLM_LORA_LAST_N (from covered layer indices, assuming a suffix)
      - VIGOR_LLM_LORA_TARGETS (from keys, e.g. q_proj,v_proj)
    """
    try:
        auto = _env_flag("VIGOR_LLM_LORA_AUTO", "0") or (_env_str("VIGOR_LLM_LORA", "").strip().lower() == "auto")
    except Exception:
        auto = False
    if not auto:
        return
    if not resume_ckpt or not os.path.isfile(resume_ckpt):
        return

    try:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
    except Exception:
        return
    if not isinstance(ckpt, dict):
        return
    sd = ckpt.get("model", None)
    if not isinstance(sd, dict) or not sd:
        return

    lora_a_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(".lora_A")]
    if not lora_a_keys:
        return

    layers = set()
    targets = set()
    r = None
    for k in lora_a_keys:
        m = re.search(r"model\\.layers\\.(\\d+)\\.", k)
        if m:
            try:
                layers.add(int(m.group(1)))
            except Exception:
                pass
        m = re.search(r"self_attn\\.([a-z_]+)\\.lora_A$", k)
        if m:
            targets.add(str(m.group(1)))
        if r is None:
            try:
                t = sd.get(k, None)
                if torch.is_tensor(t) and t.ndim == 2:
                    r = int(t.shape[0])
            except Exception:
                r = None

    last_n = None
    if layers:
        try:
            last_n = int(max(layers) - min(layers) + 1)
        except Exception:
            last_n = None

    os.environ["VIGOR_LLM_LORA"] = "1"
    if (os.environ.get("VIGOR_LLM_LORA_R", "").strip() == "") and (r is not None):
        os.environ["VIGOR_LLM_LORA_R"] = str(int(r))
    if (os.environ.get("VIGOR_LLM_LORA_LAST_N", "").strip() == "") and (last_n is not None):
        os.environ["VIGOR_LLM_LORA_LAST_N"] = str(int(last_n))
    if (os.environ.get("VIGOR_LLM_LORA_TARGETS", "").strip() == "") and targets:
        os.environ["VIGOR_LLM_LORA_TARGETS"] = ",".join(sorted(targets))

    print(
        "[Vigor][llama_stepslot][lora_auto] enabled=1 "
        f"r={os.environ.get('VIGOR_LLM_LORA_R', '')} "
        f"alpha={os.environ.get('VIGOR_LLM_LORA_ALPHA', '') or _env_float('VIGOR_LLM_LORA_ALPHA', 16.0)} "
        f"dropout={os.environ.get('VIGOR_LLM_LORA_DROPOUT', '') or _env_float('VIGOR_LLM_LORA_DROPOUT', 0.0)} "
        f"last_n={os.environ.get('VIGOR_LLM_LORA_LAST_N', '')} "
        f"targets={os.environ.get('VIGOR_LLM_LORA_TARGETS', '')} "
        f"ckpt={resume_ckpt}",
        flush=True,
    )


def _select_subset_indices(dataset, stimulus_ids: Optional[Sequence[str]], *, max_samples: int, seed: int) -> List[int]:
    n = int(len(dataset))
    if stimulus_ids:
        wanted = {str(x).strip() for x in stimulus_ids if str(x).strip()}
        if not wanted:
            return list(range(n))
        refs = getattr(dataset, "references", None)
        if refs is None or "stimulus_id" not in refs.columns:
            return list(range(n))
        sids = refs["stimulus_id"].astype(str)
        idxs = [int(i) for i in list(sids[sids.isin(wanted)].index)]
    else:
        idxs = list(range(n))

    if max_samples > 0 and len(idxs) > max_samples:
        rng = random.Random(int(seed))
        idxs = rng.sample(idxs, k=int(max_samples))
        idxs = sorted(idxs)
    return idxs


def _apply_sharding(idxs: List[int], *, shard_id: int, num_shards: int) -> List[int]:
    """
    Deterministic sharding to enable parallel full-eval trace export.

    We shard by position in the (sorted) index list to keep it stable regardless
    of the absolute dataset indices.
    """
    if shard_id < 0:
        return idxs
    ns = int(num_shards) if int(num_shards) > 0 else 1
    sid = int(shard_id)
    if sid < 0 or sid >= ns:
        raise ValueError(f"Invalid shard_id={sid} for num_shards={ns}")
    return [idx for j, idx in enumerate(idxs) if (j % ns) == sid]


def main() -> None:
    args = parse_arguments()

    llm_path = _env_str("VIGOR_LLM_MODEL_PATH", "")
    if not llm_path:
        raise RuntimeError("Missing env VIGOR_LLM_MODEL_PATH (local path to causal LLM weights).")

    resume_ckpt = _env_str("VIGOR_LLM_STEPSLOT_RESUME_CKPT", "")
    if not resume_ckpt:
        raise RuntimeError("Missing env VIGOR_LLM_STEPSLOT_RESUME_CKPT (wrapper checkpoint .pth).")

    # If the checkpoint was trained with LoRA, we must wrap LoRA modules before loading weights.
    _maybe_infer_lora_from_resume_ckpt(resume_ckpt)

    listener_init = _env_str("VIGOR_LISTENER_INIT_CKPT", "") or str(getattr(args, "resume_path", "") or "")
    if not listener_init:
        raise RuntimeError("Missing listener init ckpt: set --resume-path or env VIGOR_LISTENER_INIT_CKPT.")

    out_json = _env_str("VIGOR_TRACE_OUT_JSON", "")
    if not out_json:
        raise RuntimeError("Missing env VIGOR_TRACE_OUT_JSON (output path).")
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    case_json = _env_str("VIGOR_TRACE_STIMULUS_IDS_JSON", "")
    stimulus_ids: Optional[List[str]] = None
    if case_json:
        p = Path(case_json)
        if not p.exists():
            raise FileNotFoundError(str(p))
        stimulus_ids = _read_cases_json(p)

    topk = max(1, _env_int("VIGOR_TRACE_TOPK", 5))
    max_samples = _env_int("VIGOR_TRACE_MAX_SAMPLES", 0)
    seed = _env_int("VIGOR_TRACE_SEED", 1)
    shard_id = _env_int("VIGOR_TRACE_SHARD_ID", -1)
    num_shards = _env_int("VIGOR_TRACE_NUM_SHARDS", 1)

    # Data
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    # Single-sample loader for interpretability.
    base_test = data_loaders["test"].dataset
    idxs = _select_subset_indices(base_test, stimulus_ids, max_samples=max_samples, seed=seed)
    idxs = _apply_sharding(idxs, shard_id=shard_id, num_shards=num_shards)
    subset = Subset(base_test, idxs)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    # GPU env
    set_gpu_to_zero_position(args.gpu)
    seed_training_code(args.random_seed)
    device = torch.device("cuda")

    # Listener (BERT init) + wrapper
    n_classes = len(class_to_idx) - 1
    pad_idx = class_to_idx["pad"]
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    if _env_flag("VIGOR_STEP_MARKERS", "0"):
        step_tokens = [f"<step{i+1}>" for i in range(int(getattr(args, "order_len", 4)))]
        _ = tokenizer.add_special_tokens({"additional_special_tokens": step_tokens})

    class_name_list = list(class_to_idx.keys())
    class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].to(device)

    listener = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    if _env_flag("VIGOR_STEP_MARKERS", "0") and hasattr(listener, "language_encoder"):
        try:
            listener.language_encoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
    listener = listener.to(device)

    _ = load_state_dicts(listener_init, map_location=device, model=listener)

    llm_max_len = _env_int("VIGOR_LLM_MAX_LEN", 64)
    llm_mem_tokens = _env_int("VIGOR_LLM_MEM_TOKENS", 0)
    llm_use_bf16 = _env_flag("VIGOR_LLM_USE_BF16", "1")
    llm_cfg = LlamaStepSlotConfig(
        model_path=llm_path,
        order_len=int(getattr(args, "order_len", 4)),
        max_length=llm_max_len,
        memory_tokens=int(llm_mem_tokens),
        distill_w=0.0,
        global_distill_w=0.0,
        freeze_llm_except_step_rows=True,
        local_files_only=True,
        use_bf16=bool(llm_use_bf16),
    )
    llm = LlamaStepSlotOrderEncoder(out_dim=int(getattr(args, "inner_dim", 768)), cfg=llm_cfg).to(device)
    model = ReferIt3DNetTransformerLlamaStepSlot(listener=listener, llm=llm, cfg=llm_cfg).to(device)

    _ = load_state_dicts(resume_ckpt, map_location=device, model=model)
    model.eval()

    stop_token = _env_str("VIGOR_STOP_TOKEN", "<STOP>") or "<STOP>"
    varlen_mask_source = _env_str("VIGOR_VARLEN_MASK_SOURCE", "oracle").strip().lower()
    order_len = int(getattr(args, "order_len", 4))
    export_state = _env_flag("VIGOR_TRACE_EXPORT_SPATIAL_STATE", "0")
    progress_every = max(0, _env_int("VIGOR_TRACE_PROGRESS_EVERY", 200))

    traces: List[Dict[str, Any]] = []
    t0 = time.time()
    n_total = None
    try:
        n_total = int(len(loader))
    except Exception:
        n_total = None
    print(
        f"[export_stepwise_trace] start: n={n_total if n_total is not None else '?'} "
        f"topk={topk} order_len={order_len} mask_source={varlen_mask_source} "
        f"shard={_env_int('VIGOR_TRACE_SHARD_ID', -1)}/{_env_int('VIGOR_TRACE_NUM_SHARDS', 1)}",
        flush=True,
    )
    for batch in loader:
        if progress_every > 0 and (len(traces) % progress_every == 0):
            dt = max(1e-6, time.time() - t0)
            rate = float(len(traces)) / dt
            eta = None
            if n_total is not None and rate > 1e-9:
                eta = float(n_total - len(traces)) / rate
            eta_s = f"{eta/60.0:.1f}m" if eta is not None else "?"
            print(
                f"[export_stepwise_trace] progress {len(traces)}/{n_total if n_total is not None else '?'} "
                f"({rate:.2f} it/s, eta={eta_s})",
                flush=True,
            )
        # Ensure no loss branches are active.
        batch["inference"] = True

        # Move tensors to GPU.
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        with torch.no_grad():
            out = model(batch)
        if not isinstance(out, (list, tuple)) or len(out) < 4:
            raise RuntimeError(f"Unexpected model output type/len: {type(out)} len={len(out) if isinstance(out,(list,tuple)) else 'n/a'}")

        logits = out[3]  # [B,N]
        if not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"Unexpected logits shape: {None if not torch.is_tensor(logits) else tuple(logits.shape)}")

        ctx = _safe_int(_to_list(batch.get("context_size", [logits.size(1)]))[0], int(logits.size(1)))
        ctx = max(1, min(int(ctx), int(logits.size(1))))

        tgt_pos = int(batch["target_pos"].view(-1)[0].item())
        tgt_obj_id = _safe_int(_to_list(batch.get("target_object_id", [-1]))[0], -1)
        scene_id = str(_to_list(batch.get("scan_id", [""]))[0] or _to_list(batch.get("scene_id", [""]))[0] or "")
        utterance = str(_to_list(batch.get("utterance", [""]))[0] or "")
        stimulus_id = str(_to_list(batch.get("stimulus_id", [""]))[0] or "")

        # If the wrapper uses predicted varlen masks (e.g., gate), it may write
        # `batch["order_valid_mask"]` in-place before calling the listener.
        # Prefer that mask as the one actually used by freeze-gated updates.
        order_valid_mask_used: Optional[List[float]] = None
        try:
            ovm_used = batch.get("order_valid_mask", None)
            if torch.is_tensor(ovm_used):
                ovm_used = ovm_used.view(-1, int(order_len)).detach().cpu().float().tolist()
                if isinstance(ovm_used, list) and ovm_used and isinstance(ovm_used[0], list):
                    order_valid_mask_used = [float(x) for x in list(ovm_used[0])[:order_len]]
        except Exception:
            order_valid_mask_used = None

        steps_raw: List[str] = []
        ref_order = batch.get("referential_order", None)
        if isinstance(ref_order, list) and ref_order:
            # batch['referential_order'] is B-length list of K-length strings
            row0 = ref_order[0] if isinstance(ref_order[0], list) else ref_order
            steps_raw = [str(x) for x in list(row0)[:order_len]]
        ori_len = _safe_int(_to_list(batch.get("ori_order_len", [len([s for s in steps_raw if s])]))[0], len(steps_raw))
        ori_len = max(0, min(int(order_len), int(ori_len)))
        order_valid_mask_oracle = [1.0 if i < ori_len else 0.0 for i in range(order_len)]
        if not isinstance(order_valid_mask_used, list) or len(order_valid_mask_used) != int(order_len):
            order_valid_mask_used = list(order_valid_mask_oracle)

        # Optional gate probabilities (if enabled), for plotting/analysis.
        gate_prob: Optional[List[float]] = None
        try:
            if varlen_mask_source == "gate":
                order_embeds = getattr(model, "last_order_embeds", None)
                if torch.is_tensor(order_embeds):
                    gate_logits = model._varlen_gate_logits(order_embeds)  # type: ignore[attr-defined]
                    prob = torch.sigmoid(gate_logits).detach().cpu().float().tolist()
                    if isinstance(prob, list) and prob and isinstance(prob[0], list):
                        gate_prob = [float(x) for x in prob[0][:order_len]]
        except Exception:
            gate_prob = None

        # Get per-step logits from listener.
        step_logits_list = getattr(model.listener, "last_ref_logits_steps", None)
        if not isinstance(step_logits_list, list) or len(step_logits_list) == 0:
            raise RuntimeError("Missing listener.last_ref_logits_steps (enable step-wise logging in listener).")
        if len(step_logits_list) < order_len:
            # Best-effort pad (shouldn't happen when VIGOR_VARLEN_EARLY_STOP=0).
            step_logits_list = list(step_logits_list) + [step_logits_list[-1]] * (order_len - len(step_logits_list))

        def _summarize_logits(step_logits: torch.Tensor) -> Dict[str, Any]:
            sl = step_logits.view(1, -1)[:, :ctx].squeeze(0).float()
            probs = _softmax_probs(sl)
            topv, topi = torch.topk(probs, k=min(int(topk), int(probs.numel())))
            top_pos = [int(x) for x in topi.detach().cpu().tolist()]
            top_prob = [float(x) for x in topv.detach().cpu().tolist()]
            gt_rank = _rank_1based(probs, tgt_pos)
            gt_prob = float(probs[tgt_pos].item()) if 0 <= tgt_pos < int(probs.numel()) else float("nan")
            return {
                "top_pos": top_pos,
                "top_prob": top_prob,
                "gt_rank": int(gt_rank),
                "gt_prob": float(gt_prob),
                "entropy": float(_entropy(probs)),
            }

        per_step: List[Dict[str, Any]] = []
        for k in range(order_len):
            info = _summarize_logits(step_logits_list[k][0])
            info.update({"k": int(k + 1)})
            per_step.append(info)

        final_info = _summarize_logits(logits[0])
        pred_pos = int(torch.argmax(logits[0, :ctx]).item())
        final_info.update({"pred_pos": pred_pos})

        # Varlen freeze diagnostics: deltas between successive steps.
        step_deltas_l1: List[float] = []
        for k in range(1, order_len):
            a = step_logits_list[k - 1][0, :ctx].float()
            b = step_logits_list[k][0, :ctx].float()
            step_deltas_l1.append(float((a - b).abs().mean().item()))
        pad_deltas_l1 = []
        for k in range(1, order_len):
            # Transitions into an invalid step (k+1) should be ~0 if freeze works.
            try:
                if float(order_valid_mask_used[k]) < 0.5:
                    pad_deltas_l1.append(step_deltas_l1[k - 1])
            except Exception:
                pass

        # Geometry for visualization (bbox corners + ids), if present.
        object_ids = _to_list(batch.get("object_ids", []))
        if object_ids and isinstance(object_ids[0], list):
            object_ids = object_ids[0]
        object_ids = [int(x) for x in list(object_ids)[:ctx]] if object_ids else []

        box_corners = batch.get("box_corners", None)
        if torch.is_tensor(box_corners):
            box_corners = box_corners[0, :ctx].detach().cpu().float().tolist()
        else:
            box_corners = []

        # Optional: export compact spatial-state diagnostics (NOT full vectors).
        # This supports "spatial state is meaningful" analysis without bloating JSON.
        state_diag: Dict[str, Any] = {}
        if export_state:
            try:
                oe = batch.get("order_embeds", None)
                if torch.is_tensor(oe):
                    # order_embeds: [B,O,D] on the wrapper; listener may internally treat it as [B,O,1,D]
                    oe0 = oe[0]
                    if oe0.dim() == 3:
                        oe0 = oe0[:, 0, :]
                    oe0 = oe0[:order_len].detach().float()
                    norms = oe0.norm(dim=-1).detach().cpu().tolist()
                    state_diag["order_embed_norm"] = [float(x) for x in norms]
                    # Teacher alignment: cosine(order_embed, bert_step_embed)
                    try:
                        t = model._teacher_step_embeds(batch)[0, :order_len].detach().float()  # type: ignore[attr-defined]
                        a = torch.nn.functional.normalize(oe0, dim=-1)
                        b = torch.nn.functional.normalize(t, dim=-1)
                        cos = (a * b).sum(dim=-1).detach().cpu().tolist()
                        state_diag["order_embed_cos_teacher"] = [float(x) for x in cos]
                    except Exception:
                        state_diag["order_embed_cos_teacher"] = None
            except Exception:
                state_diag = {}

        traces.append(
            {
                "stimulus_id": stimulus_id,
                "scene_id": scene_id,
                "utterance": utterance,
                "steps_raw": steps_raw,
                "stop_token": stop_token,
                "varlen_mask_source": varlen_mask_source,
                "state_diag": state_diag,
                "ori_order_len": int(ori_len),
                "order_valid_mask_oracle": order_valid_mask_oracle,
                "order_valid_mask_used": order_valid_mask_used,
                "gate_prob": gate_prob,
                "target_pos": int(tgt_pos),
                "target_object_id": int(tgt_obj_id),
                "context_size": int(ctx),
                "object_ids": object_ids,
                "box_corners": box_corners,
                "per_step": per_step,
                "final": final_info,
                "step_deltas_l1": step_deltas_l1,
                "pad_step_deltas_l1": pad_deltas_l1,
            }
        )

    out_path.write_text(json.dumps(traces, indent=2))
    shard_note = ""
    if shard_id >= 0 and int(num_shards) > 1:
        shard_note = f" (shard {int(shard_id)}/{int(num_shards)})"
    print(f"[export_stepwise_trace] wrote n={len(traces)} -> {out_path}{shard_note}", flush=True)


if __name__ == "__main__":
    main()
