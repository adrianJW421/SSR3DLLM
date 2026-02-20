import os
import re
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy
from typing import Optional, List, Any, Dict
import random
from torchvision.ops import sigmoid_focal_loss
from baseline.dataset.dataset_code.language_info import lang_info_data
import gc
from loguru import logger
import time

try:  # pragma: no cover - optional dependency
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except ImportError:  # pragma: no cover
    hnswlib = None  # type: ignore
    _HNSW_AVAILABLE = False

from transformers import LlamaTokenizer
from peft import LoraConfig, get_peft_model
from models.misc import print_grad_status
import glob

from .modeling_llama import LlamaForCausalLM, LlamaModel
from .llama_utils import *

try:
    from models.referit3d_listener_runtime import ReferIt3DListenerRuntime as VigorRuntimeListener
except Exception:  # pragma: no cover
    VigorRuntimeListener = None  # type: ignore


_GEOM_LORA_RE = re.compile(
    r"^llm\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)(?:\.weight)?$"
)
_BASE_LORA_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$")


def _load_ckpt_payload(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _extract_state_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model", "state_dict", "model_state_dict"):
        value = payload.get(key, None)
        if isinstance(value, dict):
            return value
    return payload


def _pick_bundle_listener_state_dict(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    bundle = payload.get("ssr3dllm_bundle", None)
    if not isinstance(bundle, dict):
        return None
    listeners = bundle.get("listeners", None)
    if not isinstance(listeners, dict) or not listeners:
        return None

    default_profile = str(bundle.get("default_listener_profile", "503")).strip() or "503"
    profile_raw = (
        str(os.environ.get("SSR3DLLM_VIGOR_PROFILE", "")).strip()
        or str(os.environ.get("SSR3DLLM_BUNDLE_PROFILE", "")).strip()
        or default_profile
    )
    aliases = {"main": "503", "ub": "519"}
    profile = aliases.get(profile_raw.lower(), profile_raw)
    if profile not in listeners:
        profile = default_profile if default_profile in listeners else sorted(list(listeners.keys()))[0]

    selected = listeners.get(profile, None)
    if isinstance(selected, dict):
        return selected
    return None


def _pick_bundle_geom_state_dict(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    bundle = payload.get("ssr3dllm_bundle", None)
    if not isinstance(bundle, dict):
        return None
    geom = bundle.get("geom_adapters", None)
    if not isinstance(geom, dict) or not geom:
        return None

    default_profile = str(bundle.get("default_listener_profile", "503")).strip() or "503"
    profile_raw = (
        str(os.environ.get("SSR3DLLM_VIGOR_PROFILE", "")).strip()
        or str(os.environ.get("SSR3DLLM_BUNDLE_PROFILE", "")).strip()
        or default_profile
    )
    aliases = {"main": "503", "ub": "519"}
    profile = aliases.get(profile_raw.lower(), profile_raw)
    if profile not in geom:
        profile = default_profile if default_profile in geom else sorted(list(geom.keys()))[0]

    selected = geom.get(profile, None)
    if isinstance(selected, dict):
        return selected
    return None


def _collect_geom_lora_from_vigor(sd: Dict[str, Any]) -> Dict[tuple[int, str], Dict[str, torch.Tensor]]:
    slots: Dict[tuple[int, str], Dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        m = _GEOM_LORA_RE.match(k)
        if m is None:
            continue
        layer = int(m.group(1))
        target = str(m.group(2))
        ab = str(m.group(3))
        slot = slots.setdefault((layer, target), {})
        slot[ab] = v.detach().cpu()
    return slots


def _set_active_peft_adapter(base_model: nn.Module, adapter_name: str) -> bool:
    holder = None
    holder_ref = getattr(base_model, "_ssr3dllm_peft_parent_ref", None)
    if callable(holder_ref):
        try:
            holder = holder_ref()
        except Exception:
            holder = None
    if holder is None:
        holder = getattr(base_model, "_ssr3dllm_peft_parent", None)
    if holder is None:
        holder = base_model if hasattr(base_model, "set_adapter") else None
    if holder is None or not hasattr(holder, "set_adapter"):
        return False
    try:
        holder.set_adapter(str(adapter_name))
        return True
    except Exception:
        return False


def _get_active_peft_adapter(base_model: nn.Module) -> Optional[str]:
    holder = None
    holder_ref = getattr(base_model, "_ssr3dllm_peft_parent_ref", None)
    if callable(holder_ref):
        try:
            holder = holder_ref()
        except Exception:
            holder = None
    if holder is None:
        holder = getattr(base_model, "_ssr3dllm_peft_parent", None)
    if holder is None:
        holder = base_model if hasattr(base_model, "set_adapter") else None
    if holder is None:
        return None
    active = getattr(holder, "active_adapter", None)
    if isinstance(active, str) and active:
        return active
    if isinstance(active, (list, tuple)) and len(active) > 0:
        first = active[0]
        if isinstance(first, str) and first:
            return first
    cfg = getattr(holder, "peft_config", None)
    if isinstance(cfg, dict) and len(cfg) > 0:
        try:
            return str(next(iter(cfg.keys())))
        except Exception:
            return None
    return None


def _attach_geom_adapter_from_vigor_ckpt(
    *,
    llama_model: nn.Module,
    llama_tokenizer: LlamaTokenizer,
    geom_ckpt_path: str,
    adapter_name: str,
) -> Dict[str, int]:
    """
    Load geom routing resources from a Vigor llama-step-slot ckpt:
    - attach a separate PEFT adapter for 503 LoRA (r=8, last-4 q/v),
    - build a lightweight step-slot adapter (proj_step/proj_mem/mem_tokens),
    - copy <stepK>/<cls>/<STOP> embedding rows into current tokenizer ids.
    """
    payload = _load_ckpt_payload(geom_ckpt_path)
    sd = _extract_state_dict(payload)
    # Packed SSR3DLLM.ckpt can store geom-adapter tensors separately from the Vigor listener.
    bundle_geom = _pick_bundle_geom_state_dict(payload)
    if isinstance(bundle_geom, dict) and bundle_geom:
        sd = bundle_geom
    stats = {
        "lora_slots": 0,
        "lora_loaded": 0,
        "step_rows_loaded": 0,
        "mem_tokens": 0,
    }

    # Build geom step-slot adapter from Vigor ckpt.
    try:
        from .stepslot_adapter import SoftStepSlotAdapter

        mem_tok = sd.get("llm.mem_tokens", None)
        mem_n = int(mem_tok.shape[0]) if torch.is_tensor(mem_tok) and mem_tok.dim() == 2 else 0
        try:
            hidden_size = int(next(llama_model.parameters()).shape[-1])
        except Exception:
            hidden_size = 2048
        adapter = SoftStepSlotAdapter(hidden_size=hidden_size, out_dim=768, mem_tokens=mem_n)
        proj_step = {k[len("llm.proj_step.") :]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("llm.proj_step.") and torch.is_tensor(v)}
        proj_global = {k[len("llm.proj_global.") :]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("llm.proj_global.") and torch.is_tensor(v)}
        proj_mem = {k[len("llm.proj_mem.") :]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("llm.proj_mem.") and torch.is_tensor(v)}
        if proj_step:
            adapter.proj_step.load_state_dict(proj_step, strict=False)
        if proj_global:
            adapter.proj_global.load_state_dict(proj_global, strict=False)
        if mem_n > 0 and torch.is_tensor(mem_tok) and adapter.mem_token_embeds is not None:
            with torch.no_grad():
                adapter.mem_token_embeds.copy_(mem_tok.to(dtype=adapter.mem_token_embeds.dtype))
        if mem_n > 0 and adapter.proj_mem is not None and proj_mem:
            adapter.proj_mem.load_state_dict(proj_mem, strict=False)
        adapter = adapter.to(device=next(llama_model.parameters()).device, dtype=next(llama_model.parameters()).dtype)
        for p in adapter.parameters():
            p.requires_grad = False
        setattr(llama_model, "geom_stepslot_adapter", adapter)
        try:
            base = getattr(llama_model, "base_model", None)
            base_model = getattr(base, "model", None) if base is not None else None
            if base_model is not None:
                setattr(base_model, "geom_stepslot_adapter", adapter)
        except Exception:
            pass
        stats["mem_tokens"] = int(mem_n)
    except Exception:
        pass

    # Copy step token rows by token name (source row order in Vigor ckpt tail).
    try:
        src_emb = sd.get("llm.model.model.embed_tokens.weight", None)
        if torch.is_tensor(src_emb) and src_emb.dim() == 2:
            step_tokens = getattr(llama_tokenizer, "step_tokens", None) or []
            global_tok = getattr(llama_tokenizer, "step_global_token", "<cls>")
            stop_tok = getattr(llama_tokenizer, "stop_token", "<STOP>")
            src_tokens = list(step_tokens) + [str(global_tok), str(stop_tok)]
            n_tail = int(len(src_tokens))
            if n_tail > 0 and int(src_emb.size(0)) >= n_tail:
                src_base = int(src_emb.size(0)) - n_tail
                emb = llama_model.get_input_embeddings()
                head = llama_model.get_output_embeddings()
                if emb is not None and hasattr(emb, "weight"):
                    with torch.no_grad():
                        for idx, tok in enumerate(src_tokens):
                            ids = llama_tokenizer(tok, add_special_tokens=False)["input_ids"]
                            if not isinstance(ids, list) or len(ids) != 1:
                                continue
                            tid = int(ids[0])
                            sid = int(src_base + idx)
                            if sid < 0 or sid >= int(src_emb.size(0)):
                                continue
                            if tid < 0 or tid >= int(emb.weight.size(0)):
                                continue
                            emb.weight[tid].copy_(src_emb[sid].to(device=emb.weight.device, dtype=emb.weight.dtype))
                            if head is not None and hasattr(head, "weight") and tid < int(head.weight.size(0)):
                                head.weight[tid].copy_(src_emb[sid].to(device=head.weight.device, dtype=head.weight.dtype))
                            stats["step_rows_loaded"] += 1
    except Exception:
        pass

    # Attach a dedicated geom LoRA adapter (r=8) for last-4 q/v blocks.
    lora_slots = _collect_geom_lora_from_vigor(sd)
    stats["lora_slots"] = int(len(lora_slots))
    if len(lora_slots) <= 0:
        return stats
    if not hasattr(llama_model, "add_adapter"):
        return stats

    try:
        geom_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llama_model.add_adapter(str(adapter_name), geom_cfg)
    except Exception:
        return stats

    loaded = 0
    for name, module in llama_model.named_modules():
        m = _BASE_LORA_LAYER_RE.search(str(name))
        if m is None:
            continue
        layer = int(m.group(1))
        target = str(m.group(2))
        slot = lora_slots.get((layer, target), None)
        if slot is None:
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        try:
            if str(adapter_name) not in module.lora_A or str(adapter_name) not in module.lora_B:
                continue
            w_a = slot.get("A", None)
            w_b = slot.get("B", None)
            if (not torch.is_tensor(w_a)) or (not torch.is_tensor(w_b)):
                continue
            with torch.no_grad():
                module.lora_A[str(adapter_name)].weight.copy_(
                    w_a.to(device=module.lora_A[str(adapter_name)].weight.device, dtype=module.lora_A[str(adapter_name)].weight.dtype)
                )
                module.lora_B[str(adapter_name)].weight.copy_(
                    w_b.to(device=module.lora_B[str(adapter_name)].weight.device, dtype=module.lora_B[str(adapter_name)].weight.dtype)
                )
            loaded += 1
        except Exception:
            continue
    stats["lora_loaded"] = int(loaded)
    return stats


def load_llama_model_and_tokenizer(llama_config):
    llama_config.vicuna_weight_path = glob.glob(
        f"{llama_config.root_path}{llama_config.vicuna_version}/*pytorch_model*.bin")
    if llama_config.load_pretrain_weight and len(llama_config.vicuna_weight_path) == 0:
        print(
            "[LLM][warn] No Vicuna weights found under "
            f"{llama_config.root_path}{llama_config.vicuna_version}/; "
            "LLM will run with random weights."
        )

    llama_config.llama_dim = llama_config.hidden_size
    llama_config.tokenizer_path = f"{llama_config.root_path}{llama_config.vicuna_version}/"
    llama_config.model_path = f"{llama_config.root_path}{llama_config.vicuna_version}"

    # prepare llama tokenizer(load & and special tokens)
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_config.tokenizer_path,
                                                     use_fast=False,
                                                     legacy=False)

    added_grounding = llama_tokenizer.add_tokens(
        [
            llama_config.ref_token,
            llama_config.gs_token,
            llama_config.ge_token,
            llama_config.inref,
        ],
        special_tokens=True,
    )
    # SSR3DLLM: special control token to hint that
    # geometric reasoning / grounding is required.
    geom_token = "<geom>"
    enable_geom_token = os.environ.get("SSR3DLLM_ADD_GEOM_TOKEN", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    added_geom = 0
    if enable_geom_token:
        added_geom = llama_tokenizer.add_tokens([geom_token], special_tokens=True)

    # SSR3DLLM: optional step tokens for chain-style grounding.
    # When enabled, the LLM is supervised to generate:
    #   <step1> ... <step2> ... <step3> ... <step4> ...
    # and downstream modules can read hidden states at these positions.
    enable_step_tokens = os.environ.get("SSR3DLLM_STEP_TOKENS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    step_order_len = 4
    step_global_token = "<cls>"
    enable_stop_token = os.environ.get("SSR3DLLM_ENABLE_STOP_TOKEN", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    stop_token = os.environ.get("SSR3DLLM_STOP_TOKEN", "<STOP>").strip() or "<STOP>"
    added_steps = 0
    if enable_step_tokens:
        raw = os.environ.get("SSR3DLLM_STEP_ORDER_LEN", "").strip()
        if raw:
            try:
                step_order_len = max(1, int(raw))
            except Exception:
                step_order_len = 4
        step_tokens = [f"<step{i+1}>" for i in range(step_order_len)]
        # Also add a global marker used by llama-step-slot (optional but harmless).
        extra = step_tokens + [step_global_token]
        if enable_stop_token:
            extra = extra + [stop_token]
        added_steps = llama_tokenizer.add_tokens(extra, special_tokens=True)

    if str(os.environ.get("SSR3DLLM_TOKENIZER_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            logger.warning(
                "[LLM][tokenizer] len={} added_grounding={} added_geom={} added_steps={} enable_geom={} enable_steps={}",
                len(llama_tokenizer),
                int(added_grounding),
                int(added_geom),
                int(added_steps),
                bool(enable_geom_token),
                bool(enable_step_tokens),
            )
        except Exception:
            pass

    llama_tokenizer.ref_token_id = llama_tokenizer(
        llama_config.ref_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.gs_token_id = llama_tokenizer(
        llama_config.gs_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.ge_token_id = llama_tokenizer(
        llama_config.ge_token, add_special_tokens=False)['input_ids'][0]
    llama_tokenizer.inref_token_id = llama_tokenizer(
        llama_config.inref, add_special_tokens=False)['input_ids'][0]
    if enable_geom_token:
        llama_tokenizer.geom_token_id = llama_tokenizer(
            geom_token, add_special_tokens=False
        )["input_ids"][0]
    else:
        llama_tokenizer.geom_token_id = -1
    if enable_step_tokens:
        try:
            llama_tokenizer.step_token_ids = [
                llama_tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in step_tokens
            ]
            llama_tokenizer.step_tokens = step_tokens
            llama_tokenizer.step_global_token = step_global_token
            llama_tokenizer.step_global_token_id = llama_tokenizer(
                step_global_token, add_special_tokens=False
            )["input_ids"][0]
            if enable_stop_token:
                llama_tokenizer.stop_token = stop_token
                llama_tokenizer.stop_token_id = llama_tokenizer(
                    stop_token, add_special_tokens=False
                )["input_ids"][0]
        except Exception:
            llama_tokenizer.step_token_ids = []
            llama_tokenizer.step_tokens = []
            llama_tokenizer.step_global_token = step_global_token
            llama_tokenizer.step_global_token_id = -1
            if enable_stop_token:
                llama_tokenizer.stop_token = stop_token
                llama_tokenizer.stop_token_id = -1

    # IMPORTANT: `extra_token_head` in `models/LLM/modeling_llama.py` historically
    # assumes the special grounding tokens occupy the last 4 vocab positions.
    # Once we append additional tokens (e.g. `<geom>`), that assumption no longer
    # holds and `<ref>` can become unreachable during generation. Store the
    # explicit ids so the LM forward can overwrite the correct logits.
    llama_config.extra_token_ids = [
        int(llama_tokenizer.ref_token_id),
        int(llama_tokenizer.gs_token_id),
        int(llama_tokenizer.ge_token_id),
        int(llama_tokenizer.inref_token_id),
    ]
    llama_tokenizer.ref_token = "<ref>"
    llama_tokenizer.gs_token = "<p>"
    llama_tokenizer.ge_token = "</p>"
    llama_tokenizer.inref = "<inref>"  # there is a table ==> there is <inref>
    llama_tokenizer.geom_token = geom_token  # trigger SSR3DLLM geom reasoning (may be disabled via env)
    if enable_step_tokens:
        llama_tokenizer.step_order_len = int(step_order_len)
        if enable_stop_token:
            llama_tokenizer.stop_token = stop_token

    # init llama model
    llama_model = LLama3dForCausalLM(config=llama_config,
                                     llama_tokenizer=llama_tokenizer,
                                     gradient_checkpointing=True)

    # prepare vicuna weight
    if llama_config.load_pretrain_weight:
        vicuna_weight = {}
        assert not llama_config.vicuna_weight_path[0].split(
            ".")[-1] == "safetensor", "currently we only support torch.bin file"
        for path in llama_config.vicuna_weight_path:
            weights = torch.load(path, map_location=torch.device('cpu'))
            vicuna_weight.update(weights)
        llama_model.load_state_dict(vicuna_weight, strict=False)

    llama_model.model.wte = llama_model.resize_token_embeddings(
        len(llama_tokenizer))
    # =============== apply lora ===========================

    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                and all(
                    [
                        x not in name
                        for x in [
                            "instance2embed",
                            "hidden_state2query"
                        ]
                    ]
                )
                and any([x in name for x in lora_target_modules])
            ):
                lora_module_names.add(name)
                # print(f"add lora to {name}")
        return sorted(list(lora_module_names))

    # froze model
    lora_target_modules = find_linear_layers(
        llama_model, llama_config.lora_target_modules)
    peft_config = LoraConfig(
        r=llama_config.lora_r,
        lora_alpha=llama_config.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=llama_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llama_model = get_peft_model(llama_model, peft_config)
    # Keep a back-reference so methods on the base model can switch PEFT adapters at runtime.
    try:
        base = getattr(llama_model, "base_model", None)
        base_model = getattr(base, "model", None) if base is not None else None
        if base_model is not None:
            base_model._ssr3dllm_peft_parent_ref = weakref.ref(llama_model)
    except Exception:
        pass
    llama_model.print_trainable_parameters()
    for name, param in llama_model.named_parameters():
        if any(config_item in name for config_item in llama_config.train_layer_list):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optional: freeze the entire LLM except the <stepK> token rows in embeddings (+ lm_head).
    # This is the "STAMP-style" constraint for step-token experiments:
    # - LM loss can still teach the model to emit <stepK>
    # - Geometry losses (if any) can be routed to step tokens only
    freeze_except_step = os.environ.get("SSR3DLLM_FREEZE_LLM_EXCEPT_STEP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_step_tokens and freeze_except_step:
        try:
            step_ids = getattr(llama_tokenizer, "step_token_ids", None) or []
            step_ids = [int(x) for x in step_ids if int(x) >= 0]
            global_id = int(getattr(llama_tokenizer, "step_global_token_id", -1))
            if global_id >= 0:
                step_ids = step_ids + [global_id]
            if step_ids:
                for p in llama_model.parameters():
                    p.requires_grad = False

                emb = llama_model.get_input_embeddings()
                head = llama_model.get_output_embeddings()
                if emb is not None and hasattr(emb, "weight") and head is not None and hasattr(head, "weight"):
                    emb.weight.requires_grad = True
                    head.weight.requires_grad = True

                    step_ids_t = torch.as_tensor(step_ids, dtype=torch.long, device=emb.weight.device)
                    row_mask = torch.zeros((emb.weight.size(0), 1), dtype=emb.weight.dtype, device=emb.weight.device)
                    row_mask[step_ids_t, 0] = 1.0

                    def _mask_grad(grad):
                        try:
                            return grad * row_mask
                        except Exception:
                            return grad

                    emb.weight.register_hook(_mask_grad)
                    head.weight.register_hook(_mask_grad)
                    print(f"[LLM][step_tokens] freeze LLM except step rows: ids={step_ids}", flush=True)
        except Exception:
            pass

    # Optional: train ONLY the <stepK> token rows (and optional <cls>) in the embedding table (+ lm_head),
    # while keeping other trainable components (e.g., LoRA) intact.
    #
    # This is useful for "gradient isolation" setups:
    # - dialog/QA losses update LoRA
    # - geometry losses update listener + (<stepK> rows / stepslot adapter)
    #
    # Enable with:
    #   SSR3DLLM_STEP_TOKENS=1
    #   SSR3DLLM_TRAIN_STEP_ROWS=1
    train_step_rows = os.environ.get("SSR3DLLM_TRAIN_STEP_ROWS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_step_tokens and train_step_rows:
        try:
            step_ids = getattr(llama_tokenizer, "step_token_ids", None) or []
            step_ids = [int(x) for x in step_ids if int(x) >= 0]
            global_id = int(getattr(llama_tokenizer, "step_global_token_id", -1))
            if global_id >= 0:
                step_ids = step_ids + [global_id]
            step_ids = sorted(set(step_ids))

            emb = llama_model.get_input_embeddings()
            head = llama_model.get_output_embeddings()
            if emb is not None and hasattr(emb, "weight") and head is not None and hasattr(head, "weight") and step_ids:
                emb.weight.requires_grad = True
                head.weight.requires_grad = True

                # Store row-ids as non-persistent buffers so they move with the module device.
                ids_t = torch.as_tensor(step_ids, dtype=torch.long)
                if not hasattr(emb, "_ssr3dllm_step_row_ids"):
                    emb.register_buffer("_ssr3dllm_step_row_ids", ids_t, persistent=False)
                else:
                    emb._ssr3dllm_step_row_ids = ids_t
                if not hasattr(head, "_ssr3dllm_step_row_ids"):
                    head.register_buffer("_ssr3dllm_step_row_ids", ids_t, persistent=False)
                else:
                    head._ssr3dllm_step_row_ids = ids_t

                if not getattr(llama_model, "_ssr3dllm_step_rows_hooked", False):
                    llama_model._ssr3dllm_step_rows_hooked = True

                    def _mask_step_rows(grad, holder):
                        try:
                            row_ids = getattr(holder, "_ssr3dllm_step_row_ids", None)
                            if row_ids is None or grad is None:
                                return grad
                            row_ids = row_ids.to(device=grad.device)
                            row_mask = grad.new_zeros((grad.size(0), 1))
                            row_mask[row_ids, 0] = 1.0
                            return grad * row_mask
                        except Exception:
                            return grad

                    emb.weight.register_hook(lambda g, h=emb: _mask_step_rows(g, h))
                    head.weight.register_hook(lambda g, h=head: _mask_step_rows(g, h))
                    print(f"[LLM][step_tokens] train step rows only: ids={step_ids}", flush=True)
        except Exception:
            pass

    # Optional: load a compact step-slot adapter exported from Vigor llama-step-slot training.
    # This is an opt-in initializer so SSR3DLLM step3 can reuse the learned:
    # - <stepK> embedding rows
    # - soft memory tokens (if any) + projection layers
    adapter_path = os.environ.get("SSR3DLLM_LLM_STEPSLOT_ADAPTER", "").strip()
    if adapter_path:
        try:
            from .stepslot_adapter import SoftStepSlotAdapter

            adapter, export = SoftStepSlotAdapter.from_export(adapter_path, out_dim=768)
            trainable = os.environ.get("SSR3DLLM_LLM_STEPSLOT_ADAPTER_TRAINABLE", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            adapter = adapter.to(device=next(llama_model.parameters()).device, dtype=next(llama_model.parameters()).dtype)
            for p in adapter.parameters():
                p.requires_grad = bool(trainable)
            llama_model.stepslot_adapter = adapter
            # Also attach to the underlying base model so the adapter is visible inside
            # `LLama3dForCausalLM.forward/model_forward` when called through PeftModel.
            try:
                base = getattr(llama_model, "base_model", None)
                base_model = getattr(base, "model", None) if base is not None else None
                if base_model is not None:
                    base_model.stepslot_adapter = adapter
            except Exception:
                pass

            # Apply <stepK> embedding rows into the LLM embedding table (when step tokens are enabled).
            # This is useful when the adapter export is the *source of truth* for the step token rows,
            # but can be undesirable when loading a Lightning ckpt that already learned its own step rows.
            apply_step_token_embeds = os.environ.get("SSR3DLLM_LLM_STEPSLOT_APPLY_TOKEN_EMBEDS", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if enable_step_tokens and apply_step_token_embeds:
                try:
                    emb = llama_model.get_input_embeddings()
                    if emb is not None and hasattr(emb, "weight"):
                        with torch.no_grad():
                            for tok in list(export.step_tokens) + [str(export.global_token)]:
                                ids = llama_tokenizer(tok, add_special_tokens=False)["input_ids"]
                                if not isinstance(ids, list) or len(ids) != 1:
                                    continue
                                tid = int(ids[0])
                                vec = export.token_embeds.get(tok, None)
                                if vec is None:
                                    continue
                                if vec.numel() != int(emb.weight.size(1)):
                                    continue
                                if 0 <= tid < int(emb.weight.size(0)):
                                    emb.weight[tid].copy_(vec.to(device=emb.weight.device, dtype=emb.weight.dtype))
                    print(
                        f"[LLM][stepslot_adapter] loaded={adapter_path} trainable={int(bool(trainable))} "
                        f"mem_tokens={int(getattr(adapter, 'mem_tokens', 0))}",
                        flush=True,
                    )
                except Exception:
                    print(f"[LLM][stepslot_adapter][warn] failed to apply step token embedding rows from {adapter_path}", flush=True)
            elif enable_step_tokens and not apply_step_token_embeds:
                print(
                    f"[LLM][stepslot_adapter] loaded={adapter_path} trainable={int(bool(trainable))} "
                    f"mem_tokens={int(getattr(adapter, 'mem_tokens', 0))} apply_token_embeds=0",
                    flush=True,
                )
            else:
                print(
                    "[LLM][stepslot_adapter][warn] adapter provided but SSR3DLLM_STEP_TOKENS is disabled; "
                    "skipping step token embedding init.",
                    flush=True,
                )
        except Exception:
            print(f"[LLM][stepslot_adapter][warn] failed to load adapter: {adapter_path}", flush=True)

    # Optional: load 503 geom routing resources (dedicated LoRA + step-slot projections)
    # directly from a Vigor llama-step-slot checkpoint.
    geom_ckpt = (
        os.environ.get("SSR3DLLM_GEOM_STEPSLOT_CKPT", "").strip()
        or os.environ.get("SSR3DLLM_REFERIT3D_LISTENER_CKPT", "").strip()
    )
    geom_enable = os.environ.get("SSR3DLLM_GEOM_LORA_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    geom_adapter_name = os.environ.get("SSR3DLLM_GEOM_LORA_ADAPTER_NAME", "geom503").strip() or "geom503"
    if geom_enable and geom_ckpt:
        try:
            stats = _attach_geom_adapter_from_vigor_ckpt(
                llama_model=llama_model,
                llama_tokenizer=llama_tokenizer,
                geom_ckpt_path=geom_ckpt,
                adapter_name=geom_adapter_name,
            )
            try:
                base = getattr(llama_model, "base_model", None)
                base_model = getattr(base, "model", None) if base is not None else None
                if base_model is not None:
                    base_model._ssr3dllm_geom_adapter_name = str(geom_adapter_name)
            except Exception:
                pass
            print(
                "[LLM][geom_adapter] "
                f"ckpt={geom_ckpt} adapter={geom_adapter_name} "
                f"lora_slots={int(stats.get('lora_slots', 0))} "
                f"lora_loaded={int(stats.get('lora_loaded', 0))} "
                f"step_rows_loaded={int(stats.get('step_rows_loaded', 0))} "
                f"mem_tokens={int(stats.get('mem_tokens', 0))}",
                flush=True,
            )
        except Exception as e:
            print(f"[LLM][geom_adapter][warn] failed to load from {geom_ckpt}: {type(e).__name__}: {e}", flush=True)
    llama_model = llama_model.bfloat16()
    # print_grad_status(llama_model)
    if llama_config.use_checkpoint:
        llama_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return llama_model, llama_tokenizer

class LLama3dMetaModel:
    def __init__(
        self,
        config,
        **kwargs,  # placeholder for extra params can NOT be removed
    ):
        super(LLama3dMetaModel, self).__init__(config)
        # config & initialize model
        self.initialize_LLama3d_modules(config)
        self.config = config

        # projection layers
        # project instance to llm embed
        self.instance2embed = nn.Sequential(
            nn.Linear(self.instance_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.llama_dim),
        )
        # project last hidden state to instance query
        self.hidden_state2query = nn.Sequential(
            nn.Linear(self.llama_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.instance_dim)
        )
        self.vision_prompt_projection = nn.Sequential(
            nn.Linear(self.instance_dim, self.llama_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.llama_dim, self.llama_dim),
        )

    def initialize_LLama3d_modules(self, config):
        self.llama_dim = config.llama_dim
        self.instance_dim = config.instance_dim


class LLama3dModel(LLama3dMetaModel, LlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LLama3dModel, self).__init__(config, **kwargs)


class LLama3dForCausalLM(LlamaForCausalLM):
    def __init__(
        self,
        config,
        sample_rate=1.0,  # to avoid OOM error
        subsample=True,
        llama_tokenizer=None,
        prompts=None,
        t=0.1,
        **kwargs,
    ):
        super().__init__(config)

        # reduce batch size to avoid OOM
        self.sample_rate = sample_rate
        self.subsample = subsample

        # truncation to avoid OOM
        self.do_truncation = config.do_truncation
        self.truncation_length = config.truncation_length

        if llama_tokenizer is not None:
            self.llama_tokenizer = llama_tokenizer
        else:
            raise NotImplementedError

        # temperature for contrastive learning
        self.t = t

        self.use_single_ref_token = config.use_single_ref_token
        self.config = config

        # param for focal loss
        self.eps = 1e-12
        self.gamma = 2.0
        self.alpha = 0.25

        # config special tokens
        self.pad_token_id = llama_tokenizer.eos_token_id
        self.ref_token_id = llama_tokenizer.ref_token_id
        self.gs_token_id = llama_tokenizer.gs_token_id  # grounding start token
        self.ge_token_id = llama_tokenizer.ge_token_id  # grounding end token
        self.eos_token_id = llama_tokenizer.eos_token_id  # end of sentence token

        def convert_text_to_ids(text): return self.llama_tokenizer.convert_tokens_to_ids(
            self.llama_tokenizer.tokenize(text))
        self.model = LLama3dModel(config, **kwargs)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        if prompts is not None:
            self.prompts = prompts
        else:
            print("Warning: use default prompt")
            # "lan":"<s> SYSTEM: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
            system_prompt = "<s> SYSTEM: A chat between a curious user and a 3D AI assistant. The assistant gives helpful and polite answers to the user's questions."
            self.prompts = {
                "system_prompt": {
                    "lan": system_prompt,
                    "ids": torch.tensor(convert_text_to_ids(system_prompt))
                },
                "rules": {"user": {
                    "lan": "<s> USER:",
                    "ids": torch.tensor(convert_text_to_ids("<s> USER:"))},
                    "assistant": {
                        "lan": "ASSISTANT:",
                        "ids": torch.tensor(convert_text_to_ids("ASSISTANT:"))}
                },
                "QA_grounding": "",  # postfix
            }
        self.beam_size = config.beam_size
        # Initialize weights and apply final processing
        self.post_init()
        self.hnsw_runtime_config = {
            "enabled": False,
            "M": 16,
            "ef_construction": 200,
            "ef_search": 200,
            "top_k": 50,
            "candidate_limit": 0,
            "keep_base": True,
        }
        self._reset_match_runtime_stats()

    def _reset_match_runtime_stats(self) -> None:
        self._match_runtime_stats = {
            "mode": "disabled",
            "elapsed": 0.0,
            "calls": 0,
            "queries": 0,
            "candidates": 0,
        }

    def _accumulate_match_runtime(
        self,
        mode: str,
        elapsed_seconds: float,
        num_queries: int,
        num_candidates: int,
    ) -> None:
        stats = getattr(self, "_match_runtime_stats", None)
        if stats is None:
            self._reset_match_runtime_stats()
            stats = self._match_runtime_stats
        stats["mode"] = mode
        stats["elapsed"] = stats.get("elapsed", 0.0) + float(elapsed_seconds)
        stats["calls"] = stats.get("calls", 0) + 1
        stats["queries"] = stats.get("queries", 0) + int(num_queries)
        stats["candidates"] = stats.get("candidates", 0) + int(num_candidates)

    def get_match_runtime_stats(self) -> Dict[str, object]:
        stats = getattr(self, "_match_runtime_stats", None)
        if not stats:
            return {
                "mode": "unknown",
                "elapsed_ms": 0.0,
                "calls": 0,
                "queries": 0,
                "candidates": 0,
            }
        return {
            "mode": stats.get("mode", "unknown"),
            "elapsed_ms": float(stats.get("elapsed", 0.0)) * 1000.0,
            "calls": int(stats.get("calls", 0)),
            "queries": int(stats.get("queries", 0)),
            "candidates": int(stats.get("candidates", 0)),
        }

    def _merge_input_ids_with_instance_features(self,  batch, inference=False):
        '''
        return: 
        if inference:
            final_embedding: torch bfloat16 tensor shape = [batch_size, max_length_input, llama dim]
            final_attention_mask: torch bool tensor shape = [batch_size, max_length_input]
        else:
            final_embedding
            final_attention_mask
            final_labels
        '''
        batch_size = len(batch)
        visual_embeddings_list = []
        language_embeddings_list = []
        # step 1: process visual and text embedding features separately
        eos_token_embeds = self.model.embed_tokens(torch.tensor(
            [self.eos_token_id], device=self.device)).to(dtype=torch.bfloat16)
        for instance in batch:
            instance: grounded_3d_llm_data
            visual, lan, label = instance.instance_feature, instance.input_ids, instance.output_ids
            # Visual instance features are expected to be a 2D tensor:
            # [num_queries, instance_dim]. We no longer assume a fixed
            # number of queries (100); instead we only check the feature
            # dimension to stay compatible with different Step2 settings
            # (e.g., 100 / 150 queries).
            assert visual.ndim == 2 and visual.shape[1] == self.config.instance_dim, (
                f"instance_feature should have shape (num_queries, {self.config.instance_dim}), "
                f"got {tuple(visual.shape)}"
            )
            visual_embeddings_list.append(
                self.model.instance2embed(visual.to(dtype=torch.bfloat16)))
            visual_embeddings_list[-1] = torch.cat(
                (visual_embeddings_list[-1], eos_token_embeds), dim=0)
            language_embeddings_list.append(
                self.model.embed_tokens(lan).to(dtype=torch.bfloat16))
            # step 2: add visual token to text embeddings
            if instance.input_referent_mask.any():
                instance.input_referent = []
                find_inref = 0
                try:
                    for question_quries_id in instance.question_gt_query_ids:
                        # shuffle within one phrase
                        question_quries_id = np.array(question_quries_id)
                        np.random.shuffle(question_quries_id)
                        instance.input_referent.append(self.model.vision_prompt_projection(
                            visual[question_quries_id, :].to(dtype=torch.bfloat16)))
                        while find_inref < instance.input_referent_mask.shape[0] and instance.input_referent_mask[find_inref] != True:
                            find_inref += 1
                        assert find_inref != instance.input_referent_mask.shape[0] - \
                            1, "can not find a corresponding position for insertion"
                        assert instance.input_referent_mask[find_inref+1:find_inref+1+len(question_quries_id)].all(
                        ), "length of interval do not match number of input referent"
                        # <inref> embed ... embed <inref>
                        language_embeddings_list[-1][find_inref+1:find_inref+1 +
                                                     len(question_quries_id)] = instance.input_referent[-1]
                        find_inref += len(question_quries_id)+2
                except Exception as e:
                    print(e)
                    continue
            if not inference:
                language_embeddings_list[-1] = torch.cat(
                    (language_embeddings_list[-1], self.model.embed_tokens(label).to(dtype=torch.bfloat16)), dim=0)
        # step 3: assemble embeddings and padding
        embed_list = []
        max_length_input = 0
        embed_length = []
        for visual, lan in zip(visual_embeddings_list, language_embeddings_list):
            embed_list.append(torch.cat((visual, lan), dim=0))
            max_length_input = max(embed_list[-1].shape[0], max_length_input)
            embed_length.append(embed_list[-1].shape[0])

        pad_token_embeds = self.model.embed_tokens(torch.tensor(
            [self.pad_token_id], device=self.device)).to(dtype=torch.bfloat16)
        final_embedding = []
        final_attention_mask = torch.zeros(
            (batch_size, max_length_input), device=self.device, dtype=bool)
        for idx, embed in enumerate(embed_list):
            final_embedding.append(torch.cat(
                (embed, pad_token_embeds.repeat(max_length_input-embed.shape[0], 1)), dim=0))
            final_attention_mask[idx, :embed.shape[0]] = True
        final_embedding = torch.stack(final_embedding)
        if inference:
            return final_embedding.to(self.device, dtype=torch.bfloat16), final_attention_mask.to(self.device)
        # step 4: pad labels
        label_list = []
        for instance, end in zip(batch, embed_length):
            lan = instance.output_ids
            label = (torch.ones(max_length_input)*-100).type(torch.LongTensor)
            label[end-lan.shape[0]:end] = lan
            label_list.append(label)
        final_labels = torch.stack(label_list)
        assert final_embedding.shape[:2] == final_labels.shape
        return final_embedding.to(self.device, dtype=torch.bfloat16), final_attention_mask.to(self.device), final_labels.to(self.device).type(torch.LongTensor)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:  # for sequential generation
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def encode_stepslot_only(self, *, batch_lang_infos: List[object]) -> None:
        """
        SSR3DLLM helper: (re)compute ONLY the Vigor-style step-slot embeddings for a batch.

        This function:
        - reads teacher-forced `rel_referential_order` from each lang_info
        - runs the lightweight phrase-only encodings:
            phrase_k <stepk>   (k=1..order_len)
          and stores `lang_info.llm_step_embeds` (shape [order_len, inner_dim])
        - optionally exports `lang_info.llm_lang_embeds` using soft memory tokens

        It DOES NOT run the full LLM teacher-forcing loss / generation forward.
        """

        adapter = getattr(self, "stepslot_adapter", None)
        if adapter is None:
            raise RuntimeError("encode_stepslot_only requires a loaded stepslot_adapter (SSR3DLLM_LLM_STEPSLOT_ADAPTER).")

        mode = str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_MODE", "phrase_only")).strip().lower()
        if mode != "phrase_only":
            raise RuntimeError(f"encode_stepslot_only currently supports only phrase_only mode, got: {mode}")

        if not hasattr(self, "llama_tokenizer") or self.llama_tokenizer is None:
            raise RuntimeError("encode_stepslot_only requires self.llama_tokenizer to be set.")

        step_ids = getattr(self.llama_tokenizer, "step_token_ids", None) or []
        order_len = int(len(step_ids))
        if order_len <= 0:
            raise RuntimeError("encode_stepslot_only requires SSR3DLLM_STEP_TOKENS=1 (missing step_token_ids).")

        max_len = int(str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_MAX_LEN", "64")).strip() or "64")
        if max_len <= 0:
            max_len = 64

        export_lang_embeds = os.environ.get("SSR3DLLM_LLM_STEPSLOT_EXPORT_LANG_EMBEDS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        def _strip_geom(text: str) -> str:
            return str(text).replace("<geom>", " ").replace("  ", " ").strip()

        # -------- Step embeddings --------
        step_texts_flat: List[str] = []
        owners: List[object] = []
        for li in batch_lang_infos:
            order = getattr(li, "rel_referential_order", None)
            if not isinstance(order, list) or not order:
                continue
            phrases = [str(x).strip() for x in order if str(x).strip()]
            if not phrases:
                continue
            if len(phrases) == 1:
                phrases = phrases * order_len
            elif len(phrases) < order_len:
                phrases = phrases + [phrases[-1]] * (order_len - len(phrases))
            else:
                phrases = phrases[:order_len]
            for k in range(order_len):
                step_texts_flat.append(f"{phrases[k]} <step{k+1}>".strip())
            owners.append(li)

        if owners:
            enc = self.llama_tokenizer(
                step_texts_flat,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_len),
            )
            device = next(self.model.parameters()).device
            enc = {k: v.to(device=device) for k, v in enc.items()}

            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                out_step = self.model(**enc, output_hidden_states=True, return_dict=True)
            hs = out_step.hidden_states[-1]
            attn = enc.get("attention_mask", None)
            if attn is None:
                raise RuntimeError("tokenizer must return attention_mask for stepslot encoding")
            last_pos = attn.long().sum(dim=-1) - 1
            idx = torch.arange(hs.size(0), device=hs.device)
            step_h = hs[idx, last_pos]

            step_proj = adapter.proj_step(step_h.to(dtype=adapter.proj_step[0].weight.dtype))
            step_proj = step_proj.reshape(len(owners), order_len, -1)
            for li, embeds in zip(owners, step_proj):
                setattr(li, "llm_step_embeds", embeds)

        # -------- Global/lang embeddings (soft memory tokens) --------
        if export_lang_embeds and int(getattr(adapter, "mem_tokens", 0) or 0) > 0:
            if getattr(adapter, "mem_token_embeds", None) is None or getattr(adapter, "proj_mem", None) is None:
                raise RuntimeError("stepslot_adapter missing mem_token_embeds/proj_mem (mem_tokens > 0)")

            mem_n = int(getattr(adapter, "mem_tokens", 0))
            text_max = max(1, int(max_len) - int(mem_n))

            utterances: List[str] = []
            owners_u: List[object] = []
            for li in batch_lang_infos:
                if not hasattr(li, "rel_referential_order"):
                    continue
                u = (
                    getattr(li, "rel_distill_text", None)
                    or getattr(li, "raw_grounding_text", None)
                    or getattr(li, "question", None)
                    or ""
                )
                u = _strip_geom(str(u))
                if not u:
                    continue
                utterances.append(u)
                owners_u.append(li)

            if utterances:
                enc_u = self.llama_tokenizer(
                    utterances,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(text_max),
                )
                device = next(self.model.parameters()).device
                enc_u = {k: v.to(device=device) for k, v in enc_u.items()}
                input_ids = enc_u.get("input_ids", None)
                attn_u = enc_u.get("attention_mask", None)
                if input_ids is None or attn_u is None:
                    raise RuntimeError("tokenizer must return input_ids and attention_mask for utterances")

                base_emb = self.model.embed_tokens(input_ids)
                mem = adapter.mem_token_embeds.to(device=base_emb.device, dtype=base_emb.dtype).unsqueeze(0).expand(
                    base_emb.size(0), mem_n, -1
                )
                inputs_embeds = torch.cat([base_emb, mem], dim=1)
                attn_mem = torch.ones((base_emb.size(0), mem_n), device=attn_u.device, dtype=attn_u.dtype)
                attn_u = torch.cat([attn_u, attn_mem], dim=1)

                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    out_u = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_u,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hs_u = out_u.hidden_states[-1][:, -mem_n:, :]
                mem_proj = adapter.proj_mem(hs_u.to(dtype=adapter.proj_mem[0].weight.dtype))
                for li, m in zip(owners_u, mem_proj):
                    setattr(li, "llm_lang_embeds", m)

    def encode_stepslot_onepass_pred(
        self,
        *,
        utterances: List[str],
        adapter: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute one-pass pred-mode step embeddings:
          "{utterance}\\n<step1>\\n...\\n<stepK>"
        and read hidden states at each <stepK> token.
        """
        if not isinstance(utterances, list) or len(utterances) <= 0:
            raise RuntimeError("encode_stepslot_onepass_pred requires a non-empty utterance list.")
        if adapter is None:
            adapter = getattr(self, "geom_stepslot_adapter", None) or getattr(self, "stepslot_adapter", None)
        if adapter is None:
            raise RuntimeError("encode_stepslot_onepass_pred requires geom_stepslot_adapter or stepslot_adapter.")
        if not hasattr(self, "llama_tokenizer") or self.llama_tokenizer is None:
            raise RuntimeError("encode_stepslot_onepass_pred requires self.llama_tokenizer.")

        step_ids = getattr(self.llama_tokenizer, "step_token_ids", None) or []
        order_len = int(len(step_ids))
        if order_len <= 0:
            raise RuntimeError("encode_stepslot_onepass_pred requires SSR3DLLM_STEP_TOKENS=1.")

        max_len = int(str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_MAX_LEN", "128")).strip() or "128")
        if max_len <= 0:
            max_len = 128

        texts = []
        for u in utterances:
            uu = str(u or "").replace("<geom>", " ").replace("  ", " ").strip()
            parts = [uu] if uu else []
            for k in range(order_len):
                parts.append(f"<step{k+1}>")
            texts.append("\n".join(parts).strip())

        enc = self.llama_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_len),
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}
        input_ids = enc.get("input_ids", None)
        attn = enc.get("attention_mask", None)
        if input_ids is None or attn is None:
            raise RuntimeError("tokenizer must return input_ids and attention_mask")

        base_emb = self.model.embed_tokens(input_ids)
        base_len = int(base_emb.size(1))
        inputs_embeds = base_emb
        mem_n = int(getattr(adapter, "mem_tokens", 0) or 0)
        if mem_n > 0:
            if getattr(adapter, "mem_token_embeds", None) is None:
                raise RuntimeError("stepslot adapter mem_tokens>0 but mem_token_embeds is missing.")
            mem = adapter.mem_token_embeds.to(device=base_emb.device, dtype=base_emb.dtype).unsqueeze(0).expand(
                base_emb.size(0), mem_n, -1
            )
            inputs_embeds = torch.cat([base_emb, mem], dim=1)
            attn_mem = torch.ones((base_emb.size(0), mem_n), device=attn.device, dtype=attn.dtype)
            attn = torch.cat([attn, attn_mem], dim=1)

        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )
        hs = out.hidden_states[-1]

        B = int(input_ids.size(0))
        step_h = torch.zeros((B, order_len, hs.size(-1)), device=hs.device, dtype=hs.dtype)
        for i in range(B):
            for k, sid in enumerate([int(x) for x in step_ids]):
                where = (input_ids[i] == int(sid)).nonzero(as_tuple=False).view(-1)
                if where.numel() <= 0:
                    raise RuntimeError(
                        f"Missing <step{k+1}> in one-pass prompt (sample={i}). "
                        f"Increase SSR3DLLM_LLM_STEPSLOT_MAX_LEN (current={max_len})."
                    )
                pos = int(where[-1].item())
                step_h[i, k] = hs[i, pos]

        step_h = step_h.to(dtype=adapter.proj_step[0].weight.dtype)
        order_embeds = adapter.proj_step(step_h)

        lang_embeds = None
        if mem_n > 0:
            if getattr(adapter, "proj_mem", None) is None:
                raise RuntimeError("stepslot adapter mem_tokens>0 but proj_mem is missing.")
            mem_h = hs[:, base_len : base_len + mem_n, :].to(dtype=adapter.proj_mem[0].weight.dtype)
            lang_embeds = adapter.proj_mem(mem_h)

        return order_embeds, lang_embeds

    def calculate_grounding_loss(self, batch, use_single_ref_token):
        test_feature = None
        loss = torch.tensor(0.0)
        sim_list = []
        gt_list = []
        for instance in batch:
            for key in instance.grouped_indices.keys():
                pair = instance.grouped_indices[key]
                pred_ref = pair["query"]
                gt_ids = pair["gt"]
                if use_single_ref_token:  # only collect similarity matrixs here.
                    gt_labels = torch.zeros(
                        instance.instance_embed.shape[0], device=self.device).float()
                    gt_labels[gt_ids] = True

                    features_norm = F.normalize(pred_ref, p=2, dim=1).float()
                    embeddings_norm = F.normalize(
                        instance.instance_embed, p=2, dim=1).float()
                    # num_of_ref_query * num_of_instance
                    cosine_sim = torch.matmul(
                        features_norm, embeddings_norm.T)/self.t
                    sim_list.append(cosine_sim.squeeze())
                    gt_list.append(gt_labels)
                else:  # use multi ref tokens, loss will be accumulated
                    gt_labels = torch.zeros(
                        (pred_ref.shape[0], instance.instance_embed.shape[0]), device=self.device).float()
                    gt_labels[:, gt_ids] = True

                    features_norm = F.normalize(pred_ref, p=2, dim=1).float()
                    embeddings_norm = F.normalize(
                        instance.instance_embed, p=2, dim=1).float()
                    # num_of_ref_query * num_of_instance
                    cosine_sim = torch.matmul(
                        features_norm, embeddings_norm.T)/self.t
                    probs = torch.sigmoid(cosine_sim)
                    neg_cost = -(1 - probs + self.eps).log() * \
                        (1 - self.alpha) * probs.pow(self.gamma)
                    pos_cost = -(probs + self.eps).log() * \
                        self.alpha * (1 - probs).pow(self.gamma)
                    cls_cost = torch.einsum(
                        'nc,mc->nm', pos_cost, gt_labels) + torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
                    try:
                        row_ind, col_ind = linear_sum_assignment(
                            cls_cost.cpu().detach().numpy(), maximize=False)
                    except Exception as e:
                        print("calculate_grounding_loss", e, "\n", features_norm, "\n",
                              "embeddings_norm\n", embeddings_norm, '\n', "you may try to use bfloat16")
                        loss = torch.tensor(0.0)
                        return loss
                    matched_loss = cls_cost[row_ind, col_ind].sum()
                    if len(row_ind) > 0:
                        loss = loss + matched_loss/len(row_ind)

        if use_single_ref_token:  # use the collected similarity matrixs to calculate loss
            if sim_list:
                # NOTE: different instances in a batch may have different numbers
                # of proposals (e.g., when using GT-based proposals). To avoid
                # shape mismatches when stacking, compute focal loss per sample
                # and accumulate with `reduction='sum'`.
                total_loss = torch.tensor(0.0, device=self.device)
                total_pos = torch.tensor(0.0, device=self.device)
                for sim, gt in zip(sim_list, gt_list):
                    # `sim` and `gt` can be 1D or 2D; sigmoid_focal_loss supports both.
                    loss_i = sigmoid_focal_loss(
                        sim, gt, alpha=self.alpha, gamma=self.gamma, reduction='sum'
                    )
                    total_loss = total_loss + loss_i
                    total_pos = total_pos + gt.sum()
                loss = total_loss / (total_pos + 1.0)
        return loss

    def model_forward(self,
                      batch_input_text_list: list,  # input text
                      batch_output_text_list: list,  # output text
                      batch_instance_queries_hidden_state: list,
                      batch_instance_queries_normalized_embed: list,
                      batch_eval_types: list,
                      batch_gt_inst_ids: list,
                      **kwargs
                      ):
        r"""
        Args:
            batch_input_text_list:
                A list that contains all input texts, without any special tokens.

            batch_output_text_list:
                A list that contains all output texts, without any special tokens.

            batch_instance_queries_hidden_state:
                A list that stores the feature representations of all instance queries.

            batch_instance_queries_normalized_embed:
                A list that stores normalized embeddings for retrieval targets.

            batch_eval_types:
                A list containing information about each query, usually formatted as 'dataset_name:extra_information_1(split/idx):extra_information_2(split/idx):...'.
                Note: Some datasets require specific instances as input (using a input referent). The 'eval_type' configuration determines whether to add a input referent. However, even if 'positives_answer' and 'query_ids_answer' are present, suggesting that a input referent can be added, it should not be added.

            batch_gt_inst_ids:
                A list that stores all information relevant for grounding.
                Format: (batch_idx, item, max_gt_iou) if not self.training else (batch_idx, item).
                    item: Must contain the following attributes:
                        - positives_answer: List of tuples [(start1, end1), (start2, end2), ...]
                        - query_ids_answer: List of lists [(query ids for interval 1), ...]
                        - positives_question: List of tuples [(start1, end1), (start2, end2), ...]
                        - query_ids_question: List of lists [(query ids for interval 1), ...]

                        Example:
                        query: "Find all chairs close to the wooden table."
                        item.positives_question == [[25,41]]            ==> query[25:41] == 'the wooden table'
                        item.query_ids_question == [[21]]

                        answer: "OK, the following chairs are what you want."
                        item.positives_answer == [[18,24]]              ==> answer[18:24] == 'chairs'
                        item.query_ids_answer == [[2,3,24,48,12,9,8,13,95]]

                    max_gt_iou: Used only for single object description tests in scan2cap, which assesses the accuracy of bounding boxes.

        Returns:
            {
            "lm_loss": lm_loss,
            "match_loss": match_loss,
            "model_output": model_output,
            **loss_data_type(each type of loss)
            }
        ```"""
        batch = []
        if self.do_truncation:
            num_of_truncated = 0
        for input_text, output_text, instance_feat, instance_embed, eval_type, gt_instance_ids in zip(
            batch_input_text_list,
            batch_output_text_list,
            batch_instance_queries_hidden_state,
            batch_instance_queries_normalized_embed,
            batch_eval_types,
            batch_gt_inst_ids,
        ):
            # During training, batch_gt_inst_ids is a tuple:
            #   (batch_idx, lang_info_data)  (see trainer.training_step)
            # We keep a pointer to the underlying lang_info_data so that
            # LLM-derived context vectors can later be written back to it.
            gt_lang_info: lang_info_data = gt_instance_ids[1]

            answer_grounding_token_pos, answer_gt_query_ids = (
                gt_lang_info.positives_answer,
                gt_lang_info.query_ids_answer,
            )
            question_grounding_token_pos, question_gt_query_ids = (
                gt_lang_info.positives_question,
                gt_lang_info.query_ids_question,
            )

            instance = grounded_3d_llm_data(input_text=input_text,
                                            output_text=output_text,
                                            instance_embed=instance_embed,
                                            instance_feature=instance_feat,
                                            answer_gt_query_ids=answer_gt_query_ids,
                                            answer_grounding_token_pos=answer_grounding_token_pos,
                                            question_gt_query_ids=question_gt_query_ids,
                                            question_grounding_token_pos=question_grounding_token_pos,
                                            eval_type=eval_type,
                                            lang_info=gt_lang_info,
                                            )

            try:
                input_ids, lm_labels, ref_token_mask, input_referent_mask = instance.build_input_from_segments(use_system_prompt=self.config.use_system_prompt,
                                                                                                              tokenizer=self.llama_tokenizer,
                                                                                                              input_text=instance.input_text,
                                                                                                              output_text=instance.output_text,
                                                                                                              prompts=self.prompts,
                                                                                                              use_input_referent=instance.use_input_referent,
                                                                                                              answer_grounding_token_pos=instance.answer_grounding_token_pos,
                                                                                                              question_grounding_token_pos=instance.question_grounding_token_pos,
                                                                                                              use_single_ref_token=self.use_single_ref_token
                                                                                                              )
            except Exception as e:
                print(e)
                continue

            if self.do_truncation:
                if input_ids.shape[0] + lm_labels.shape[0] + instance_feat.shape[0] > self.truncation_length:
                    left_output_length = self.truncation_length - \
                        input_ids.shape[0] - instance_feat.shape[0]
                    if left_output_length <= 0:
                        print(f"Ignore the long QA: ``{eval_type}''.")
                        continue
                    ref_token_mask = ref_token_mask[:left_output_length]
                    # lm label should not end with seg token
                    for i in range(len(ref_token_mask) - 1, -1, -1):
                        if ref_token_mask[i]:
                            left_output_length -= 1
                        else:
                            break
                    ref_token_mask = ref_token_mask[:left_output_length]
                    lm_labels = lm_labels[:left_output_length]
                    num_of_truncated += 1
                    # print('truncated', eval_type, input_text, output_text)
            instance.output_ids = lm_labels.to(self.device)

            instance.input_ids = input_ids.to(self.device)
            instance.ref_token_mask = ref_token_mask.to(self.device)
            instance.input_referent_mask = input_referent_mask.to(self.device)
            batch.append(instance)
        # for i in batch:
        #     if 'alpaca' in i.eval_type: continue
        #     print('-------------------------------------------------------------------------------')
        #     print(f'>>>>>>>>>>>>{i.eval_type}')
        #     print("INPUT TEXT>>> ", i.input_text)
        #     print("OUTPUT TEXT>>> ", i.output_text)
        #     print('=====')
        #     print("INPUT LLM decode>>> ", self.llama_tokenizer.decode(i.input_ids ) )
        #     print("OUTPUT LLM decode>>> ",  self.llama_tokenizer.decode(i.output_ids))
        #     print('=====')
        #     print("answer_grounding_token_pos: (pos, phrase, gt_query_ids)")
        #     if i.answer_grounding_token_pos:
        #         assert len(i.answer_grounding_token_pos) == len(i.answer_gt_query_ids)
        #         for j, k in zip(i.answer_grounding_token_pos, i.answer_gt_query_ids):
        #             print(j, i.output_text[j[0]:j[1]], k)
        #     print("question_grounding_token_pos: (pos, phrase, gt_query_ids)")
        #     if i.question_grounding_token_pos:
        #         assert len(i.question_grounding_token_pos) == len(i.question_gt_query_ids)
        #         for j, k in zip(i.question_grounding_token_pos, i.question_gt_query_ids):
        #             print(j, i.input_text[j[0]:j[1]], k)
        #     print("use_input_referent: ", i.use_input_referent)
        #     print("input_referent_mask: ", i.input_referent_mask.any())
        #     print('------------------------------------------------------------------------------------------')
        # from IPython import embed; embed()
        max_lang_size = getattr(self.config, 'max_lang_size', 200 if self.config.vicuna_version ==
                                "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" or self.config.vicuna_version == "Tiny-Vicuna-1B" else 100)
        min_lang_size = min(getattr(self.config, 'min_lang_size', 100 if self.config.vicuna_version ==
                            "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T" or self.config.vicuna_version == "Tiny-Vicuna-1B" else 50), max_lang_size)
        if self.subsample:
            num_samples = min(max(min(
                int(len(batch) * self.sample_rate), max_lang_size), min_lang_size), len(batch))
            batch = random.sample(batch, num_samples)

        if not batch:
            # It is possible that all candidate samples were filtered out above
            # (e.g. build_input_from_segments failed or truncation dropped them).
            # In that case, we must early-return a safe zero loss, otherwise
            # `output` is undefined and the forward will crash.
            print("warning: no valid batch content, llm will not be updated")
            device = None
            try:
                device = self.device
            except Exception:
                pass
            if device is None:
                try:
                    device = next(self.parameters()).device
                except Exception:
                    device = torch.device("cpu")

            # DDP safety: make the "zero loss" connected to a trainable parameter
            # so that downstream backward() does not error out (e.g. in gradient
            # isolation code paths that call backward(non_geom_loss) directly).
            dummy_param = None
            for p in self.parameters():
                if getattr(p, "requires_grad", False):
                    dummy_param = p
                    break
            zero = (
                dummy_param.sum() * 0.0
                if dummy_param is not None
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            return {
                "lm_loss": zero,
                "match_loss": zero,
                "model_output": None,
            }

        inputs_embeds, attention_mask, labels = self._merge_input_ids_with_instance_features(
            batch=batch)

        gc.collect()
        torch.cuda.empty_cache()
        print(" ============================================ ")
        if num_of_truncated > 0:
            print(f"{num_of_truncated} of output is truncated")
        try:
            with torch.autocast("cuda"):
                print(f"llm input embeds shape: {inputs_embeds.shape}")
                output = super().forward(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True,
                )
        except Exception as e:
            print("failed to forward")
            print(inputs_embeds.shape)
            print(e)
            raise e

        output_last_hidden_states = output.hidden_states[-1].bfloat16()
        model_output = output
        assert output_last_hidden_states.shape[:2] == labels.shape

        # ------------------------------------------------------------------
        # Optional: Llama step-slot embeddings (phrase-only), matching the
        # `mask3d-vigor-llama-step-slot` pipeline.
        #
        # This does NOT rely on the generated answer tokens. Instead we encode
        # each step independently as:
        #   phrase_k <stepk>
        # and take the final hidden at the last position (the <stepk> token),
        # then project with the exported adapter to Vigor inner_dim (typically 768).
        #
        # Enable with:
        #   SSR3DLLM_STEP_TOKENS=1
        #   SSR3DLLM_LLM_STEPSLOT_ADAPTER=/path/to/adapter.pt
        #   SSR3DLLM_LLM_STEPSLOT_MODE=phrase_only
        # Optional token-level memory for bypassing BERT:
        #   SSR3DLLM_LLM_STEPSLOT_EXPORT_LANG_EMBEDS=1
        # ------------------------------------------------------------------
        stepslot_mode = str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_MODE", "")).strip().lower()
        export_lang_embeds = str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_EXPORT_LANG_EMBEDS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        adapter = getattr(self, "stepslot_adapter", None)
        if stepslot_mode == "phrase_only" and adapter is not None:
            try:
                def _strip_geom_token_local(text: str) -> str:
                    return str(text).replace("<geom>", " ").replace("  ", " ").strip()

                step_ids = getattr(self.llama_tokenizer, "step_token_ids", None) or []
                order_len = int(len(step_ids))
                if order_len <= 0:
                    raise RuntimeError("SSR3DLLM_STEP_TOKENS is required for SSR3DLLM_LLM_STEPSLOT_MODE=phrase_only")

                max_len = int(str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_MAX_LEN", "64")).strip() or "64")
                if max_len <= 0:
                    max_len = 64

                # Collect step texts for samples that provide Vigor-style referential order.
                step_texts_flat = []
                owner_infos = []
                for instance in batch:
                    li = getattr(instance, "lang_info", None)
                    order = getattr(li, "rel_referential_order", None) if li is not None else None
                    if not isinstance(order, list) or not order:
                        continue
                    # Pad/truncate to order_len.
                    phrases = [str(x).strip() for x in order if str(x).strip()]
                    if not phrases:
                        continue
                    if len(phrases) == 1:
                        phrases = phrases * order_len
                    elif len(phrases) < order_len:
                        phrases = phrases + [phrases[-1]] * (order_len - len(phrases))
                    else:
                        phrases = phrases[:order_len]
                    for k in range(order_len):
                        step_texts_flat.append(f"{phrases[k]} <step{k+1}>".strip())
                    owner_infos.append(li)

                if owner_infos:
                    enc = self.llama_tokenizer(
                        step_texts_flat,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=int(max_len),
                    )
                    device = output_last_hidden_states.device
                    enc = {k: v.to(device=device) for k, v in enc.items()}

                    with torch.autocast("cuda", enabled=(device.type == "cuda")):
                        out_step = self.model(**enc, output_hidden_states=True, return_dict=True)
                    hs = out_step.hidden_states[-1]
                    attn = enc.get("attention_mask", None)
                    if attn is None:
                        raise RuntimeError("tokenizer must return attention_mask for stepslot encoding")
                    last_pos = attn.long().sum(dim=-1) - 1
                    idx = torch.arange(hs.size(0), device=hs.device)
                    step_h = hs[idx, last_pos]

                    # Project to Vigor inner_dim.
                    step_proj = adapter.proj_step(step_h.to(dtype=adapter.proj_step[0].weight.dtype))
                    step_proj = step_proj.reshape(len(owner_infos), order_len, -1)

                    for li, embeds in zip(owner_infos, step_proj):
                        li.llm_step_embeds = embeds  # [order_len, inner_dim]

                # Optional: token-level memory embeddings for bypassing BERT lang encoder.
                if export_lang_embeds and int(getattr(adapter, "mem_tokens", 0) or 0) > 0:
                    if getattr(adapter, "mem_token_embeds", None) is None or getattr(adapter, "proj_mem", None) is None:
                        raise RuntimeError("stepslot_adapter missing mem_token_embeds/proj_mem (mem_tokens > 0)")

                    mem_n = int(getattr(adapter, "mem_tokens", 0))
                    if mem_n <= 0:
                        raise RuntimeError("mem_tokens must be > 0 for SSR3DLLM_LLM_STEPSLOT_EXPORT_LANG_EMBEDS=1")

                    # Use a distill/utterance-like text when available; fall back to question.
                    utterances = []
                    owners_u = []
                    for instance in batch:
                        li = getattr(instance, "lang_info", None)
                        if li is None:
                            continue
                        if not hasattr(li, "rel_referential_order"):
                            continue
                        # Prefer rel_distill_text (if present), otherwise fall back to the raw grounding text
                        # captured from the dataset, then to `question` (legacy).
                        u = (
                            getattr(li, "rel_distill_text", None)
                            or getattr(li, "raw_grounding_text", None)
                            or getattr(li, "question", None)
                            or ""
                        )
                        u = _strip_geom_token_local(str(u))
                        if not u:
                            # Treat as a bad sample for llm_lang_embeds export (do not fabricate text).
                            if not getattr(self, "_stepslot_lang_missing_warned", False):
                                self._stepslot_lang_missing_warned = True
                                print(
                                    "[LLM][stepslot_phrase_only][warn] Missing utterance/distill text for some "
                                    "samples; llm_lang_embeds will NOT be exported for them.",
                                    flush=True,
                                )
                            continue
                        utterances.append(u)
                        owners_u.append(li)

                    if utterances:
                        text_max = max(1, int(max_len) - int(mem_n))
                        enc_u = self.llama_tokenizer(
                            utterances,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=int(text_max),
                        )
                        device = output_last_hidden_states.device
                        enc_u = {k: v.to(device=device) for k, v in enc_u.items()}
                        input_ids = enc_u.get("input_ids", None)
                        attn_u = enc_u.get("attention_mask", None)
                        if input_ids is None or attn_u is None:
                            raise RuntimeError("tokenizer must return input_ids and attention_mask for utterances")

                        base_emb = self.model.embed_tokens(input_ids)
                        mem = adapter.mem_token_embeds.to(device=base_emb.device, dtype=base_emb.dtype).unsqueeze(0).expand(
                            base_emb.size(0), mem_n, -1
                        )
                        inputs_embeds = torch.cat([base_emb, mem], dim=1)
                        attn_mem = torch.ones((base_emb.size(0), mem_n), device=attn_u.device, dtype=attn_u.dtype)
                        attn_u = torch.cat([attn_u, attn_mem], dim=1)

                        with torch.autocast("cuda", enabled=(device.type == "cuda")):
                            out_u = self.model(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attn_u,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                        hs_u = out_u.hidden_states[-1][:, -mem_n:, :]
                        mem_proj = adapter.proj_mem(hs_u.to(dtype=adapter.proj_mem[0].weight.dtype))
                        for li, m in zip(owners_u, mem_proj):
                            li.llm_lang_embeds = m  # [mem_n, inner_dim]
            except Exception as e:
                strict = str(os.environ.get("SSR3DLLM_LLM_STEPSLOT_STRICT", "1")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if strict:
                    raise
                if not getattr(self, "_stepslot_phrase_only_warned", False):
                    self._stepslot_phrase_only_warned = True
                    print(f"[LLM][stepslot_phrase_only][warn] {type(e).__name__}: {e}", flush=True)

        # get corresponding ref query & build per-sample LLM text context
        for instance, output_last_hidden_state in zip(batch, output_last_hidden_states):
            instance: grounded_3d_llm_data
            s, e = instance.output_range
            # ---- LLM text context for SSR3DLLM ----
            # We summarise the answer segment [s:e] of the last hidden layer,
            # project it into instance feature space via hidden_state2query,
            # and attach the result back to the originating lang_info_data.
            if hasattr(instance, "lang_info") and instance.lang_info is not None:
                segment = output_last_hidden_state[s:e]
                if segment.shape[0] == 0:
                    # Fallback to the whole sequence if answer span is empty.
                    segment = output_last_hidden_state
                # [llama_dim] summary over tokens
                summary = segment.mean(dim=0)
                # Map to instance feature dimension (mask_dim / instance_dim).
                text_init_vec = self.model.hidden_state2query(
                    summary.unsqueeze(0)
                ).squeeze(0)
                # Also expose token-level features for geometry decoder cross-attention.
                # Shape: [L, mask_dim] where L is the number of tokens in the chosen segment.
                try:
                    text_tokens = self.model.hidden_state2query(segment)
                    instance.lang_info.llm_text_tokens = text_tokens.detach()
                except Exception:
                    pass
                # Detach so that geometric losses using this vector do not
                # backpropagate gradients into the LLM.
                try:
                    instance.lang_info.llm_text_init = text_init_vec.detach()
                except Exception:
                    pass

                # ---- Step-token embeddings (for chain-style grounding) ----
                # If `<stepK>` tokens are enabled in the tokenizer, extract one
                # embedding per step token from the answer segment and project
                # it to instance feature space (mask_dim / instance_dim).
                #
                # IMPORTANT: unlike `llm_text_init`, we intentionally DO NOT
                # detach these embeddings so geometry losses can backpropagate
                # into the `<stepK>` token rows (optionally gradient-masked by
                # SSR3DLLM_FREEZE_LLM_EXCEPT_STEP).
                try:
                    step_ids = getattr(self.llama_tokenizer, "step_token_ids", None) or []
                    if step_ids and getattr(instance, "output_ids", None) is not None:
                        out_ids = instance.output_ids
                        if torch.is_tensor(out_ids):
                            out_ids_t = out_ids.to(device=output_last_hidden_state.device)
                        else:
                            out_ids_t = torch.as_tensor(out_ids, device=output_last_hidden_state.device, dtype=torch.long)

                        seg_hs = output_last_hidden_state[s:e]
                        step_embeds = []
                        step_pos = []
                        for sid in [int(x) for x in step_ids]:
                            idxs = (out_ids_t == int(sid)).nonzero(as_tuple=False).view(-1)
                            if idxs.numel() > 0:
                                pos = int(idxs[0].item())
                                # For chain-step tokens, we want the hidden state AT the
                                # <stepK> position so gradients flow into the <stepK>
                                # token embedding row (when SSR3DLLM_FREEZE_LLM_EXCEPT_STEP=1).
                                # This also requires the step text to appear BEFORE <stepK>
                                # in the supervised output (see vigor_steps json builder).
                                hs_pos = max(pos, 0)
                                hs_pos = min(hs_pos, int(seg_hs.size(0) - 1))
                                hs = seg_hs[hs_pos].unsqueeze(0)  # [1, llama_dim]
                                step_vec = self.model.hidden_state2query(hs).squeeze(0)  # [mask_dim]
                                step_embeds.append(step_vec)
                                step_pos.append(pos)
                            else:
                                step_embeds.append(
                                    torch.zeros(
                                        (self.model.instance_dim,),
                                        device=output_last_hidden_state.device,
                                        dtype=output_last_hidden_state.dtype,
                                    )
                                )
                                step_pos.append(-1)
                        if step_embeds:
                            instance.lang_info.llm_step_embeds = torch.stack(step_embeds, dim=0)
                            instance.lang_info.llm_step_pos = step_pos
                except Exception:
                    pass

            instance.grouped_indices = instance.group_true_indices(output_last_hidden_state=output_last_hidden_state[s:e],
                                                                   MLP=self.model.hidden_state2query,
                                                                   ref_token_id=self.llama_tokenizer.ref_token_id)

        loss_data_type = get_loss_for_each_type(output, batch)
        output = model_output.logits
        lm_loss = model_output.loss

        # Optional: fully disable original Grounded 3D-LLM matching grounding (<ref> + match_loss),
        # keep language loss only and let the geometry module handle grounding.
        disable_grounding = os.environ.get("SSR3DLLM_DISABLE_LLM_GROUNDING", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_grounding:
            match_loss = torch.zeros_like(lm_loss)
        else:
            match_loss = self.calculate_grounding_loss(
                batch, use_single_ref_token=self.use_single_ref_token)

        if torch.isnan(lm_loss + match_loss):
            print('Nan loss!')
            lm_loss = match_loss = model_output.logits.sum() * 0.
        return {
            "lm_loss": lm_loss,
            "match_loss": match_loss,
            "model_output": model_output,
            **loss_data_type
        }

    def evaluate(
        self,
        input_text_list: list,
        batch_instance_queries_hidden_state: list,
        batch_instance_queries_normalized_embed: list,
        batch_eval_types: list,
        max_new_tokens=150,
        use_mini_batch=True,
        mini_batch_size=10,
        batch_gt_inst_ids: list = None,
        batch_box_info: list = None,
        batch_obj_mask: list = None,
        batch_order_valid_mask: list = None,
        batch_out_text=None,
        output_logits=False,
        top_p=1.,
        repetition_penalty=1.2,
        length_penalty=1,
        text_only_output=False,  # simplify output (only text)
    ):
        # SSR3DLLM: route "<geom>" prompts to a pretrained Vigor listener (geometry head),
        # bypassing LLM generation + <ref> matching. This enables evaluating the
        # geometric head directly on ScanRefer/ReferIt3D-style metrics.
        route_geom = os.environ.get("SSR3DLLM_ROUTE_GEOM_VIGOR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if route_geom and any(("<geom>" in str(t)) for t in (input_text_list or [])):
            if VigorRuntimeListener is None:
                raise RuntimeError("SSR3DLLM_ROUTE_GEOM_VIGOR=1 but VigorRuntimeListener import failed.")
            use_llm_order = os.environ.get("SSR3DLLM_ROUTE_GEOM_USE_LLM_ORDER", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            strict_llm_order = os.environ.get("SSR3DLLM_ROUTE_GEOM_LLM_ORDER_STRICT", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            if not hasattr(self, "_ssr3dllm_vigor_runtime") or self._ssr3dllm_vigor_runtime is None:
                self._ssr3dllm_vigor_runtime = VigorRuntimeListener(device=torch.device(self.device))
            vigor: VigorRuntimeListener = self._ssr3dllm_vigor_runtime

            if batch_out_text is None:
                batch_out_text = ["NONE"] * len(input_text_list)

            # Try to switch to geom-specific PEFT adapter for "<geom>" routing.
            prev_adapter = _get_active_peft_adapter(self)
            target_adapter = str(getattr(self, "_ssr3dllm_geom_adapter_name", "") or "")
            switched = False
            if use_llm_order and target_adapter:
                switched = _set_active_peft_adapter(self, target_adapter)

            try:
                out_json = []
                output_logits_list = [] if output_logits else None
                warned_once = False
                box_info_list = (
                    list(batch_box_info)
                    if isinstance(batch_box_info, (list, tuple))
                    else [None] * len(input_text_list)
                )
                if len(box_info_list) < len(input_text_list):
                    box_info_list = box_info_list + [None] * (len(input_text_list) - len(box_info_list))
                obj_mask_list = (
                    list(batch_obj_mask)
                    if isinstance(batch_obj_mask, (list, tuple))
                    else [None] * len(input_text_list)
                )
                if len(obj_mask_list) < len(input_text_list):
                    obj_mask_list = obj_mask_list + [None] * (len(input_text_list) - len(obj_mask_list))
                order_valid_list = (
                    list(batch_order_valid_mask)
                    if isinstance(batch_order_valid_mask, (list, tuple))
                    else [None] * len(input_text_list)
                )
                if len(order_valid_list) < len(input_text_list):
                    order_valid_list = order_valid_list + [None] * (len(input_text_list) - len(order_valid_list))
                for bid, (input_text, instance_queries_hidden_state, gt, eval_type) in enumerate(
                    zip(input_text_list, batch_instance_queries_hidden_state, batch_out_text, batch_eval_types)
                ):
                    text = str(input_text).replace("<geom>", "").strip()
                    q = instance_queries_hidden_state
                    if not torch.is_tensor(q):
                        q = torch.as_tensor(q)
                    box_info = box_info_list[bid] if bid < len(box_info_list) else None
                    if box_info is not None and not torch.is_tensor(box_info):
                        box_info = torch.as_tensor(box_info)
                    obj_mask = obj_mask_list[bid] if bid < len(obj_mask_list) else None
                    if obj_mask is not None and not torch.is_tensor(obj_mask):
                        obj_mask = torch.as_tensor(obj_mask)
                    order_valid = order_valid_list[bid] if bid < len(order_valid_list) else None
                    if order_valid is not None and not torch.is_tensor(order_valid):
                        order_valid = torch.as_tensor(order_valid)

                    logits = None
                    if use_llm_order:
                        try:
                            geom_adapter = getattr(self, "geom_stepslot_adapter", None) or getattr(self, "stepslot_adapter", None)
                            order_embeds, lang_embeds = self.encode_stepslot_onepass_pred(
                                utterances=[text],
                                adapter=geom_adapter,
                            )
                            order_len = int(order_embeds.size(1))
                            pred_class_mask = torch.ones(
                                (1, order_len, int(q.size(0))),
                                device=order_embeds.device,
                                dtype=torch.float32,
                            )
                            if obj_mask is not None:
                                om = obj_mask
                                if om.dim() == 2:
                                    om = om.view(-1)
                                if om.dim() == 1 and int(om.numel()) == int(q.size(0)):
                                    pred_class_mask = pred_class_mask * (om.view(1, 1, -1).to(device=pred_class_mask.device, dtype=pred_class_mask.dtype))
                            order_valid_mask = torch.ones((1, order_len), device=order_embeds.device, dtype=torch.float32)
                            if order_valid is not None:
                                ov = order_valid
                                if ov.dim() == 1 and int(ov.numel()) == int(order_len):
                                    order_valid_mask = ov.view(1, -1).to(
                                        device=order_embeds.device, dtype=torch.float32
                                    )
                            if lang_embeds is not None:
                                logits = vigor.forward_logits_with_order_embeds(
                                    lang_embeds=lang_embeds,
                                    order_embeds=order_embeds,
                                    order_valid_mask=order_valid_mask,
                                    mask3d_queries=q,
                                    box_info=box_info,
                                    obj_mask=obj_mask,
                                    pred_class_mask=pred_class_mask,
                                )
                            else:
                                lang_tokens = vigor.tokenizer([text], return_tensors="pt", padding=True)
                                logits = vigor.forward_logits_with_order_embeds(
                                    lang_tokens=lang_tokens,
                                    order_embeds=order_embeds,
                                    order_valid_mask=order_valid_mask,
                                    mask3d_queries=q,
                                    box_info=box_info,
                                    obj_mask=obj_mask,
                                    pred_class_mask=pred_class_mask,
                                )
                            logits = logits.squeeze(0).detach().to("cpu")
                        except Exception as e:
                            if strict_llm_order:
                                raise
                            if not warned_once:
                                warned_once = True
                                print(
                                    f"[LLM][geom_route][warn] fallback to listener.predict_logits due to "
                                    f"{type(e).__name__}: {e}",
                                    flush=True,
                                )
                            logits = None

                    if logits is None:
                        logits = vigor.predict_logits(text=text, mask3d_queries=q, box_info=box_info, obj_mask=obj_mask)

                    probs = torch.sigmoid(logits).unsqueeze(0)  # [1,N]
                    pred_idx = int(torch.argmax(logits).item()) if logits.numel() > 0 else -1
                    score = float(probs[0, pred_idx].item()) if pred_idx >= 0 and probs.numel() > 0 else 0.0

                    out_json.append(
                        {
                            "output_language": "<geom>",
                            "input_language": str(input_text),
                            "grounding_result": [pred_idx] if pred_idx >= 0 else None,
                            "score": [score] if pred_idx >= 0 else None,
                            "gt": gt,
                        }
                    )
                    if output_logits and output_logits_list is not None and "detection" in str(eval_type):
                        output_logits_list.append((bid, probs))

                if text_only_output:
                    return out_json[0]["output_language"] if out_json else ""
                if output_logits:
                    return out_json, (output_logits_list or [])
                return out_json
            finally:
                if switched and prev_adapter:
                    _set_active_peft_adapter(self, prev_adapter)

        self._reset_match_runtime_stats()
        batch = []
        mini_batch = []

        # placeholder for GT
        if batch_out_text is None:
            batch_out_text = ["NONE"]*len(input_text_list)
        if batch_gt_inst_ids is None:
            batch_gt_inst_ids = [None]*len(input_text_list)

        assert len(batch_out_text) == len(input_text_list)
        for input_text, instance_queries_hidden_state, instance_queries_normalized_embed, gt, eval_type, gt_instance_ids in zip(input_text_list,
                                                                                                                                batch_instance_queries_hidden_state,
                                                                                                                                batch_instance_queries_normalized_embed,
                                                                                                                                batch_out_text,
                                                                                                                                batch_eval_types,
                                                                                                                                batch_gt_inst_ids
                                                                                                                                ):

            gt_instance_predicted_iou = None
            question_grounding_token_pos = None
            question_gt_query_ids = None

            if 'chat' not in eval_type: # for chat demo
                if "scan2cap" in eval_type:
                    gt_instance_predicted_iou = gt_instance_ids[2]
                gt_instance_ids = gt_instance_ids[1]
                question_grounding_token_pos, question_gt_query_ids = gt_instance_ids.positives_question, gt_instance_ids.query_ids_question

            instance = grounded_3d_llm_data(input_text=input_text,
                                            instance_embed=instance_queries_normalized_embed,
                                            instance_feature=instance_queries_hidden_state,
                                            question_gt_query_ids=question_gt_query_ids,
                                            question_grounding_token_pos=question_grounding_token_pos,
                                            eval_type=eval_type,
                                            gt_instance_predicted_iou=gt_instance_predicted_iou
                                            )

            input_ids, input_referent_mask = instance.build_input_from_segments(use_system_prompt=self.config.use_system_prompt,
                                                                               tokenizer=self.llama_tokenizer,
                                                                               input_text=instance.input_text,
                                                                               prompts=self.prompts,
                                                                               use_input_referent=instance.use_input_referent,
                                                                               inference=True,
                                                                               question_grounding_token_pos=instance.question_grounding_token_pos,
                                                                               use_single_ref_token=self.use_single_ref_token
                                                                               )

            instance.input_ids = input_ids.to(self.device)
            instance.input_referent_mask = input_referent_mask.to(self.device)
            instance.output_hidden_states = []
            instance.gt = gt
            instance.eval_type = eval_type

            if not use_mini_batch:
                inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=[instance],
                                                                                             inference=True)
                instance.inputs_embeds = inputs_embeds
                instance.attention_mask = attention_mask
                batch.append(instance)
            else:
                mini_batch.append(instance)
                if len(mini_batch) == mini_batch_size:
                    inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=mini_batch,
                                                                                                 inference=True)
                    mini_batch = MiniBatchData(batch=mini_batch)
                    mini_batch.inputs_embeds = inputs_embeds
                    mini_batch.attention_mask = attention_mask
                    batch.append(mini_batch)
                    mini_batch = []

        if mini_batch and use_mini_batch:
            inputs_embeds, attention_mask = self._merge_input_ids_with_instance_features(batch=mini_batch,
                                                                                         inference=True)
            mini_batch = MiniBatchData(batch=mini_batch)
            mini_batch.inputs_embeds = inputs_embeds
            mini_batch.attention_mask = attention_mask
            batch.append(mini_batch)

        # to avoid OOM error
        with torch.no_grad():
            for instance in batch:
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    common_params = {
                        'inputs_embeds': instance.inputs_embeds,
                        'attention_mask': instance.attention_mask,
                        'max_new_tokens': max_new_tokens,
                        'output_hidden_states': True,
                        'return_dict_in_generate': True,
                        'num_beams': self.beam_size,
                        'output_scores': True,
                        'do_sample': False,
                        'min_length': 1,
                        'top_p': top_p,
                        'repetition_penalty': repetition_penalty,
                        'length_penalty': length_penalty
                    }
                    if self.config.vicuna_version in ["TinyLlama-1.1B-intermediate-step-1195k-token-2.5T", "Tiny-Vicuna-1B"]:
                        common_params['pad_token_id'] = self.llama_tokenizer.eos_token_id

                    with torch.autocast("cuda"):
                        outputs = self.generate(**common_params)
                        output_ids = outputs.sequences

                except Exception as e:
                    print(e)
                    print(instance.inputs_embeds.shape)
                    raise

                last_hidden_states = extract_decoder_hidden_states(outputs)
                if use_mini_batch:
                    assert len(instance.batch) == output_ids.shape[0]
                    for _instance, ids, lash_hidden_state in zip(instance.batch, output_ids, last_hidden_states):
                        _instance.output_ids = ids.tolist()
                        _instance.output_hidden_states = self.model.hidden_state2query(
                            lash_hidden_state.squeeze(dim=0).bfloat16()).cpu()
                        _instance.output_text = self.llama_tokenizer.decode(
                            ids)
                else:
                    instance.output_ids = output_ids.squeeze().tolist()
                    instance.output_hidden_states = self.model.hidden_state2query(
                        last_hidden_states.squeeze(dim=0).bfloat16()).cpu()
                    instance.output_text = self.llama_tokenizer.decode(
                        instance.output_ids)
        torch.cuda.empty_cache()
        if use_mini_batch:
            flattened_batch = []
            for instance in batch:
                flattened_batch += instance.batch
            batch = flattened_batch

        template = {
            "grounding_start": None,
            "ref_token_pos": [],
            "ref_token_feature": [],
            "grounding_end": None,
            "closed": True,
            "match_result": None,
        }

        if output_logits:
            output_logits_list = []

        for bid, instance in enumerate(batch):
            intervals = []
            open_interval = False
            assistant_ids_len = self.prompts["rules"]["assistant"]["ids"].shape[0]
            current_interval = None
            assert len(
                instance.output_ids) == instance.output_hidden_states.shape[0]
            for idx, id in enumerate(instance.output_ids):
                if int(id) == self.gs_token_id:
                    open_interval = True
                    current_interval = copy.deepcopy(template)
                    current_interval["grounding_start"] = idx-assistant_ids_len
                elif open_interval and id == self.ref_token_id:
                    current_interval["ref_token_feature"].append(
                        instance.output_hidden_states[idx])
                    current_interval["ref_token_pos"].append(
                        idx-assistant_ids_len)
                elif open_interval and int(id) == self.ge_token_id:
                    current_interval["grounding_end"] = idx-assistant_ids_len
                    current_interval["closed"] = True
                    open_interval = False
                    # get instance id
                    if current_interval["ref_token_feature"]:
                        features_norm = F.normalize(
                            torch.stack(current_interval["ref_token_feature"]).float(), p=2, dim=1
                        )
                        embeddings_norm = F.normalize(
                            instance.instance_embed.cpu().float(), p=2, dim=1
                        )
                        use_hnsw = (
                            self.hnsw_runtime_config["enabled"]
                            and self.use_single_ref_token
                            and _HNSW_AVAILABLE
                        )
                        match_mode = "hnsw" if use_hnsw else "exact"
                        match_start = time.perf_counter()
                        try:
                            if use_hnsw:
                                cosine_sim = self._hnsw_similarity(features_norm, embeddings_norm)
                            else:
                                cosine_sim = torch.matmul(features_norm, embeddings_norm.T)
                        except Exception as exc:  # pragma: no cover
                            logger.warning("Falling back to exact similarity due to HNSW error: {}", exc)
                            match_mode = "exact"
                            match_start = time.perf_counter()
                            cosine_sim = torch.matmul(features_norm, embeddings_norm.T)
                        match_elapsed = time.perf_counter() - match_start
                        self._accumulate_match_runtime(
                            match_mode,
                            match_elapsed,
                            features_norm.shape[0],
                            embeddings_norm.shape[0],
                        )
                        probs = cosine_sim
                        if not self.use_single_ref_token:
                            row_ind, col_ind = linear_sum_assignment(
                                probs.cpu().detach().numpy(), maximize=True)
                            best_match = col_ind.tolist()
                            best_match_probs = probs[row_ind, col_ind].tolist()
                        else:
                            probs = torch.sigmoid(probs / self.t)
                            if output_logits and "detection" in instance.eval_type:
                                output_logits_list.append((bid, probs))
                            mask = probs > self.config.prediction_threshold
                            if mask.any():
                                best_match = torch.nonzero(
                                    mask).squeeze(dim=1).tolist()
                                best_match = [i[1] for i in best_match]
                                best_match_probs = probs[mask].tolist()
                            else:
                                max_prob, max_index = torch.max(probs, dim=1)
                                best_match = max_index.tolist()
                                best_match_probs = max_prob.tolist()
                        current_interval["match_result"] = best_match
                        current_interval["probs"] = best_match_probs
                    else:
                        current_interval["match_result"] = None
                        current_interval["probs"] = None
                    intervals.append(current_interval)
            # unclosed interval
            if open_interval == True:
                current_interval["grounding_end"] = instance.output_hidden_states.shape[0]-1
                current_interval["closed"] = False
            instance.intervals = intervals
        # prepare output
        out_json = []
        for instance in batch:
            item = {}
            item["output_language"] = instance.output_text
            item["input_language"] = instance.input_text
            item["grounding_result"] = []
            item["score"] = []
            item["gt"] = instance.gt
            if 'scan2cap' in instance.eval_type or 'objdesc' in instance.eval_type:
                item['gt_predicted_iou'] = instance.gt_instance_predicted_iou
            if instance.intervals:
                for i in instance.intervals:
                    item["grounding_result"].append(i["match_result"])
                    item["score"].append(i["probs"])
            else:
                item["grounding_result"] = None
            out_json.append(item)
        if text_only_output:
            return out_json[0]["output_language"]
        if output_logits:
            return out_json, output_logits_list
        return out_json

    def configure_runtime_hnsw(
        self,
        enabled: bool,
        top_k: int = 50,
        ef_search: int = 200,
        M: int = 16,
        ef_construction: Optional[int] = None,
        candidate_limit: int = 0,
        keep_base: bool = True,
    ) -> None:
        """
        Configure on-the-fly HNSW retrieval that replaces the full cosine similarity.
        """
        if ef_construction is None:
            ef_construction = max(ef_search, 200)
        top_k = max(1, int(top_k))
        ef_search = max(1, int(ef_search))
        M = max(4, int(M))
        ef_construction = max(ef_construction, top_k, 100)

        if enabled and not _HNSW_AVAILABLE:
            logger.warning("HNSW runtime requested but hnswlib is not installed; fallback to exact search.")
            enabled = False

        self.hnsw_runtime_config = {
            "enabled": bool(enabled),
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "top_k": top_k,
            "candidate_limit": max(0, int(candidate_limit)),
            "keep_base": bool(keep_base),
        }

    def _hnsw_similarity(self, query: torch.Tensor, database: torch.Tensor) -> torch.Tensor:
        """
        Approximate similarity matrix using HNSW (inner product over normalized vectors).
        """
        if not _HNSW_AVAILABLE or database.shape[0] == 0:
            raise RuntimeError("HNSW similarity requested but hnswlib is unavailable or database is empty.")

        cfg = self.hnsw_runtime_config
        top_k = min(cfg["top_k"], database.shape[0])
        if top_k <= 0:
            top_k = database.shape[0]

        db_tensor = database.detach().cpu().to(torch.float32).contiguous()
        query_tensor = query.detach().cpu().to(torch.float32).contiguous()

        db_np = np.asarray(db_tensor.numpy(), dtype=np.float32)
        query_np = np.asarray(query_tensor.numpy(), dtype=np.float32)

        index = hnswlib.Index(space="ip", dim=db_np.shape[1])
        index.init_index(
            max_elements=db_np.shape[0],
            ef_construction=cfg["ef_construction"],
            M=cfg["M"],
        )
        index.add_items(db_np, np.arange(db_np.shape[0]))
        index.set_ef(cfg["ef_search"])

        labels, distances = index.knn_query(query_np, k=top_k)

        sim = torch.full(
            (query.shape[0], database.shape[0]),
            fill_value=-1e9,
            device=query.device,
            dtype=torch.float32,
        )
        for row_idx in range(labels.shape[0]):
            cand_indices = labels[row_idx].astype(np.int64, copy=False)
            cand_scores = distances[row_idx].astype(np.float32, copy=False)
            torch_scores = torch.from_numpy(cand_scores).to(sim.device, sim.dtype)
            sim[row_idx, cand_indices] = torch_scores
        return sim
