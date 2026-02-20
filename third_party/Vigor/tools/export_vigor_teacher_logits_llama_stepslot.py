#!/usr/bin/env python3
"""
Export Vigor teacher logits for distillation (LLAMA-stepslot wrapper).

This matches the evaluation/training pipeline in:
  referit3d/scripts/train_referit3d_llama_stepslot.py

Key differences vs tools/export_vigor_teacher_logits.py:
  - Teacher model is `ReferIt3DNetTransformerLlamaStepSlot` (LLM produces order_embeds).
  - We still export logits in SSR3DLLM TeacherLogitsDB KV format:
      {teacher_key: [Q]} or {teacher_key: {"final": [Q], "steps": [T,Q], "ori_len": ...}}

Intended use:
  - Offline export teacher logits once (train/test).
  - Train SSR3DLLM with:
      SSR3DLLM_DISTILL_VIGOR_WEIGHT>0
      SSR3DLLM_VIGOR_TEACHER_LOGITS_PATH=/path/to/*_kv.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import BertTokenizer

# SSR3DLLM distillation helpers (teacher_key + query-logits mapping)
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
from utils.teacher_distill import (  # type: ignore
    make_teacher_key,
    scatter_context_logits_to_queries,
)

# Ensure local referit3d is importable
vigor_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(vigor_root))

from referit3d.in_out.neural_net_oriented import (  # type: ignore
    load_scan_related_data,
    load_referential_data,
    compute_auxiliary_data,
)
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders  # type: ignore
from referit3d.in_out.arguments import parse_arguments as vigor_parse_args  # type: ignore
from referit3d.models.referit3d_net import ReferIt3DNet_transformer  # type: ignore
from referit3d.models.utils import load_state_dicts  # type: ignore

from referit3d.models.llama_stepslot import (  # type: ignore
    LlamaStepSlotConfig,
    LlamaStepSlotOrderEncoder,
    ReferIt3DNetTransformerLlamaStepSlot,
)


def _maybe_add_step_tokens(tokenizer: BertTokenizer, order_len: int):
    tokens = [f"<step{i+1}>" for i in range(int(order_len))]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in tokens]
    return tokens, ids, int(added)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Vigor teacher logits (llama-stepslot wrapper) for SSR3DLLM distillation."
    )
    p.add_argument("-scannet-file", required=True, type=str)
    p.add_argument("-referit3D-file", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str, help="Path to llama-stepslot wrapper ckpt (.pth)")
    p.add_argument("--output", required=True, type=str, help="Path to save exported logits (torch .pt)")
    p.add_argument("--split", default="test", choices=["train", "test"], help="Which split to export.")
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--n-workers", default=0, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--teacher-name", default="vigor", type=str, help="Name used in teacher_key.")
    p.add_argument("--num-queries", default=100, type=int, help="Mask3D query count (usually 100).")
    p.add_argument(
        "--save-format",
        default="kv",
        choices=["kv", "full"],
        help="Save format: 'kv' saves {teacher_key: query_logits} (small, recommended); "
        "'full' saves a list[dict] with extra fields (large).",
    )
    p.add_argument(
        "--export-steps",
        action="store_true",
        help="If set, also export per-step referential logits (T x Q).",
    )
    p.add_argument(
        "--no-collapse-padded-steps",
        action="store_true",
        help="By default, collapse Vigor's padded order_len=4 scheme (e.g. ori_len=2 => [A,A,T,T]) "
        "into effective-length steps (e.g. [A,T]). Set this flag to keep raw padded steps.",
    )
    # LLM config (must match the wrapper ckpt architecture)
    p.add_argument(
        "--llm-model-path",
        default="",
        type=str,
        help="Local path to causal LLM weights. If empty, use env VIGOR_LLM_MODEL_PATH.",
    )
    p.add_argument("--llm-max-len", default=64, type=int)
    p.add_argument("--llm-mem-tokens", default=16, type=int, help="Soft memory token count (must match ckpt).")
    p.add_argument("--llm-use-bf16", action="store_true", help="Load causal LLM in bf16 (recommended).")
    # Mask3D options (if used in training)
    p.add_argument("--mask3d-feature-root", default=None, type=str)
    p.add_argument("--mask3d-feature-root-test", default=None, type=str)
    p.add_argument("--mask3d-feature-dim", default=128, type=int)
    p.add_argument("--use-scannet200-obj-cls", action="store_true")
    # Sample limiting
    p.add_argument("--max-samples", default=-1, type=int, help="If >0, limit number of samples exported.")
    return p.parse_args()


def build_loaders(args: argparse.Namespace):
    # Vigor's load_referential_data expects a *train* csv path and will derive the
    # corresponding test csv by string replacement (train -> test).
    referit_file = args.referit3D_file
    referit_name = Path(referit_file).name
    if ("_test_" in referit_name or referit_name.startswith(("sr3d_test", "nr3d_test"))) and (
        "_train_" not in referit_name and not referit_name.startswith(("sr3d_train", "nr3d_train"))
    ):
        candidate = str(Path(referit_file).with_name(referit_name.replace("_test_", "_train_")))
        if Path(candidate).exists():
            print(
                f"[export_vigor_teacher_logits_llama][WARN] got a test csv in -referit3D-file: {referit_file}\n"
                f"  -> using train csv instead (Vigor will auto-load its paired test csv): {candidate}",
                flush=True,
            )
            referit_file = candidate

    vigor_args_list = [
        "-scannet-file",
        args.scannet_file,
        "-referit3D-file",
        referit_file,
        "--log-dir",
        "/tmp/vigor_export_llama",
        "--batch-size",
        str(args.batch_size),
        "--n-workers",
        str(args.n_workers),
        "--multilabel-pretraining",
        "True",
        "--lang-multilabel",
        "True",
        "--cascading",
        "True",
        "--order-len",
        "4",
    ]
    if args.mask3d_feature_root:
        vigor_args_list += [
            "--mask3d-feature-root",
            args.mask3d_feature_root,
            "--mask3d-feature-dim",
            str(args.mask3d_feature_dim),
        ]
        if args.mask3d_feature_root_test:
            vigor_args_list += [
                "--mask3d-feature-root-test",
                args.mask3d_feature_root_test,
            ]
    if args.use_scannet200_obj_cls:
        vigor_args_list += ["--use-scannet200-obj-cls", "True"]

    v_args = vigor_parse_args(vigor_args_list)
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(v_args.scannet_file)
    referit_data = load_referential_data(v_args, v_args.referit3D_file, scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, v_args)
    loaders = make_data_loaders(v_args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    return loaders, v_args, class_to_idx


def _collapse_vigor_order_len4_steps(steps_q: torch.Tensor, ori_len: int) -> torch.Tensor:
    """
    Collapse Vigor's order_len=4 padded steps into effective-length steps.

    Padding rules (see listening_dataset.py):
      - ori_len=1: [X, X, X, X]
      - ori_len=2: [A, A, T, T]
      - ori_len=3: [A, B, C, C]
      - ori_len=4: [A, B, C, D]
    We return a tensor with shape [ori_len, Q].
    """
    if not torch.is_tensor(steps_q) or steps_q.dim() != 2:
        return steps_q
    T, _Q = int(steps_q.size(0)), int(steps_q.size(1))
    if T < 4:
        return steps_q[: max(min(int(ori_len), T), 1)]
    ori_len = int(max(ori_len, 1))
    if ori_len == 1:
        return steps_q[:4].mean(dim=0, keepdim=True)
    if ori_len == 2:
        return torch.stack([steps_q[0:2].mean(dim=0), steps_q[2:4].mean(dim=0)], dim=0)
    if ori_len == 3:
        return torch.stack([steps_q[0], steps_q[1], steps_q[2:4].mean(dim=0)], dim=0)
    return steps_q[:4]


@torch.no_grad()
def main() -> None:
    # Speed: disable llama-step-slot distillation terms during export (they do not affect logits).
    os.environ.setdefault("VIGOR_LLM_DISTILL_W", "0")
    os.environ.setdefault("VIGOR_LLM_GLOBAL_DISTILL_W", "0")

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    llm_path = str(args.llm_model_path or "").strip() or str(os.environ.get("VIGOR_LLM_MODEL_PATH", "")).strip()
    if not llm_path:
        raise RuntimeError(
            "Missing causal LLM weights path. Set --llm-model-path or env VIGOR_LLM_MODEL_PATH."
        )

    loaders, v_args, class_to_idx = build_loaders(args)
    split_loader = loaders[args.split]

    # ---- build listener (Vigor) ----
    n_classes = len(class_to_idx) - 1  # ignore pad
    pad_idx = class_to_idx["pad"]

    tokenizer = BertTokenizer.from_pretrained(v_args.bert_pretrain_path)
    # Always add <stepK> tokens so step-slot checkpoints can load without shape mismatch.
    step_tokens, step_token_ids, added = _maybe_add_step_tokens(tokenizer, int(getattr(v_args, "order_len", 4)))
    setattr(v_args, "vigor_step_tokens", step_tokens)
    setattr(v_args, "vigor_step_token_ids", step_token_ids)
    os.environ.setdefault("VIGOR_STEP_MARKERS", "1")
    print(f"[export_vigor_teacher_logits_llama][step_tokens] added={added} ids={step_token_ids}", flush=True)

    class_name_list = list(class_to_idx.keys())
    class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].to(device)

    listener = ReferIt3DNet_transformer(v_args, n_classes, class_name_tokens, ignore_index=pad_idx).to(device)
    # Keep consistent with training script when step tokens are added.
    try:
        listener.language_encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # ---- build llama stepslot wrapper ----
    llm_cfg = LlamaStepSlotConfig(
        model_path=llm_path,
        order_len=int(getattr(v_args, "order_len", 4)),
        max_length=int(args.llm_max_len),
        memory_tokens=int(args.llm_mem_tokens),
        distill_w=0.0,
        global_distill_w=0.0,
        freeze_llm_except_step_rows=True,
        local_files_only=True,
        use_bf16=bool(args.llm_use_bf16),
    )
    llm = LlamaStepSlotOrderEncoder(out_dim=int(getattr(v_args, "inner_dim", 768)), cfg=llm_cfg).to(device)
    model = ReferIt3DNetTransformerLlamaStepSlot(listener=listener, llm=llm, cfg=llm_cfg).to(device)

    # Load wrapper checkpoint (strict by default; configure with env VIGOR_STRICT_LOAD/VIGOR_ALLOWED_*).
    resumed_epoch = load_state_dicts(args.checkpoint, map_location=device, model=model)
    print(
        f"[export_vigor_teacher_logits_llama][ckpt] loaded resumed_epoch={resumed_epoch} path={args.checkpoint}",
        flush=True,
    )
    model.eval()

    export = []
    kv_table: dict[str, torch.Tensor | dict] = {}
    scene_cache: dict[str, dict] = {}
    total = 0

    for batch in tqdm(split_loader, desc=f"export {args.split}"):
        # Move tensor fields to device
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        # Provide lang_tokens for robustness (wrapper may still need it if mem_tokens=0).
        lang_tokens = tokenizer(batch["tokens"], return_tensors="pt", padding=True)
        lang_tokens = {k: v.to(device) for k, v in lang_tokens.items()}
        batch["lang_tokens"] = lang_tokens

        # Forward
        out = model(batch)
        scannet_class_logits = None
        if isinstance(out, (list, tuple)):
            if len(out) == 6:
                _, class_logits, lang_logits, logits, scannet_class_logits, _ = out
            elif len(out) == 5:
                _, class_logits, lang_logits, logits, scannet_class_logits = out
            else:
                _, class_logits, lang_logits, logits = out
        else:
            raise RuntimeError("Unexpected model output type")

        # Optional: per-step logits from the *listener* (not the wrapper).
        step_ctx_logits = None
        if args.export_steps:
            step_ctx_logits = getattr(model.listener, "last_ref_logits_steps", None)

        # Save per-sample data
        B = int(logits.size(0))
        for i in range(B):
            scan_id = batch["scan_id"][i]
            tokens = batch["tokens"][i] if isinstance(batch.get("tokens", None), list) else ""
            # Build a stable "utterance-like" string for teacher_key (must match training-time key).
            # batch["tokens"][i] is usually a list of token strings.
            if isinstance(tokens, (list, tuple)):
                text = " ".join([str(t) for t in tokens])
            else:
                text = str(tokens)

            target_pos = int(batch["target_pos"][i].item())
            target_gt_id = None
            inst_ids = batch.get("instance_ids", None)
            if isinstance(inst_ids, torch.Tensor) and inst_ids.dim() == 2 and 0 <= target_pos < int(inst_ids.size(1)):
                try:
                    target_gt_id = int(inst_ids[i, target_pos].item())
                except Exception:
                    target_gt_id = None

            teacher_key = None
            if target_gt_id is not None and isinstance(scan_id, str):
                teacher_key = make_teacher_key(
                    teacher_name=str(args.teacher_name),
                    scene_id=scan_id,
                    target_gt_id=int(target_gt_id),
                    text=text,
                )

            # Map teacher referential logits over ReferIt3D context slots -> Mask3D query logits.
            query_logits = None
            query_step_logits = None
            mask3d_path = None
            if "mask3d_feature_path" in batch:
                try:
                    mask3d_path = batch["mask3d_feature_path"][i]
                except Exception:
                    mask3d_path = None
            if isinstance(scan_id, str) and isinstance(mask3d_path, str):
                if scan_id not in scene_cache:
                    try:
                        feat = torch.load(mask3d_path, map_location="cpu")
                    except Exception:
                        feat = None
                    if isinstance(feat, dict):
                        gt_to_query = feat.get("gt_to_query_map") or {}
                        norm_map = {}
                        if isinstance(gt_to_query, dict):
                            for k, v in gt_to_query.items():
                                try:
                                    norm_map[int(k)] = int(v)
                                except Exception:
                                    continue
                        scene_cache[scan_id] = {"gt_to_query_map": norm_map}
                    else:
                        scene_cache[scan_id] = {"gt_to_query_map": {}}
                gt_to_query_map = scene_cache[scan_id].get("gt_to_query_map") or {}
                if isinstance(inst_ids, torch.Tensor) and inst_ids.dim() == 2:
                    qlogits = scatter_context_logits_to_queries(
                        context_logits=logits[i].detach().cpu().view(-1),
                        context_instance_ids=inst_ids[i].detach().cpu().view(-1),
                        gt_to_query_map=gt_to_query_map,
                        num_queries=int(args.num_queries),
                    )
                    query_logits = qlogits

                    if args.export_steps and isinstance(step_ctx_logits, list) and step_ctx_logits:
                        step_q = []
                        for t in range(len(step_ctx_logits)):
                            try:
                                step_logits_t = step_ctx_logits[t][i].detach().cpu().view(-1)
                            except Exception:
                                continue
                            step_q.append(
                                scatter_context_logits_to_queries(
                                    context_logits=step_logits_t,
                                    context_instance_ids=inst_ids[i].detach().cpu().view(-1),
                                    gt_to_query_map=gt_to_query_map,
                                    num_queries=int(args.num_queries),
                                )
                            )
                        if step_q:
                            query_step_logits = torch.stack(step_q, dim=0)  # [T, Q]

            item = {
                "teacher_name": str(args.teacher_name),
                "teacher_key": teacher_key,
                "scan_id": scan_id,
                "target_gt_id": target_gt_id,
                "text": text,
                "target_pos": batch["target_pos"][i].cpu(),
                "class_labels": batch["class_labels"][i].cpu(),
                "target_class": batch["target_class"][i].cpu(),
                "anchor_ind": batch["anchor_ind"][i].cpu() if "anchor_ind" in batch else None,
                "ordered_multilabel_gt": batch["ordered_multilabel_gt"][i].cpu() if "ordered_multilabel_gt" in batch else None,
                "pred_class_mask": batch["pred_class_mask"][i].cpu() if "pred_class_mask" in batch else None,
                "referential_logits": logits[i].detach().cpu(),
                "query_logits": query_logits,
            }
            if scannet_class_logits is not None:
                item["scannet_class_logits"] = scannet_class_logits[i].detach().cpu()
            if query_step_logits is not None:
                item["query_step_logits"] = query_step_logits
            if "ori_order_len" in batch and torch.is_tensor(batch["ori_order_len"]):
                try:
                    item["ori_order_len"] = batch["ori_order_len"][i].detach().cpu()
                except Exception:
                    pass

            if args.save_format == "kv":
                if isinstance(teacher_key, str) and teacher_key and isinstance(query_logits, torch.Tensor):
                    if args.export_steps and isinstance(query_step_logits, torch.Tensor):
                        if (
                            (not args.no_collapse_padded_steps)
                            and isinstance(item.get("ori_order_len", None), torch.Tensor)
                            and int(getattr(v_args, "order_len", 4)) == 4
                        ):
                            try:
                                ori_len_i = int(item["ori_order_len"].item())
                            except Exception:
                                ori_len_i = None
                            if isinstance(ori_len_i, int) and ori_len_i > 0:
                                query_step_logits = _collapse_vigor_order_len4_steps(query_step_logits, ori_len_i)
                        entry = {"final": query_logits.detach().cpu(), "steps": query_step_logits.detach().cpu()}
                        if "ori_order_len" in item:
                            entry["ori_len"] = item["ori_order_len"]
                        kv_table[teacher_key] = entry
                    else:
                        kv_table[teacher_key] = query_logits.detach().cpu()
            else:
                export.append(item)

            total += 1
            if args.max_samples > 0 and total >= args.max_samples:
                break
        if args.max_samples > 0 and total >= args.max_samples:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_format == "kv":
        torch.save(kv_table, out_path)
        print(f"[export_vigor_teacher_logits_llama] saved {len(kv_table)} keys -> {out_path}", flush=True)
    else:
        torch.save(export, out_path)
        print(f"[export_vigor_teacher_logits_llama] saved {len(export)} samples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
