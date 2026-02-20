#!/usr/bin/env python
# coding: utf-8

"""
Mask3D-Vigor Llama-step (minimal closed loop):

- Listener: Vigor ReferIt3D transformer (Mask3D input), initialized from a BERT step-slot checkpoint.
- Step guidance: a causal LLM encodes each step independently:
    "{utterance} {step_text} <stepK>"
  and we use the hidden state at <stepK> as `order_embeds`.
- Training signal: (1) referential loss through the frozen listener, plus
  (2) optional distillation loss to BERT step-slot embeddings (teacher).

This script intentionally avoids DataParallel (requires B=1-per-replica string lists).
Run with a single GPU (CUDA_VISIBLE_DEVICES set by the caller).
"""

import sys
from pathlib import Path

_vigor_root = Path(__file__).resolve().parents[2]  # .../Vigor
sys.path.insert(0, str(_vigor_root))

import os
import os.path as osp
import time
import warnings
import re

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import tqdm
from termcolor import colored

from transformers import BertTokenizer

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import (
    load_scan_related_data,
    load_referential_data,
    compute_auxiliary_data,
)
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.models.referit3d_net import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions

from referit3d.models.llama_stepslot import (
    LlamaStepSlotConfig,
    LlamaStepSlotOrderEncoder,
    ReferIt3DNetTransformerLlamaStepSlot,
)


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


def _env_float_or_none(name: str) -> float | None:
    v = os.environ.get(name, None)
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_int_list(s: str) -> list[int]:
    raw = str(s or "").strip()
    if not raw:
        return []
    raw = raw.replace(" ", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    # unique + sorted for MultiStepLR
    return sorted(set(out))


def _maybe_infer_lora_from_resume_ckpt(resume_ckpt: str) -> None:
    """
    Auto-enable LoRA adapters to match a resume checkpoint (useful for probe/eval scripts).

    Enable via:
      - VIGOR_LLM_LORA_AUTO=1   (recommended)
        or
      - VIGOR_LLM_LORA=auto

    If the checkpoint contains LoRA keys, we infer:
      - VIGOR_LLM_LORA=1
      - VIGOR_LLM_LORA_R (from lora_A shape)
      - VIGOR_LLM_LORA_LAST_N (from covered layer indices, assuming a suffix)
      - VIGOR_LLM_LORA_TARGETS (from keys, e.g. q_proj,v_proj)

    We do NOT infer alpha/dropout; keep whatever the environment specifies (defaults are ok).
    """
    try:
        auto = _env_flag("VIGOR_LLM_LORA_AUTO", "0") or (_env_str("VIGOR_LLM_LORA", "").strip().lower() == "auto")
    except Exception:
        auto = False
    if not auto:
        return
    if not resume_ckpt or not osp.isfile(resume_ckpt):
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

    # Assume LoRA is applied on a suffix of layers and includes the last layer present in keys.
    last_n = None
    if layers:
        try:
            max_layer = max(layers)
            min_layer = min(layers)
            if max_layer >= min_layer:
                last_n = int(max_layer - min_layer + 1)
        except Exception:
            last_n = None

    # Only set env vars if the user did not specify explicit values already.
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
        f"last_n={os.environ.get('VIGOR_LLM_LORA_LAST_N', '')} "
        f"targets={os.environ.get('VIGOR_LLM_LORA_TARGETS', '')} "
        f"ckpt={resume_ckpt}",
        flush=True,
    )


def _maybe_add_step_tokens(tokenizer, order_len: int):
    tokens = [f"<step{i+1}>" for i in range(int(order_len))]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in tokens]
    return tokens, ids, int(added)


def log_train_test_information(logger, epoch, train_meters, test_meters, timings, best_test_acc, best_test_epoch, args):
    logger.info("Epoch:{}".format(epoch))
    for phase in ["train", "test"]:
        meters = train_meters if phase == "train" else test_meters
        info = "{}: Total-Loss {:.4f}, Listening-Acc {:.4f}".format(
            phase,
            meters[phase + "_total_loss"],
            meters[phase + "_referential_acc"],
        )

        # Optional: llama-step-slot distillation diagnostics.
        d_step_key = phase + "_llm_distill_step"
        d_glb_key = phase + "_llm_distill_global"
        d_step_mp_key = phase + "_llm_distill_step_mp"
        d_glb_mp_key = phase + "_llm_distill_global_mp"
        d_mp_op_key = phase + "_llm_distill_mp_op"
        if d_step_key in meters:
            info += ", Distill-Step {:.4f}".format(meters[d_step_key])
        if d_glb_key in meters:
            info += ", Distill-Global {:.4f}".format(meters[d_glb_key])
        if d_step_mp_key in meters:
            info += ", Distill-Step-MP {:.4f}".format(meters[d_step_mp_key])
        if d_glb_mp_key in meters:
            info += ", Distill-Global-MP {:.4f}".format(meters[d_glb_mp_key])
        if d_mp_op_key in meters:
            info += ", Distill-MP2OP {:.4f}".format(meters[d_mp_op_key])

        if getattr(args, "use_scannet200_obj_cls", False):
            key = phase + "_scannet_object_cls_acc"
            if key in meters:
                info += ", ScanNet200-Obj-Acc: {:.4f}".format(meters[key])
        elif args.obj_cls_alpha > 0 and (phase + "_object_cls_acc") in meters:
            info += ", Object-Clf-Acc: {:.4f}".format(meters[phase + "_object_cls_acc"])

        if args.lang_cls_alpha > 0 and (phase + "_txt_cls_acc") in meters:
            info += ", Text-Clf-Acc: {:.4f}".format(meters[phase + "_txt_cls_acc"])

        logger.info(info)
        logger.info("{}: Epoch-time {:.3f}".format(phase, timings[phase]))

    logger.info("Best so far {:.3f} (@epoch {})".format(best_test_acc, best_test_epoch))


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")

    args = parse_arguments()

    # Force single GPU: this script relies on Python string lists in the batch.
    if int(getattr(args, "n_gpus", 1) or 1) != 1:
        raise RuntimeError("train_referit3d_llama_stepslot.py requires --n-gpus 1 (no DataParallel).")

    llm_path = _env_str("VIGOR_LLM_MODEL_PATH", "")
    if not llm_path:
        raise RuntimeError("Missing env VIGOR_LLM_MODEL_PATH (local path to causal LLM weights).")

    # Optional: resume full (wrapper) checkpoint, including the LLM adapter/memory tokens.
    # This is required for Phase-B (distill-off) fine-tuning to actually start from
    # the Phase-A learned LLM-side parameters.
    #
    # - If `VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT=1`, optimizer/scheduler states are restored
    #   and training continues from (saved epoch + 1).
    # - Otherwise, only model weights are loaded and epoch counting restarts from 1.
    llm_stepslot_resume_ckpt = _env_str("VIGOR_LLM_STEPSLOT_RESUME_CKPT", "")
    llm_stepslot_resume_with_opt = _env_flag("VIGOR_LLM_STEPSLOT_RESUME_WITH_OPT", "0")
    _maybe_infer_lora_from_resume_ckpt(llm_stepslot_resume_ckpt)

    llm_max_len = _env_int("VIGOR_LLM_MAX_LEN", 64)
    llm_mem_tokens = _env_int("VIGOR_LLM_MEM_TOKENS", 0)
    llm_distill_w = float(_env_str("VIGOR_LLM_DISTILL_W", "1.0") or "1.0")
    llm_distill_type = _env_str("VIGOR_LLM_DISTILL_TYPE", "cos").lower()
    llm_global_distill_w = float(_env_str("VIGOR_LLM_GLOBAL_DISTILL_W", "1.0") or "1.0")
    llm_global_distill_type = _env_str("VIGOR_LLM_GLOBAL_DISTILL_TYPE", "cos").lower()
    llm_use_bf16 = _env_flag("VIGOR_LLM_USE_BF16", "1")

    print(
        "[Vigor][llama_stepslot][args] "
        f"mode={args.mode} order_len={getattr(args, 'order_len', None)} "
        f"mask3d_feature_root={getattr(args, 'mask3d_feature_root', None)} "
        f"VIGOR_USE_PRED_BOX_INFO={os.environ.get('VIGOR_USE_PRED_BOX_INFO', '0')} "
        f"VIGOR_PRED_CLASS_MASK_MODE={os.environ.get('VIGOR_PRED_CLASS_MASK_MODE', 'normal')} "
        f"llm_path={llm_path} llm_max_len={llm_max_len} llm_mem_tokens={llm_mem_tokens} "
        f"distill_w={llm_distill_w} distill_type={llm_distill_type} "
        f"global_w={llm_global_distill_w} global_type={llm_global_distill_type}",
        flush=True,
    )

    # Data
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    # GPU env
    set_gpu_to_zero_position(args.gpu)
    seed_training_code(args.random_seed)
    print(f"[GPU INFO] visible GPUs: {torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    device = torch.device("cuda")

    # Listener
    n_classes = len(class_to_idx) - 1
    pad_idx = class_to_idx["pad"]
    class_name_list = list(class_to_idx.keys())

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    if _env_flag("VIGOR_STEP_MARKERS", "0"):
        step_tokens, step_token_ids, added = _maybe_add_step_tokens(tokenizer, getattr(args, "order_len", 4))
        args.vigor_step_tokens = step_tokens
        args.vigor_step_token_ids = step_token_ids
        print(f"[Vigor][step_tokens] enabled tokens={step_tokens} added={added}", flush=True)

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

    # Initialize listener weights from a BERT step-slot checkpoint.
    listener_init = _env_str("VIGOR_LISTENER_INIT_CKPT", "") or str(getattr(args, "resume_path", "") or "")
    if not listener_init:
        raise RuntimeError("Missing listener init ckpt: set --resume-path or env VIGOR_LISTENER_INIT_CKPT.")
    warnings.warn("Loading listener init checkpoint (BERT step-slot). Optimizer state is NOT loaded.")
    _ = load_state_dicts(listener_init, map_location=device, model=listener)

    # LLM step-slot encoder + wrapper model
    llm_cfg = LlamaStepSlotConfig(
        model_path=llm_path,
        order_len=int(getattr(args, "order_len", 4)),
        max_length=llm_max_len,
        memory_tokens=int(llm_mem_tokens),
        distill_w=llm_distill_w,
        distill_type=llm_distill_type,
        global_distill_w=llm_global_distill_w,
        global_distill_type=llm_global_distill_type,
        freeze_llm_except_step_rows=True,
        local_files_only=True,
        use_bf16=bool(llm_use_bf16),
    )
    llm = LlamaStepSlotOrderEncoder(out_dim=int(getattr(args, "inner_dim", 768)), cfg=llm_cfg).to(device)
    model = ReferIt3DNetTransformerLlamaStepSlot(listener=listener, llm=llm, cfg=llm_cfg).to(device)

    # Optimizer: only train parameters with requires_grad=True.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found (check LLM freezing settings).")

    # Optional: separate learning rates for listener vs LLM-side params.
    # This is useful when Phase-B unfreezes the full listener: we typically want a smaller LR
    # for the large listener and a slightly larger LR for LoRA/step-token adapters.
    listener_lr = _env_float_or_none("VIGOR_LISTENER_LR")
    llm_lr = _env_float_or_none("VIGOR_LLM_LR")
    default_lr = float(getattr(args, "init_lr", 1e-4))
    if (listener_lr is not None) or (llm_lr is not None):
        listener_ids = {id(p) for p in model.listener.parameters()}
        listener_params = [p for p in model.parameters() if p.requires_grad and (id(p) in listener_ids)]
        other_params = [p for p in model.parameters() if p.requires_grad and (id(p) not in listener_ids)]
        groups = []
        if listener_params:
            groups.append({"params": listener_params, "lr": float(listener_lr if listener_lr is not None else default_lr)})
        if other_params:
            groups.append({"params": other_params, "lr": float(llm_lr if llm_lr is not None else default_lr)})
        if not groups:
            groups = [{"params": trainable, "lr": default_lr}]
        optimizer = optim.Adam(groups)
        print(
            f"[Vigor][llama_stepslot][optim] param_groups=1 default_lr={default_lr} "
            f"listener_lr={listener_lr} llm_lr={llm_lr} "
            f"n_listener={len(listener_params)} n_other={len(other_params)}",
            flush=True,
        )
    else:
        optimizer = optim.Adam(trainable, lr=default_lr)
    # LR schedule: allow per-experiment override via env (useful for multi-phase scripts).
    # Example:
    #   export VIGOR_LR_MILESTONES="12,16,18"
    #   export VIGOR_LR_GAMMA="0.65"
    default_milestones = [40, 50, 60, 70, 80, 90]
    milestones = default_milestones
    ms = _env_str("VIGOR_LR_MILESTONES", "")
    parsed = _parse_int_list(ms)
    if parsed:
        milestones = parsed
    gamma = _env_float("VIGOR_LR_GAMMA", 0.65)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=float(gamma))
    print(f"[Vigor][llama_stepslot][lr] milestones={milestones} gamma={float(gamma)}", flush=True)

    start_training_epoch = 1
    best_test_acc = -1.0
    best_test_epoch = -1
    last_test_acc = -1.0
    last_test_epoch = -1

    # Resume wrapper checkpoint (model-only or full states) if provided.
    resumed_epoch = None
    if llm_stepslot_resume_ckpt:
        if args.mode == "train" and llm_stepslot_resume_with_opt:
            resumed_epoch = load_state_dicts(
                llm_stepslot_resume_ckpt,
                map_location=device,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            if resumed_epoch is not None:
                start_training_epoch = int(resumed_epoch) + 1
        else:
            resumed_epoch = load_state_dicts(llm_stepslot_resume_ckpt, map_location=device, model=model)
            start_training_epoch = 1

        print(
            f"[Vigor][llama_stepslot][resume] ckpt={llm_stepslot_resume_ckpt} "
            f"with_opt={int(llm_stepslot_resume_with_opt)} resumed_epoch={resumed_epoch} "
            f"start_epoch={start_training_epoch}",
            flush=True,
        )

        # When resuming optimizer/scheduler states, the checkpoint LR overrides `args.init_lr`.
        # For multi-phase scripts, it's often desirable to keep moments but reset LR.
        # Enable with:
        #   export VIGOR_OVERRIDE_OPT_LR=1   (uses args.init_lr)
        # or set explicitly:
        #   export VIGOR_FORCE_LR=2e-5
        if args.mode == "train" and llm_stepslot_resume_with_opt:
            override_opt_lr = _env_flag("VIGOR_OVERRIDE_OPT_LR", "0")
            force_lr = _env_float_or_none("VIGOR_FORCE_LR")
            if override_opt_lr or (force_lr is not None):
                new_lr = float(force_lr if force_lr is not None else float(getattr(args, "init_lr", 1e-4)))
                for g in optimizer.param_groups:
                    g["lr"] = new_lr
                try:
                    lr_scheduler.base_lrs = [g["lr"] for g in optimizer.param_groups]
                    lr_scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
                except Exception:
                    pass
                print(f"[Vigor][llama_stepslot][resume] overridden_lr={new_lr}", flush=True)

        # Initialize `best_test_acc` to the checkpoint performance so Phase-B doesn't
        # overwrite best_model on the first epoch if it temporarily degrades.
        #
        # In pure evaluation mode, avoid the extra (duplicate) evaluation pass.
        if args.mode == "train":
            try:
                init_meters = evaluate_on_dataset(
                    model, data_loaders["test"], {}, device, pad_idx, args=args, tokenizer=tokenizer
                )
                init_acc = float(init_meters.get("test_referential_acc", float("nan")))
                if init_acc == init_acc:  # not NaN
                    best_test_acc = init_acc
                    best_test_epoch = int(resumed_epoch) if resumed_epoch is not None else 0
                    print(
                        f"[Vigor][llama_stepslot][resume] init_test_acc={init_acc:.4f} best_epoch={best_test_epoch}",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[Vigor][llama_stepslot][resume][warn] failed to eval init ckpt: {type(e).__name__}: {e}",
                    flush=True,
                )

    if args.mode == "train":
        logger = create_logger(args.log_dir)
        logger.info("Starting Llama-step-slot training (listener frozen).")

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc="epochs") as bar:
            timings = {}
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr(), flush=True)

                tic = time.time()
                train_meters = single_epoch_train(
                    model, data_loaders["train"], {}, optimizer, device, pad_idx, args=args, tokenizer=tokenizer, epoch=epoch
                )
                toc = time.time()
                timings["train"] = (toc - tic) / 60.0

                tic = time.time()
                test_meters = evaluate_on_dataset(
                    model, data_loaders["test"], {}, device, pad_idx, args=args, tokenizer=tokenizer
                )
                toc = time.time()
                timings["test"] = (toc - tic) / 60.0

                eval_acc = float(test_meters.get("test_referential_acc", float("nan")))
                last_test_acc = eval_acc
                last_test_epoch = epoch
                lr_scheduler.step()

                save_state_dicts(
                    osp.join(args.checkpoint_dir, "last_model.pth"),
                    epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                if best_test_acc < eval_acc:
                    logger.info(colored("Test accuracy, improved @epoch {}".format(epoch), "green"))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch
                    save_state_dicts(
                        osp.join(args.checkpoint_dir, "best_model.pth"),
                        epoch,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                    )
                else:
                    logger.info(colored("Test accuracy, did not improve @epoch {}".format(epoch), "red"))

                log_train_test_information(
                    logger, epoch, train_meters, test_meters, timings, best_test_acc, best_test_epoch, args
                )

                train_meters.update(test_meters)
                bar.refresh()

        with open(osp.join(args.checkpoint_dir, "final_result.txt"), "w") as f_out:
            f_out.write(("Best accuracy: {:.4f} (@epoch {})".format(best_test_acc, best_test_epoch)))
            f_out.write(("Last accuracy: {:.4f} (@epoch {})".format(last_test_acc, last_test_epoch)))

        logger.info("Finished training successfully.")

    elif args.mode == "evaluate":
        try:
            test_n = int(len(data_loaders["test"].dataset))
        except Exception:
            test_n = -1
        if test_n == 0:
            print(
                "[Vigor][eval][warn] Empty test split (0 samples). "
                "This usually means your *_test_*.csv is empty or all samples fell into the train split. "
                "Regenerate the CSV with more scenes (ensure ScanNet val scenes exist) before running eval.",
                flush=True,
            )
            raise SystemExit(0)
        meters = evaluate_on_dataset(model, data_loaders["test"], {}, device, pad_idx, args=args, tokenizer=tokenizer)
        # mk-only probe mode: only care about `[Vigor][varlen_mk]` printed from evaluate_on_dataset().
        # Skip the normal ReferIt3D accuracy summary + heavy analyze_predictions().
        try:
            mk_only = str(__import__("os").environ.get("VIGOR_VARLEN_MK_ONLY", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        except Exception:
            mk_only = False
        if mk_only:
            print("[Vigor][varlen_mk] done (mk-only probe).", flush=True)
            raise SystemExit(0)
        print("Reference-Accuracy: {:.4f}".format(meters["test_referential_acc"]))
        if "test_queryacc" in meters:
            print("QueryAcc: {:.4f}".format(meters.get("test_queryacc", 0.0)))
            if "test_queryacc_easy" in meters or "test_queryacc_hard" in meters:
                print(
                    "QueryAcc Easy/Hard: {:.4f} / {:.4f}".format(
                        meters.get("test_queryacc_easy", 0.0),
                        meters.get("test_queryacc_hard", 0.0),
                    )
                )
            if "test_queryacc_vdep" in meters or "test_queryacc_vindep" in meters:
                print(
                    "QueryAcc V-Dep/V-Indep: {:.4f} / {:.4f}".format(
                        meters.get("test_queryacc_vdep", 0.0),
                        meters.get("test_queryacc_vindep", 0.0),
                    )
                )
        if "test_bbox_acc_iou_25" in meters and "test_bbox_acc_iou_50" in meters:
            print(
                "BBox-Acc@IoU(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_bbox_acc_iou_25", 0.0),
                    meters.get("test_bbox_acc_iou_50", 0.0),
                )
            )
        if "test_bbox_acc_iou_25_unique" in meters and "test_bbox_acc_iou_25_multiple" in meters:
            print(
                "BBox-Acc@IoU Unique(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_bbox_acc_iou_25_unique", 0.0),
                    meters.get("test_bbox_acc_iou_50_unique", 0.0),
                )
            )
            print(
                "BBox-Acc@IoU Multiple(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_bbox_acc_iou_25_multiple", 0.0),
                    meters.get("test_bbox_acc_iou_50_multiple", 0.0),
                )
            )
        if "test_bbox_mean_iou" in meters:
            print("BBox-Mean-IoU: {:.4f}".format(meters.get("test_bbox_mean_iou", 0.0)))
        # Decomposed bbox diagnostics: decision correctness vs box quality.
        if (
            "test_bbox_acc_iou_25_correct" in meters
            and "test_bbox_acc_iou_50_correct" in meters
            and "test_bbox_acc_iou_25_wrong" in meters
            and "test_bbox_acc_iou_50_wrong" in meters
        ):
            n_all = int(meters.get("test_bbox_n", 0) or 0)
            n_c = int(meters.get("test_bbox_n_correct", 0) or 0)
            n_w = int(meters.get("test_bbox_n_wrong", 0) or 0)
            print(f"BBox-Stats: n={n_all} correct={n_c} wrong={n_w}")
            print(
                "BBox-Acc@IoU Correct(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_bbox_acc_iou_25_correct", 0.0),
                    meters.get("test_bbox_acc_iou_50_correct", 0.0),
                )
            )
            print(
                "BBox-Acc@IoU Wrong(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_bbox_acc_iou_25_wrong", 0.0),
                    meters.get("test_bbox_acc_iou_50_wrong", 0.0),
                )
            )
        if "test_bbox_acc_iou_25_target" in meters and "test_bbox_acc_iou_50_target" in meters:
            n_t = int(meters.get("test_bbox_n_target", 0) or 0)
            print(
                "TargetBox-Acc@IoU(0.25/0.50): {:.4f} / {:.4f} (n={})".format(
                    meters.get("test_bbox_acc_iou_25_target", 0.0),
                    meters.get("test_bbox_acc_iou_50_target", 0.0),
                    n_t,
                )
            )
        if "test_f1_iou_25" in meters and "test_f1_iou_50" in meters:
            print(
                "F1@IoU(0.25/0.50): {:.4f} / {:.4f}".format(
                    meters.get("test_f1_iou_25", 0.0),
                    meters.get("test_f1_iou_50", 0.0),
                )
            )
        if getattr(args, "use_scannet200_obj_cls", False) and "test_scannet_object_cls_acc" in meters:
            print("ScanNet200-Obj-Accuracy: {:.4f}".format(meters["test_scannet_object_cls_acc"]))
        elif "test_object_cls_acc" in meters:
            print("Object-Clf-Accuracy: {:.4f}".format(meters["test_object_cls_acc"]))
        print("Text-Clf-Accuracy {:.4f}:".format(meters.get("test_txt_cls_acc", 0.0)))

        out_file = osp.join(args.checkpoint_dir, "test_result.txt")
        # External datasets may not support ReferIt3D-specific analysis. Allow skipping it explicitly.
        try:
            skip_analyze = str(os.environ.get("VIGOR_SKIP_ANALYZE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            skip_analyze = False
        # Some custom datasets may bypass the standard ReferIt3D "multiple-distractors" setup.
        # If `analyze_predictions` cannot run (e.g. due to unexpected stimulus format), keep eval robust.
        try:
            test_ds_len = int(len(data_loaders["test"].dataset))
        except Exception:
            test_ds_len = -1
        if test_ds_len <= 0:
            raise SystemExit(0)
        if skip_analyze:
            raise SystemExit(0)
        try:
            res = analyze_predictions(
                model,
                data_loaders["test"].dataset,
                class_to_idx,
                pad_idx,
                device,
                args,
                out_file=out_file,
                tokenizer=tokenizer,
            )
            print(res)
        except Exception as e:
            print(f"[Vigor][eval][warn] analyze_predictions failed: {type(e).__name__}: {e}", flush=True)
            raise SystemExit(0)
