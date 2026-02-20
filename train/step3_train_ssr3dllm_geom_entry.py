#!/usr/bin/env python

import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]  # release repo root
sys.path.insert(0, str(repo_root))               # Add repo root first so local packages take precedence.
sys.path.insert(0, str(repo_root / "src"))       # Also add src/ if the project keeps source code there.

import warnings

# Reduce noisy Lightning warnings about `self.log(..., sync_dist=True)` in DDP.
# We keep the behavior unchanged; this only silences the advisory.
try:
    from pytorch_lightning.utilities.warnings import PossibleUserWarning

    warnings.filterwarnings(
        "ignore",
        category=PossibleUserWarning,
        message=r"It is recommended to use `self\.log\(",
    )
except Exception:
    pass

import argparse
from datetime import datetime

from config import (
    clone_config,
    apply_overrides,
    load_yaml_config,
    refresh_links,
)
from main_run import run_train, run_test


def build_cfg(mode: str):
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    exp_name = os.environ.get("EXPNAME", f"step3_ssr3dllm_geom_{num_gpus}GPUS")
    data_cfg_path = os.environ.get("DATA_CONFIG", "baseline/core/conf/data/indoor_grounding.yaml")
    model_cfg_path = os.environ.get("MODEL_CONFIG", "baseline/core/conf/model/mask3d_lang.yaml")
    trainer_cfg_path = os.environ.get("TRAINER_CONFIG", "baseline/core/conf/trainer/trainer50.yaml")
    llm_config = os.environ.get("LLM_CONFIG", "baseline/core/conf/llm/tiny_vicuna_len512.json")
    llm_data_config = os.environ.get("LLM_DATA_CONFIG", "").strip()
    topk = int(os.environ.get("CURR_TOPK", "750"))
    pretrained = os.environ.get("PRETRAINED", "")
    resume_ckpt = os.environ.get("SSR3DLLM_RESUME_CKPT", "").strip()
    test_ckpt = os.environ.get("SSR3DLLM_TEST_CKPT", "").strip()
    geom_only = os.environ.get("SSR3DLLM_GEOM_ONLY", "0") not in {"0", "", "false", "False"}

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "")
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    cfg = clone_config()

    # Fail early on missing configs to avoid silently falling back to defaults
    # (which can lead to misleading "Data statistics" mixes like rel3dref dominating).
    def _must_exist(path_s: str, name: str) -> None:
        if not path_s:
            return
        p = Path(path_s)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"[step3_train_ssr3dllm_geom_entry] {name} not found: {p}")

    _must_exist(data_cfg_path, "DATA_CONFIG")
    _must_exist(model_cfg_path, "MODEL_CONFIG")
    _must_exist(trainer_cfg_path, "TRAINER_CONFIG")
    _must_exist(llm_config, "LLM_CONFIG")
    _must_exist(llm_data_config, "LLM_DATA_CONFIG")

    general_overrides = {
        "experiment_name": exp_name,
        "project_name": "scannet200",
        "gpus": num_gpus,
        "train_mode": mode == "train",
        "filter_scene00": False,
        "topk_per_image": topk,
        "llm_config": llm_config,
        # Sampling ratios for per-scene language sources (detection/scanrefer/m3dref/rel3dref/...).
        # Default comes from `baseline/core/conf/config_base.yaml`, but allow overriding via env for experiments.
        "llm_data_config": llm_data_config or cfg.general.llm_data_config,
        "enable_ssr3dllm_geom": True,
        "ssr3dllm_geom_only": geom_only,
        "ssr3dllm_geom_weight": 1.0,
        "ssr3dllm_ref_loss_weight": _env_float("SSR3DLLM_REF_LOSS_WEIGHT", 1.0),
        "ssr3dllm_anchor_loss_weight": _env_float("SSR3DLLM_ANCHOR_LOSS_WEIGHT", 1.0),
        "ssr3dllm_relcls_loss_weight": _env_float("SSR3DLLM_RELCLS_LOSS_WEIGHT", 1.0),
        "ssr3dllm_chain_loss_weight": _env_float("SSR3DLLM_CHAIN_LOSS_WEIGHT", 1.0),
        # Offline teacher distillation (optional; requires env paths in geom head).
        "ssr3dllm_distill_vigor_weight": _env_float("SSR3DLLM_DISTILL_VIGOR_WEIGHT", 0.0),
        "ssr3dllm_distill_temperature": _env_float("SSR3DLLM_DISTILL_TEMPERATURE", 1.0),
        "ssr3dllm_bert_model": "pretrained/bert-base-uncased",
        "timestamp": datetime.now().strftime("%m-%d-%H-%M-%S"),
    }

    # Optional: cap the number of per-scene evaluation queries (fast sanity checks).
    # NOTE: this affects only evaluation sampling in `trainer.prepare_llm()` when not training.
    # Env:
    #   SSR3DLLM_MAX_EVAL_QUERIES=100
    max_eval_raw = str(os.environ.get("SSR3DLLM_MAX_EVAL_QUERIES", "")).strip()
    if max_eval_raw:
        try:
            general_overrides["max_eval_queries"] = int(max_eval_raw)
        except Exception:
            pass
    if mode == "train":
        # Two checkpoint modes:
        # 1) Fresh init from `PRETRAINED` (weights-only load in get_parameters()).
        # 2) Full resume from a Lightning `.ckpt` (restores optimizer/scheduler/global_step).
        #    Enable by setting `SSR3DLLM_RESUME_CKPT=/path/to/last-epoch.ckpt`.
        general_overrides["checkpoint"] = None if resume_ckpt else pretrained
    else:
        # For `--mode test`, allow loading a checkpoint for evaluation.
        # This uses the same weights-loading path as other scripts (not Lightning resume).
        general_overrides["checkpoint"] = test_ckpt or pretrained or None

    apply_overrides(cfg, {
        "general": general_overrides,
        "optimizer": {
            "lr": _env_float("SSR3DLLM_OPTIM_LR", 8e-4),
        },
    })

    apply_overrides(cfg.data, load_yaml_config(data_cfg_path, cfg))
    apply_overrides(cfg.model, load_yaml_config(model_cfg_path, cfg))
    apply_overrides(cfg.trainer, load_yaml_config(trainer_cfg_path, cfg))
    if mode == "train" and resume_ckpt:
        try:
            apply_overrides(cfg.trainer, {"resume_from_checkpoint": resume_ckpt})
        except Exception:
            pass

    # Optional: limit batches to speed up debug/eval without touching CLI args.
    def _parse_limit(raw: str):
        s = str(raw).strip()
        if not s:
            return None
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return None

    limit_test = _parse_limit(os.environ.get("SSR3DLLM_LIMIT_TEST_BATCHES", ""))
    if limit_test is not None:
        apply_overrides(cfg.trainer, {"limit_test_batches": limit_test})
    limit_val = _parse_limit(os.environ.get("SSR3DLLM_LIMIT_VAL_BATCHES", ""))
    if limit_val is not None:
        apply_overrides(cfg.trainer, {"limit_val_batches": limit_val})
    limit_train = _parse_limit(os.environ.get("SSR3DLLM_LIMIT_TRAIN_BATCHES", ""))
    if limit_train is not None:
        apply_overrides(cfg.trainer, {"limit_train_batches": limit_train})

    # Optional: completely skip internal Lightning validation/test loops.
    # This is useful when you only care about external evaluations (e.g. Vigor step-slot)
    # driven by existing checkpoints.
    skip_internal_eval = os.environ.get("SSR3DLLM_SKIP_INTERNAL_EVAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if skip_internal_eval:
        if mode == "train":
            apply_overrides(cfg.trainer, {"limit_val_batches": 0})
            apply_overrides(cfg.trainer, {"num_sanity_val_steps": 0})
        else:
            # Lightning may skip `test_epoch_end` entirely if there are zero batches.
            # Keep a single (cheap) batch so we can still run external evals in epoch_end hooks
            # (e.g. SSR3DLLM_VIGOR_STEPSLOT_EVAL), while avoiding expensive internal metrics.
            apply_overrides(cfg.trainer, {"limit_test_batches": 1})
    # Sanity-check behavior:
    # - By default follow the trainer YAML (so smoke configs can catch val bugs early).
    # - Allow overriding via env var SSR3DLLM_NUM_SANITY_VAL_STEPS when needed.
    sanity_raw = os.environ.get("SSR3DLLM_NUM_SANITY_VAL_STEPS", "").strip()
    if sanity_raw:
        try:
            apply_overrides(cfg.trainer, {"num_sanity_val_steps": int(sanity_raw)})
        except Exception:
            pass

    apply_overrides(cfg.model, {
        "use_rel3d_geom": True,
        "rel3d_use_continuous_geom": True,
    })

    refresh_links(cfg)

    # Step-token SFT: adjust checkpoint monitor/filename to use rel3d metrics,
    # since AP metrics may be disabled/irrelevant in this experiment.
    def _flag(name: str) -> bool:
        v = os.environ.get(name, "0")
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    if _flag("SSR3DLLM_STEP_TOKENS") and _flag("SSR3DLLM_REL3D_OUTPUT_STEPS") and _flag("SSR3DLLM_DISABLE_LLM_GROUNDING"):
        if cfg.callbacks and isinstance(cfg.callbacks, list) and len(cfg.callbacks) > 0:
            ckpt_cb = cfg.callbacks[0]
            if isinstance(ckpt_cb, dict) and ckpt_cb.get("monitor") == "val_mean_ap_50":
                ckpt_cb["monitor"] = "val_ssr3dllm_target_acc"
                ckpt_cb["mode"] = "max"
                ckpt_cb["filename"] = "{epoch}-{val_ssr3dllm_target_acc:.3f}"

    # Optional: allow redirecting ALL artifacts/checkpoints to a different root.
    # Example: export SSR3DLLM_SAVE_DIR_ROOT="/abs/path/to/outputs/ssr3dllm"
    save_root = os.environ.get("SSR3DLLM_SAVE_DIR_ROOT", "").strip()
    if save_root:
        cfg.general.save_dir = f"{save_root.rstrip('/')}/{cfg.general.experiment_name}"
        if cfg.callbacks:
            cfg.callbacks[0]["dirpath"] = cfg.general.save_dir
        if cfg.logging:
            cfg.logging[0]["save_dir"] = cfg.general.save_dir

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    args = parser.parse_args()

    cfg = build_cfg(args.mode)
    if args.mode == "train":
        run_train(cfg)
    else:
        run_test(cfg)


if __name__ == "__main__":
    main()
