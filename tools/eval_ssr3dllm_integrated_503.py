#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate integrated SSR3DLLM checkpoint on ReferIt3D (503 protocol).")
    p.add_argument("--checkpoint", required=True, help="Integrated SSR3DLLM checkpoint (.ckpt).")
    p.add_argument("--profile", default="503", help="Listener profile for bundled ckpt: 503|519|main|ub.")
    p.add_argument("--datasets", default="nr3d,sr3d", help="Comma-separated datasets: nr3d,sr3d.")
    p.add_argument("--nr3d-train-csv", required=True, help="Path to nr3d train csv (test csv auto-derived).")
    p.add_argument("--sr3d-train-csv", required=True, help="Path to sr3d train csv (test csv auto-derived).")
    p.add_argument("--scannet-file", required=True, help="ReferIt3D aligned ScanNet pkl.")
    p.add_argument("--bert-path", required=True, help="BERT path used by listener runtime.")
    p.add_argument("--mask3d-feature-root", required=True, help="Mask3D train features root.")
    p.add_argument("--mask3d-feature-root-test", required=True, help="Mask3D test features root.")
    p.add_argument("--scannet-processed-root", required=True, help="Processed ScanNet root for BaselineModelAPI.")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0, help="0 means full test split.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output-json", default="", help="Optional output JSON path.")
    return p.parse_args()


def _resolve_test_csv(train_csv: Path) -> Path:
    text = str(train_csv)
    if "train" in text:
        text = text.replace("train", "test")
    text = re.sub(r"_\d+\.\d+", "", text)
    return Path(text)


def _to_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if torch.is_tensor(x):
        if x.dim() == 0:
            return [x.item()]
        return list(x)
    return [x]


def _extract_pred_qid(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, list) and first:
            try:
                return int(first[0])
            except Exception:
                return None
        try:
            return int(first)
        except Exception:
            return None
    return None


def _mapping_get(mapping: Dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    key_str = str(key)
    if key_str in mapping:
        return mapping[key_str]
    return None


def _resolve_query_idx(mapping: Dict[Any, Any], inst_id: int) -> Optional[int]:
    q_idx = _mapping_get(mapping, inst_id)
    if q_idx is None:
        has_zero = (_mapping_get(mapping, 0) is not None)
        has_one = (_mapping_get(mapping, 1) is not None)
        cand = int(inst_id) + 1
        if inst_id >= 0 and (not has_zero) and has_one:
            q_idx = _mapping_get(mapping, cand)
    if q_idx is None:
        return None
    try:
        return int(q_idx)
    except Exception:
        return None


def _load_context_features(mask3d_feature_path: str, instance_ids: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
    try:
        payload = torch.load(mask3d_feature_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    queries = payload.get("object_queries", None)
    if queries is None:
        return None
    if isinstance(queries, np.ndarray):
        queries = torch.from_numpy(queries)
    if not torch.is_tensor(queries) or queries.dim() != 2:
        return None

    mapping = payload.get("gt_to_query_map", {}) or {}
    if not isinstance(mapping, dict):
        mapping = {}

    inst_ids = torch.as_tensor(instance_ids).long().view(-1)
    n_ctx = int(inst_ids.numel())
    feat_dim = int(queries.shape[1])
    out_q = torch.zeros((n_ctx, feat_dim), dtype=torch.float32)
    out_box = torch.zeros((n_ctx, 4), dtype=torch.float32)
    queries = queries.float()
    pred_box_info = payload.get("pred_box_info", None)
    if isinstance(pred_box_info, np.ndarray):
        pred_box_info = torch.from_numpy(pred_box_info)
    if torch.is_tensor(pred_box_info):
        pred_box_info = pred_box_info.float()
    else:
        pred_box_info = None

    for j in range(n_ctx):
        inst_id = int(inst_ids[j].item())
        if inst_id < 0:
            continue
        q_idx = _resolve_query_idx(mapping, inst_id)
        if q_idx is None:
            continue
        if 0 <= int(q_idx) < int(queries.shape[0]):
            out_q[j] = queries[int(q_idx)]
            if pred_box_info is not None and pred_box_info.dim() == 2 and pred_box_info.size(1) == 4:
                if 0 <= int(q_idx) < int(pred_box_info.shape[0]):
                    out_box[j] = pred_box_info[int(q_idx)]
    return {"queries": out_q, "box_info": out_box}


def _build_api(args: argparse.Namespace):
    from baseline.api.baseline_interface import BaselineModelAPI

    config_root = Path("baseline/core/conf")
    scannet_root = Path(args.scannet_processed_root).expanduser().resolve()
    label_db = scannet_root / "label_database.yaml"
    color_stat = scannet_root / "color_mean_std.yaml"

    ds_override: Dict[str, Any] = {"data_dir": str(scannet_root)}
    if label_db.is_file():
        ds_override["label_db_filepath"] = str(label_db)
    if color_stat.is_file():
        ds_override["color_mean_std"] = str(color_stat)

    extra_overrides: Dict[str, Any] = {
        "data": {
            "train_dataset": dict(ds_override),
            "validation_dataset": dict(ds_override),
            "test_dataset": dict(ds_override),
        }
    }

    cfg = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "split_type": str(args.split),
        "scannet_processed_root": str(scannet_root),
        "config_path": str(config_root),
        "data_config": str(config_root / "data/indoor_dialog.yaml"),
        "model_config": str(config_root / "model/mask3d_lang.yaml"),
        "trainer_config": str(config_root / "trainer/trainer50.yaml"),
        "llm_config": str(config_root / "llm/tiny_vicuna_len512_bs4.json"),
        "llm_data_config": str(config_root / "llm/det10.json"),
        "topk_per_image": 750,
        "extra_overrides": extra_overrides,
    }
    return BaselineModelAPI(cfg)


def _resolve_profile_alias(profile: str) -> str:
    p = str(profile).strip().lower()
    if p == "main":
        return "503"
    if p == "ub":
        return "519"
    return str(profile).strip()


def _preflight_bundle_checkpoint(path: Path, profile: str) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload type: {type(payload)}")
    bundle = payload.get("ssr3dllm_bundle", None)
    if not isinstance(bundle, dict):
        raise RuntimeError(
            "checkpoint has no 'ssr3dllm_bundle'; this is not an integrated packed ckpt."
        )

    listeners = bundle.get("listeners", None)
    geom = bundle.get("geom_adapters", None)
    if not isinstance(listeners, dict) or not listeners:
        raise RuntimeError("ssr3dllm_bundle.listeners is missing/empty.")
    if not isinstance(geom, dict) or not geom:
        raise RuntimeError(
            "ssr3dllm_bundle.geom_adapters is missing/empty. "
            "This usually means you are using an old packed ckpt (listener-only bundle)."
        )

    default_profile = str(bundle.get("default_listener_profile", "503")).strip() or "503"
    want = _resolve_profile_alias(profile)
    if want not in listeners:
        want = default_profile if default_profile in listeners else sorted(list(listeners.keys()))[0]
    if want not in geom:
        raise RuntimeError(
            f"geom_adapters profile '{want}' not found in bundle. "
            f"available={sorted(list(geom.keys()))}"
        )

    listener_sd = listeners.get(want, {})
    geom_sd = geom.get(want, {})
    if not isinstance(listener_sd, dict) or not listener_sd:
        raise RuntimeError(f"listeners['{want}'] is empty.")
    if not isinstance(geom_sd, dict) or not geom_sd:
        raise RuntimeError(f"geom_adapters['{want}'] is empty.")

    has_proj = any(str(k).startswith("llm.proj_step.") for k in geom_sd.keys())
    has_mem = ("llm.mem_tokens" in geom_sd)
    has_lora = any("lora_A" in str(k) or "lora_B" in str(k) for k in geom_sd.keys())
    has_step_rows = ("llm.model.model.embed_tokens.weight" in geom_sd)
    if not (has_proj and has_lora and has_step_rows):
        raise RuntimeError(
            "geom_adapters exists but looks incomplete for 503/one-pass routing: "
            f"has_proj_step={int(has_proj)} has_lora={int(has_lora)} "
            f"has_step_rows={int(has_step_rows)} has_mem_tokens={int(has_mem)}"
        )

    info = {
        "bundle_format": str(bundle.get("format", "")),
        "bundle_default_profile": default_profile,
        "bundle_selected_profile": want,
        "listener_tensor_count": int(len(listener_sd)),
        "geom_tensor_count": int(len(geom_sd)),
        "geom_has_proj_step": bool(has_proj),
        "geom_has_lora": bool(has_lora),
        "geom_has_step_rows": bool(has_step_rows),
        "geom_has_mem_tokens": bool(has_mem),
    }
    print(
        "[integrated_503_eval][ckpt_preflight] "
        + " ".join([f"{k}={v}" for k, v in info.items()]),
        flush=True,
    )
    return info


def _build_vigor_args(
    *,
    scannet_file: str,
    referit3d_file: str,
    mask3d_feature_root: str,
    mask3d_feature_root_test: str,
    n_workers: int,
    batch_size: int,
) -> Namespace:
    return Namespace(
        scannet_file=str(scannet_file),
        referit3D_file=str(referit3d_file),
        log_dir=None,
        resume_path=None,
        config_file=None,
        max_distractors=51,
        max_seq_len=24,
        points_per_object=1024,
        unit_sphere_norm=True,
        mentions_target_class_only=True,
        min_word_freq=3,
        max_test_objects=88,
        mode="evaluate",
        max_train_epochs=100,
        n_workers=int(n_workers),
        random_seed=2020,
        init_lr=0.0005,
        bert_pretrain_path="",
        view_number=4,
        rotate_number=4,
        label_lang_sup=True,
        aggregate_type="avg",
        encoder_layer_num=3,
        decoder_layer_num=4,
        decoder_nhead_num=8,
        lang_multilabel=True,
        multilabel_pretraining=True,
        cascading=True,
        no_pretrain_ordering=False,
        order_len=4,
        disable_coor_loss=False,
        disable_text_loss=True,
        disable_multilabel_loss=False,
        object_latent_dim=768,
        inner_dim=768,
        dropout_rate=0.15,
        lang_cls_alpha=0.0,
        obj_cls_alpha=0.5,
        gpu="0",
        n_gpus=1,
        batch_size=int(batch_size),
        save_args=False,
        experiment_tag=None,
        cluster_pid=None,
        mask3d_feature_root=str(mask3d_feature_root),
        mask3d_feature_root_test=str(mask3d_feature_root_test),
        mask3d_feature_dim=128,
        augment_with_sr3d=None,
        vocab_file=None,
        fine_tune=False,
        s_vs_n_weight=None,
        use_scannet200_obj_cls=True,
    )


def _prepare_vigor_loader(
    *,
    scannet_file: str,
    referit_train_csv: str,
    mask3d_feature_root: str,
    mask3d_feature_root_test: str,
    n_workers: int,
    batch_size: int,
):
    vigor_root = (Path(__file__).resolve().parents[1] / "third_party" / "Vigor").resolve()
    if str(vigor_root) not in sys.path:
        sys.path.insert(0, str(vigor_root))

    from referit3d.in_out.neural_net_oriented import (  # type: ignore
        compute_auxiliary_data,
        load_referential_data,
        load_scan_related_data,
    )
    from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders  # type: ignore

    vargs = _build_vigor_args(
        scannet_file=scannet_file,
        referit3d_file=referit_train_csv,
        mask3d_feature_root=mask3d_feature_root,
        mask3d_feature_root_test=mask3d_feature_root_test,
        n_workers=n_workers,
        batch_size=batch_size,
    )

    all_scans, scans_split, class_to_idx = load_scan_related_data(str(scannet_file), verbose=True, add_pad=True)
    referit_data = load_referential_data(vargs, str(referit_train_csv), scans_split)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans, vargs)
    data_loaders = make_data_loaders(vargs, referit_data, vocab, class_to_idx, all_scans, mean_rgb)
    return data_loaders["test"]


@torch.no_grad()
def _eval_dataset(
    *,
    api: Any,
    data_loader: Iterable[Dict[str, Any]],
    dataset_name: str,
    max_samples: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    model_device = next(api.llama_model.parameters()).device
    total = 0
    correct = 0
    skipped = 0
    with_pred_box = 0
    pred_on_padding = 0
    samples: List[Dict[str, Any]] = []

    for batch in data_loader:
        utterances = _to_list(batch.get("utterance", []))
        feature_paths = _to_list(batch.get("mask3d_feature_path", []))
        instance_ids = torch.as_tensor(batch.get("instance_ids"))
        target_pos = torch.as_tensor(batch.get("target_pos")).long().view(-1)
        ori_order_len = batch.get("ori_order_len", None)
        if ori_order_len is not None:
            ori_order_len = torch.as_tensor(ori_order_len).long().view(-1)

        if instance_ids.dim() == 1:
            instance_ids = instance_ids.unsqueeze(0)
        bsz = min(len(utterances), len(feature_paths), int(instance_ids.shape[0]), int(target_pos.shape[0]))
        for i in range(bsz):
            if int(max_samples) > 0 and total >= int(max_samples):
                break
            utterance = str(utterances[i]).strip()
            feat_path = str(feature_paths[i]).strip()
            if not utterance or not feat_path:
                skipped += 1
                continue

            feat_ctx = _load_context_features(feat_path, instance_ids[i])
            if feat_ctx is None:
                skipped += 1
                continue

            query_ctx = feat_ctx["queries"]
            box_ctx = feat_ctx["box_info"]
            obj_mask = (torch.as_tensor(instance_ids[i]).view(-1) >= 0).to(torch.float32)
            order_valid_mask = None
            if ori_order_len is not None and int(i) < int(ori_order_len.numel()):
                o = int(ori_order_len[i].item())
                o = max(0, min(4, o))
                order_valid_mask = torch.zeros((4,), dtype=torch.float32)
                if o > 0:
                    order_valid_mask[:o] = 1.0
            if torch.is_tensor(box_ctx) and int(torch.count_nonzero(box_ctx).item()) > 0:
                with_pred_box += 1

            q = query_ctx.to(device=model_device, dtype=torch.float32)
            q_norm = F.normalize(q, p=2, dim=1)
            box_info = box_ctx.to(device=model_device, dtype=torch.float32)
            obj_mask = obj_mask.to(device=model_device, dtype=torch.float32)
            if order_valid_mask is not None:
                order_valid_mask = order_valid_mask.to(device=model_device, dtype=torch.float32)
            prompt = f"<geom> {utterance}"
            out = api.llama_model.evaluate(
                input_text_list=[prompt],
                batch_instance_queries_hidden_state=[q],
                batch_instance_queries_normalized_embed=[q_norm],
                batch_eval_types=["chat"],
                use_mini_batch=False,
                batch_box_info=[box_info],
                batch_obj_mask=[obj_mask],
                batch_order_valid_mask=[order_valid_mask] if order_valid_mask is not None else [None],
                max_new_tokens=int(max_new_tokens),
                text_only_output=False,
            )
            pred_qid = _extract_pred_qid(out[0].get("grounding_result", None)) if isinstance(out, list) and out else None
            gt_qid = int(target_pos[i].item())
            if pred_qid is not None:
                try:
                    if int(pred_qid) < 0 or int(pred_qid) >= int(obj_mask.numel()) or float(obj_mask[int(pred_qid)].item()) <= 0.0:
                        pred_on_padding += 1
                except Exception:
                    pred_on_padding += 1
            is_correct = (pred_qid is not None) and (int(pred_qid) == int(gt_qid))

            total += 1
            if is_correct:
                correct += 1
            if len(samples) < 20:
                samples.append(
                    {
                        "utterance": utterance,
                        "gt_qid": int(gt_qid),
                        "pred_qid": None if pred_qid is None else int(pred_qid),
                        "correct": bool(is_correct),
                    }
                )
        if int(max_samples) > 0 and total >= int(max_samples):
            break

    acc = (float(correct) / float(total)) if total > 0 else 0.0
    print(
        f"[integrated_503_eval] dataset={dataset_name} used={total} correct={correct} skipped={skipped} "
        f"with_pred_box={with_pred_box} pred_on_padding={pred_on_padding}",
        flush=True,
    )
    print(f"[integrated_503_eval][{dataset_name}] Reference-Accuracy: {acc:.4f}", flush=True)
    return {
        "dataset": dataset_name,
        "num_samples_used": int(total),
        "num_correct": int(correct),
        "num_skipped": int(skipped),
        "num_with_pred_box": int(with_pred_box),
        "num_pred_on_padding": int(pred_on_padding),
        "reference_accuracy": float(acc),
        "samples_preview": samples,
    }


def main() -> None:
    args = _parse_args()
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    preflight = _preflight_bundle_checkpoint(ckpt, str(args.profile))

    nr3d_train = Path(args.nr3d_train_csv).expanduser().resolve()
    sr3d_train = Path(args.sr3d_train_csv).expanduser().resolve()
    nr3d_test = _resolve_test_csv(nr3d_train)
    sr3d_test = _resolve_test_csv(sr3d_train)
    if not nr3d_test.is_file():
        raise FileNotFoundError(f"nr3d test csv not found: {nr3d_test}")
    if not sr3d_test.is_file():
        raise FileNotFoundError(f"sr3d test csv not found: {sr3d_test}")

    # Force integrated "<geom>" route + packed-listener profile.
    os.environ["SSR3DLLM_ROUTE_GEOM_VIGOR"] = "1"
    os.environ["SSR3DLLM_ROUTE_GEOM_USE_LLM_ORDER"] = "1"
    os.environ["SSR3DLLM_ROUTE_GEOM_LLM_ORDER_STRICT"] = "0"
    os.environ["SSR3DLLM_GEOM_LORA_ENABLE"] = "1"
    os.environ["SSR3DLLM_GEOM_STEPSLOT_CKPT"] = str(ckpt)
    os.environ["SSR3DLLM_REFERIT3D_LISTENER_CKPT"] = str(ckpt)
    os.environ["SSR3DLLM_REFERIT3D_LISTENER_BERT"] = str(Path(args.bert_path).expanduser().resolve())
    os.environ["SSR3DLLM_REFERIT_SCANNET_FILE"] = str(Path(args.scannet_file).expanduser().resolve())
    os.environ["SCANNET_PKL"] = str(Path(args.scannet_file).expanduser().resolve())
    os.environ["SSR3DLLM_VIGOR_PROFILE"] = str(args.profile).strip()
    os.environ.setdefault("SSR3DLLM_STEP_TOKENS", "1")
    os.environ.setdefault("SSR3DLLM_STEP_ORDER_LEN", "4")
    os.environ.setdefault("SSR3DLLM_ENABLE_STOP_TOKEN", "1")
    os.environ.setdefault("SSR3DLLM_STOP_TOKEN", "<STOP>")
    os.environ.setdefault("SSR3DLLM_LLM_STEPSLOT_MAX_LEN", "128")
    # Match Vigor file-backed eval path: in in-memory Mask3D mode, still build
    # multi-view rotated box features (instead of naively repeating one view).
    os.environ.setdefault("SSR3DLLM_VIGOR_INMEMORY_BOX_MULTIVIEW", "1")
    os.environ.setdefault("VIGOR_USE_PRED_BOX_INFO", "1")
    os.environ.setdefault("VIGOR_PRED_CLASS_MASK_MODE", "all_ones")

    api = _build_api(args)

    run_sets = [x.strip().lower() for x in str(args.datasets).split(",") if x.strip()]
    results: Dict[str, Any] = {
        "checkpoint": str(ckpt),
        "profile": str(args.profile),
        "bundle_preflight": preflight,
        "inmemory_box_multiview": str(os.environ.get("SSR3DLLM_VIGOR_INMEMORY_BOX_MULTIVIEW", "")),
        "max_samples": int(args.max_samples),
        "datasets": {},
    }

    if "nr3d" in run_sets:
        loader = _prepare_vigor_loader(
            scannet_file=str(args.scannet_file),
            referit_train_csv=str(nr3d_train),
            mask3d_feature_root=str(args.mask3d_feature_root),
            mask3d_feature_root_test=str(args.mask3d_feature_root_test),
            n_workers=int(args.n_workers),
            batch_size=int(args.batch_size),
        )
        results["datasets"]["nr3d"] = _eval_dataset(
            api=api,
            data_loader=loader,
            dataset_name="nr3d",
            max_samples=int(args.max_samples),
            max_new_tokens=int(args.max_new_tokens),
        )

    if "sr3d" in run_sets:
        loader = _prepare_vigor_loader(
            scannet_file=str(args.scannet_file),
            referit_train_csv=str(sr3d_train),
            mask3d_feature_root=str(args.mask3d_feature_root),
            mask3d_feature_root_test=str(args.mask3d_feature_root_test),
            n_workers=int(args.n_workers),
            batch_size=int(args.batch_size),
        )
        results["datasets"]["sr3d"] = _eval_dataset(
            api=api,
            data_loader=loader,
            dataset_name="sr3d",
            max_samples=int(args.max_samples),
            max_new_tokens=int(args.max_new_tokens),
        )

    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
