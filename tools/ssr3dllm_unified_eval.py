#!/usr/bin/env python3
"""
Single-question SSR3DLLM evaluator with "<geom>" routing support.

This tool loads a Step3/Step4-style Grounded3D-LLM checkpoint through
`BaselineModelAPI`, extracts scene queries, and runs one prompt through the
LLM evaluator:
  - normal prompt: language answer
  - prompt containing "<geom>": geometry routing path (when enabled via env)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSR3DLLM unified single-question eval")
    parser.add_argument("--checkpoint", required=True, help="Path to Grounded3D-LLM Step3/Step4 ckpt")
    parser.add_argument("--scene-id", required=True, help="ScanNet scene id, e.g. scene0000_00")
    parser.add_argument("--question", required=True, help="Input question text (supports <geom>)")
    parser.add_argument(
        "--prompt-profile",
        default="paper",
        choices=["raw", "appendix", "paper"],
        help="Prompt normalization profile for ask-mode demo",
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--scannet-processed-root",
        default="data/SCANNET200_ROOT",
        help="Processed ScanNet root containing train/validation npy files",
    )
    parser.add_argument("--config-root", default="baseline/core/conf")
    parser.add_argument("--data-config", default="", help="Override data config yaml")
    parser.add_argument("--model-config", default="", help="Override model config yaml")
    parser.add_argument("--trainer-config", default="", help="Override trainer config yaml")
    parser.add_argument("--llm-config", default="", help="Override llm config json")
    parser.add_argument("--llm-data-config", default="", help="Override llm data config json")
    parser.add_argument("--topk-per-image", type=int, default=750)
    # Align defaults with `LLama3d.evaluate(...)` so ask-mode matches paper eval behavior.
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument(
        "--decode-profile",
        default="paper",
        choices=["raw", "paper"],
        help="Decode/output cleaning profile (paper matches models/metrics/evaluate_LLM.py token filtering).",
    )
    return parser


def _resolve_or_default(value: str, default_value: str) -> str:
    value = str(value or "").strip()
    return value if value else default_value


def _load_api(args: argparse.Namespace):
    from baseline.api.baseline_interface import BaselineModelAPI

    config_root = Path(args.config_root)
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

    cfg: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_type": args.split,
        "scannet_processed_root": str(scannet_root),
        "config_path": str(config_root),
        "data_config": _resolve_or_default(args.data_config, str(config_root / "data/indoor_dialog.yaml")),
        "model_config": _resolve_or_default(args.model_config, str(config_root / "model/mask3d_lang.yaml")),
        "trainer_config": _resolve_or_default(args.trainer_config, str(config_root / "trainer/trainer50.yaml")),
        "llm_config": _resolve_or_default(args.llm_config, str(config_root / "llm/tiny_vicuna_len512_bs4.json")),
        "llm_data_config": _resolve_or_default(args.llm_data_config, str(config_root / "llm/det10.json")),
        "topk_per_image": int(args.topk_per_image),
        "extra_overrides": extra_overrides,
    }
    return BaselineModelAPI(cfg)


def _extract_first_qid(raw_grounding: Any) -> Optional[int]:
    if raw_grounding is None:
        return None
    if isinstance(raw_grounding, list) and raw_grounding:
        first = raw_grounding[0]
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


def _clean_text(text: str) -> str:
    text = str(text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_scene_desc(question: str) -> bool:
    q = _clean_text(question).lower().strip(" .!?")
    if not q:
        return False
    prefixes = (
        "describe this room",
        "describe the room",
        "describe this scene",
        "describe the scene",
        "describe this space",
        "summarize this room",
        "summarize the room",
        "what is in this room",
        "what's in this room",
        "what does this room look like",
    )
    if q.startswith(prefixes):
        return True
    return ("describe" in q) and any(token in q for token in ("room", "scene", "space"))


def _normalize_prompt(question: str, profile: str) -> str:
    raise NotImplementedError("use _normalize_prompt_ex")


def _load_lan_templates(config_root: str) -> Dict[str, Any]:
    path = Path(config_root) / "models" / "LLM" / "lan_template.json"
    if not path.is_file():
        path = Path("baseline/core/models/LLM/lan_template.json")
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _stable_pick_index(scene_id: str, question: str, n: int) -> int:
    if n <= 0:
        return 0
    key = f"{scene_id}|{question}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(n)


def _pick_global_scene_prompt(
    *,
    scene_id: str,
    question: str,
    config_root: str,
    fixed_override: str,
) -> Tuple[str, str, int]:
    if fixed_override:
        return fixed_override.strip(), "fixed_override", 0
    templates = _load_lan_templates(config_root)
    values = templates.get("globalscenecap", None) if isinstance(templates, dict) else None
    if not isinstance(values, list) or not values:
        fallback = "Can you provide a brief description of this indoor scene?"
        return fallback, "fallback_default", 0
    idx = _stable_pick_index(scene_id=scene_id, question=question, n=len(values))
    text = str(values[idx]).strip()
    if not text:
        text = "Can you provide a brief description of this indoor scene?"
    return text, "lan_template.globalscenecap", int(idx)


def _normalize_prompt_ex(question: str, profile: str, scene_id: str, config_root: str) -> Tuple[str, Dict[str, Any]]:
    raw = _clean_text(question)
    meta: Dict[str, Any] = {
        "profile": profile,
        "source": "raw",
        "template_index": None,
    }
    if profile == "raw":
        return raw, meta

    has_geom = "<geom>" in raw
    if has_geom:
        body = _clean_text(raw.replace("<geom>", ""))
        out = "<geom>" if not body else f"<geom> {body}"
        meta["source"] = "geom_normalized"
        return out, meta

    if _looks_like_scene_desc(raw):
        fixed_override = str(os.environ.get("SSR3DLLM_APPENDIX_SCENE_PROMPT", "")).strip()
        if profile == "appendix" and fixed_override:
            meta["source"] = "fixed_override"
            return fixed_override, meta
        picked, source, idx = _pick_global_scene_prompt(
            scene_id=scene_id,
            question=raw,
            config_root=config_root,
            fixed_override=fixed_override if profile == "appendix" else "",
        )
        meta["source"] = source
        meta["template_index"] = idx
        return picked, meta

    return raw, meta


def _build_full_prompt_text(*, llm, effective_question: str) -> str:
    """
    Reconstruct the *string-level* prompt that `build_input_from_segments(...)` encodes.
    This is for debugging "full input string" mismatches against paper pipelines.
    """
    try:
        prompts = getattr(llm, "prompts", None) or {}
        sys_lan = str(((prompts.get("system_prompt") or {}).get("lan")) or "").strip()
        user_lan = str((((prompts.get("rules") or {}).get("user") or {}).get("lan")) or "").strip()
        asst_lan = str((((prompts.get("rules") or {}).get("assistant") or {}).get("lan")) or "").strip()
        use_system = bool(getattr(getattr(llm, "config", None), "use_system_prompt", True))
        parts = []
        if use_system and sys_lan:
            parts.append(sys_lan)
        if user_lan:
            parts.append(user_lan)
        parts.append(str(effective_question))
        if asst_lan:
            parts.append(asst_lan)
        return " ".join([p for p in parts if str(p).strip()]).strip()
    except Exception:
        # Best-effort fallback; never crash ask-mode.
        return str(effective_question).strip()


def _decode_clean_paper(text: str, max_length: int = 256) -> str:
    """
    Match the qualitative "paper" decode cleaning:
    - remove special tokens / chat prefixes
    - collapse whitespace
    - truncate (to avoid extremely long degenerate continuations)
    This mirrors the spirit of `models/metrics/evaluate_LLM.py:special_token_filter`.
    """
    s = str(text or "")
    replacements = {
        "ASSISTANT:": "",
        "ASSISTANT: ": "",
        "\n": " ",
        "<s>": "",
        "</s>": "",
        "<unk>": "",
        "<p>": "",
        "</p>": "",
        "<ref>": "",
        "<|endoftext|>": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if max_length and len(s) > int(max_length):
        s = s[: int(max_length)].rstrip()
    return s


@torch.no_grad()
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    api = _load_api(args)
    if api.llama_model is None:
        raise RuntimeError("LLM is not initialized in BaselineModelAPI.")

    scene_pack = api.get_scene_data_for_verification(args.scene_id)
    if not isinstance(scene_pack, dict):
        raise RuntimeError(f"failed to load scene features for {args.scene_id}")

    object_queries = scene_pack.get("object_queries", None)
    if not torch.is_tensor(object_queries) or object_queries.dim() != 2:
        raise RuntimeError("invalid object_queries from scene features")

    model_device = next(api.llama_model.parameters()).device
    object_queries = object_queries.to(device=model_device, dtype=torch.float32)
    object_queries_norm = F.normalize(object_queries, p=2, dim=1)

    effective_question, prompt_meta = _normalize_prompt_ex(
        question=args.question,
        profile=args.prompt_profile,
        scene_id=args.scene_id,
        config_root=args.config_root,
    )

    results = api.llama_model.evaluate(
        input_text_list=[effective_question],
        batch_instance_queries_hidden_state=[object_queries],
        batch_instance_queries_normalized_embed=[object_queries_norm],
        batch_eval_types=["chat"],
        use_mini_batch=False,
        max_new_tokens=int(args.max_new_tokens),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        length_penalty=float(args.length_penalty),
        text_only_output=False,
    )

    if not isinstance(results, list) or not results:
        raise RuntimeError("unexpected evaluate output")

    item = results[0]
    grounding_raw = item.get("grounding_result")
    top_qid = _extract_first_qid(grounding_raw)

    pred_class = None
    pred_score = None
    if top_qid is not None:
        forward = api._run_full_forward(args.scene_id)  # noqa: SLF001 - intentional for metadata lookup
        if isinstance(forward, dict):
            pred_classes = forward.get("pred_classes", None)
            pred_scores = forward.get("pred_scores", None)
            try:
                if torch.is_tensor(pred_classes) and 0 <= top_qid < int(pred_classes.shape[0]):
                    pred_class = int(pred_classes[top_qid].item())
                if torch.is_tensor(pred_scores) and 0 <= top_qid < int(pred_scores.shape[0]):
                    pred_score = float(pred_scores[top_qid].item())
            except Exception:
                pred_class = pred_class
                pred_score = pred_score

    payload = {
        "scene_id": args.scene_id,
        "question": args.question,
        "question_effective": effective_question,
        "prompt_profile": args.prompt_profile,
        "prompt_source": prompt_meta.get("source"),
        "prompt_template_index": prompt_meta.get("template_index"),
        "answer_text_raw": item.get("output_language"),
        "answer_text": (
            item.get("output_language")
            if args.decode_profile == "raw"
            else _decode_clean_paper(item.get("output_language"))
        ),
        "grounding_result_raw": grounding_raw,
        "grounding_score_raw": item.get("score"),
        "grounding_top_query_id": top_qid,
        "grounding_top_query_pred_class": pred_class,
        "grounding_top_query_pred_score": pred_score,
        "route_geom_enabled": str(effective_question).find("<geom>") >= 0,
    }
    if os.environ.get("SSR3DLLM_DEBUG_FULL_PROMPT", "").strip().lower() in {"1", "true", "yes", "on"}:
        payload["prompt_full_text"] = _build_full_prompt_text(llm=api.llama_model, effective_question=effective_question)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
