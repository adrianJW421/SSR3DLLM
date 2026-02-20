#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Vigor-eval compatible checkpoint views from a packed "
            "SSR3DLLM checkpoint bundle."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Packed SSR3DLLM ckpt path")
    parser.add_argument("--profile", default="503", help="Bundle profile: 503|519|main|ub")
    parser.add_argument("--out-wrapper", required=True, help="Output eval-adapt wrapper ckpt (.pth)")
    parser.add_argument(
        "--out-listener",
        default="",
        help="Optional output listener-only ckpt (.pth), stored as key='model'",
    )
    parser.add_argument(
        "--out-language",
        default="",
        help=(
            "Optional output language-view ckpt (.ckpt). "
            "It keeps the packed checkpoint payload but removes `ssr3dllm_bundle`."
        ),
    )
    parser.add_argument("--epoch", type=int, default=0, help="Epoch value saved in output ckpt")
    return parser.parse_args()


def _clone_tensors(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
    return out


def _select_profile(bundle: Dict[str, Any], raw_profile: str) -> str:
    listeners = bundle.get("listeners", {})
    if not isinstance(listeners, dict) or not listeners:
        raise RuntimeError("invalid bundle: missing non-empty `listeners`")

    aliases = {"main": "503", "ub": "519"}
    requested = aliases.get(str(raw_profile).strip().lower(), str(raw_profile).strip())
    default_profile = str(bundle.get("default_listener_profile", "503")).strip() or "503"

    if requested in listeners:
        return requested
    if default_profile in listeners:
        return default_profile
    return sorted(list(listeners.keys()))[0]


def _pick_embed_base(
    *,
    wrapper_state: Dict[str, torch.Tensor],
    base_state: Dict[str, Any],
    emb_dim: int,
    min_rows: int,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    # First priority: if wrapper already has full LLM embed_tokens, keep that.
    candidate = wrapper_state.get("llm.model.model.embed_tokens.weight", None)
    if torch.is_tensor(candidate) and candidate.dim() == 2:
        if int(candidate.size(1)) == int(emb_dim) and int(candidate.size(0)) >= int(min_rows):
            return "wrapper.llm.model.model.embed_tokens.weight", candidate.detach().cpu().clone()

    # Second priority: find a compatible embed_tokens tensor in base checkpoint.
    best_key: Optional[str] = None
    best_tensor: Optional[torch.Tensor] = None
    best_score = -10**9
    for key, value in base_state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if value.dim() != 2:
            continue
        if "embed_tokens.weight" not in key:
            continue
        if int(value.size(1)) != int(emb_dim) or int(value.size(0)) < int(min_rows):
            continue

        low = key.lower()
        score = int(value.size(0))
        if low.endswith("llm.model.model.embed_tokens.weight"):
            score += 100
        if "llm" in low:
            score += 20
        if score > best_score:
            best_score = score
            best_key = key
            best_tensor = value.detach().cpu().clone()
    return best_key, best_tensor


def _build_wrapper_state(
    *,
    listener_state: Dict[str, torch.Tensor],
    geom_state: Dict[str, torch.Tensor],
    base_state: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    wrapper: Dict[str, torch.Tensor] = {}

    # Normalize listener keys to the wrapper schema expected by Vigor eval scripts.
    for key, value in listener_state.items():
        if key.startswith("llm.") or key == "stop_embed":
            wrapper[key] = value
        else:
            wrapper[f"listener.{key}"] = value

    # Geom adapters override matching LLM-side keys.
    tail_rows = None
    for key, value in geom_state.items():
        if key == "llm.model.model.embed_tokens.weight":
            tail_rows = value.detach().cpu().clone()
            continue
        wrapper[key] = value

    embed_report: Dict[str, Any] = {
        "source_key": None,
        "tail_rows": 0,
        "full_rows": 0,
        "rebuilt": False,
    }
    if torch.is_tensor(tail_rows) and tail_rows.dim() == 2:
        tail_cpu = tail_rows.detach().cpu().clone()
        need_rows = int(tail_cpu.size(0))
        emb_dim = int(tail_cpu.size(1))
        embed_report["tail_rows"] = need_rows

        src_key, full = _pick_embed_base(
            wrapper_state=wrapper,
            base_state=base_state,
            emb_dim=emb_dim,
            min_rows=need_rows,
        )
        if torch.is_tensor(full) and full.dim() == 2 and int(full.size(1)) == emb_dim and int(full.size(0)) >= need_rows:
            full[-need_rows:] = tail_cpu.to(dtype=full.dtype)
            wrapper["llm.model.model.embed_tokens.weight"] = full
            embed_report["source_key"] = src_key
            embed_report["full_rows"] = int(full.size(0))
            embed_report["rebuilt"] = True
        else:
            # Last resort: keep only tail rows (still loadable under strict=False).
            wrapper["llm.model.model.embed_tokens.weight"] = tail_cpu
            embed_report["source_key"] = None
            embed_report["full_rows"] = int(tail_cpu.size(0))
            embed_report["rebuilt"] = False

    return wrapper, embed_report


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_wrapper = Path(args.out_wrapper).expanduser().resolve()
    out_listener = Path(args.out_listener).expanduser().resolve() if args.out_listener else None
    out_language = Path(args.out_language).expanduser().resolve() if args.out_language else None

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint format: {ckpt_path}")

    bundle = payload.get("ssr3dllm_bundle", None)
    if not isinstance(bundle, dict):
        raise RuntimeError("checkpoint missing `ssr3dllm_bundle`")

    profile = _select_profile(bundle, args.profile)
    listeners = bundle.get("listeners", {})
    geom_adapters = bundle.get("geom_adapters", {})

    listener_raw = listeners.get(profile, None)
    geom_raw = geom_adapters.get(profile, None)
    if not isinstance(listener_raw, dict) or not listener_raw:
        raise RuntimeError(f"bundle listeners profile '{profile}' is missing/empty")
    if not isinstance(geom_raw, dict) or not geom_raw:
        raise RuntimeError(f"bundle geom_adapters profile '{profile}' is missing/empty")

    listener_state = _clone_tensors(listener_raw)
    geom_state = _clone_tensors(geom_raw)
    base_state = payload.get("state_dict", {})
    if not isinstance(base_state, dict):
        base_state = {}

    wrapper_state, embed_report = _build_wrapper_state(
        listener_state=listener_state,
        geom_state=geom_state,
        base_state=base_state,
    )

    out_wrapper.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": int(args.epoch), "model": wrapper_state}, str(out_wrapper))

    if out_listener is not None:
        out_listener.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": int(args.epoch), "model": listener_state}, str(out_listener))

    if out_language is not None:
        language_payload = dict(payload)
        language_payload.pop("ssr3dllm_bundle", None)
        out_language.parent.mkdir(parents=True, exist_ok=True)
        torch.save(language_payload, str(out_language))

    report = {
        "checkpoint": str(ckpt_path),
        "profile_requested": str(args.profile),
        "profile_selected": str(profile),
        "out_wrapper": str(out_wrapper),
        "out_listener": (str(out_listener) if out_listener is not None else ""),
        "out_language": (str(out_language) if out_language is not None else ""),
        "listener_tensors": int(len(listener_state)),
        "geom_tensors": int(len(geom_state)),
        "wrapper_tensors": int(len(wrapper_state)),
        "embed_rebuild": embed_report,
        "bundle_format": str(bundle.get("format", "")),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
