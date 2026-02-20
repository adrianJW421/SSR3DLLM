"""
Runtime wrapper to use a pretrained ReferIt3D listener (vendored under `third_party/Vigor`)
as a geometry head inside SSR3DLLM.

This is intended for the "<geom>" routing path where we want:
  - inputs: in-memory Mask3D query features (usually Q=100, dim=128)
  - output: referential logits over the provided query set

The wrapper builds Vigor's ReferIt3DNet_transformer in "Mask3D features" mode and
loads a checkpoint from an env var.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

# Ensure release repo root is importable.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

_vigor_root_env = (
    os.environ.get("SSR3DLLM_REFERIT3D_LISTENER_ROOT", "").strip()
    or os.environ.get("SSR3DLLM_VIGOR_ROOT", "").strip()
)
if _vigor_root_env:
    _p = Path(_vigor_root_env).expanduser()
    vigor_root = (_p if _p.is_absolute() else (repo_root / _p)).resolve()
else:
    vigor_root = repo_root / "third_party" / "Vigor"

def _with_vigor_referit3d_imports(fn):
    """
    Execute `fn()` in a temporary import context where Vigor's `referit3d` package
    is importable as the top-level module name `referit3d`.

    This is required because Vigor's ScanNet preprocessed pkls can contain pickled
    objects referencing the module path `referit3d.*`, and `cPickle.load()` will
    attempt to import that module during unpickling.

    Note: we also have other `referit3d` trees in this repo, so we avoid
    permanently clobbering any already-imported `referit3d*` modules.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        saved: Dict[str, Any] = {}
        for k in list(sys.modules.keys()):
            if k == "referit3d" or k.startswith("referit3d."):
                saved[k] = sys.modules[k]
                del sys.modules[k]

        vigor_root_s = str(vigor_root)
        inserted = False
        if sys.path[:1] != [vigor_root_s]:
            sys.path.insert(0, vigor_root_s)
            inserted = True
        try:
            yield
        finally:
            for k in list(sys.modules.keys()):
                if k == "referit3d" or k.startswith("referit3d."):
                    del sys.modules[k]
            for k, v in saved.items():
                sys.modules[k] = v
            if inserted:
                try:
                    if sys.path[:1] == [vigor_root_s]:
                        sys.path.pop(0)
                except Exception:
                    pass

    with _ctx():
        return fn()


def _import_vigor_referit3d_net() -> Any:
    """
    Import Vigor's `ReferIt3DNet_transformer` without permanently clobbering the
    benchmark `referit3d` package (they share the same top-level name).

    Strategy:
    - Temporarily remove any already-imported `referit3d*` modules.
    - Temporarily push Vigor root to sys.path[0].
    - Import `referit3d.models.referit3d_net` (resolves to Vigor).
    - Restore the original `referit3d*` modules and sys.path.
    """
    import importlib

    if not vigor_root.exists():
        raise ModuleNotFoundError(
            f"Cannot locate Vigor third_party root at '{vigor_root}'. "
            "Set SSR3DLLM_VIGOR_ROOT or ensure third_party/Vigor exists."
        )
    if not (vigor_root / "referit3d").exists():
        raise ModuleNotFoundError(
            f"Cannot find Vigor's 'referit3d' package under '{vigor_root}'. "
            "Ensure the third_party/Vigor checkout is complete."
        )

    def _do_import():
        mod = importlib.import_module("referit3d.models.referit3d_net")
        return getattr(mod, "ReferIt3DNet_transformer")

    return _with_vigor_referit3d_imports(_do_import)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return str(v).strip() if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    try:
        return bool(int(s))
    except Exception:
        return bool(default)


_LISTENER_KEY_MARKERS = [
    # Core Vigor listener modules (parameter prefixes).
    "mask3d_proj_in.",
    "mask3d_adapter.",
    "language_encoder.",
    "language_clf.",
    "anchor_clf.",
    "object_language_clf.",
    "obj_feature_mapping.",
    "box_feature_mapping.",
    "feat_to_multilabel_clf.",
    "feat_to_coor_reg.",
    "scannet_obj_clf.",
    "refer_encoder.",
]


def _strip_to_listener_key(k: str) -> str:
    """
    Normalize a checkpoint key into the canonical listener param key-space.

    Supports:
    - bare listener checkpoints (e.g. "mask3d_proj_in.weight")
    - wrapper checkpoints (e.g. "listener.mask3d_proj_in.weight", "model.listener....")
    - SSR3DLLM Lightning checkpoints where listener is nested (e.g.
      "ssr3dllm_geom_head.vigor_listener.mask3d_proj_in.weight")
    """
    kk = k[7:] if k.startswith("module.") else k
    if kk.startswith("listener."):
        kk = kk[len("listener.") :]
    elif kk.startswith("model.listener."):
        kk = kk[len("model.listener.") :]
    # If nested, strip everything before the first known listener marker.
    best = None
    for m in _LISTENER_KEY_MARKERS:
        idx = kk.find(m)
        if idx >= 0:
            cand = kk[idx:]
            if best is None or idx < best[0]:
                best = (idx, cand)
    if best is not None:
        return best[1]
    return kk


def _extract_listener_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract listener-related tensors into a normalized key-space.

    Note:
    - Some Lightning checkpoints can contain multiple submodules whose keys, after
      `_strip_to_listener_key()`, collide in the same canonical namespace (e.g. multiple
      `language_encoder.*` blocks from different wrappers).
    - If we keep the "first seen" tensor, we can silently load the *wrong* weights
      while still matching shapes, leading to near-random accuracy.
    - We therefore keep the *best* candidate per canonical key using a heuristic
      preference score on the original checkpoint key.
    """
    def _score_source_key(full_key: str) -> Tuple[int, int]:
        """
        Prefer keys that likely belong to the SSR3DLLM geom-head Vigor listener.
        Tie-breaker: shorter keys usually mean less wrapper nesting.
        """
        lk = str(full_key).lower()
        score = 0
        if "ssr3dllm_geom_head" in lk:
            score += 5
        if ".vigor_listener." in lk or lk.endswith("vigor_listener"):
            score += 4
        if "vigor" in lk:
            score += 2
        if "listener" in lk:
            score += 2
        return (score, -len(full_key))

    best: Dict[str, Tuple[Tuple[int, int], torch.Tensor]] = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if not torch.is_tensor(v):
            continue
        kk = _strip_to_listener_key(k)
        sc = _score_source_key(k)
        prev = best.get(kk, None)
        if prev is None or sc > prev[0]:
            best[kk] = (sc, v)

    return {k: tv for k, (_, tv) in best.items()}


def _pick_listener_state_dict_from_bundle(payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None
    bundle = payload.get("ssr3dllm_bundle", None)
    if not isinstance(bundle, dict):
        return None, None
    listeners = bundle.get("listeners", None)
    if not isinstance(listeners, dict) or not listeners:
        return None, None

    default_profile = str(bundle.get("default_listener_profile", "503")).strip() or "503"
    profile_raw = _env_str("SSR3DLLM_VIGOR_PROFILE", _env_str("SSR3DLLM_BUNDLE_PROFILE", default_profile))
    aliases = {"main": "503", "ub": "519"}
    profile = aliases.get(str(profile_raw).strip().lower(), str(profile_raw).strip())
    if profile not in listeners:
        profile = default_profile if default_profile in listeners else sorted(list(listeners.keys()))[0]

    selected = listeners.get(profile, None)
    if not isinstance(selected, dict):
        raise ValueError(
            "Invalid ssr3dllm_bundle format: listeners[profile] must be a tensor dict. "
            f"profile={profile}"
        )
    return selected, profile


class ReferIt3DListenerRuntime(nn.Module):
    """
    ReferIt3D listener runtime adapter (third-party implementation vendored under `third_party/Vigor`).

    Env vars:
      - SSR3DLLM_REFERIT3D_LISTENER_ROOT: optional override of Vigor root (defaults to `third_party/Vigor`)
      - SSR3DLLM_REFERIT3D_LISTENER_CKPT: listener checkpoint to load (fallback: SSR3DLLM_VIGOR_LISTENER_CKPT)
      - SSR3DLLM_REFERIT3D_LISTENER_BERT: bert path/name (fallback: SSR3DLLM_VIGOR_BERT)
      - SSR3DLLM_VIGOR_ORDER_LEN: number of steps (default: 4)
      - SSR3DLLM_VIGOR_VIEW_NUMBER: number of views (default: 4)
      - SSR3DLLM_VIGOR_ROTATE_NUMBER: rotations (default: 4)
      - SSR3DLLM_VIGOR_MASK3D_FEATURE_DIM: input dim for Mask3D queries (default: 128)
      - SSR3DLLM_VIGOR_USE_FULL_CLASS_TOKENS: build class-name tokens from the ScanNet pkl (default: 0)
      - SSR3DLLM_VIGOR_SCANNET_FILE: keep_all_points_with_global_scan_alignment.pkl used to build class_to_idx
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self._bn_modules: List[nn.Module] = []

        bert_path = _env_str(
            "SSR3DLLM_REFERIT3D_LISTENER_BERT",
            _env_str("SSR3DLLM_VIGOR_BERT", "pretrained/bert-base-uncased"),
        )
        view_number = _env_int("SSR3DLLM_VIGOR_VIEW_NUMBER", 4)
        rotate_number = _env_int("SSR3DLLM_VIGOR_ROTATE_NUMBER", 4)

        ckpt = _env_str(
            "SSR3DLLM_REFERIT3D_LISTENER_CKPT",
            _env_str("SSR3DLLM_VIGOR_LISTENER_CKPT", ""),
        )

        # Infer model hyper-params from checkpoint when available (most robust).
        inferred = self._infer_from_checkpoint(ckpt)
        n_classes = int(inferred.get("n_classes", 607))
        inner_dim = int(inferred.get("inner_dim", 768))
        object_dim = int(inferred.get("object_dim", 768))
        mask3d_dim = int(inferred.get("mask3d_dim", _env_int("SSR3DLLM_VIGOR_MASK3D_FEATURE_DIM", 128)))
        order_len = int(inferred.get("order_len", _env_int("SSR3DLLM_VIGOR_ORDER_LEN", 4)))
        label_lang_sup = bool(inferred.get("label_lang_sup", True))
        lang_multilabel = bool(inferred.get("lang_multilabel", False))
        use_scannet200_obj_cls = bool(inferred.get("use_scannet200_obj_cls", False))
        # Best-effort info log (avoid multi-rank spam).
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")).strip() or "0")
        except Exception:
            rank = 0
        if rank == 0:
            try:
                print(
                    "[ReferIt3DListenerRuntime] inferred "
                    f"n_classes={n_classes} inner_dim={inner_dim} object_dim={object_dim} "
                    f"mask3d_dim={mask3d_dim} order_len={order_len} "
                    f"label_lang_sup={int(label_lang_sup)} lang_multilabel={int(lang_multilabel)} "
                    f"use_scannet200_obj_cls={int(use_scannet200_obj_cls)}",
                    flush=True,
                )
            except Exception:
                pass

        # NOTE:
        # - For pure inference usage inside SSR3DLLM, we keep class tokens minimal by default
        #   to avoid tokenizing hundreds of class names and loading ScanNet metadata.
        # - But when we fine-tune the Vigor listener (SSR3DLLM_VIGOR_FINETUNE=1), using minimal
        #   class-name tokens ("pad/dummy") breaks the language-clf / cross-class behavior and
        #   can quickly destroy the pretrained listener (as observed in Step4 stage3).
        # Therefore: default to full class-name tokens when FINETUNE=1.
        finetune_listener = _env_bool("SSR3DLLM_VIGOR_FINETUNE", False)
        use_full_class_tokens = _env_bool("SSR3DLLM_VIGOR_USE_FULL_CLASS_TOKENS", finetune_listener)
        scannet_file = _env_str("SSR3DLLM_VIGOR_SCANNET_FILE", "")
        if not scannet_file:
            scannet_file = _env_str("SSR3DLLM_REFERIT_SCANNET_FILE", "")
        if not scannet_file:
            scannet_file = _env_str("SCANNET_PKL", "")
        if not scannet_file and use_full_class_tokens:
            raise ValueError(
                "SSR3DLLM_VIGOR_USE_FULL_CLASS_TOKENS=1 requires SSR3DLLM_VIGOR_SCANNET_FILE "
                "(or SSR3DLLM_REFERIT_SCANNET_FILE / SCANNET_PKL) to be set."
            )

        # Minimal args namespace required by ReferIt3DNet_transformer.
        ReferIt3DNet_transformer = _import_vigor_referit3d_net()  # type: ignore[assignment]
        args = Namespace()
        args.bert_pretrain_path = bert_path
        args.view_number = view_number
        args.rotate_number = rotate_number
        args.label_lang_sup = label_lang_sup
        args.aggregate_type = "avg"
        args.encoder_layer_num = _env_int("SSR3DLLM_VIGOR_TEXT_ENCODER_LAYERS", 3)
        args.decoder_layer_num = _env_int("SSR3DLLM_VIGOR_DECODER_LAYERS", 4)
        args.decoder_nhead_num = _env_int("SSR3DLLM_VIGOR_DECODER_NHEAD", 8)
        args.object_latent_dim = object_dim
        args.inner_dim = inner_dim
        args.dropout_rate = _env_float("SSR3DLLM_VIGOR_DROPOUT", 0.1)
        args.lang_cls_alpha = 0.0
        args.obj_cls_alpha = 0.0
        args.lang_multilabel = lang_multilabel
        args.multilabel_pretraining = False
        args.disable_text_loss = True
        args.disable_multilabel_loss = True
        args.use_scannet200_obj_cls = False
        # If the checkpoint was trained with ScanNet200 object classification head enabled,
        # enable it here as well so weights can be loaded cleanly (even though we do not
        # rely on obj-cls logits for the geom-head inference path).
        args.use_scannet200_obj_cls = use_scannet200_obj_cls
        args.order_len = order_len
        # Trigger "Mask3D features" mode (we use the in-memory query path).
        args.mask3d_feature_root = "__inmemory__"
        args.mask3d_feature_dim = mask3d_dim

        tokenizer = BertTokenizer.from_pretrained(bert_path)
        if use_full_class_tokens:
            # Match Vigor training: build class-name tokens from the full ScanNet class list
            # derived from the ReferIt3D ScanNet preprocessed pkl.
            def _do_load_scan_related_data():
                import importlib

                mod = importlib.import_module("referit3d.in_out.neural_net_oriented")
                fn = getattr(mod, "load_scan_related_data")
                try:
                    return fn(scannet_file, verbose=(rank == 0), add_pad=True)
                except TypeError:
                    # Older Vigor forks may not accept `add_pad` kwarg.
                    return fn(scannet_file, verbose=(rank == 0))

            _, _, class_to_idx = _with_vigor_referit3d_imports(_do_load_scan_related_data)

            if "pad" not in class_to_idx:
                class_to_idx = dict(class_to_idx)
                class_to_idx["pad"] = len(class_to_idx)

            expected = len(class_to_idx) - 1
            if expected != n_classes:
                allow = _env_bool("SSR3DLLM_VIGOR_ALLOW_CLASS_MISMATCH", False)
                msg = (
                    "[ReferIt3DListenerRuntime] WARNING: class_to_idx size mismatch: "
                    f"expected_n_classes={expected} (from scannet_file) ckpt_n_classes={n_classes} "
                    f"scannet_file={scannet_file}"
                )
                if rank == 0:
                    print(msg, flush=True)
                if not allow:
                    # Fall back to minimal tokens to avoid silently running with a mismatched class inventory.
                    use_full_class_tokens = False

        if use_full_class_tokens:
            class_name_list = list(class_to_idx.keys())
            class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
            for k in class_name_tokens.data:
                class_name_tokens.data[k] = class_name_tokens.data[k].to(device)
            pad_idx = int(class_to_idx["pad"])
            if rank == 0:
                try:
                    print(
                        "[ReferIt3DListenerRuntime] class_tokens=full "
                        f"n_class_names={len(class_name_list)} pad_idx={pad_idx} scannet_file={scannet_file}",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            # Vigor constructor requires class_name_tokens even if we only care about referential logits.
            # For non-eval usage, keep it tiny to avoid tokenizing hundreds of class names and loading ScanNet.
            class_to_idx = {"pad": 0, "dummy": 1}
            class_name_tokens = tokenizer(list(class_to_idx.keys()), return_tensors="pt", padding=True)
            for k in class_name_tokens.data:
                class_name_tokens.data[k] = class_name_tokens.data[k].to(device)
            pad_idx = class_to_idx["pad"]
            if rank == 0:
                try:
                    print(
                        "[ReferIt3DListenerRuntime] class_tokens=minimal "
                        f"n_class_names={len(class_to_idx)} pad_idx={pad_idx}",
                        flush=True,
                    )
                except Exception:
                    pass

        self.model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx).to(device)
        self.tokenizer = tokenizer
        self.order_len = order_len

        if ckpt:
            self.load_checkpoint(ckpt)

        # Cache BN modules so we can optionally freeze BN running stats during training.
        # This is important for Step4: BN buffers drifting on mixed dialog data can break
        # the listener's eval-time behavior even if most weights are unchanged.
        try:
            self._bn_modules = [
                m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
            ]
        except Exception:
            self._bn_modules = []

        self.model.eval()

    @staticmethod
    def _import_vigor_load_scan_related_data() -> Any:
        """
        Import Vigor's `load_scan_related_data` without clobbering the benchmark `referit3d` package.
        """
        import importlib

        saved: Dict[str, Any] = {}
        for k in list(sys.modules.keys()):
            if k == "referit3d" or k.startswith("referit3d."):
                saved[k] = sys.modules[k]
                del sys.modules[k]

        vigor_root_s = str(vigor_root)
        inserted = False
        if sys.path[:1] != [vigor_root_s]:
            sys.path.insert(0, vigor_root_s)
            inserted = True
        try:
            mod = importlib.import_module("referit3d.in_out.neural_net_oriented")
            return getattr(mod, "load_scan_related_data")
        finally:
            for k in list(sys.modules.keys()):
                if k == "referit3d" or k.startswith("referit3d."):
                    del sys.modules[k]
            for k, v in saved.items():
                sys.modules[k] = v
            if inserted:
                try:
                    if sys.path[:1] == [vigor_root_s]:
                        sys.path.pop(0)
                except Exception:
                    pass

    @staticmethod
    def _infer_from_checkpoint(path: str) -> Dict[str, Any]:
        """
        Best-effort infer Vigor model hyper-params from a checkpoint state_dict.
        This avoids class-count mismatches such as language_clf output dim.
        """
        if not path:
            return {}
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            return {}
        try:
            data = torch.load(str(p), map_location="cpu")
        except Exception:
            return {}

        sd = None
        selected_listener_sd, _ = _pick_listener_state_dict_from_bundle(data)
        if isinstance(selected_listener_sd, dict):
            sd = selected_listener_sd
        if sd is None and isinstance(data, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in data and isinstance(data[k], dict):
                    sd = data[k]
                    break
        if sd is None and isinstance(data, dict):
            sd = data
        if not isinstance(sd, dict):
            return {}

        key_to_tensor = _extract_listener_state_dict(sd)
        keys = list(key_to_tensor.keys())

        out: Dict[str, Any] = {}
        # Prefer reading dims from feature mappers (most direct / unambiguous).
        w_obj_map = key_to_tensor.get("obj_feature_mapping.0.weight", None)
        if torch.is_tensor(w_obj_map) and w_obj_map.dim() == 2:
            # Linear(object_dim -> inner_dim): weight is [inner_dim, object_dim]
            out["inner_dim"] = int(w_obj_map.shape[0])
            out["object_dim"] = int(w_obj_map.shape[1])
        w_box_map = key_to_tensor.get("box_feature_mapping.0.weight", None)
        if torch.is_tensor(w_box_map) and w_box_map.dim() == 2:
            # Linear(4 -> inner_dim): weight is [inner_dim, 4]
            out.setdefault("inner_dim", int(w_box_map.shape[0]))

        # Infer class count and (fallback) inner_dim from language_clf.
        # Vigor uses `get_mlp_head(input_size=inner_dim, hidden_size=inner_dim, ...)`,
        # whose last Linear is `Linear(inner_dim//2 -> n_classes)`, so the weight shape
        # is [n_classes, inner_dim//2].
        w_lang = key_to_tensor.get("language_clf.4.weight", None)
        if torch.is_tensor(w_lang) and w_lang.dim() == 2:
            out["n_classes"] = int(w_lang.shape[0])
            out.setdefault("inner_dim", int(w_lang.shape[1]) * 2)
        w2 = key_to_tensor.get("mask3d_proj_in.weight", None)
        if torch.is_tensor(w2) and w2.dim() == 2:
            out.setdefault("object_dim", int(w2.shape[0]))
            out["mask3d_dim"] = int(w2.shape[1])
        # Infer order_len from refer_encoder.{i}.*
        max_idx = -1
        for k in keys:
            if k.startswith("refer_encoder."):
                rest = k[len("refer_encoder.") :]
                idx_str = rest.split(".", 1)[0]
                try:
                    max_idx = max(max_idx, int(idx_str))
                except Exception:
                    continue
        if max_idx >= 0:
            out["order_len"] = int(max_idx + 1)
        # Infer whether obj_clf exists (means label_lang_sup=False).
        has_obj_clf = any(k.startswith("obj_clf.") for k in keys)
        out["label_lang_sup"] = not has_obj_clf
        out["lang_multilabel"] = any(k.startswith("anchor_clf.") for k in keys)
        out["use_scannet200_obj_cls"] = any(k.startswith("scannet_obj_clf.") for k in keys)
        return out

    def load_checkpoint(self, path: str) -> Tuple[list[str], list[str]]:
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        data = torch.load(str(p), map_location="cpu")
        state_dict = None
        selected_profile = None
        selected_listener_sd, selected_profile = _pick_listener_state_dict_from_bundle(data)
        if isinstance(selected_listener_sd, dict):
            state_dict = selected_listener_sd
        if state_dict is None and isinstance(data, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in data and isinstance(data[k], dict):
                    state_dict = data[k]
                    break
        if state_dict is None and isinstance(data, dict):
            state_dict = data
        if state_dict is None:
            raise ValueError(f"Unsupported checkpoint format: {path}")

        cleaned = _extract_listener_state_dict(state_dict)
        # If the checkpoint was trained with added BERT special tokens (e.g. <stepK> markers),
        # the embedding matrix will be larger than the runtime default and would be filtered out
        # as a shape mismatch, leaving a partially loaded BERT.
        #
        # Align vocab size to the checkpoint so we can load `word_embeddings.weight` too.
        try:
            emb_key = "language_encoder.embeddings.word_embeddings.weight"
            if emb_key in cleaned:
                w_ckpt = cleaned.get(emb_key)
                w_model = self.model.state_dict().get(emb_key)
                if torch.is_tensor(w_ckpt) and torch.is_tensor(w_model) and w_ckpt.ndim == 2 and w_model.ndim == 2:
                    ckpt_vocab, ckpt_dim = int(w_ckpt.shape[0]), int(w_ckpt.shape[1])
                    model_vocab, model_dim = int(w_model.shape[0]), int(w_model.shape[1])
                    if ckpt_dim == model_dim and ckpt_vocab != model_vocab:
                        # Best-effort: if the delta matches order_len, assume <stepK> tokens were appended.
                        delta = ckpt_vocab - model_vocab
                        if delta > 0 and hasattr(self, "tokenizer") and isinstance(self.tokenizer, BertTokenizer):
                            try:
                                if int(delta) == int(self.order_len):
                                    step_tokens = [f"<step{i+1}>" for i in range(int(self.order_len))]
                                    self.tokenizer.add_special_tokens({"additional_special_tokens": step_tokens})
                            except Exception:
                                pass
                        # Ensure the BERT embedding table matches checkpoint vocab size.
                        try:
                            self.model.language_encoder.resize_token_embeddings(int(ckpt_vocab))
                        except Exception:
                            pass
        except Exception:
            pass

        # Filter out keys with shape mismatch; strict=False still errors on mismatched shapes.
        model_sd = self.model.state_dict()
        filtered = {}
        # Direct matches first.
        for k, v in cleaned.items():
            if k in model_sd:
                try:
                    if tuple(model_sd[k].shape) != tuple(v.shape):
                        continue
                except Exception:
                    continue
                filtered[k] = v

        # Suffix matches for nested Lightning checkpoints where keys may still carry extra prefixes.
        # (This should be rare after _strip_to_listener_key, but keeps robustness.)
        if len(filtered) < len(model_sd):
            for mk in model_sd.keys():
                if mk in filtered:
                    continue
                cands = [(ck, cv) for ck, cv in cleaned.items() if ck.endswith(mk)]
                if not cands:
                    continue
                # Prefer keys that contain common nesting prefixes.
                def _score(k: str) -> Tuple[int, int]:
                    score = 0
                    lk = k.lower()
                    if "ssr3dllm_geom_head" in lk:
                        score += 3
                    if "vigor" in lk:
                        score += 2
                    if "listener" in lk:
                        score += 2
                    # Prefer shorter keys (less extra prefix).
                    return (score, -len(k))

                cands.sort(key=lambda kv: _score(kv[0]), reverse=True)
                ck, cv = cands[0]
                try:
                    if tuple(model_sd[mk].shape) != tuple(cv.shape):
                        continue
                except Exception:
                    continue
                filtered[mk] = cv

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        # Best-effort info log (avoid multi-rank spam).
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")).strip() or "0")
        except Exception:
            rank = 0
        if rank == 0:
            try:
                print(
                    "[ReferIt3DListenerRuntime] loaded_ckpt="
                    f"{str(p)} kept={len(filtered)}/{len(cleaned)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
                if selected_profile is not None:
                    print(
                        f"[ReferIt3DListenerRuntime] bundle_profile={selected_profile}",
                        flush=True,
                    )
            except Exception:
                pass
            if _env_int("SSR3DLLM_VIGOR_CKPT_VERBOSE", 0) == 1:
                try:
                    if missing:
                        print(
                            f"[ReferIt3DListenerRuntime] missing_keys({len(missing)}): {list(missing)[:50]}",
                            flush=True,
                        )
                    if unexpected:
                        print(
                            f"[ReferIt3DListenerRuntime] unexpected_keys({len(unexpected)}): {list(unexpected)[:50]}",
                            flush=True,
                        )
                except Exception:
                    pass
        return missing, unexpected

    @torch.no_grad()
    def predict_logits(
        self,
        text: str,
        mask3d_queries: torch.Tensor,
        *,
        box_info: Optional[torch.Tensor] = None,
        obj_mask: Optional[torch.Tensor] = None,
        order_texts: Optional[List[str]] = None,
        pred_class_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text: question / prompt string (without <geom>)
            mask3d_queries: [N, D] or [1, N, D] in float
        Returns:
            logits: [N] float tensor (on CPU)
        """
        if mask3d_queries.dim() == 2:
            mask3d_queries = mask3d_queries.unsqueeze(0)
        mask3d_queries = mask3d_queries.to(self.device).to(torch.float32)
        B, N, _ = mask3d_queries.shape

        # Build lang tokens (utterance) and order tokens (referential-order steps).
        lang_tokens = self.tokenizer([text], return_tensors="pt", padding=True)
        lang_tokens = {k: v.to(self.device) for k, v in lang_tokens.items()}
        if order_texts is None:
            order_texts = [text] * int(self.order_len)
        else:
            # Normalize to a fixed-length list.
            order_texts = [str(t) for t in order_texts]
            if len(order_texts) < int(self.order_len):
                order_texts = order_texts + [order_texts[-1] if order_texts else text] * (
                    int(self.order_len) - len(order_texts)
                )
            elif len(order_texts) > int(self.order_len):
                order_texts = order_texts[: int(self.order_len)]
        order_tokens = self.tokenizer(order_texts, return_tensors="pt", padding=True)
        # Shape [B, order_len, L] to match DP-safe path.
        for k in list(order_tokens.keys()):
            v = order_tokens[k]
            if torch.is_tensor(v) and v.dim() == 2:
                order_tokens[k] = v.reshape(1, int(self.order_len), v.size(1))
        order_tokens = {k: v.to(self.device) for k, v in order_tokens.items()}

        if box_info is None:
            box_info = torch.zeros((B, N, 4), device=self.device)
        else:
            if box_info.dim() == 2:
                box_info = box_info.unsqueeze(0)
            box_info = box_info.to(self.device).to(torch.float32)
            if box_info.shape[0] != B or box_info.shape[1] != N or box_info.shape[2] != 4:
                raise ValueError(f"box_info must be [{B},{N},4], got {tuple(box_info.shape)}")

        if pred_class_mask is None:
            pred_class_mask = torch.ones((1, int(self.order_len), N), device=self.device, dtype=torch.float32)
        else:
            if pred_class_mask.dim() == 2:
                pred_class_mask = pred_class_mask.unsqueeze(0)
            pred_class_mask = pred_class_mask.to(self.device)
            if pred_class_mask.shape[0] != 1 or pred_class_mask.shape[1] != int(self.order_len) or pred_class_mask.shape[2] != N:
                raise ValueError(
                    f"pred_class_mask must be [1,{int(self.order_len)},{N}] or [{int(self.order_len)},{N}], "
                    f"got {tuple(pred_class_mask.shape)}"
                )

        if obj_mask is None:
            obj_mask = torch.ones((B, N), device=self.device, dtype=torch.float32)
        else:
            if obj_mask.dim() == 1:
                obj_mask = obj_mask.unsqueeze(0)
            obj_mask = obj_mask.to(self.device).to(torch.float32)
            if obj_mask.shape[0] != B or obj_mask.shape[1] != N:
                raise ValueError(f"obj_mask must be [{B},{N}] or [{N}], got {tuple(obj_mask.shape)}")

        batch = {
            "inference": True,
            "mask3d_object_queries": mask3d_queries,
            "box_info": box_info,
            "obj_mask": obj_mask,
            "lang_tokens": lang_tokens,
            "order_tokens": order_tokens,
            "pred_class_mask": pred_class_mask,
            # Dummy fields (not used when inference=True).
            "class_labels": torch.zeros((1, N), device=self.device, dtype=torch.long),
            "target_pos": torch.zeros((1,), device=self.device, dtype=torch.long),
            "target_class": torch.zeros((1,), device=self.device, dtype=torch.long),
        }

        _, _, _, logits, _, _ = self.model(batch)
        logits = logits.squeeze(0).detach().to("cpu")
        return logits

    def forward_logits_with_order_embeds(
        self,
        *,
        lang_tokens: Optional[Dict[str, torch.Tensor]] = None,
        lang_embeds: Optional[torch.Tensor] = None,
        order_embeds: torch.Tensor,
        order_valid_mask: Optional[torch.Tensor] = None,
        mask3d_queries: torch.Tensor,
        box_info: Optional[torch.Tensor] = None,
        obj_mask: Optional[torch.Tensor] = None,
        pred_class_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward Vigor listener using precomputed order embeddings (e.g. LLM <stepK> hidden states).

        Args:
            lang_tokens: tokenized utterance dict.
            order_embeds: [B, order_len, D] or [B, order_len, L, D] where D==inner_dim.
            mask3d_queries: [N, mask3d_dim] or [B, N, mask3d_dim].
        Returns:
            logits: [B, N] float tensor on the same device as the model.
        """
        if mask3d_queries.dim() == 2:
            mask3d_queries = mask3d_queries.unsqueeze(0)
        mask3d_queries = mask3d_queries.to(self.device).to(torch.float32)
        B, N, _ = mask3d_queries.shape

        # Normalize order_embeds shape.
        # Keep listener inputs in fp32 (listener params are fp32; bf16 can crash in Linear).
        order_embeds = torch.as_tensor(order_embeds, device=self.device).to(torch.float32)
        if order_embeds.dim() == 3:
            order_embeds = order_embeds.unsqueeze(2)
        if order_embeds.dim() != 4:
            raise ValueError(
                f"order_embeds must be [B,O,D] or [B,O,L,D], got {tuple(order_embeds.shape)}"
            )
        if int(order_embeds.size(0)) != int(B):
            raise ValueError(
                f"order_embeds batch size mismatch: got {int(order_embeds.size(0))}, expected {int(B)}"
            )

        if box_info is None:
            box_info = torch.zeros((B, N, 4), device=self.device)
        else:
            if box_info.dim() == 2:
                box_info = box_info.unsqueeze(0)
            box_info = box_info.to(self.device).to(torch.float32)
            if box_info.shape[0] != B or box_info.shape[1] != N or box_info.shape[2] != 4:
                raise ValueError(f"box_info must be [{B},{N},4], got {tuple(box_info.shape)}")

        if pred_class_mask is None:
            pred_class_mask = torch.ones((B, int(self.order_len), N), device=self.device, dtype=torch.float32)
        else:
            if pred_class_mask.dim() == 2:
                pred_class_mask = pred_class_mask.unsqueeze(0)
            pred_class_mask = pred_class_mask.to(self.device)
            if (
                pred_class_mask.shape[0] != B
                or pred_class_mask.shape[1] != int(self.order_len)
                or pred_class_mask.shape[2] != N
            ):
                raise ValueError(
                    f"pred_class_mask must be [{B},{int(self.order_len)},{N}] or [{int(self.order_len)},{N}], "
                    f"got {tuple(pred_class_mask.shape)}"
                )

        if obj_mask is None:
            obj_mask = torch.ones((B, N), device=self.device, dtype=torch.float32)
        else:
            if obj_mask.dim() == 1:
                obj_mask = obj_mask.unsqueeze(0)
            obj_mask = obj_mask.to(self.device).to(torch.float32)
            if obj_mask.shape[0] != B or obj_mask.shape[1] != N:
                raise ValueError(f"obj_mask must be [{B},{N}] or [{N}], got {tuple(obj_mask.shape)}")

        if lang_embeds is None and (not isinstance(lang_tokens, dict) or not lang_tokens):
            raise ValueError("Must provide either lang_tokens or lang_embeds.")
        if lang_tokens is not None:
            lang_tokens = {k: v.to(self.device) for k, v in lang_tokens.items()}
        if lang_embeds is not None:
            # Listener (BERT + MLP heads) is trained/evaluated in fp32; bf16 can crash
            # in `nn.Linear` (e.g., "expected scalar type Float but found BFloat16").
            lang_embeds = torch.as_tensor(lang_embeds, device=self.device).to(torch.float32)

        batch = {
            "inference": True,
            "mask3d_object_queries": mask3d_queries,
            "box_info": box_info,
            "obj_mask": obj_mask,
            "order_embeds": order_embeds,
            "pred_class_mask": pred_class_mask,
            # Dummy fields (not used when inference=True).
            "class_labels": torch.zeros((B, N), device=self.device, dtype=torch.long),
            "target_pos": torch.zeros((B,), device=self.device, dtype=torch.long),
            "target_class": torch.zeros((B,), device=self.device, dtype=torch.long),
        }
        if order_valid_mask is not None:
            ovm = torch.as_tensor(order_valid_mask, device=self.device, dtype=torch.float32)
            if ovm.dim() == 1:
                if int(ovm.numel()) != int(self.order_len):
                    raise ValueError(
                        f"order_valid_mask must have order_len={int(self.order_len)} elements, got {int(ovm.numel())}"
                    )
                ovm = ovm.view(1, int(self.order_len)).repeat(int(B), 1)
            elif ovm.dim() == 2:
                if int(ovm.size(1)) != int(self.order_len):
                    raise ValueError(
                        f"order_valid_mask must have shape [B,{int(self.order_len)}], got {tuple(ovm.shape)}"
                    )
                if int(ovm.size(0)) == 1 and int(B) > 1:
                    ovm = ovm.repeat(int(B), 1)
                if int(ovm.size(0)) != int(B):
                    raise ValueError(
                        f"order_valid_mask batch size mismatch: got {int(ovm.size(0))} expected {int(B)}"
                    )
            else:
                raise ValueError(f"order_valid_mask must be [B,O] or [O], got {tuple(ovm.shape)}")
            batch["order_valid_mask"] = ovm
        if lang_embeds is not None:
            batch["lang_embeds"] = lang_embeds
        else:
            batch["lang_tokens"] = lang_tokens

        _, _, _, logits, _, _ = self.model(batch)
        return logits

    def forward_train_with_order_embeds(
        self,
        *,
        lang_tokens: Optional[Dict[str, torch.Tensor]] = None,
        lang_embeds: Optional[torch.Tensor] = None,
        order_embeds: torch.Tensor,
        order_valid_mask: Optional[torch.Tensor] = None,
        mask3d_queries: torch.Tensor,
        box_info: torch.Tensor,
        pred_class_mask: torch.Tensor,
        target_pos: torch.Tensor,
        scannet_class_labels: Optional[torch.Tensor] = None,
        # Training knobs (read from env by default to keep the callsite clean).
        lang_cls_alpha: Optional[float] = None,
        obj_cls_alpha: Optional[float] = None,
        enable_text_loss: Optional[bool] = None,
        enable_obj_loss: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Train-time forward that enables Vigor's BASIC_LOSS path (inference=False).

        This is used to reproduce the learnability of the original Vigor step-slot
        training under `pred_class_mask=all_ones`, by providing stronger supervision
        signals (referential + optional obj/text losses).

        Returns:
            Dict with:
              - loss_total: scalar
              - logits: [B,N]
              - loss_ref_ce: scalar (always computed from logits/target_pos)
              - loss_obj_ce: scalar (0 if disabled/unavailable)
              - loss_lang_ce: scalar (0 if disabled/unavailable)
        """
        if mask3d_queries.dim() == 2:
            mask3d_queries = mask3d_queries.unsqueeze(0)
        mask3d_queries = mask3d_queries.to(self.device).to(torch.float32)
        B, N, _ = mask3d_queries.shape

        # Normalize order_embeds shape to [B,O,L,D]
        order_embeds = torch.as_tensor(order_embeds, device=self.device)
        if order_embeds.dim() == 3:
            order_embeds = order_embeds.unsqueeze(2)
        if order_embeds.dim() != 4:
            raise ValueError(
                f"order_embeds must be [B,O,D] or [B,O,L,D], got {tuple(order_embeds.shape)}"
            )
        if int(order_embeds.size(0)) != int(B):
            raise ValueError(
                f"order_embeds batch size mismatch: got {int(order_embeds.size(0))}, expected {int(B)}"
            )

        # box_info / pred_class_mask must be strict in train mode (no silent fallbacks).
        if box_info is None:
            raise ValueError("box_info is required for train-mode Vigor forward.")
        if pred_class_mask is None:
            raise ValueError("pred_class_mask is required for train-mode Vigor forward.")
        if target_pos is None:
            raise ValueError("target_pos is required for train-mode Vigor forward.")
        if lang_embeds is None and (not isinstance(lang_tokens, dict) or not lang_tokens):
            raise ValueError("Must provide either lang_tokens or lang_embeds in train-mode Vigor forward.")

        box_info = torch.as_tensor(box_info, device=self.device, dtype=torch.float32)
        if box_info.dim() == 2:
            box_info = box_info.unsqueeze(0)
        if box_info.shape[0] != B or box_info.shape[1] != N or box_info.shape[2] != 4:
            raise ValueError(f"box_info must be [{B},{N},4], got {tuple(box_info.shape)}")

        pred_class_mask = torch.as_tensor(pred_class_mask, device=self.device)
        if pred_class_mask.dim() == 2:
            pred_class_mask = pred_class_mask.unsqueeze(0)
        if pred_class_mask.shape[0] != B or pred_class_mask.shape[2] != N:
            raise ValueError(
                f"pred_class_mask must be [{B},O,{N}] (or [O,{N}] for B=1), got {tuple(pred_class_mask.shape)}"
            )

        target_pos = torch.as_tensor(target_pos, device=self.device, dtype=torch.long).view(-1)
        if int(target_pos.numel()) != int(B):
            raise ValueError(f"target_pos must have B={B} elements, got {int(target_pos.numel())}")

        # Training knobs (env defaults).
        if lang_cls_alpha is None:
            lang_cls_alpha = _env_float("SSR3DLLM_VIGOR_TRAIN_LANG_ALPHA", 0.5)
        if obj_cls_alpha is None:
            obj_cls_alpha = _env_float("SSR3DLLM_VIGOR_TRAIN_OBJ_ALPHA", 0.5)
        if enable_text_loss is None:
            enable_text_loss = _env_int("SSR3DLLM_VIGOR_TRAIN_ENABLE_TEXT_LOSS", 0) == 1
        if enable_obj_loss is None:
            enable_obj_loss = _env_int("SSR3DLLM_VIGOR_TRAIN_ENABLE_OBJ_LOSS", 1) == 1

        # For SSR3DLLM phase1 we typically don't use anchor multilabel supervision;
        # disabling it here avoids requiring `anchor_ind` at the callsite.
        disable_anchor = _env_int("SSR3DLLM_VIGOR_TRAIN_DISABLE_ANCHOR", 1) == 1

        if lang_tokens is not None:
            lang_tokens = {k: v.to(self.device) for k, v in lang_tokens.items()}
        if lang_embeds is not None:
            lang_embeds = torch.as_tensor(lang_embeds, device=self.device)

        ovm = None
        if order_valid_mask is not None:
            ovm = torch.as_tensor(order_valid_mask, device=self.device, dtype=torch.float32)
            if ovm.dim() == 1:
                if int(ovm.numel()) != int(self.order_len):
                    raise ValueError(
                        f"order_valid_mask must have order_len={int(self.order_len)} elements, got {int(ovm.numel())}"
                    )
                ovm = ovm.view(1, int(self.order_len)).repeat(int(B), 1)
            elif ovm.dim() == 2:
                if int(ovm.size(1)) != int(self.order_len):
                    raise ValueError(
                        f"order_valid_mask must have shape [B,{int(self.order_len)}], got {tuple(ovm.shape)}"
                    )
                if int(ovm.size(0)) == 1 and int(B) > 1:
                    ovm = ovm.repeat(int(B), 1)
                if int(ovm.size(0)) != int(B):
                    raise ValueError(
                        f"order_valid_mask batch size mismatch: got {int(ovm.size(0))} expected {int(B)}"
                    )
            else:
                raise ValueError(f"order_valid_mask must be [B,O] or [O], got {tuple(ovm.shape)}")

        # Prepare auxiliary labels.
        if scannet_class_labels is not None:
            scannet_class_labels = torch.as_tensor(scannet_class_labels, device=self.device, dtype=torch.long)
            # Accept [B,N] or [1,B,N] (legacy), but enforce final shape [B,N].
            if scannet_class_labels.dim() == 3 and int(scannet_class_labels.size(0)) == 1:
                scannet_class_labels = scannet_class_labels.squeeze(0)
            if scannet_class_labels.dim() != 2:
                raise ValueError(
                    f"scannet_class_labels must be 2D [B,N], got {tuple(scannet_class_labels.shape)}"
                )
            if int(scannet_class_labels.size(0)) != int(B) or int(scannet_class_labels.size(1)) != int(N):
                raise ValueError(
                    f"scannet_class_labels must be [{B},{N}], got {tuple(scannet_class_labels.shape)}"
                )

        # Store/restore mutable flags to avoid affecting inference calls.
        old_train = bool(self.model.training)
        old_lang_alpha = float(getattr(self.model, "lang_cls_alpha", 0.0))
        old_obj_alpha = float(getattr(self.model, "obj_cls_alpha", 0.0))
        old_disable_text = bool(getattr(self.model, "disable_text_loss", True))
        old_lang_multilabel = bool(getattr(self.model, "lang_multilabel", False))

        try:
            # Enable BASIC_LOSS path.
            self.model.train(True)
            # Optional: freeze BN running stats while keeping the rest of the model in train mode.
            # Enable with:
            #   export SSR3DLLM_VIGOR_TRAIN_FREEZE_BN=1
            # or:
            #   export SSR3DLLM_VIGOR_LISTENER_FREEZE_BN=1
            freeze_bn = _env_bool("SSR3DLLM_VIGOR_TRAIN_FREEZE_BN", False) or _env_bool(
                "SSR3DLLM_VIGOR_LISTENER_FREEZE_BN", False
            )
            if freeze_bn and self._bn_modules:
                for m in self._bn_modules:
                    try:
                        m.eval()
                    except Exception:
                        pass
            setattr(self.model, "lang_cls_alpha", float(lang_cls_alpha if enable_text_loss else 0.0))
            setattr(self.model, "obj_cls_alpha", float(obj_cls_alpha if enable_obj_loss else 0.0))
            setattr(self.model, "disable_text_loss", (not bool(enable_text_loss)))
            if disable_anchor:
                setattr(self.model, "lang_multilabel", False)

            # Dummy fields required by the model signature.
            # - class_labels/target_class are only used when the corresponding losses are enabled.
            batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor] | torch.Tensor] = {
                "inference": False,
                "mask3d_object_queries": mask3d_queries,
                "box_info": box_info,
                "order_embeds": order_embeds,
                "order_valid_mask": ovm,
                "pred_class_mask": pred_class_mask.to(torch.float32),
                "class_labels": torch.zeros((B, N), device=self.device, dtype=torch.long),
                "target_pos": target_pos,
                "target_class": torch.zeros((B,), device=self.device, dtype=torch.long),
            }
            if lang_embeds is not None:
                batch["lang_embeds"] = lang_embeds
            else:
                batch["lang_tokens"] = lang_tokens
            if scannet_class_labels is not None:
                batch["scannet_class_labels"] = scannet_class_labels

            total_loss, _, lang_logits, logits, scannet_logits, _ = self.model(batch)  # type: ignore[misc]

            # Always compute ref CE for logging/debug.
            ref_ce = F.cross_entropy(logits, target_pos)

            obj_ce = torch.zeros((), device=self.device)
            if enable_obj_loss and (scannet_logits is not None) and (scannet_class_labels is not None):
                labels = scannet_class_labels
                valid = labels >= 0
                if valid.any():
                    obj_ce = F.cross_entropy(scannet_logits[valid], labels[valid])

            lang_ce = torch.zeros((), device=self.device)
            if enable_text_loss and (lang_logits is not None):
                # If Vigor is in ScanNet200 text-clf mode, compute the target label from scannet_class_labels.
                if bool(getattr(self.model, "use_scannet200_text_cls", False)) and (scannet_class_labels is not None):
                    labels = scannet_class_labels
                    idx = torch.arange(B, device=self.device)
                    tgt_cls = labels[idx, target_pos].clamp(min=-1)
                    lang_ce = F.cross_entropy(lang_logits, tgt_cls, ignore_index=-1)
                else:
                    # No reliable target_class label in SSR3DLLM path; keep disabled unless caller switches modes.
                    lang_ce = torch.zeros((), device=self.device)

            return {
                "loss_total": total_loss,
                "logits": logits,
                "loss_ref_ce": ref_ce,
                "loss_obj_ce": obj_ce,
                "loss_lang_ce": lang_ce,
            }
        finally:
            # Restore original flags.
            try:
                self.model.train(old_train)
                setattr(self.model, "lang_cls_alpha", old_lang_alpha)
                setattr(self.model, "obj_cls_alpha", old_obj_alpha)
                setattr(self.model, "disable_text_loss", old_disable_text)
                setattr(self.model, "lang_multilabel", old_lang_multilabel)
            except Exception:
                pass


VigorRuntimeListener = ReferIt3DListenerRuntime

__all__ = ["ReferIt3DListenerRuntime", "VigorRuntimeListener"]
