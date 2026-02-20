import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
                nn.Linear(input_size, hidden_size//2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size//2, eps=1e-12),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def my_get_siamese_features(net, in_features, numbers):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    n_scenes,n_items = in_features.shape[:2]
    out_features = []
    for i in range(n_scenes):
        cc=net(in_features[i,:numbers[i]])
        dd=torch.ones(n_items,762).cuda()
        dd[:numbers[i]]=cc
        out_features.append(dd)
    out_features = torch.stack(out_features)
    return out_features

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    strict_model = str(os.environ.get("VIGOR_STRICT_LOAD", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
    verbose = str(os.environ.get("VIGOR_VERBOSE_LOAD", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
    allowed_missing_prefixes = tuple(
        p.strip()
        for p in str(
            os.environ.get(
                "VIGOR_ALLOWED_MISSING_PREFIXES",
                # Optional modules that may not exist in older checkpoints.
	                (
	                    "halt_head.,module.halt_head.,"
	                    "scannet_obj_clf.,module.scannet_obj_clf.,"
	                    "anchor_clf.,module.anchor_clf.,"
	                    # VarLen gate head is optional and may not exist in older checkpoints.
	                    "varlen_gate.,module.varlen_gate.,"
	                    # `stop_embed` is a single nn.Parameter (no trailing dot in the key).
	                    "stop_embed,stop_embed.,module.stop_embed,module.stop_embed."
	                ),
	            )
	        ).split(",")
        if p.strip()
    )
    allowed_unexpected_prefixes = tuple(
        p.strip()
        for p in str(
            os.environ.get(
                "VIGOR_ALLOWED_UNEXPECTED_PREFIXES",
                # Modules that can appear in some checkpoints (e.g. *_pre warmup) but not others.
                "feat_to_coor_reg.,module.feat_to_coor_reg.",
            )
        ).split(",")
        if p.strip()
    )

    def _env_flag(name: str, default: str = "0") -> bool:
        v = os.environ.get(name, default)
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _get_word_embedding_weight(model: nn.Module):
        try:
            m = model.module if isinstance(model, nn.DataParallel) else model
            return m.language_encoder.embeddings.word_embeddings.weight
        except Exception:
            return None

    def _maybe_resize_word_embeddings_in_state_dict(state_dict: dict, model: nn.Module):
        """
        Allow loading checkpoints into a model whose tokenizer/embeddings have been
        extended (e.g. adding <stepK> tokens, <STOP>, etc).

        PyTorch load_state_dict errors on shape mismatch even with strict=False.
        We patch checkpoint embedding weights by copying the overlapping rows into the
        *current* resized embedding tensor, keeping the current init for new rows.

        Supported:
          - Vigor BERT: language_encoder.embeddings.word_embeddings.weight
          - Llama-step-slot wrapper: llm.model.model.embed_tokens.weight + llm.model.lm_head.weight
        """
        if not (_env_flag("VIGOR_STEP_MARKERS", "0") or _env_flag("VIGOR_ALLOW_EMB_RESIZE", "0")):
            return state_dict, False
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict, False

        changed = False

        key = "language_encoder.embeddings.word_embeddings.weight"
        if key not in state_dict:
            key = "module.language_encoder.embeddings.word_embeddings.weight"
        if key in state_dict:
            w = state_dict.get(key, None)
            emb_w = _get_word_embedding_weight(model)
            if (torch.is_tensor(w)) and (emb_w is not None) and torch.is_tensor(emb_w):
                if w.ndim == 2 and emb_w.ndim == 2 and int(w.size(1)) == int(emb_w.size(1)) and int(w.size(0)) != int(emb_w.size(0)):
                    patched = emb_w.detach().to("cpu").clone()
                    n = min(int(w.size(0)), int(patched.size(0)))
                    patched[:n] = w[:n].detach().to(device=patched.device, dtype=patched.dtype)
                    state_dict[key] = patched
                    changed = True
                    if verbose:
                        print(
                            f"[Vigor][load_state_dicts] patched word_embeddings from {tuple(w.shape)} to {tuple(patched.shape)}",
                            flush=True,
                        )

        # Patch LLM embeddings if present (Llama-step-slot wrapper checkpoints).
        # This enables adding new special tokens (e.g. <STOP>) after a checkpoint was saved.
        try:
            model_sd = model.state_dict()
        except Exception:
            model_sd = None

        if isinstance(model_sd, dict) and model_sd:
            llm_keys = [
                "llm.model.model.embed_tokens.weight",
                "llm.model.lm_head.weight",
                "module.llm.model.model.embed_tokens.weight",
                "module.llm.model.lm_head.weight",
            ]
            for k in llm_keys:
                if k not in state_dict or k not in model_sd:
                    continue
                src = state_dict.get(k)
                dst_ref = model_sd.get(k)
                if not torch.is_tensor(src) or not torch.is_tensor(dst_ref):
                    continue
                if src.ndim != 2 or dst_ref.ndim != 2:
                    continue
                if int(src.size(1)) != int(dst_ref.size(1)):
                    continue
                if int(src.size(0)) == int(dst_ref.size(0)):
                    continue
                dst = dst_ref.detach().to("cpu").clone()
                src_cpu = src.detach().to("cpu")
                n0 = min(int(src_cpu.size(0)), int(dst.size(0)))
                dst[:n0] = src_cpu[:n0].to(dtype=dst.dtype)
                state_dict[k] = dst
                changed = True
                if verbose:
                    print(
                        f"[Vigor][load_state_dicts] patched {k} from {tuple(src.shape)} to {tuple(dst.shape)}",
                        flush=True,
                    )

        return state_dict, changed

    def _maybe_resize_classifier_heads_in_state_dict(state_dict: dict, model: nn.Module):
        """
        Allow loading checkpoints where the number of object classes differs (e.g. 524 vs 607).
        This affects the final linear layers in classifier heads such as `language_clf.4`.
        Similar to word-embedding resize, PyTorch errors on size mismatch even with strict=False,
        so we patch checkpoint tensors to match the *current* model shapes.
        """
        if not (
            _env_flag("VIGOR_ALLOW_HEAD_RESIZE", "0")
            or _env_flag("VIGOR_ALLOW_CLS_RESIZE", "0")
            or _env_flag("VIGOR_STEP_MARKERS", "0")
        ):
            return state_dict, False
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict, False

        # Only patch known heads where class-count changes are expected.
        candidates = [
            "language_clf.4.weight",
            "language_clf.4.bias",
            "module.language_clf.4.weight",
            "module.language_clf.4.bias",
            # Optional ScanNet200 object classifier head (may exist / differ across checkpoints).
            "scannet_obj_clf.net.8.weight",
            "scannet_obj_clf.net.8.bias",
            "module.scannet_obj_clf.net.8.weight",
            "module.scannet_obj_clf.net.8.bias",
        ]

        model_sd = None
        try:
            model_sd = model.state_dict()
        except Exception:
            model_sd = None

        if not isinstance(model_sd, dict) or not model_sd:
            return state_dict, False

        changed = False
        text_cls_scannet200 = _env_flag("VIGOR_TEXT_CLS_SCANNET200", "0")

        for k in candidates:
            if k not in state_dict or k not in model_sd:
                continue
            src = state_dict.get(k)
            dst_ref = model_sd.get(k)
            if not torch.is_tensor(src) or not torch.is_tensor(dst_ref):
                continue
            if tuple(src.shape) == tuple(dst_ref.shape):
                continue

            # Patch on CPU; keep the destination init for new rows.
            dst = dst_ref.detach().to("cpu").clone()
            src_cpu = src.detach().to("cpu")

            if src_cpu.ndim == 2 and dst.ndim == 2:
                # (out, in): copy overlapping rows/cols.
                # Special case: when switching language_clf to ScanNet200 label space,
                # do NOT copy from an unrelated checkpoint head (different class ordering).
                if text_cls_scannet200 and ("language_clf.4.weight" in k):
                    state_dict[k] = dst
                    changed = True
                    if verbose:
                        print(
                            f"[Vigor][load_state_dicts] reinit {k} due to VIGOR_TEXT_CLS_SCANNET200=1 "
                            f"(ckpt {tuple(src.shape)} -> model {tuple(dst.shape)})",
                            flush=True,
                        )
                    continue

                n0 = min(int(src_cpu.size(0)), int(dst.size(0)))
                n1 = min(int(src_cpu.size(1)), int(dst.size(1)))
                dst[:n0, :n1] = src_cpu[:n0, :n1].to(dtype=dst.dtype)
                state_dict[k] = dst
                changed = True
                if verbose:
                    print(
                        f"[Vigor][load_state_dicts] patched {k} from {tuple(src.shape)} to {tuple(dst.shape)}",
                        flush=True,
                    )
                continue

            if src_cpu.ndim == 1 and dst.ndim == 1:
                if text_cls_scannet200 and ("language_clf.4.bias" in k):
                    state_dict[k] = dst
                    changed = True
                    if verbose:
                        print(
                            f"[Vigor][load_state_dicts] reinit {k} due to VIGOR_TEXT_CLS_SCANNET200=1 "
                            f"(ckpt {tuple(src.shape)} -> model {tuple(dst.shape)})",
                            flush=True,
                        )
                    continue
                n0 = min(int(src_cpu.size(0)), int(dst.size(0)))
                dst[:n0] = src_cpu[:n0].to(dtype=dst.dtype)
                state_dict[k] = dst
                changed = True
                if verbose:
                    print(
                        f"[Vigor][load_state_dicts] patched {k} from {tuple(src.shape)} to {tuple(dst.shape)}",
                        flush=True,
                    )
                continue

        return state_dict, changed

    def _maybe_remap_lora_base_linear_weights_in_state_dict(state_dict: dict, model: nn.Module):
        """
        Compatibility for switching between:
          - plain Linear modules saved as "...q_proj.weight"
          - LoRA-wrapped modules saved/expected as "...q_proj.base.weight" (+ lora_A/B)

        When loading an older checkpoint into a newer LoRA-wrapped model (or when expanding
        LoRA coverage), we remap:
          "<prefix>.weight" -> "<prefix>.base.weight"
          "<prefix>.bias"   -> "<prefix>.base.bias"
        only if the destination key exists in the current model and shapes are compatible.
        """
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict, False
        model_sd = None
        try:
            model_sd = model.state_dict()
        except Exception:
            model_sd = None
        if not isinstance(model_sd, dict) or not model_sd:
            return state_dict, False

        changed = False

        def _try_map(dst_key: str, src_key: str) -> None:
            nonlocal changed
            if dst_key in state_dict:
                return
            if src_key not in state_dict:
                return
            if dst_key not in model_sd:
                return
            src = state_dict.get(src_key)
            dst_ref = model_sd.get(dst_key)
            if not torch.is_tensor(src) or not torch.is_tensor(dst_ref):
                return
            if tuple(src.shape) != tuple(dst_ref.shape):
                return
            state_dict[dst_key] = src
            # Remove the old key so it doesn't show up as "unexpected".
            try:
                del state_dict[src_key]
            except Exception:
                pass
            changed = True

        # Two-way remap:
        #  1) plain -> LoRA base (model expects `.base.*`, ckpt provides `.weight/.bias`)
        #  2) LoRA base -> plain (model expects `.weight/.bias`, ckpt provides `.base.*`)
        for k in list(model_sd.keys()):
            if not isinstance(k, str):
                continue
            if k.endswith(".base.weight"):
                _try_map(k, k.replace(".base.weight", ".weight"))
            elif k.endswith(".base.bias"):
                _try_map(k, k.replace(".base.bias", ".bias"))
            elif k.endswith(".weight"):
                _try_map(k, k.replace(".weight", ".base.weight"))
            elif k.endswith(".bias"):
                _try_map(k, k.replace(".bias", ".base.bias"))

        if verbose and changed:
            print("[Vigor][load_state_dicts] remapped Linear <-> LoRA base.* weights", flush=True)
        return state_dict, changed

    def _state_dict_has_prefix(state_dict, prefix):
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return False
        keys = list(state_dict.keys())
        head = keys[: min(50, len(keys))]
        return sum(1 for k in head if isinstance(k, str) and k.startswith(prefix)) > (len(head) * 0.6)

    def _normalize_state_dict_keys_for_model(state_dict, model):
        if not isinstance(state_dict, dict) or not isinstance(model, nn.Module) or len(state_dict) == 0:
            return state_dict, False

        ckpt_has_module = _state_dict_has_prefix(state_dict, "module.")
        model_has_module = _state_dict_has_prefix(model.state_dict(), "module.")

        if ckpt_has_module and not model_has_module:
            stripped = {}
            changed = False
            for k, v in state_dict.items():
                if isinstance(k, str) and k.startswith("module."):
                    stripped[k[len("module."):]] = v
                    changed = True
                else:
                    stripped[k] = v
            return stripped, changed

        if (not ckpt_has_module) and model_has_module:
            wrapped = {}
            for k, v in state_dict.items():
                wrapped[f"module.{k}"] = v
            return wrapped, True

        return state_dict, False

    for key, value in kwargs.items():
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing key='{key}'. Available keys={list(checkpoint.keys())}")

        # nn.Module: report missing/unexpected keys (and fail fast on shape mismatches).
        if isinstance(value, nn.Module):
            state_dict, normalized = _normalize_state_dict_keys_for_model(checkpoint[key], value)
            if verbose and normalized:
                print(f"[Vigor][load_state_dicts] key='{key}': normalized state_dict keys for DataParallel prefix")
            # Handle tokenizer/embedding extension (e.g. step-marker special tokens).
            state_dict, _ = _maybe_resize_word_embeddings_in_state_dict(state_dict, value)
            # Handle classifier head resize when class-count differs (e.g. 524 vs 607).
            state_dict, _ = _maybe_resize_classifier_heads_in_state_dict(state_dict, value)
            # Handle LoRA wrapper compatibility (Linear.weight -> LoRA.base.weight).
            state_dict, _ = _maybe_remap_lora_base_linear_weights_in_state_dict(state_dict, value)
            try:
                # Always load with strict=False so we can selectively allow missing keys
                # for newly added optional modules (e.g. adaptive-halting heads) while
                # still failing on unexpected gaps when `VIGOR_STRICT_LOAD=1`.
                incompatible = value.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model state_dict from checkpoint='{checkpoint_file}' (strict={strict_model})."
                ) from e

            if verbose:
                missing = getattr(incompatible, "missing_keys", []) or []
                unexpected = getattr(incompatible, "unexpected_keys", []) or []
                if missing or unexpected:
                    print(
                        f"[Vigor][load_state_dicts] key='{key}' strict={strict_model} "
                        f"missing={len(missing)} unexpected={len(unexpected)}"
                    )
                    if len(missing) > 0:
                        print(f"[Vigor][load_state_dicts] missing_keys[:20]={missing[:20]}")
                    if len(unexpected) > 0:
                        print(f"[Vigor][load_state_dicts] unexpected_keys[:20]={unexpected[:20]}")

            # Enforce strictness after filtering allowed missing prefixes.
            if strict_model:
                missing = getattr(incompatible, "missing_keys", []) or []
                unexpected = getattr(incompatible, "unexpected_keys", []) or []
                if allowed_missing_prefixes:
                    missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
                if allowed_unexpected_prefixes:
                    unexpected = [k for k in unexpected if not any(k.startswith(p) for p in allowed_unexpected_prefixes)]
                # LoRA adapters are optional (enabled/disabled by env) and may be absent in older checkpoints.
                # Keep strict-loading for the base model weights while allowing LoRA keys to differ.
                missing = [k for k in missing if ".lora_" not in str(k)]
                unexpected = [k for k in unexpected if ".lora_" not in str(k)]
                if missing or unexpected:
                    raise RuntimeError(
                        f"Failed to load model state_dict from checkpoint='{checkpoint_file}' "
                        f"(strict={strict_model}): missing={len(missing)} unexpected={len(unexpected)}"
                    )
            continue

        # Optimizer / scheduler / others: do not swallow errors.
        try:
            value.load_state_dict(checkpoint[key])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load state_dict for key='{key}' from checkpoint='{checkpoint_file}'."
            ) from e

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
