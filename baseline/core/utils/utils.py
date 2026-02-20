import sys

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

import torch
from loguru import logger
import os

def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.checkpoint, map_location='cpu')["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    missing_keys = []
    for key in list(state_dict.keys()):
        if correct_dict.pop(key, None) is None:
            missing_keys.append(key)

    def _partial_copy_vocab_like(key: str, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor | None:
        """
        Handle common shape mismatches caused by tokenizer/vocab growth:
        - embed_tokens / wte / lm_head: [vocab, hidden]
        Copy the overlapping prefix rows and keep the remaining rows from dst init.
        """
        try:
            if not (torch.is_tensor(src) and torch.is_tensor(dst)):
                return None
            if src.ndim != dst.ndim:
                return None
            # Only allow targeted vocab-like matrices to avoid silently masking real bugs.
            key_l = str(key).lower()
            allow = (
                "embed_tokens.weight" in key_l
                or key_l.endswith(".wte.weight")
                or key_l.endswith(".lm_head.weight")
            )
            if not allow:
                return None
            if src.ndim == 2 and src.shape[1] == dst.shape[1]:
                n = int(min(src.shape[0], dst.shape[0]))
                out = dst.clone()
                out[:n, :] = src[:n, :].to(dtype=dst.dtype)
                return out
            if src.ndim == 1:
                n = int(min(src.shape[0], dst.shape[0]))
                out = dst.clone()
                out[:n] = src[:n].to(dtype=dst.dtype)
                return out
        except Exception:
            return None
        return None

    # if parametrs have different shape, try safe partial load for vocab-like weights;
    # otherwise randomly initialize (keep model init).
    state_dict = torch.load(cfg.general.checkpoint, map_location='cpu')["state_dict"]
    correct_dict = dict(model.state_dict())
    shape_mismatch_keys = []
    for key in correct_dict.keys():
        if key not in state_dict:
            missing_keys.append(key)
            state_dict.update({key: correct_dict[key]})
        elif state_dict[key].shape != correct_dict[key].shape:
            shape_mismatch_keys.append(key)
            patched = _partial_copy_vocab_like(key, state_dict[key], correct_dict[key])
            state_dict.update({key: patched if patched is not None else correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    excessive_keys = []
    for key in state_dict.keys():
        if key in correct_dict.keys():
            new_state_dict.update({key: state_dict[key]})
        else:
            excessive_keys.append(key)
    model.load_state_dict(new_state_dict)

    if missing_keys or shape_mismatch_keys or excessive_keys:
        logger.debug(
            "Checkpoint mismatches summary | missing={} | shape={} | excessive={}",
            len(missing_keys),
            len(shape_mismatch_keys),
            len(excessive_keys),
        )
        verbose = str(torch.get_default_dtype())  # keep torch imported / appease linters
        _ = verbose
        if str(os.environ.get("CKPT_MISMATCH_VERBOSE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                show = int(os.environ.get("CKPT_MISMATCH_SHOW", "25"))
            except Exception:
                show = 25
            show = max(1, int(show))
            if missing_keys:
                logger.warning("Missing keys (showing {}): {}", min(show, len(missing_keys)), missing_keys[:show])
            if shape_mismatch_keys:
                logger.warning(
                    "Shape-mismatch keys (showing {}): {}",
                    min(show, len(shape_mismatch_keys)),
                    shape_mismatch_keys[:show],
                )
            if excessive_keys:
                logger.warning("Excessive keys (showing {}): {}", min(show, len(excessive_keys)), excessive_keys[:show])
    return cfg, model
