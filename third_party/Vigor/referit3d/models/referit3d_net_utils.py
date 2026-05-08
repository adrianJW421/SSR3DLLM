"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import tqdm
import torch.nn.functional as F
import torch.nn as nn
import time
from pathlib import Path

from ..utils.evaluation import AverageMeter


def _env_flag(name: str, default: str = "0") -> bool:
    v = str(__import__("os").environ.get(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(__import__("os").environ.get(name, str(default))).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(__import__("os").environ.get(name, str(default))).strip())
    except Exception:
        return float(default)


def _safe_get_referential_token(ref_order, sample_idx: int, step_idx: int) -> str:
    """
    Vigor's DataLoader returns `referential_order` as a Python object; depending on
    how it was collated, it can be either:
      - list[batch] of list[order_len] strings, or
      - list[order_len] of list[batch] strings.
    This helper makes the training loop robust to both layouts.
    """
    if ref_order is None:
        return ""
    try:
        return ref_order[sample_idx][step_idx]
    except Exception:
        try:
            return ref_order[step_idx][sample_idx]
        except Exception:
            return ""


def _safe_get_ori_order_len(batch: dict, sample_idx: int, default: int) -> int:
    """
    Robustly read `ori_order_len` for a given sample from a collated batch.

    Vigor data loaders may keep `ori_order_len` as:
      - a torch.Tensor [B]
      - a list[int] length B
      - a scalar int (rare)
    """
    if not isinstance(batch, dict) or "ori_order_len" not in batch:
        force = _env_int("VIGOR_FORCE_ORI_ORDER_LEN", -1)
        if force and int(force) > 0:
            return int(min(int(force), int(default)))
        return int(default)
    try:
        force = _env_int("VIGOR_FORCE_ORI_ORDER_LEN", -1)
        if force and int(force) > 0:
            return int(min(int(force), int(default)))
        v = batch["ori_order_len"]
        if torch.is_tensor(v):
            if v.numel() == 1:
                return int(v.view(-1)[0].item())
            return int(v.view(-1)[int(sample_idx)].item())
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return int(default)
            return int(v[int(sample_idx)])
        return int(v)
    except Exception:
        return int(default)


def _reshape_order_tokens(order_tokens: dict, batch_size: int, order_len: int) -> dict:
    """
    Make `order_tokens` DataParallel-friendly.

    `tokenizer(..., return_tensors='pt')` produces tensors shaped [B*order_len, L].
    Under DataParallel, other batch tensors are scattered along B, so scattering
    [B*order_len, L] independently can desync per-replica B and break downstream
    reshapes. We reshape to [B, order_len, L] so DataParallel always scatters
    consistently by the true batch dimension.
    """
    reshaped = {}
    for k, v in order_tokens.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.size(0) == batch_size * order_len:
            reshaped[k] = v.reshape(batch_size, order_len, v.size(1))
        else:
            reshaped[k] = v
    return reshaped


def _build_step_marker_text(step_idx: int, utterance: str, step_text: str, order_len: int) -> str:
    """
    Build per-step text so that the special token <stepK> is always the first token
    after [CLS]. This makes it easy to extract its hidden state as a "step slot".
    """
    k = int(step_idx) + 1
    if k < 1:
        k = 1
    if k > int(order_len):
        k = int(order_len)
    u = str(utterance or "").strip()
    s = str(step_text or "").strip()
    # Keep the marker at the beginning; rest can be any text (teacher forced).
    return f"<step{k}> {u} {s}".strip()


def _env_str(name: str, default: str = "") -> str:
    return str(__import__("os").environ.get(name, default)).strip()


def _order_perm(order_len: int, seed: int, salt: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(salt))
    return torch.randperm(int(order_len), generator=g)


def _order_perm_varlen(order_len: int, prefix_len: int, seed: int, salt: int, mode: str) -> torch.Tensor:
    """
    Build a permutation for step order that preserves the padded tail (e.g. STOP slots).

    When varlen/STOP is enabled, we should NOT move padded slots into the valid prefix,
    otherwise the perturbation changes the effective chain length instead of testing
    step-order dependence.
    """
    order_len = int(order_len)
    prefix_len = int(prefix_len)
    mode = str(mode or "").strip().lower()
    if order_len <= 0:
        return torch.zeros((0,), dtype=torch.long)
    prefix_len = max(0, min(order_len, prefix_len))
    if prefix_len <= 1:
        return torch.arange(order_len, dtype=torch.long)
    if mode == "reverse":
        head = torch.arange(prefix_len - 1, -1, -1, dtype=torch.long)
    else:
        head = _order_perm(prefix_len, seed=seed, salt=salt)
    tail = torch.arange(prefix_len, order_len, dtype=torch.long)
    return torch.cat([head, tail], dim=0)


def _apply_pred_class_mask_mode(batch: dict, mode: str) -> None:
    """
    Mutate batch['pred_class_mask'] in-place for ablations.
    mode:
      - "normal": no-op
      - "all_ones": allow all existing context objects at every step
    """
    m = str(mode or "").strip().lower()
    if m in {"", "none", "normal"}:
        return
    if m not in {"all_ones"}:
        return
    try:
        pcm = batch.get("pred_class_mask", None)
        if not torch.is_tensor(pcm) or pcm.dim() != 3:
            return
        B, T, N = pcm.shape
        obj_mask = batch.get("obj_mask", None)
        if torch.is_tensor(obj_mask) and obj_mask.dim() >= 2 and obj_mask.size(0) == B:
            valid = (obj_mask[:, :N] > 0).to(dtype=pcm.dtype)
        else:
            valid = torch.ones((B, N), dtype=pcm.dtype, device=pcm.device)
        pcm[:] = valid.unsqueeze(1).expand(B, T, N)
        batch["pred_class_mask"] = pcm
    except Exception:
        return


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    dino_enabled = str(
        os.environ.get(
            "VIGOR_MASK3D_DINO_ENABLE",
            "1" if str(os.environ.get("VIGOR_MASK3D_DINO_SAMPLE_CACHE_ROOT", "")).strip() else "0",
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    if dino_enabled:
        batch_keys += ["mask3d_dino_features", "mask3d_dino_valid_mask"]

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, args, tokenizer=None,epoch=None):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """

    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    scannet_cls_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    distill_step_mtr = AverageMeter()
    distill_global_mtr = AverageMeter()
    distill_step_mp_mtr = AverageMeter()
    distill_global_mp_mtr = AverageMeter()
    mp_op_distill_mtr = AverageMeter()
    mk_loss_mtr = AverageMeter()
    gate_loss_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)
    batch_idx = 0
    pbar = tqdm.tqdm(data_loader)
    for batch in pbar:
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Convert tokenizer outputs to a plain dict so DataParallel can scatter them.
        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        lang_tokens = {k: v.to(device) for k, v in lang_tokens.items()}

        # Optional ablation: perturb step order / blank step text.
        order_perturb = _env_str("VIGOR_ORDER_PERTURB", "none").lower()
        mask_mode = _env_str("VIGOR_PRED_CLASS_MASK_MODE", "normal").lower()
        varlen_enabled = _env_flag("VIGOR_VARLEN_CHAIN", "0")
        try:
            shuffle_seed = int(_env_str("VIGOR_ORDER_SHUFFLE_SEED", "0") or "0")
        except Exception:
            shuffle_seed = 0

        # Keep step-dependent supervision tensors consistent with the text order.
        if order_perturb in {"shuffle", "reverse"}:
            try:
                B = int(batch['target_pos'].size(0))
                for b in range(B):
                    prefix_len = int(args.order_len)
                    if varlen_enabled:
                        prefix_len = _safe_get_ori_order_len(batch, b, default=int(args.order_len))
                    perm = _order_perm_varlen(int(args.order_len), int(prefix_len), shuffle_seed, b, order_perturb)
                    if "pred_class_mask" in batch and torch.is_tensor(batch["pred_class_mask"]) and batch["pred_class_mask"].dim() == 3:
                        batch["pred_class_mask"][b] = batch["pred_class_mask"][b].index_select(0, perm)
                    if "ordered_multilabel_gt" in batch and torch.is_tensor(batch["ordered_multilabel_gt"]) and batch["ordered_multilabel_gt"].dim() == 3:
                        batch["ordered_multilabel_gt"][b] = batch["ordered_multilabel_gt"][b].index_select(0, perm)
                    # Also permute referential_order itself so llama-step-slot sees the same perturbation.
                    try:
                        ro = batch.get("referential_order", None)
                        if isinstance(ro, list) and b < len(ro) and isinstance(ro[b], list):
                            row = list(ro[b])
                            if len(row) >= int(args.order_len):
                                ro[b] = [row[int(i)] for i in perm.tolist()]
                    except Exception:
                        pass
            except Exception:
                pass

        # Build flattened referential-order texts.
        # Use tensor batch size as ground truth to avoid rare list-length mismatch.
        order = []
        B = int(batch['target_pos'].size(0))
        for i in range(B):
            perm = None
            if order_perturb in {"shuffle", "reverse"}:
                prefix_len = int(args.order_len)
                if varlen_enabled:
                    prefix_len = _safe_get_ori_order_len(batch, i, default=int(args.order_len))
                perm = _order_perm_varlen(int(args.order_len), int(prefix_len), shuffle_seed, i, order_perturb).tolist()
            for j in range(args.order_len):
                jj = perm[j] if perm is not None else j
                tok = _safe_get_referential_token(batch.get('referential_order', None), i, jj)
                if order_perturb in {"blank", "marker"}:
                    tok = ""
                # Counterfactual: force STOP tail by overriding padded steps to STOP.
                # This is evaluation-only plumbing; it does not change ground-truth `ori_order_len`
                # unless VIGOR_FORCE_ORI_ORDER_LEN is set.
                if varlen_enabled and _env_flag("VIGOR_FORCE_STOP_TAIL", "0"):
                    prefix_len = _safe_get_ori_order_len(batch, i, default=int(args.order_len))
                    if j >= int(prefix_len):
                        tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                if _env_flag("VIGOR_STEP_MARKERS", "0"):
                    utt = batch["tokens"][i] if isinstance(batch.get("tokens", None), list) and i < len(batch["tokens"]) else ""
                    if order_perturb == "marker":
                        utt = ""
                    tok = _build_step_marker_text(j, utt, tok, order_len=args.order_len)
                order.append(tok)

        order_tokens = tokenizer(order, return_tensors='pt', padding=True)
        order_tokens = _reshape_order_tokens(order_tokens, batch_size=B, order_len=args.order_len)
        order_tokens = {k: v.to(device) for k, v in order_tokens.items()}
        
        batch['order_tokens'] = order_tokens
        _apply_pred_class_mask_mode(batch, mask_mode)
        batch['pred_class_mask'] = batch['pred_class_mask'].to(device)
        if args.lang_multilabel:
            batch['anchor_ind'] = batch['anchor_ind'].to(device)
        if args.multilabel_pretraining:
            batch['ordered_multilabel_gt'] = batch['ordered_multilabel_gt'].to(device)
            batch['center_coors'] = batch['center_coors'].to(device)
            batch['corner_coors'] = batch['corner_coors'].to(device)
            batch['obj_mask'] = batch['obj_mask'].to(device).squeeze()

        batch['lang_tokens'] = lang_tokens

        # Forward pass
        out = model(batch, epoch)
        SCANNET_CLASS_LOGITS = None
        scannet_labels = None
        if isinstance(out, (list, tuple)):
            if len(out) == 6:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS, scannet_labels = out
            elif len(out) == 5:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS = out
            else:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out
        else:
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out
        LOSS = LOSS.mean()

        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        # Optional: LLM distillation diagnostics (available only for llama-step-slot wrapper).
        d_step = getattr(base_model, "last_distill_step", None)
        d_glb = getattr(base_model, "last_distill_global", None)
        d_step_mp = getattr(base_model, "last_distill_step_mp", None)
        d_glb_mp = getattr(base_model, "last_distill_global_mp", None)
        d_mp_op = getattr(base_model, "last_mp_op_distill", None)
        d_mk = getattr(base_model, "last_mk_loss", None)
        d_gate = getattr(base_model, "last_gate_loss", None)
        if d_step is not None:
            try:
                distill_step_mtr.update(float(d_step), batch_size)
            except Exception:
                pass
        if d_glb is not None:
            try:
                distill_global_mtr.update(float(d_glb), batch_size)
            except Exception:
                pass
        if d_step_mp is not None:
            try:
                distill_step_mp_mtr.update(float(d_step_mp), batch_size)
            except Exception:
                pass
        if d_glb_mp is not None:
            try:
                distill_global_mp_mtr.update(float(d_glb_mp), batch_size)
            except Exception:
                pass
        if d_mp_op is not None:
            try:
                mp_op_distill_mtr.update(float(d_mp_op), batch_size)
            except Exception:
                pass
        if d_mk is not None:
            try:
                mk_loss_mtr.update(float(d_mk), batch_size)
            except Exception:
                pass
        if d_gate is not None:
            try:
                gate_loss_mtr.update(float(d_gate), batch_size)
            except Exception:
                pass
        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)
        if (d_step is not None) or (d_glb is not None) or (d_step_mp is not None) or (d_mp_op is not None) or (d_mk is not None) or (d_gate is not None):
            try:
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss_mtr.avg:.3f}",
                        "acc": f"{ref_acc_mtr.avg:.3f}",
                        "d_step": f"{distill_step_mtr.val:.4f}" if d_step is not None else "-",
                        "d_glb": f"{distill_global_mtr.val:.4f}" if d_glb is not None else "-",
                        "d_mp": f"{distill_step_mp_mtr.val:.4f}" if d_step_mp is not None else "-",
                        "d_op": f"{mp_op_distill_mtr.val:.4f}" if d_mp_op is not None else "-",
                        "mk": f"{mk_loss_mtr.val:.4f}" if d_mk is not None else "-",
                        "gate": f"{gate_loss_mtr.val:.4f}" if d_gate is not None else "-",
                    },
                    refresh=False,
                )
            except Exception:
                pass

        # 原始 607 类 object head 的准确率（仅在未启用 ScanNet200 头时统计）
        if args.obj_cls_alpha > 0 and not getattr(args, "use_scannet200_obj_cls", False):
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        # Optional: ScanNet200-based object classification accuracy
        if getattr(args, "use_scannet200_obj_cls", False) and SCANNET_CLASS_LOGITS is not None:
            labels = scannet_labels if scannet_labels is not None else batch.get('scannet_class_labels', None)
            if labels is not None:
                preds = SCANNET_CLASS_LOGITS.argmax(dim=-1)
                valid = labels >= 0
                if valid.any():
                    correct = (preds[valid] == labels[valid]).double().mean().item()
                    scannet_cls_mtr.update(correct, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            target_class = batch['target_class']
            if batch_guess.shape[0] != target_class.shape[0]:
                m = min(int(batch_guess.shape[0]), int(target_class.shape[0]))
                batch_guess = batch_guess[:m]
                target_class = target_class[:m]
            cls_b_acc = torch.mean((batch_guess == target_class).double()).item()
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_scannet_object_cls_acc'] = scannet_cls_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    if distill_step_mtr.count > 0:
        metrics['train_llm_distill_step'] = distill_step_mtr.avg
    if distill_global_mtr.count > 0:
        metrics['train_llm_distill_global'] = distill_global_mtr.avg
    if distill_step_mp_mtr.count > 0:
        metrics['train_llm_distill_step_mp'] = distill_step_mp_mtr.avg
    if distill_global_mp_mtr.count > 0:
        metrics['train_llm_distill_global_mp'] = distill_global_mp_mtr.avg
    if mp_op_distill_mtr.count > 0:
        metrics['train_llm_distill_mp_op'] = mp_op_distill_mtr.avg
    if mk_loss_mtr.count > 0:
        metrics['train_stop_mk_loss'] = mk_loss_mtr.avg
    if gate_loss_mtr.count > 0:
        metrics['train_varlen_gate_loss'] = gate_loss_mtr.avg
    return metrics


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False, tokenizer=None):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    scannet_cls_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    distill_step_mtr = AverageMeter()
    distill_global_mtr = AverageMeter()
    distill_step_mp_mtr = AverageMeter()
    distill_global_mp_mtr = AverageMeter()
    mp_op_distill_mtr = AverageMeter()
    mk_loss_mtr = AverageMeter()
    gate_loss_mtr = AverageMeter()
    bbox_iou_mtr = AverageMeter()
    bbox_iou_25_mtr = AverageMeter()
    bbox_iou_50_mtr = AverageMeter()
    # Diagnostics: decompose bbox IoU by decision correctness.
    # - "correct": predicted index == GT target index (isolates box quality when decision is right)
    # - "wrong": predicted index != GT target index (isolates decision errors)
    bbox_iou_correct_mtr = AverageMeter()
    bbox_iou_25_correct_mtr = AverageMeter()
    bbox_iou_50_correct_mtr = AverageMeter()
    bbox_iou_wrong_mtr = AverageMeter()
    bbox_iou_25_wrong_mtr = AverageMeter()
    bbox_iou_50_wrong_mtr = AverageMeter()
    # Diagnostic: target-only box quality (IoU between target's pred-box and target's GT box),
    # independent of which object the model predicted.
    bbox_iou_target_mtr = AverageMeter()
    bbox_iou_25_target_mtr = AverageMeter()
    bbox_iou_50_target_mtr = AverageMeter()
    bbox_iou_25_unique_mtr = AverageMeter()
    bbox_iou_50_unique_mtr = AverageMeter()
    bbox_iou_25_multiple_mtr = AverageMeter()
    bbox_iou_50_multiple_mtr = AverageMeter()
    # Optional: formulation-aligned query-hit proxy (P2) using exported `gt_to_query_map`.
    queryacc_proxy = _env_flag("VIGOR_QUERYACC_PROXY", "0")
    queryacc_mtr = AverageMeter()
    queryacc_easy_mtr = AverageMeter()
    queryacc_hard_mtr = AverageMeter()
    queryacc_vdep_mtr = AverageMeter()
    queryacc_vindep_mtr = AverageMeter()
    f1_iou_25_mtr = AverageMeter()
    f1_iou_50_mtr = AverageMeter()
    # Optional probe: can we infer the varlen mask m_k from STOP similarity (no GT at inference)?
    varlen_mk_eval = _env_flag("VIGOR_VARLEN_MK_EVAL", "0")
    varlen_mk_only = _env_flag("VIGOR_VARLEN_MK_ONLY", "0")
    varlen_mk_only_max_batches = max(0, _env_int("VIGOR_VARLEN_MK_ONLY_MAX_BATCHES", 0))
    varlen_mk_sweep = _env_flag("VIGOR_VARLEN_MK_SWEEP", "1")
    varlen_mk_tau = _env_float("VIGOR_VARLEN_MK_TAU", 0.90)
    varlen_mk_use_post_stop = _env_flag("VIGOR_VARLEN_MK_USE_POST_STOP", "0")
    varlen_mk_cos_all = []
    varlen_mk_gt_all = []
    varlen_mk_debug = _env_flag("VIGOR_VARLEN_MK_DEBUG", "0")
    varlen_mk_last_error = None

    # Optional probe: learned varlen gate head (STOP probability) statistics.
    varlen_gate_eval = _env_flag("VIGOR_VARLEN_GATE_EVAL", "0")
    varlen_gate_sweep = _env_flag("VIGOR_VARLEN_GATE_SWEEP", "1")
    varlen_gate_tau = _env_float("VIGOR_VARLEN_GATE_TAU", 0.5)
    varlen_gate_prob_all = []
    varlen_gate_gt_all = []
    varlen_gate_last_error = None

    # Generic "probe-only" mode (skip grounding): reuse the mk-only mechanism for gate probing too.
    varlen_probe_only = bool(varlen_mk_only) and bool(varlen_mk_eval or varlen_gate_eval)

    # Optional speed benchmark (OP vs MP): measures walltime of `base_model(batch)` in eval.
    speed_bench = _env_flag("VIGOR_SPEED_BENCH", "0")
    speed_warmup = max(0, _env_int("VIGOR_SPEED_WARMUP", 5))
    speed_max_batches = max(0, _env_int("VIGOR_SPEED_MAX_BATCHES", 50))
    speed_early_stop = _env_flag("VIGOR_SPEED_EARLY_STOP", "1")
    speed_sync_cuda = _env_flag("VIGOR_SPEED_SYNC_CUDA", "1")
    speed_time_s = 0.0
    speed_samples = 0
    speed_batches = 0

    # Set the model in eval mode
    model.eval()
    # When using DataParallel, do eval on a single GPU to avoid
    # last-batch size / referential-order reshape mismatches.
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_device = next(base_model.parameters()).device

    if randomize:
        np.random.seed()
    else:
        np.random.seed(args.random_seed)

    # For probe-only mode we don't need most batch tensors on GPU (and we skip the listener).
    batch_keys = [] if varlen_probe_only else make_batch_keys(args)

    debug_align = _env_flag("VIGOR_DEBUG_ALIGN", "0")
    debug_batches = max(0, _env_int("VIGOR_DEBUG_ALIGN_BATCHES", 2))
    debug_samples = max(0, _env_int("VIGOR_DEBUG_ALIGN_SAMPLES", 2))
    debug_halt = _env_flag("VIGOR_DEBUG_HALT", "0")
    debug_cache = {}
    batch_idx = 0
    # Cache gt_to_query_map per scene (when VIGOR_QUERYACC_PROXY=1).
    gt_map_cache = {}
    vdep_words = {
        "front",
        "behind",
        "back",
        "right",
        "left",
        "facing",
        "leftmost",
        "rightmost",
        "looking",
        "across",
    }

    pbar = tqdm.tqdm(data_loader)
    for batch in pbar:
        # Probe-only: compute mk/gate statistics without running grounding/eval metrics.
        if varlen_probe_only:
            # Determine batch size without relying on any tensor keys.
            tokens = batch.get("tokens", None)
            if isinstance(tokens, list):
                B = int(len(tokens))
            elif "target_pos" in batch and torch.is_tensor(batch["target_pos"]):
                B = int(batch["target_pos"].size(0))
            else:
                B = 0

            # Compute & cache order embeddings without invoking the listener.
            if hasattr(base_model, "mk_only_forward"):
                base_model.mk_only_forward(batch)
            else:
                # Fallback: run a full forward (slower) but still allows mk probing.
                _ = base_model(batch)

            # Collect GT mask once; used for mk and/or gate probing.
            emb = None
            if varlen_mk_use_post_stop:
                emb = getattr(base_model, "last_order_embeds_post_stop", None)
            if emb is None:
                emb = getattr(base_model, "last_order_embeds", None)
            if not torch.is_tensor(emb):
                emb = None
            gt = None
            try:
                if emb is not None:
                    gt = batch.get("order_valid_mask", None)
                    if (gt is None) and ("ori_order_len" in batch):
                        ori_len_t = torch.as_tensor(batch["ori_order_len"], device=emb.device).long().view(-1)
                        steps = torch.arange(int(args.order_len), device=emb.device).view(1, -1)
                        gt = (steps < ori_len_t.view(-1, 1)).to(dtype=torch.float32)
                    if gt is None:
                        ro = batch.get("referential_order", None)
                        if ro is not None:
                            stop_tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                            rows = []
                            for b in range(int(B)):
                                row = []
                                for k in range(int(args.order_len)):
                                    s = str(_safe_get_referential_token(ro, b, k) or "").strip()
                                    valid = (s != "") and (s != stop_tok)
                                    row.append(1.0 if valid else 0.0)
                                rows.append(row)
                            if rows:
                                gt = torch.as_tensor(rows, device=emb.device, dtype=torch.float32)
            except Exception:
                gt = None

            # mk probe
            if varlen_mk_eval and emb is not None and torch.is_tensor(gt):
                try:
                    stop = getattr(base_model, "stop_embed", None)
                    if not torch.is_tensor(stop):
                        raise RuntimeError("missing stop_embed")
                    stop_n = F.normalize(stop.detach().float().view(1, 1, -1), dim=-1)
                    emb_n = F.normalize(emb.detach().float(), dim=-1)
                    cos = (emb_n * stop_n).sum(dim=-1)
                    varlen_mk_cos_all.append(cos.detach().to("cpu"))
                    varlen_mk_gt_all.append(gt.detach().to("cpu"))
                except Exception:
                    pass

            # gate probe
            if varlen_gate_eval and emb is not None and torch.is_tensor(gt):
                try:
                    if hasattr(base_model, "_varlen_gate_logits"):
                        logits = base_model._varlen_gate_logits(emb)  # [B,O]
                    elif hasattr(base_model, "varlen_gate"):
                        logits = base_model.varlen_gate(emb).squeeze(-1)
                    else:
                        raise RuntimeError("missing gate head")
                    prob = torch.sigmoid(logits.detach().float())
                    varlen_gate_prob_all.append(prob.to("cpu"))
                    varlen_gate_gt_all.append(gt.detach().to("cpu"))
                except Exception:
                    pass

            batch_idx += 1
            if varlen_mk_only_max_batches > 0 and batch_idx >= varlen_mk_only_max_batches:
                break
            continue

        # Move data to gpu (full eval path)
        for k in batch_keys:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(base_device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Convert tokenizer outputs to a plain dict so DataParallel can scatter them.
        if tokenizer is not None:
            lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
            lang_tokens = {k: v.to(base_device) for k, v in lang_tokens.items()}
            batch['lang_tokens'] = lang_tokens

        order_perturb = _env_str("VIGOR_ORDER_PERTURB", "none").lower()
        mask_mode = _env_str("VIGOR_PRED_CLASS_MASK_MODE", "normal").lower()
        varlen_enabled = _env_flag("VIGOR_VARLEN_CHAIN", "0")
        try:
            shuffle_seed = int(_env_str("VIGOR_ORDER_SHUFFLE_SEED", "0") or "0")
        except Exception:
            shuffle_seed = 0

        # In eval, optionally perturb the step-dependent masks in the same way as the texts.
        if order_perturb in {"shuffle", "reverse"}:
            try:
                for b in range(int(B)):
                    prefix_len = int(args.order_len)
                    if varlen_enabled and ("ori_order_len" in batch) and torch.is_tensor(batch["ori_order_len"]):
                        try:
                            prefix_len = int(batch["ori_order_len"][b].item())
                        except Exception:
                            prefix_len = int(args.order_len)
                    perm = _order_perm_varlen(int(args.order_len), int(prefix_len), shuffle_seed, b, order_perturb)
                    if "pred_class_mask" in batch and torch.is_tensor(batch["pred_class_mask"]) and batch["pred_class_mask"].dim() == 3:
                        batch["pred_class_mask"][b] = batch["pred_class_mask"][b].index_select(0, perm)
                    if "ordered_multilabel_gt" in batch and torch.is_tensor(batch["ordered_multilabel_gt"]) and batch["ordered_multilabel_gt"].dim() == 3:
                        batch["ordered_multilabel_gt"][b] = batch["ordered_multilabel_gt"][b].index_select(0, perm)
                    # Also permute referential_order itself so llama-step-slot sees the same perturbation.
                    try:
                        ro = batch.get("referential_order", None)
                        if isinstance(ro, list) and b < len(ro) and isinstance(ro[b], list):
                            row = list(ro[b])
                            if len(row) >= int(args.order_len):
                                ro[b] = [row[int(i)] for i in perm.tolist()]
                    except Exception:
                        pass
            except Exception:
                pass

        order = []
        B = int(batch['target_pos'].size(0))
        for i in range(B):
            perm = None
            if order_perturb in {"shuffle", "reverse"}:
                prefix_len = int(args.order_len)
                if varlen_enabled and ("ori_order_len" in batch) and torch.is_tensor(batch["ori_order_len"]):
                    try:
                        prefix_len = int(batch["ori_order_len"][i].item())
                    except Exception:
                        prefix_len = int(args.order_len)
                perm = _order_perm_varlen(int(args.order_len), int(prefix_len), shuffle_seed, i, order_perturb).tolist()
            for j in range(args.order_len):
                jj = perm[j] if perm is not None else j
                tok = _safe_get_referential_token(batch.get('referential_order', None), i, jj)
                if order_perturb in {"blank", "marker"}:
                    tok = ""
                if _env_flag("VIGOR_STEP_MARKERS", "0"):
                    utt = batch["tokens"][i] if isinstance(batch.get("tokens", None), list) and i < len(batch["tokens"]) else ""
                    if order_perturb == "marker":
                        utt = ""
                    tok = _build_step_marker_text(j, utt, tok, order_len=args.order_len)
                order.append(tok)

        order_tokens = tokenizer(order, return_tensors='pt', padding=True)
        order_tokens = _reshape_order_tokens(order_tokens, batch_size=B, order_len=args.order_len)
        order_tokens = {k: v.to(base_device) for k, v in order_tokens.items()}
        
        batch['order_tokens'] = order_tokens
        _apply_pred_class_mask_mode(batch, mask_mode)
        batch['pred_class_mask'] = batch['pred_class_mask'].to(base_device)
        if args.lang_multilabel:
            batch['anchor_ind'] = batch['anchor_ind'].to(base_device)
        if args.multilabel_pretraining:
            batch['ordered_multilabel_gt'] = batch['ordered_multilabel_gt'].to(base_device)
            batch['center_coors'] = batch['center_coors'].to(base_device)
            batch['corner_coors'] = batch['corner_coors'].to(base_device)
            batch['obj_mask'] = batch['obj_mask'].to(base_device).squeeze()

        # Optional alignment debug prints (main process only).
        # This helps diagnose cases where Vigor accuracy collapses to ~random due to:
        # - gt_to_query_map key-space mismatch (instance_ids not found),
        # - pred_class_mask excluding the target slot (class-name normalization mismatch),
        # - predbox mode producing zero box_info due to mapping misses.
        if debug_align and (batch_idx < debug_batches):
            try:
                B = int(batch["target_pos"].size(0))
            except Exception:
                B = 0
            try:
                scan_ids = batch.get("scan_id", None)
            except Exception:
                scan_ids = None
            try:
                instance_ids = batch.get("instance_ids", None)
            except Exception:
                instance_ids = None
            try:
                mask3d_paths = batch.get("mask3d_feature_path", None)
            except Exception:
                mask3d_paths = None
            pred_mask = batch.get("pred_class_mask", None)
            obj_mask = batch.get("obj_mask", None)
            box_info = batch.get("box_info", None)
            ref_order = batch.get("referential_order", None)

            print(
                f"[Vigor][debug_align] batch_idx={int(batch_idx)} "
                f"B={B} order_len={getattr(args, 'order_len', None)} "
                f"VIGOR_USE_PRED_BOX_INFO={__import__('os').environ.get('VIGOR_USE_PRED_BOX_INFO','0')}",
                flush=True,
            )

            max_b = min(int(B), int(debug_samples))
            for b in range(max_b):
                try:
                    tgt = int(batch["target_pos"][b].item())
                except Exception:
                    tgt = -1
                try:
                    if torch.is_tensor(obj_mask) and obj_mask.dim() >= 2:
                        # obj_mask: [B, max_context]
                        ctx = int((obj_mask[b] > 0).sum().item())
                    else:
                        ctx = -1
                except Exception:
                    ctx = -1

                sid = None
                try:
                    if isinstance(scan_ids, (list, tuple)) and b < len(scan_ids):
                        sid = scan_ids[b]
                except Exception:
                    sid = None

                tgt_inst = None
                try:
                    if torch.is_tensor(instance_ids) and instance_ids.dim() >= 2 and tgt >= 0:
                        tgt_inst = int(instance_ids[b, tgt].item())
                except Exception:
                    tgt_inst = None

                # pred_class_mask diagnostics.
                step_stats = []
                try:
                    if torch.is_tensor(pred_mask) and pred_mask.dim() == 3 and ctx > 0 and tgt >= 0:
                        O = int(pred_mask.shape[1])
                        for t in range(O):
                            m = pred_mask[b, t, :ctx]
                            m_sum = int((m > 0).sum().item())
                            tgt_ok = int((pred_mask[b, t, tgt] > 0).item())
                            step_stats.append((m_sum, tgt_ok))
                except Exception:
                    step_stats = []

                # referential order strings (best-effort).
                order_preview = []
                try:
                    for t in range(int(getattr(args, "order_len", 0) or 0)):
                        tok = _safe_get_referential_token(ref_order, b, t)
                        if tok:
                            order_preview.append(str(tok))
                except Exception:
                    order_preview = []

                # Mask3D mapping diagnostics (best-effort, a few samples only).
                map_hit = None
                tgt_in_map = None
                map_key_range = None
                if isinstance(mask3d_paths, (list, tuple)) and b < len(mask3d_paths):
                    p = str(mask3d_paths[b])
                    if p and p not in debug_cache:
                        try:
                            debug_cache[p] = torch.load(p, map_location="cpu")
                        except Exception:
                            debug_cache[p] = None
                    feat = debug_cache.get(p, None)
                    try:
                        gt_map = (feat.get("gt_to_query_map") if isinstance(feat, dict) else None) or {}
                        if isinstance(gt_map, dict) and len(gt_map) > 0:
                            keys = []
                            try:
                                keys = [int(k) for k in gt_map.keys()]
                            except Exception:
                                keys = []
                            if keys:
                                map_key_range = (min(keys), max(keys))
                    except Exception:
                        gt_map = {}
                    try:
                        if torch.is_tensor(instance_ids) and instance_ids.dim() >= 2 and isinstance(gt_map, dict):
                            ids = instance_ids[b, : max(ctx, 0)].detach().to("cpu").numpy().tolist() if ctx > 0 else []
                            ids = [int(x) for x in ids if int(x) >= 0]
                            if ids:
                                hits = sum(1 for x in ids if x in gt_map)
                                map_hit = float(hits) / float(len(ids))
                            if tgt_inst is not None:
                                tgt_in_map = int(tgt_inst in gt_map)
                    except Exception:
                        map_hit = map_hit

                # Pred-box diagnostics: does target slot have non-zero box?
                tgt_box = None
                try:
                    if torch.is_tensor(box_info) and box_info.dim() == 3 and tgt >= 0:
                        tb = box_info[b, tgt].detach().to("cpu").float().numpy().tolist()
                        tgt_box = [float(x) for x in tb]
                except Exception:
                    tgt_box = None

                print(
                    f"[Vigor][debug_align] sample={b} scan_id={sid} ctx={ctx} "
                    f"target_pos={tgt} target_inst_id={tgt_inst} "
                    f"order={order_preview} "
                    f"mask(step_sum,target_ok)={step_stats} "
                    f"gt_map_hit={map_hit} target_in_gt_map={tgt_in_map} gt_map_key_range={map_key_range} "
                    f"target_box_info={tgt_box}",
                    flush=True,
                )

        # Forward pass
        dt = None
        if speed_bench:
            try:
                if speed_sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            t0 = time.perf_counter()
            out = base_model(batch)
            try:
                if speed_sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            dt = time.perf_counter() - t0
        else:
            out = base_model(batch)
        SCANNET_CLASS_LOGITS = None
        scannet_labels = None
        if isinstance(out, (list, tuple)):
            if len(out) == 6:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS, scannet_labels = out
            elif len(out) == 5:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS = out
            else:
                LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out
        else:
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out
        LOSS = LOSS.mean()
        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS

        # Optional adaptive-halting debug (prints one batch).
        if debug_halt and batch_idx == 0:
            try:
                w = getattr(base_model, "last_halt_weights", None)
                h = getattr(base_model, "last_halt_logits", None)
                ori_len = batch.get("ori_order_len", None)
                if torch.is_tensor(w) and torch.is_tensor(h):
                    w_cpu = w.detach().to("cpu")
                    h_cpu = h.detach().to("cpu")
                    stop_idx = None
                    if ori_len is not None:
                        stop_idx = torch.as_tensor(ori_len).long().view(-1) - 1
                        stop_idx = stop_idx.clamp(min=0, max=int(w_cpu.size(1) - 1))
                    max_b = min(int(w_cpu.size(0)), int(debug_samples))
                    for b in range(max_b):
                        ww = [float(x) for x in w_cpu[b].tolist()]
                        hh = [float(x) for x in h_cpu[b].tolist()]
                        si = int(stop_idx[b].item()) if stop_idx is not None else -1
                        print(f"[Vigor][debug_halt] sample={b} stop_idx={si} halt_weights={ww} halt_logits={hh}", flush=True)
            except Exception:
                pass

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects

        if speed_bench and (dt is not None) and batch_idx >= speed_warmup:
            speed_time_s += float(dt)
            speed_samples += int(batch_size)
            speed_batches += 1

        total_loss_mtr.update(LOSS.item(), batch_size)

        # Optional: LLM distillation diagnostics (available only for llama-step-slot wrapper).
        d_step = getattr(base_model, "last_distill_step", None)
        d_glb = getattr(base_model, "last_distill_global", None)
        d_step_mp = getattr(base_model, "last_distill_step_mp", None)
        d_glb_mp = getattr(base_model, "last_distill_global_mp", None)
        d_mp_op = getattr(base_model, "last_mp_op_distill", None)
        d_mk = getattr(base_model, "last_mk_loss", None)
        d_gate = getattr(base_model, "last_gate_loss", None)
        if d_step is not None:
            try:
                distill_step_mtr.update(float(d_step), batch_size)
            except Exception:
                pass
        if d_glb is not None:
            try:
                distill_global_mtr.update(float(d_glb), batch_size)
            except Exception:
                pass
        if d_step_mp is not None:
            try:
                distill_step_mp_mtr.update(float(d_step_mp), batch_size)
            except Exception:
                pass
        if d_glb_mp is not None:
            try:
                distill_global_mp_mtr.update(float(d_glb_mp), batch_size)
            except Exception:
                pass
        if d_mp_op is not None:
            try:
                mp_op_distill_mtr.update(float(d_mp_op), batch_size)
            except Exception:
                pass
        if d_mk is not None:
            try:
                mk_loss_mtr.update(float(d_mk), batch_size)
            except Exception:
                pass
        if d_gate is not None:
            try:
                gate_loss_mtr.update(float(d_gate), batch_size)
            except Exception:
                pass
        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        # Optional: query-hit proxy (P2-s) for models evaluated under ReferIt3D contexts.
        # This maps both the predicted instance id and the GT instance id to Mask3D query ids
        # via the exported `gt_to_query_map`, and reports whether they match.
        if queryacc_proxy:
            try:
                scan_ids = batch.get("scan_id", None)
                inst_ids = batch.get("instance_ids", None)
                cls_labels_q = batch.get("class_labels", None)
                obj_mask_q = batch.get("obj_mask", None)
                utterances = batch.get("utterance", None)

                if (scan_ids is not None) and torch.is_tensor(inst_ids) and isinstance(scan_ids, (list, tuple)):
                    feat_root = getattr(args, "mask3d_feature_root_test", None) or getattr(args, "mask3d_feature_root", None)
                    feat_root = str(feat_root or "").strip()
                    feat_root_p = Path(feat_root) if feat_root else None
                    if feat_root_p is not None:
                        # Convert cls_labels/obj_mask best-effort.
                        if (cls_labels_q is not None) and (not torch.is_tensor(cls_labels_q)):
                            try:
                                cls_labels_q = torch.as_tensor(cls_labels_q)
                            except Exception:
                                cls_labels_q = None
                        if (obj_mask_q is not None) and (not torch.is_tensor(obj_mask_q)):
                            try:
                                obj_mask_q = torch.as_tensor(obj_mask_q)
                            except Exception:
                                obj_mask_q = None

                        for b in range(int(batch_size)):
                            pred_idx = int(predictions[b].item())
                            tgt_idx = int(target[b].item())
                            scene_id = str(scan_ids[b])

                            gt_map = gt_map_cache.get(scene_id, None)
                            if gt_map is None:
                                try:
                                    d = torch.load(str(feat_root_p / f"{scene_id}.pt"), map_location="cpu")
                                except Exception:
                                    d = None
                                m = None
                                if isinstance(d, dict):
                                    m = d.get("gt_to_query_map", None)
                                gt_map = m if isinstance(m, dict) else {}
                                gt_map_cache[scene_id] = gt_map

                            # Map ScanNet instance id -> query id (allow +1 fallback for legacy indexing).
                            def _map_q(obj_id: int):
                                if obj_id in gt_map:
                                    return gt_map.get(obj_id, None)
                                if (obj_id + 1) in gt_map:
                                    return gt_map.get(obj_id + 1, None)
                                return None

                            obj_tgt = int(inst_ids[b, tgt_idx].item())
                            obj_pred = int(inst_ids[b, pred_idx].item())
                            q_star = _map_q(obj_tgt)
                            q_hat = _map_q(obj_pred)
                            hit = int((q_star is not None) and (q_hat is not None) and (int(q_star) == int(q_hat)))
                            queryacc_mtr.update(hit, 1)

                            # Easy/Hard split: count same-class candidates in valid context.
                            easy = None
                            try:
                                if torch.is_tensor(cls_labels_q) and cls_labels_q.dim() >= 2:
                                    tgt_cls = int(cls_labels_q[b, tgt_idx].item())
                                    if torch.is_tensor(obj_mask_q) and obj_mask_q.dim() >= 2:
                                        valid = (obj_mask_q[b].view(-1) > 0).detach().cpu().numpy().astype(bool)
                                    else:
                                        valid = np.ones((int(cls_labels_q.shape[1]),), dtype=bool)
                                    cls_row = cls_labels_q[b].detach().cpu().numpy()
                                    same = (cls_row == tgt_cls) & valid & (cls_row >= 0)
                                    n_same = int(same.astype(np.int64).sum())
                                    easy = bool(n_same <= 2)
                            except Exception:
                                easy = None
                            if easy is True:
                                queryacc_easy_mtr.update(hit, 1)
                            elif easy is False:
                                queryacc_hard_mtr.update(hit, 1)

                            # View-dependent split (token heuristic; fall back to raw utterance splitting).
                            vdep = None
                            try:
                                utt = None
                                if isinstance(utterances, (list, tuple)):
                                    utt = str(utterances[b])
                                elif isinstance(batch.get("tokens", None), (list, tuple)):
                                    utt = str(batch["tokens"][b])
                                if utt is not None:
                                    toks = set(str(utt).lower().replace(";", " ").replace(",", " ").split())
                                    vdep = len(toks.intersection(vdep_words)) > 0
                            except Exception:
                                vdep = None
                            if vdep is True:
                                queryacc_vdep_mtr.update(hit, 1)
                            elif vdep is False:
                                queryacc_vindep_mtr.update(hit, 1)
            except Exception:
                pass

        # Optional: bbox IoU accuracy (ScanRefer/M3DRef-style Acc@IoU) when boxes are available.
        # This uses the candidate boxes (GT instance boxes in the scene); it measures overlap between
        # the predicted candidate's box and the GT target box.
        try:
            box_corners = batch.get("box_corners", None)
            if box_corners is None:
                box_corners = batch.get("corner_coors", None)  # legacy key
            gt_box_corners = batch.get("gt_box_corners", None)
            obj_mask = batch.get("obj_mask", None)

            # DataLoader collation can yield numpy arrays / lists depending on upstream types.
            # Convert best-effort so bbox metrics don't silently drop.
            if (box_corners is not None) and (not torch.is_tensor(box_corners)):
                try:
                    box_corners = torch.as_tensor(box_corners)
                except Exception:
                    box_corners = None
            if (gt_box_corners is not None) and (not torch.is_tensor(gt_box_corners)):
                try:
                    gt_box_corners = torch.as_tensor(gt_box_corners)
                except Exception:
                    gt_box_corners = None
            if (obj_mask is not None) and (not torch.is_tensor(obj_mask)):
                try:
                    obj_mask = torch.as_tensor(obj_mask)
                except Exception:
                    obj_mask = None

            if torch.is_tensor(box_corners) and box_corners.dim() == 4 and box_corners.size(-2) == 8 and box_corners.size(-1) == 3:
                from ..in_out.cuboid import iou_3d  # local import to avoid slowing down non-box eval paths
                def _iou_val(a_np, g_np) -> float:
                    v = iou_3d(a_np, g_np)
                    # `iou_3d` returns either a float (no-overlap early return)
                    # or a tuple (iou, intersection, vol_a, vol_b).
                    if isinstance(v, tuple):
                        return float(v[0])
                    return float(v)

                # For ScanRefer-style Unique/Multiple splits: count how many same-class instances exist
                # in the (valid) context; if ==1 -> Unique, else -> Multiple.
                cls_labels = batch.get("scannet_class_labels", None)
                if not torch.is_tensor(cls_labels):
                    cls_labels = batch.get("class_labels", None)
                if (cls_labels is not None) and (not torch.is_tensor(cls_labels)):
                    try:
                        cls_labels = torch.as_tensor(cls_labels)
                    except Exception:
                        cls_labels = None

                ious = []
                ious_tgt = []
                correct_flags = []
                subset = []  # "unique" | "multiple" | None
                has_gt = torch.is_tensor(gt_box_corners) and gt_box_corners.shape == box_corners.shape
                for b in range(int(batch_size)):
                    pred_idx = int(predictions[b].item())
                    tgt_idx = int(target[b].item())
                    is_correct = int(pred_idx == tgt_idx)
                    correct_flags.append(is_correct)
                    # Guard: if pred idx is padded/invalid, treat IoU as 0.
                    if torch.is_tensor(obj_mask) and obj_mask.dim() >= 2:
                        try:
                            if float(obj_mask[b, pred_idx].item()) <= 0:
                                ious.append(0.0)
                                if has_gt:
                                    a_t = box_corners[b, tgt_idx].detach().to("cpu").numpy()
                                    g_t = gt_box_corners[b, tgt_idx].detach().to("cpu").numpy()
                                    ious_tgt.append(_iou_val(a_t, g_t))
                                subset.append(None)
                                continue
                        except Exception:
                            pass
                    a = box_corners[b, pred_idx].detach().to("cpu").numpy()
                    if has_gt:
                        g = gt_box_corners[b, tgt_idx].detach().to("cpu").numpy()
                        a_t = box_corners[b, tgt_idx].detach().to("cpu").numpy()
                        g_t = g
                        ious_tgt.append(_iou_val(a_t, g_t))
                    else:
                        g = box_corners[b, tgt_idx].detach().to("cpu").numpy()
                    ious.append(_iou_val(a, g))
                    # Subset tag
                    tag = None
                    try:
                        if torch.is_tensor(cls_labels) and cls_labels.dim() >= 2:
                            tgt_cls = int(cls_labels[b, tgt_idx].item())
                            if tgt_cls >= 0:
                                if torch.is_tensor(obj_mask) and obj_mask.dim() >= 2:
                                    valid = (obj_mask[b] > 0)
                                else:
                                    valid = torch.ones_like(cls_labels[b], dtype=torch.bool)
                                same = (cls_labels[b] == tgt_cls) & valid & (cls_labels[b] >= 0)
                                n_same = int(same.long().sum().item())
                                tag = "unique" if n_same <= 1 else "multiple"
                    except Exception:
                        tag = None
                    subset.append(tag)
                if ious:
                    ious_np = np.asarray(ious, dtype=np.float32)
                    acc25_np = (ious_np >= 0.25).astype(np.float32)
                    acc50_np = (ious_np >= 0.50).astype(np.float32)
                    correct_np = np.asarray(correct_flags, dtype=np.int64)[: ious_np.shape[0]] > 0

                    mean_iou = float(ious_np.mean())
                    acc25 = float(acc25_np.mean())
                    acc50 = float(acc50_np.mean())
                    bbox_iou_mtr.update(mean_iou, batch_size)
                    bbox_iou_25_mtr.update(acc25, batch_size)
                    bbox_iou_50_mtr.update(acc50, batch_size)

                    # Decompose by correctness (best-effort; lengths should match batch_size).
                    try:
                        if correct_np.any():
                            bbox_iou_correct_mtr.update(float(ious_np[correct_np].mean()), int(correct_np.sum()))
                            bbox_iou_25_correct_mtr.update(float(acc25_np[correct_np].mean()), int(correct_np.sum()))
                            bbox_iou_50_correct_mtr.update(float(acc50_np[correct_np].mean()), int(correct_np.sum()))
                        wrong_np = ~correct_np
                        if wrong_np.any():
                            bbox_iou_wrong_mtr.update(float(ious_np[wrong_np].mean()), int(wrong_np.sum()))
                            bbox_iou_25_wrong_mtr.update(float(acc25_np[wrong_np].mean()), int(wrong_np.sum()))
                            bbox_iou_50_wrong_mtr.update(float(acc50_np[wrong_np].mean()), int(wrong_np.sum()))
                    except Exception:
                        pass

                    # Target-only box quality: IoU between target's predicted box (AABB) and target GT box.
                    try:
                        if ious_tgt:
                            t_np = np.asarray(ious_tgt, dtype=np.float32)
                            t25 = (t_np >= 0.25).astype(np.float32)
                            t50 = (t_np >= 0.50).astype(np.float32)
                            bbox_iou_target_mtr.update(float(t_np.mean()), int(t_np.shape[0]))
                            bbox_iou_25_target_mtr.update(float(t25.mean()), int(t_np.shape[0]))
                            bbox_iou_50_target_mtr.update(float(t50.mean()), int(t_np.shape[0]))
                    except Exception:
                        pass

                    # Unique/Multiple breakdown (best-effort; only when we can infer class labels)
                    try:
                        sub = np.asarray([s or "" for s in subset])
                        is_u = sub == "unique"
                        is_m = sub == "multiple"
                        if int(is_u.sum()) > 0:
                            bbox_iou_25_unique_mtr.update(float(acc25_np[is_u].mean()), int(is_u.sum()))
                            bbox_iou_50_unique_mtr.update(float(acc50_np[is_u].mean()), int(is_u.sum()))
                        if int(is_m.sum()) > 0:
                            bbox_iou_25_multiple_mtr.update(float(acc25_np[is_m].mean()), int(is_m.sum()))
                            bbox_iou_50_multiple_mtr.update(float(acc50_np[is_m].mean()), int(is_m.sum()))
                    except Exception:
                        pass
        except Exception:
            pass

        # Optional: strict multi-target F1@IoU for M3DRef-like samples.
        # Requires `multi_target_mask` (float, [B,max_ctx]) where 1 indicates a GT target.
        try:
            mt = batch.get("multi_target_mask", None)
            if torch.is_tensor(mt) and mt.dim() == 2 and torch.is_tensor(box_corners) and torch.is_tensor(gt_box_corners):
                from ..in_out.cuboid import iou_3d
                def _iou_val(a_np, g_np) -> float:
                    v = iou_3d(a_np, g_np)
                    if isinstance(v, tuple):
                        return float(v[0])
                    return float(v)

                # Use oracle cardinality by default: K_pred = |GT| (can override via env).
                pred_k_mode = _env_str("VIGOR_M3DREF_F1_PRED_K", "gt").strip().lower()
                thr = _env_float("VIGOR_M3DREF_F1_SCORE_THR", 0.0)
                score_kind = _env_str("VIGOR_M3DREF_F1_SCORE_KIND", "softmax").strip().lower()
                logits_cpu = res["logits"].detach().to("cpu")
                pred_cpu = predictions.detach().to("cpu")
                obj_mask_cpu = batch.get("obj_mask", None)
                if torch.is_tensor(obj_mask_cpu):
                    obj_mask_cpu = obj_mask_cpu.detach().to("cpu")

                for b in range(int(batch_size)):
                    # GT indices (valid objects only)
                    valid = None
                    if torch.is_tensor(obj_mask_cpu) and obj_mask_cpu.dim() >= 2:
                        valid = (obj_mask_cpu[b, :, 0] > 0) if obj_mask_cpu.dim() == 3 else (obj_mask_cpu[b] > 0)
                    gt_idx = (mt[b] > 0.5).nonzero(as_tuple=False).view(-1).detach().to("cpu")
                    if valid is not None:
                        gt_idx = gt_idx[valid[gt_idx]]
                    gt_list = [int(x.item()) for x in gt_idx]
                    if not gt_list:
                        continue

                    # Predicted set indices
                    if pred_k_mode in {"gt", "oracle"}:
                        k_pred = int(len(gt_list))
                    else:
                        try:
                            k_pred = int(pred_k_mode)
                        except Exception:
                            k_pred = int(len(gt_list))
                    if k_pred <= 0:
                        continue

                    # Mask invalid logits
                    scores = logits_cpu[b].clone()
                    if valid is not None:
                        scores[~valid] = -1e9
                    # Optional threshold mode: if pred_k_mode == "thr", use score threshold.
                    if pred_k_mode in {"thr", "threshold"}:
                        if score_kind in {"softmax", "sm"}:
                            s = torch.softmax(scores, dim=0)
                        elif score_kind in {"sigmoid", "sg"}:
                            s = torch.sigmoid(scores)
                        else:
                            s = scores
                        pred_list = [int(i) for i in (s >= float(thr)).nonzero(as_tuple=False).view(-1).tolist()]
                    else:
                        k_use = min(int(k_pred), int(scores.numel()))
                        topk = torch.topk(scores, k=k_use, dim=0).indices
                        pred_list = [int(x.item()) for x in topk]

                    if not pred_list:
                        continue

                    # Compute IoU matrix pred x gt
                    pairs = []
                    for pi in pred_list:
                        a = box_corners[b, pi].detach().to("cpu").numpy()
                        for gi in gt_list:
                            g = gt_box_corners[b, gi].detach().to("cpu").numpy()
                            pairs.append((_iou_val(a, g), pi, gi))

                    # Greedy max matching by IoU
                    pairs.sort(key=lambda x: x[0], reverse=True)
                    def _tp_at(th: float) -> int:
                        used_p = set()
                        used_g = set()
                        tp = 0
                        for iou, pi, gi in pairs:
                            if iou < th:
                                break
                            if pi in used_p or gi in used_g:
                                continue
                            used_p.add(pi)
                            used_g.add(gi)
                            tp += 1
                        return tp

                    tp25 = _tp_at(0.25)
                    tp50 = _tp_at(0.50)
                    p25 = tp25 / float(len(pred_list))
                    r25 = tp25 / float(len(gt_list))
                    f125 = (2 * p25 * r25 / (p25 + r25)) if (p25 + r25) > 0 else 0.0
                    p50 = tp50 / float(len(pred_list))
                    r50 = tp50 / float(len(gt_list))
                    f150 = (2 * p50 * r50 / (p50 + r50)) if (p50 + r50) > 0 else 0.0

                    f1_iou_25_mtr.update(float(f125), 1)
                    f1_iou_50_mtr.update(float(f150), 1)
        except Exception:
            pass
        if (d_step is not None) or (d_glb is not None) or (d_step_mp is not None) or (d_mp_op is not None) or (d_mk is not None) or (d_gate is not None):
            try:
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss_mtr.avg:.3f}",
                        "acc": f"{ref_acc_mtr.avg:.3f}",
                        "d_step": f"{distill_step_mtr.val:.4f}" if d_step is not None else "-",
                        "d_glb": f"{distill_global_mtr.val:.4f}" if d_glb is not None else "-",
                        "d_mp": f"{distill_step_mp_mtr.val:.4f}" if d_step_mp is not None else "-",
                        "d_op": f"{mp_op_distill_mtr.val:.4f}" if d_mp_op is not None else "-",
                        "mk": f"{mk_loss_mtr.val:.4f}" if d_mk is not None else "-",
                        "gate": f"{gate_loss_mtr.val:.4f}" if d_gate is not None else "-",
                    },
                    refresh=False,
                )
            except Exception:
                pass

        # 原始 607 类 object head 的准确率（仅在未启用 ScanNet200 头时统计）
        if args.obj_cls_alpha > 0 and not getattr(args, "use_scannet200_obj_cls", False):
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        # Optional ScanNet200-based accuracy
        if getattr(args, "use_scannet200_obj_cls", False) and SCANNET_CLASS_LOGITS is not None:
            labels = scannet_labels if scannet_labels is not None else batch.get('scannet_class_labels', None)
            if labels is not None:
                preds = SCANNET_CLASS_LOGITS.argmax(dim=-1)
                valid = labels >= 0
                if valid.any():
                    correct = (preds[valid] == labels[valid]).double().mean().item()
                    scannet_cls_mtr.update(correct, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            target_class = batch['target_class']
            if batch_guess.shape[0] != target_class.shape[0]:
                m = min(int(batch_guess.shape[0]), int(target_class.shape[0]))
                batch_guess = batch_guess[:m]
                target_class = target_class[:m]
            cls_b_acc = torch.mean((batch_guess == target_class).double()).item()
            txt_acc_mtr.update(cls_b_acc, batch_size)

        batch_idx += 1

        if speed_bench and speed_early_stop and speed_max_batches > 0 and speed_batches >= speed_max_batches:
            break

        # Varlen m_k prediction probe: infer chain length from cos(order_embed_k, stop_embed).
        # Note: this probe can run even when varlen gating is disabled, as long as we can
        # derive a GT validity mask (ori_order_len / order_valid_mask / referential_order STOP padding).
        if varlen_mk_eval:
            try:
                emb = None
                if varlen_mk_use_post_stop:
                    emb = getattr(base_model, "last_order_embeds_post_stop", None)
                if emb is None:
                    emb = getattr(base_model, "last_order_embeds", None)
                stop = getattr(base_model, "stop_embed", None)
                if (not torch.is_tensor(emb)) or (not torch.is_tensor(stop)):
                    raise RuntimeError("missing order_embeds/stop_embed (mk probe)")
                if emb.dim() != 3:
                    raise RuntimeError(f"order_embeds has unexpected shape: {tuple(emb.size())}")
                B_mk = int(emb.size(0))
                # GT mask: use batch-provided order_valid_mask (preferred) or derive from ori_order_len.
                gt = batch.get("order_valid_mask", None)
                if (gt is None) and ("ori_order_len" in batch):
                    try:
                        ori_len_t = torch.as_tensor(batch["ori_order_len"], device=emb.device).long().view(-1)
                        steps = torch.arange(int(args.order_len), device=emb.device).view(1, -1)
                        gt = (steps < ori_len_t.view(-1, 1)).to(dtype=torch.float32)
                    except Exception:
                        gt = None
                # Fallback: derive from referential_order padding (common in ViGOR csvs).
                if gt is None:
                    ro = batch.get("referential_order", None)
                    if ro is not None:
                        stop_tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                        rows = []
                        for b in range(int(B_mk)):
                            row = []
                            for k in range(int(args.order_len)):
                                s = str(_safe_get_referential_token(ro, b, k) or "").strip()
                                valid = (s != "") and (s != stop_tok)
                                row.append(1.0 if valid else 0.0)
                            rows.append(row)
                        if rows:
                            gt = torch.as_tensor(rows, device=emb.device, dtype=torch.float32)
                if not torch.is_tensor(gt):
                    raise RuntimeError("missing GT mask (ori_order_len/order_valid_mask)")

                # Cosine similarity per slot: [B,O]
                stop_n = F.normalize(stop.detach().float().view(1, 1, -1), dim=-1)
                emb_n = F.normalize(emb.detach().float(), dim=-1)
                cos = (emb_n * stop_n).sum(dim=-1)

                varlen_mk_cos_all.append(cos.detach().to("cpu"))
                varlen_mk_gt_all.append(gt.detach().to("cpu"))
            except Exception as e:
                # Keep eval robust; mk probe is optional.
                if varlen_mk_last_error is None:
                    varlen_mk_last_error = f"{type(e).__name__}: {e}"
                pass

        # Varlen gate probe: infer chain length from learned STOP probabilities.
        if varlen_gate_eval:
            try:
                emb = getattr(base_model, "last_order_embeds", None)
                if not torch.is_tensor(emb):
                    raise RuntimeError("missing order_embeds (gate probe)")
                if emb.dim() != 3:
                    raise RuntimeError(f"order_embeds has unexpected shape: {tuple(emb.size())}")
                B_gate = int(emb.size(0))

                if hasattr(base_model, "_varlen_gate_logits"):
                    logits = base_model._varlen_gate_logits(emb)  # [B,O]
                elif hasattr(base_model, "varlen_gate"):
                    logits = base_model.varlen_gate(emb).squeeze(-1)
                else:
                    raise RuntimeError("missing gate head (gate probe)")
                prob = torch.sigmoid(logits.detach().float())

                gt = batch.get("order_valid_mask", None)
                if (gt is None) and ("ori_order_len" in batch):
                    try:
                        ori_len_t = torch.as_tensor(batch["ori_order_len"], device=emb.device).long().view(-1)
                        steps = torch.arange(int(args.order_len), device=emb.device).view(1, -1)
                        gt = (steps < ori_len_t.view(-1, 1)).to(dtype=torch.float32)
                    except Exception:
                        gt = None
                if gt is None:
                    ro = batch.get("referential_order", None)
                    if ro is not None:
                        stop_tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                        rows = []
                        for b in range(int(B_gate)):
                            row = []
                            for k in range(int(args.order_len)):
                                s = str(_safe_get_referential_token(ro, b, k) or "").strip()
                                valid = (s != "") and (s != stop_tok)
                                row.append(1.0 if valid else 0.0)
                            rows.append(row)
                        if rows:
                            gt = torch.as_tensor(rows, device=emb.device, dtype=torch.float32)
                if not torch.is_tensor(gt):
                    raise RuntimeError("missing GT mask (ori_order_len/order_valid_mask/referential_order)")

                varlen_gate_prob_all.append(prob.to("cpu"))
                varlen_gate_gt_all.append(gt.detach().to("cpu"))
            except Exception as e:
                if varlen_gate_last_error is None:
                    varlen_gate_last_error = f"{type(e).__name__}: {e}"
                pass

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_scannet_object_cls_acc'] = scannet_cls_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    # Optional: formulation-aligned query-hit proxy (P2). Stored only when enabled.
    if queryacc_proxy and queryacc_mtr.count > 0:
        metrics["test_queryacc"] = queryacc_mtr.avg
        metrics["test_queryacc_n"] = int(queryacc_mtr.count)
        if queryacc_easy_mtr.count > 0:
            metrics["test_queryacc_easy"] = queryacc_easy_mtr.avg
            metrics["test_queryacc_easy_n"] = int(queryacc_easy_mtr.count)
        if queryacc_hard_mtr.count > 0:
            metrics["test_queryacc_hard"] = queryacc_hard_mtr.avg
            metrics["test_queryacc_hard_n"] = int(queryacc_hard_mtr.count)
        if queryacc_vdep_mtr.count > 0:
            metrics["test_queryacc_vdep"] = queryacc_vdep_mtr.avg
            metrics["test_queryacc_vdep_n"] = int(queryacc_vdep_mtr.count)
        if queryacc_vindep_mtr.count > 0:
            metrics["test_queryacc_vindep"] = queryacc_vindep_mtr.avg
            metrics["test_queryacc_vindep_n"] = int(queryacc_vindep_mtr.count)
    if bbox_iou_mtr.count > 0:
        metrics["test_bbox_mean_iou"] = bbox_iou_mtr.avg
        metrics["test_bbox_acc_iou_25"] = bbox_iou_25_mtr.avg
        metrics["test_bbox_acc_iou_50"] = bbox_iou_50_mtr.avg
        metrics["test_bbox_n"] = int(bbox_iou_mtr.count)
        if bbox_iou_25_unique_mtr.count > 0:
            metrics["test_bbox_acc_iou_25_unique"] = bbox_iou_25_unique_mtr.avg
            metrics["test_bbox_acc_iou_50_unique"] = bbox_iou_50_unique_mtr.avg
        if bbox_iou_25_multiple_mtr.count > 0:
            metrics["test_bbox_acc_iou_25_multiple"] = bbox_iou_25_multiple_mtr.avg
            metrics["test_bbox_acc_iou_50_multiple"] = bbox_iou_50_multiple_mtr.avg
    if bbox_iou_correct_mtr.count > 0:
        metrics["test_bbox_mean_iou_correct"] = bbox_iou_correct_mtr.avg
        metrics["test_bbox_acc_iou_25_correct"] = bbox_iou_25_correct_mtr.avg
        metrics["test_bbox_acc_iou_50_correct"] = bbox_iou_50_correct_mtr.avg
        metrics["test_bbox_n_correct"] = int(bbox_iou_correct_mtr.count)
    if bbox_iou_wrong_mtr.count > 0:
        metrics["test_bbox_mean_iou_wrong"] = bbox_iou_wrong_mtr.avg
        metrics["test_bbox_acc_iou_25_wrong"] = bbox_iou_25_wrong_mtr.avg
        metrics["test_bbox_acc_iou_50_wrong"] = bbox_iou_50_wrong_mtr.avg
        metrics["test_bbox_n_wrong"] = int(bbox_iou_wrong_mtr.count)
    if bbox_iou_target_mtr.count > 0:
        metrics["test_bbox_mean_iou_target"] = bbox_iou_target_mtr.avg
        metrics["test_bbox_acc_iou_25_target"] = bbox_iou_25_target_mtr.avg
        metrics["test_bbox_acc_iou_50_target"] = bbox_iou_50_target_mtr.avg
        metrics["test_bbox_n_target"] = int(bbox_iou_target_mtr.count)
    if f1_iou_25_mtr.count > 0:
        metrics["test_f1_iou_25"] = f1_iou_25_mtr.avg
        metrics["test_f1_iou_50"] = f1_iou_50_mtr.avg
    if distill_step_mtr.count > 0:
        metrics['test_llm_distill_step'] = distill_step_mtr.avg
    if distill_global_mtr.count > 0:
        metrics['test_llm_distill_global'] = distill_global_mtr.avg
    if distill_step_mp_mtr.count > 0:
        metrics['test_llm_distill_step_mp'] = distill_step_mp_mtr.avg
    if distill_global_mp_mtr.count > 0:
        metrics['test_llm_distill_global_mp'] = distill_global_mp_mtr.avg
    if mp_op_distill_mtr.count > 0:
        metrics['test_llm_distill_mp_op'] = mp_op_distill_mtr.avg
    if mk_loss_mtr.count > 0:
        metrics['test_stop_mk_loss'] = mk_loss_mtr.avg
    if gate_loss_mtr.count > 0:
        metrics['test_varlen_gate_loss'] = gate_loss_mtr.avg

    if speed_bench and speed_batches > 0 and speed_samples > 0:
        onepass = _env_flag("VIGOR_LLM_STEPSLOT_ONEPASS", "0")
        onepass_mode = _env_str("VIGOR_LLM_ONEPASS_INPUT_MODE", "teacher").strip().lower()
        mp_mode = _env_str("VIGOR_LLM_MULTIPASS_INPUT_MODE", "teacher").strip().lower()
        ms_per_batch = 1000.0 * float(speed_time_s) / float(speed_batches)
        ms_per_sample = 1000.0 * float(speed_time_s) / float(speed_samples)
        print(
            f"[Vigor][speed] onepass={int(onepass)} onepass_mode={onepass_mode} multipass_mode={mp_mode} "
            f"warmup={speed_warmup} measured_batches={speed_batches} measured_samples={speed_samples} "
            f"ms_per_batch={ms_per_batch:.2f} ms_per_sample={ms_per_sample:.2f}",
            flush=True,
        )

    # Finalize varlen m_k probe (optional).
    if varlen_mk_eval and varlen_mk_cos_all and varlen_mk_gt_all:
        try:
            cos = torch.cat(varlen_mk_cos_all, dim=0).numpy()  # [N,O]
            gt = torch.cat(varlen_mk_gt_all, dim=0).numpy()  # [N,O]
            gt_bool = gt > 0.5
            N, O = int(cos.shape[0]), int(cos.shape[1])

            # Summary separation stats.
            valid_cos = cos[gt_bool]
            invalid_cos = cos[~gt_bool]
            v_mean = float(valid_cos.mean()) if valid_cos.size > 0 else float("nan")
            i_mean = float(invalid_cos.mean()) if invalid_cos.size > 0 else float("nan")

            # Sweep thresholds to predict L_hat and m_hat.
            taus = [float(varlen_mk_tau)]
            if varlen_mk_sweep:
                # Cosine is in [-1,1]. Focus on the practical range.
                taus = [round(x, 3) for x in np.linspace(-0.2, 1.0, 61).tolist()]

            best_slot = (-1.0, float("nan"))
            best_len = (-1.0, float("nan"))
            fixed_slot = float("nan")
            fixed_len = float("nan")

            idx = np.arange(O).reshape(1, O)
            gt_len = gt_bool.sum(axis=1)  # [N]

            for tau in taus:
                stop_hit = cos >= float(tau)  # [N,O]
                any_hit = stop_hit.any(axis=1)
                first_stop = np.where(any_hit, stop_hit.argmax(axis=1), O)  # [N], O means "no stop"
                pred = idx < first_stop.reshape(N, 1)  # [N,O] predicted valid mask
                slot_acc = float((pred == gt_bool).mean())
                len_acc = float((first_stop == gt_len).mean())
                if slot_acc > best_slot[0]:
                    best_slot = (slot_acc, float(tau))
                if len_acc > best_len[0]:
                    best_len = (len_acc, float(tau))

            # Fixed threshold evaluation (even when sweeping).
            try:
                tau0 = float(varlen_mk_tau)
                stop_hit = cos >= tau0
                any_hit = stop_hit.any(axis=1)
                first_stop = np.where(any_hit, stop_hit.argmax(axis=1), O)
                pred = idx < first_stop.reshape(N, 1)
                fixed_slot = float((pred == gt_bool).mean())
                fixed_len = float((first_stop == gt_len).mean())
            except Exception:
                fixed_slot = float("nan")
                fixed_len = float("nan")

            metrics["test_varlen_mk_cos_valid_mean"] = v_mean
            metrics["test_varlen_mk_cos_invalid_mean"] = i_mean
            metrics["test_varlen_mk_slot_acc"] = float(best_slot[0])
            metrics["test_varlen_mk_len_acc"] = float(best_len[0])
            metrics["test_varlen_mk_best_tau_slot"] = float(best_slot[1])
            metrics["test_varlen_mk_best_tau_len"] = float(best_len[1])
            metrics["test_varlen_mk_slot_acc_fixed_tau"] = float(fixed_slot) if fixed_slot == fixed_slot else float("nan")
            metrics["test_varlen_mk_len_acc_fixed_tau"] = float(fixed_len) if fixed_len == fixed_len else float("nan")
            metrics["test_varlen_mk_fixed_tau"] = float(varlen_mk_tau)
            metrics["test_varlen_mk_samples"] = int(N)

            print(
                "[Vigor][varlen_mk] "
                f"samples={N} O={O} "
                f"cos(valid_mean={v_mean:.4f}, invalid_mean={i_mean:.4f}) "
                f"fixed(slot={fixed_slot:.4f}, len={fixed_len:.4f})@tau={float(varlen_mk_tau):.3f} "
                f"slot_acc={best_slot[0]:.4f}@tau={best_slot[1]:.3f} "
                f"len_acc={best_len[0]:.4f}@tau={best_len[1]:.3f} "
                f"use_post_stop={int(varlen_mk_use_post_stop)} sweep={int(varlen_mk_sweep)}",
                flush=True,
            )
        except Exception:
            pass
    elif varlen_mk_eval:
        print(
            "[Vigor][varlen_mk][warn] probe enabled but no stats collected. "
            "Need GT mask from batch.ori_order_len / batch.order_valid_mask / referential_order STOP padding.",
            flush=True,
        )
        if varlen_mk_debug and varlen_mk_last_error:
            print(f"[Vigor][varlen_mk][debug] first_error={varlen_mk_last_error}", flush=True)

    # Summarize gate probe.
    if varlen_gate_eval and varlen_gate_prob_all and varlen_gate_gt_all:
        try:
            prob = torch.cat(varlen_gate_prob_all, dim=0).numpy()  # [N,O]
            gt = torch.cat(varlen_gate_gt_all, dim=0).numpy()
            N, O = prob.shape
            gt_bool = gt >= 0.5
            valid_mean = float(prob[gt_bool].mean()) if gt_bool.any() else float("nan")
            invalid_mean = float(prob[~gt_bool].mean()) if (~gt_bool).any() else float("nan")
            idx = np.arange(O).reshape(1, O)
            gt_len = gt_bool.sum(axis=1)

            # MAP decode (no threshold): choose boundary L maximizing log-likelihood
            # under a prefix/suffix factorization.
            eps = 1e-6
            p = np.clip(prob, eps, 1.0 - eps)
            logp = np.log(p)
            log1p = np.log(1.0 - p)
            prefix = np.zeros((N, O + 1), dtype=np.float64)
            prefix[:, 1:] = np.cumsum(log1p, axis=1)
            suffix = np.zeros((N, O + 1), dtype=np.float64)
            suffix[:, :O] = np.cumsum(logp[:, ::-1], axis=1)[:, ::-1]
            scores = prefix + suffix  # [N,O+1]
            L_map = np.argmax(scores, axis=1)  # [N]
            pred_map = idx < L_map.reshape(N, 1)
            map_slot = float((pred_map == gt_bool).mean())
            map_len = float((L_map == gt_len).mean())

            taus = [float(varlen_gate_tau)]
            if varlen_gate_sweep:
                taus = [i / 100.0 for i in range(50, 96, 2)]  # 0.50..0.94

            best_slot = (float("-inf"), float(varlen_gate_tau))
            best_len = (float("-inf"), float(varlen_gate_tau))
            fixed_slot = float("nan")
            fixed_len = float("nan")

            for tau in taus:
                stop_hit = prob >= float(tau)
                any_hit = stop_hit.any(axis=1)
                first_stop = np.where(any_hit, stop_hit.argmax(axis=1), O)
                pred = idx < first_stop.reshape(N, 1)
                slot_acc = float((pred == gt_bool).mean())
                len_acc = float((first_stop == gt_len).mean())
                if slot_acc > best_slot[0]:
                    best_slot = (slot_acc, float(tau))
                if len_acc > best_len[0]:
                    best_len = (len_acc, float(tau))

            # Fixed threshold evaluation.
            tau0 = float(varlen_gate_tau)
            stop_hit = prob >= tau0
            any_hit = stop_hit.any(axis=1)
            first_stop = np.where(any_hit, stop_hit.argmax(axis=1), O)
            pred = idx < first_stop.reshape(N, 1)
            fixed_slot = float((pred == gt_bool).mean())
            fixed_len = float((first_stop == gt_len).mean())

            metrics["test_varlen_gate_prob_valid_mean"] = valid_mean
            metrics["test_varlen_gate_prob_invalid_mean"] = invalid_mean
            metrics["test_varlen_gate_slot_acc"] = float(best_slot[0])
            metrics["test_varlen_gate_len_acc"] = float(best_len[0])
            metrics["test_varlen_gate_best_tau_slot"] = float(best_slot[1])
            metrics["test_varlen_gate_best_tau_len"] = float(best_len[1])
            metrics["test_varlen_gate_slot_acc_fixed_tau"] = float(fixed_slot)
            metrics["test_varlen_gate_len_acc_fixed_tau"] = float(fixed_len)
            metrics["test_varlen_gate_fixed_tau"] = float(varlen_gate_tau)
            metrics["test_varlen_gate_slot_acc_map"] = float(map_slot)
            metrics["test_varlen_gate_len_acc_map"] = float(map_len)
            metrics["test_varlen_gate_samples"] = int(N)

            print(
                "[Vigor][varlen_gate] "
                f"samples={N} O={O} "
                f"prob(valid_mean={valid_mean:.4f}, invalid_mean={invalid_mean:.4f}) "
                f"map(slot={map_slot:.4f}, len={map_len:.4f}) "
                f"fixed(slot={fixed_slot:.4f}, len={fixed_len:.4f})@tau={float(varlen_gate_tau):.3f} "
                f"slot_acc={best_slot[0]:.4f}@tau={best_slot[1]:.3f} "
                f"len_acc={best_len[0]:.4f}@tau={best_len[1]:.3f} "
                f"sweep={int(varlen_gate_sweep)}",
                flush=True,
            )
        except Exception:
            pass
    elif varlen_gate_eval:
        print(
            "[Vigor][varlen_gate][warn] probe enabled but no stats collected. "
            "Need order_embeds + GT mask from batch.ori_order_len / batch.order_valid_mask / referential_order STOP padding.",
            flush=True,
        )
        if varlen_mk_debug and varlen_gate_last_error:
            print(f"[Vigor][varlen_gate][debug] first_error={varlen_gate_last_error}", flush=True)
    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True,tokenizer=None):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    export_varlen = _env_flag("VIGOR_VARLEN_EXPORT_PRED", "0")
    if export_varlen:
        # Export per-sample oracle and predicted varlen masks / lengths for B-tier diagnostics.
        # This must not affect evaluation metrics.
        res["varlen_oracle_mask"] = list()  # list[np.ndarray] of shape [B,O]
        res["varlen_pred_mask"] = list()  # list[np.ndarray] of shape [B,O]
        res["varlen_L_oracle"] = list()  # list[np.ndarray] of shape [B]
        res["varlen_L_pred"] = list()  # list[np.ndarray] of shape [B]
        try:
            res["varlen_mask_source"] = str(__import__("os").environ.get("VIGOR_VARLEN_MASK_SOURCE", "")).strip()
            res["varlen_gate_tau"] = float(_env_float("VIGOR_VARLEN_GATE_TAU", 0.5))
            res["varlen_gate_decode"] = str(__import__("os").environ.get("VIGOR_VARLEN_GATE_DECODE", "")).strip()
            res["varlen_gate_mono"] = int(_env_flag("VIGOR_VARLEN_GATE_MONO", "0"))
        except Exception:
            pass
        if isinstance(model, nn.DataParallel):
            print(
                "[Vigor][varlen_export][warn] VIGOR_VARLEN_EXPORT_PRED=1 but model is DataParallel; "
                "export may be unreliable. Consider running with n_gpus=1 (no DataParallel).",
                flush=True,
            )

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)
        
        # Flatten referential order with robust batch-size.
        order = []
        B = int(batch['target_pos'].size(0))
        for i in range(B):
            for j in range(args.order_len):
                order.append(_safe_get_referential_token(batch.get('referential_order', None), i, j))

        order_tokens = tokenizer(order, return_tensors='pt', padding=True)
        order_tokens = _reshape_order_tokens(order_tokens, batch_size=B, order_len=args.order_len)
        order_tokens = {k: v.to(device) for k, v in order_tokens.items()}
        
        batch['order_tokens'] = order_tokens
        batch['pred_class_mask'] = batch['pred_class_mask'].to(device)
        if args.lang_multilabel:
            batch['anchor_ind'] = batch['anchor_ind'].to(device)
        if args.multilabel_pretraining:
            batch['ordered_multilabel_gt'] = batch['ordered_multilabel_gt'].to(device)
            batch['center_coors'] = batch['center_coors'].to(device)
            batch['corner_coors'] = batch['corner_coors'].to(device)
            batch['obj_mask'] = batch['obj_mask'].to(device).squeeze()

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        lang_tokens = {k: v.to(device) for k, v in lang_tokens.items()}
        batch['lang_tokens'] = lang_tokens

        out_tuple = model(batch)
        if isinstance(out_tuple, (list, tuple)):
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out_tuple[:4]
        else:
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out_tuple
        LOSS = LOSS.mean()
        out = {}
        out['logits'] = LOGITS
        out['class_logits'] = CLASS_LOGITS
        out['lang_logits'] = LANG_LOGITS

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        if export_varlen:
            # Prefer the underlying module for exported attributes.
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            oracle_m = getattr(base_model, "last_order_valid_mask", None)
            pred_m = getattr(base_model, "last_order_valid_mask_pred", None)
            try:
                if torch.is_tensor(oracle_m):
                    om = (oracle_m.detach().float().cpu().numpy() > 0.5).astype(np.int8)
                    if torch.is_tensor(pred_m):
                        pm = (pred_m.detach().float().cpu().numpy() > 0.5).astype(np.int8)
                    else:
                        pm = om
                    res["varlen_oracle_mask"].append(om)
                    res["varlen_pred_mask"].append(pm)
                    res["varlen_L_oracle"].append(om.sum(axis=1).astype(np.int16))
                    res["varlen_L_pred"].append(pm.sum(axis=1).astype(np.int16))
            except Exception:
                pass

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])
    if export_varlen and res.get("varlen_L_oracle"):
        try:
            res["varlen_oracle_mask"] = np.vstack(res["varlen_oracle_mask"])
            res["varlen_pred_mask"] = np.vstack(res["varlen_pred_mask"])
            res["varlen_L_oracle"] = np.hstack(res["varlen_L_oracle"])
            res["varlen_L_pred"] = np.hstack(res["varlen_L_pred"])
        except Exception:
            pass
    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=2020):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                if isinstance(batch[k],list):
                    continue
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass (support updated signature)
        out_tuple = model(batch)
        if isinstance(out_tuple, (list, tuple)):
            _, class_logits, _, logits = out_tuple[:4]
            res = {"logits": logits, "class_logits": class_logits}
        else:
            res = out_tuple

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
