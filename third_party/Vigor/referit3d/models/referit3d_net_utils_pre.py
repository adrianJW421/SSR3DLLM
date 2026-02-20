"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import tqdm
import torch.nn.functional as F

from ..utils.evaluation import AverageMeter


def _env_flag(name: str, default: str = "0") -> bool:
    v = str(__import__("os").environ.get(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _build_step_marker_text(step_idx: int, utterance: str, step_text: str, order_len: int) -> str:
    k = int(step_idx) + 1
    if k < 1:
        k = 1
    if k > int(order_len):
        k = int(order_len)
    u = str(utterance or "").strip()
    s = str(step_text or "").strip()
    return f"<step{k}> {u} {s}".strip()


def _reshape_order_tokens(order_tokens: dict, batch_size: int, order_len: int) -> dict:
    """
    Make `order_tokens` DataParallel-friendly.

    Tokenizers produce tensors shaped [B*order_len, L]. Under DataParallel, other
    batch tensors are scattered along B, so scattering [B*order_len, L] can
    desync per-replica B and break downstream reshapes. Reshape to [B, order_len, L].
    """
    reshaped = {}
    for k, v in order_tokens.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.size(0) == batch_size * order_len:
            reshaped[k] = v.reshape(batch_size, order_len, v.size(1))
        else:
            reshaped[k] = v
    return reshaped


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

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
    txt_acc_mtr = AverageMeter()
    scannet_valid_frac_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # Tokenize language into plain dict tensors for DataParallel safety.
        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        lang_tokens = {k: v.to(device) for k, v in lang_tokens.items()}

        # Flatten referential-order texts with robust batch-size.
        order = []
        B = int(batch['target_pos'].size(0))
        for i in range(B):
            for j in range(args.order_len):
                try:
                    tok = batch['referential_order'][j][i]
                except Exception:
                    tok = ""
                if _env_flag("VIGOR_STEP_MARKERS", "0"):
                    utt = batch["tokens"][i] if isinstance(batch.get("tokens", None), list) and i < len(batch["tokens"]) else ""
                    tok = _build_step_marker_text(j, utt, tok, order_len=args.order_len)
                order.append(tok)

        order_tokens = tokenizer(order, return_tensors='pt', padding=True)
        order_tokens = _reshape_order_tokens(order_tokens, batch_size=B, order_len=args.order_len)
        order_tokens = {k: v.to(device) for k, v in order_tokens.items()}

        batch['order_tokens'] = order_tokens
        batch['pred_class_mask'] = batch['pred_class_mask'].to(device)
        batch['order_labels'] = batch['order_labels'].to(device)
        if args.lang_multilabel:
            batch['anchor_ind'] = batch['anchor_ind'].to(device)
        if args.multilabel_pretraining:
            batch['ordered_multilabel_gt'] = batch['ordered_multilabel_gt'].to(device)
            batch['rel_coors'] = batch['rel_coors'].to(device)
            batch['center_coors'] = batch['center_coors'].to(device)
            batch['corner_coors'] = batch['corner_coors'].to(device)
            batch['obj_mask'] = batch['obj_mask'].to(device).squeeze()

        batch['lang_tokens'] = lang_tokens

        # Forward pass (support updated signature)
        out = model(batch, epoch)
        SCANNET_CLASS_LOGITS = None
        scannet_labels = None
        if isinstance(out, (list, tuple)):
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out[:4]
            if len(out) >= 5:
                SCANNET_CLASS_LOGITS = out[4]
            if len(out) >= 6:
                scannet_labels = out[5]
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

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            if getattr(args, "use_scannet200_obj_cls", False) and SCANNET_CLASS_LOGITS is not None:
                labels = scannet_labels if scannet_labels is not None else batch.get("scannet_class_labels", None)
                if labels is None:
                    labels = torch.full_like(batch["class_labels"], -1)
                valid = labels >= 0
                # Track how many context slots actually have a valid ScanNet200 label.
                # AverageMeter is used with n=total_slots so avg == total_valid/total_slots.
                total_slots = int(valid.numel())
                if total_slots > 0:
                    valid_frac = float(valid.sum().item()) / float(total_slots)
                    scannet_valid_frac_mtr.update(valid_frac, total_slots)
                if valid.any():
                    preds = torch.argmax(SCANNET_CLASS_LOGITS, dim=-1)
                    cls_b_acc = torch.mean((preds[valid] == labels[valid]).double()).item()
                else:
                    cls_b_acc = 0.0
                cls_acc_mtr.update(cls_b_acc, batch_size)
            else:
                cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
                cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['train_scannet_valid_frac'] = scannet_valid_frac_mtr.avg
    return metrics


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False, tokenizer=None):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    scannet_valid_frac_mtr = AverageMeter()

    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        lang_tokens = {k: v.to(device) for k, v in lang_tokens.items()}
        batch['lang_tokens'] = lang_tokens

        order = []
        B = int(batch['target_pos'].size(0))
        for i in range(B):
            for j in range(args.order_len):
                try:
                    tok = batch['referential_order'][j][i]
                except Exception:
                    tok = ""
                if _env_flag("VIGOR_STEP_MARKERS", "0"):
                    utt = batch["tokens"][i] if isinstance(batch.get("tokens", None), list) and i < len(batch["tokens"]) else ""
                    tok = _build_step_marker_text(j, utt, tok, order_len=args.order_len)
                order.append(tok)

        order_tokens = tokenizer(order, return_tensors='pt', padding=True)
        order_tokens = _reshape_order_tokens(order_tokens, batch_size=B, order_len=args.order_len)
        order_tokens = {k: v.to(device) for k, v in order_tokens.items()}
        
        batch['order_tokens'] = order_tokens
        batch['pred_class_mask'] = batch['pred_class_mask'].to(device)
        batch['order_labels'] = batch['order_labels'].to(device)
        if args.lang_multilabel:
            batch['anchor_ind'] = batch['anchor_ind'].to(device)
        if args.multilabel_pretraining:
            batch['ordered_multilabel_gt'] = batch['ordered_multilabel_gt'].to(device)
            batch['rel_coors'] = batch['rel_coors'].to(device)
            batch['center_coors'] = batch['center_coors'].to(device)
            batch['corner_coors'] = batch['corner_coors'].to(device)
            batch['obj_mask'] = batch['obj_mask'].to(device).squeeze()

        # Forward pass (support updated signature)
        out = model(batch)
        SCANNET_CLASS_LOGITS = None
        scannet_labels = None
        if isinstance(out, (list, tuple)):
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out[:4]
            if len(out) >= 5:
                SCANNET_CLASS_LOGITS = out[4]
            if len(out) >= 6:
                scannet_labels = out[5]
        else:
            LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = out
        LOSS = LOSS.mean()
        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            if getattr(args, "use_scannet200_obj_cls", False) and SCANNET_CLASS_LOGITS is not None:
                labels = scannet_labels if scannet_labels is not None else batch.get("scannet_class_labels", None)
                if labels is None:
                    labels = torch.full_like(batch["class_labels"], -1)
                valid = labels >= 0
                total_slots = int(valid.numel())
                if total_slots > 0:
                    valid_frac = float(valid.sum().item()) / float(total_slots)
                    scannet_valid_frac_mtr.update(valid_frac, total_slots)
                if valid.any():
                    preds = torch.argmax(SCANNET_CLASS_LOGITS, dim=-1)
                    cls_b_acc = torch.mean((preds[valid] == labels[valid]).double()).item()
                else:
                    cls_b_acc = 0.0
                cls_acc_mtr.update(cls_b_acc, batch_size)
            else:
                cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
                cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    metrics['test_scannet_valid_frac'] = scannet_valid_frac_mtr.avg
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

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

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
