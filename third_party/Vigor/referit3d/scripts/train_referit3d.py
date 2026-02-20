#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path

# Ensure the local Vigor referit3d package is importable when running this
# script directly, without relying on global installation or CWD.
_vigor_root = Path(__file__).resolve().parents[2]  # .../Vigor
sys.path.insert(0, str(_vigor_root))               # contains the 'referit3d' package

import torch
import torch.multiprocessing as mp
import tqdm
import time
import warnings
import os
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.models.referit3d_net import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_add_step_tokens(tokenizer, order_len: int):
    """
    Add per-step special tokens for the "step-slot" experiment.
    These tokens are used as fixed anchor positions whose embeddings can be trained
    while freezing the rest of BERT (STAMP-style).
    """
    tokens = [f"<step{i+1}>" for i in range(int(order_len))]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in tokens]
    return tokens, ids, int(added)

def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(
                phase,
                meters[phase + '_total_loss'],
                meters[phase + '_referential_acc'],
            )

            # 原始 607 类 object head 的准确率（仅在未启用 ScanNet200 头时打印）
            if args.obj_cls_alpha > 0 and not getattr(args, "use_scannet200_obj_cls", False):
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            # ScanNet200 头的准确率（仅在启用 use_scannet200-obj-cls 时打印）
            if getattr(args, "use_scannet200_obj_cls", False):
                key = phase + '_scannet_object_cls_acc'
                if key in meters:
                    info += ', ScanNet200-Obj-Acc: {:.4f}'.format(meters[key])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


if __name__ == '__main__':
    
    # Avoid "Too many open files" in DataLoader workers on some systems.
    # Use file_system sharing instead of file_descriptor.
    mp.set_sharing_strategy("file_system")
    
    # Parse arguments
    args = parse_arguments()
    print(
        "[Vigor][args] "
        f"mode={args.mode} "
        f"mask3d_feature_root={getattr(args, 'mask3d_feature_root', None)} "
        f"mask3d_feature_root_test={getattr(args, 'mask3d_feature_root_test', None)} "
        f"mask3d_feature_dim={getattr(args, 'mask3d_feature_dim', None)} "
        f"use_scannet200_obj_cls={getattr(args, 'use_scannet200_obj_cls', False)} "
        f"cascading={getattr(args, 'cascading', False)} "
        f"order_len={getattr(args, 'order_len', None)} "
        f"lang_multilabel={getattr(args, 'lang_multilabel', False)} "
        f"multilabel_pretraining={getattr(args, 'multilabel_pretraining', False)} "
        f"label_lang_sup={getattr(args, 'label_lang_sup', False)} "
        f"VIGOR_USE_PRED_BOX_INFO={os.environ.get('VIGOR_USE_PRED_BOX_INFO', '0')}"
    )
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    # Prepare GPU environment
    # Respect the GPU list provided; this sets CUDA_VISIBLE_DEVICES.
    set_gpu_to_zero_position(args.gpu)
    seed_training_code(args.random_seed)

    # After masking, see how many GPUs are actually visible.
    available_gpus = torch.cuda.device_count()
    print(f"[GPU INFO] visible GPUs: {available_gpus}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    device = torch.device('cuda')

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    if _env_flag("VIGOR_STEP_MARKERS", "0"):
        step_tokens, step_token_ids, added = _maybe_add_step_tokens(tokenizer, getattr(args, "order_len", 4))
        # Stash ids on args so the model can selectively unfreeze them.
        args.vigor_step_tokens = step_tokens
        args.vigor_step_token_ids = step_token_ids
        print(f"[Vigor][step_tokens] enabled tokens={step_tokens} added={added}")
    class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = available_gpus if available_gpus > 0 else len(args.gpu.strip(',').split(','))

    model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    # If we added special tokens to the tokenizer, resize BERT embeddings accordingly.
    try:
        if _env_flag("VIGOR_STEP_MARKERS", "0") and hasattr(model, "language_encoder"):
            model.language_encoder.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    if gpu_num > 1:
        device_ids = list(range(gpu_num))
        print(f"[GPU INFO] Using DataParallel on device_ids={device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    # <1>
    base = model.module if gpu_num > 1 else model
    param_list=[
        {'params':base.language_encoder.parameters(),'lr':args.init_lr*0.1},
        {'params':base.obj_feature_mapping.parameters(), 'lr': args.init_lr},
        {'params':base.box_feature_mapping.parameters(), 'lr': args.init_lr},
        {'params':base.language_clf.parameters(), 'lr': args.init_lr},
        {'params':base.object_language_clf.parameters(), 'lr': args.init_lr},
        {'params':base.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
    ]
    # encoder branch
    if base.use_mask3d_features:
        param_list.append({'params':base.mask3d_proj_in.parameters(), 'lr': args.init_lr})
        param_list.append({'params':base.mask3d_adapter.parameters(), 'lr': args.init_lr*0.5})
    else:
        param_list.append({'params':base.object_encoder.parameters(), 'lr':args.init_lr})

    if args.multilabel_pretraining:
        param_list.append({'params':base.feat_to_multilabel_clf.parameters(), 'lr': args.init_lr})
    if not args.label_lang_sup:
        param_list.append({'params':base.obj_clf.parameters(), 'lr': args.init_lr})
    if args.lang_multilabel and hasattr(base, "anchor_clf"):
        param_list.append({'params':base.anchor_clf.parameters(), 'lr': args.init_lr})

    optimizer = optim.Adam(param_list,lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[40, 50, 60, 70, 80, 90], gamma=0.65)
    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    # Training.
    if args.mode == 'train':
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr())
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args, tokenizer=tokenizer,epoch=epoch)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']

                last_test_acc = eval_acc
                last_test_epoch = epoch

                lr_scheduler.step()

                save_state_dicts(osp.join(args.checkpoint_dir, 'last_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                else:
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)

                bar.refresh()

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            f_out.write(('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch)))
            f_out.write(('Last accuracy: {:.4f} (@epoch {})'.format(last_test_acc, last_test_epoch)))

        logger.info('Finished training successfully.')

    elif args.mode == 'evaluate':

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
        print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        if getattr(args, "use_scannet200_obj_cls", False) and 'test_scannet_object_cls_acc' in meters:
            print('ScanNet200-Obj-Accuracy: {:.4f}'.format(meters['test_scannet_object_cls_acc']))
        elif 'test_object_cls_acc' in meters:
            print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        print('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))
        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  args, out_file=out_file,tokenizer=tokenizer)
        print(res)
