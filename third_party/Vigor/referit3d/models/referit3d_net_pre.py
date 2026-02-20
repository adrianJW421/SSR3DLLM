import torch
from torch import nn
import numpy as np
from .utils import get_siamese_features, get_mlp_head
import math
from pathlib import Path
import os
try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

import yaml

from transformers import BertModel, BertConfig
from referit3d.models import MLP

from .encoder_decoder_layers import RefEcoderLayer

class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index):

        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim
        
        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        # Optional ScanNet200-based object classification instead of Vigor 607-class space.
        self.use_scannet200_obj_cls = getattr(args, "use_scannet200_obj_cls", False)

        # Optional: use pre-extracted Mask3D features (object_queries) instead of PointNet++.
        self.use_mask3d_features = getattr(args, "mask3d_feature_root", None) not in [None, ""]
        if self.use_mask3d_features:
            self.mask3d_feature_root = Path(args.mask3d_feature_root)
            in_dim = getattr(args, "mask3d_feature_dim", None) or self.object_dim
            self.mask3d_proj_in = nn.Linear(in_dim, self.object_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.object_dim,
                nhead=self.decoder_nhead_num,
                dim_feedforward=self.object_dim * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            self.mask3d_adapter = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.object_encoder = None
        else:
            self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                            sa_n_samples=[[32], [32], [None]],
                                            sa_radii=[[0.2], [0.4], [None]],
                                            sa_mlps=[[[3, 64, 64, 128]],
                                                    [[128, 128, 128, 256]],
                                                    [[256, 256, self.object_dim, self.object_dim]]])


        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        # Optional step-slot mode (see referit3d_net.py for details).
        self.step_marker_ids = getattr(args, "vigor_step_token_ids", None)
        self.use_step_markers = str(os.environ.get("VIGOR_STEP_MARKERS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.use_step_slot_only = str(os.environ.get("VIGOR_STEP_SLOT_ONLY", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
        freeze_except_step = str(os.environ.get("VIGOR_FREEZE_BERT_EXCEPT_STEP", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        if self.use_step_markers and freeze_except_step and self.step_marker_ids:
            try:
                for p in self.language_encoder.parameters():
                    p.requires_grad = False
                emb = self.language_encoder.embeddings.word_embeddings
                emb.weight.requires_grad = True
                step_ids = torch.as_tensor(list(self.step_marker_ids), dtype=torch.long, device=emb.weight.device)
                row_mask = torch.zeros((emb.weight.size(0), 1), dtype=emb.weight.dtype, device=emb.weight.device)
                row_mask[step_ids, 0] = 1.0

                def _mask_grad(grad):
                    try:
                        return grad * row_mask
                    except Exception:
                        return grad

                emb.weight.register_hook(_mask_grad)
                print(
                    f"[Vigor][step_tokens] freeze BERT except step embeddings: ids={list(self.step_marker_ids)}",
                    flush=True,
                )
            except Exception:
                pass

        # Classifier heads
        self.language_clf = get_mlp_head(self.inner_dim, self.inner_dim, n_obj_classes, dropout=self.dropout_rate)
        self.object_language_clf = get_mlp_head(self.inner_dim, self.inner_dim, 1, dropout=self.dropout_rate)

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.class_name_tokens = class_name_tokens
        self._class_name_tokens_cached_device = None

        self.lang_multilabel = args.lang_multilabel
        self.multilabel_pretraining = args.multilabel_pretraining
        self.logit_loss = nn.CrossEntropyLoss()
        if not self.lang_multilabel:
            self.lang_logits_loss = nn.CrossEntropyLoss()
        else:
            self.lang_logits_loss = nn.BCEWithLogitsLoss()
        if self.multilabel_pretraining:
            self.ml_feature_constraint_loss = nn.CrossEntropyLoss()
            self.coor_reg_loss = nn.MSELoss()
            self.feat_to_multilabel_clf = get_mlp_head(self.inner_dim, self.inner_dim, 1, dropout=self.dropout_rate)
            self.feat_to_coor_reg = get_mlp_head(self.inner_dim, self.inner_dim, 3, dropout=self.dropout_rate)

        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Optional ScanNet200-based object classification head / loss.
        self.scannet_obj_clf = None
        self.scannet_obj_logits_loss = None
        self.scannet_num_classes = None
        self.scannet_id_to_contig = None
        if self.use_scannet200_obj_cls:
            try:
                repo_root = Path(__file__).resolve().parents[4]  # .../SSR3DLLM
                label_db_path = repo_root.parent / "label_database.yaml"
                if not label_db_path.exists():
                    label_db_path = repo_root / "label_database.yaml"
                if label_db_path.exists():
                    with open(label_db_path, "r") as f:
                        label_db = yaml.safe_load(f)
                    keys = []
                    for k in (label_db or {}).keys():
                        if isinstance(k, int):
                            keys.append(k)
                            continue
                        try:
                            keys.append(int(k))
                        except Exception:
                            continue
                    keys = sorted(set(keys))
                    if keys:
                        # ScanNet200 label ids are sparse (max id can be > 1000). We remap them
                        # to a contiguous [0..199] space for stable training/metrics.
                        self.scannet_id_to_contig = {int(k): i for i, k in enumerate(keys)}
                        self.scannet_num_classes = len(keys)
                        self.scannet_obj_clf = MLP(
                            self.inner_dim,
                            [self.object_dim, self.object_dim, self.scannet_num_classes],
                            dropout_rate=self.dropout_rate,
                        )
                        self.scannet_obj_logits_loss = nn.CrossEntropyLoss(ignore_index=-1)
            except Exception:
                self.use_scannet200_obj_cls = False

        self.order_len = args.order_len

        self.refer_encoder = nn.ModuleList()
        for _ in range(self.order_len):
            self.refer_encoder.append(RefEcoderLayer(
                self.inner_dim, n_heads=self.decoder_nhead_num, dim_feedforward=2048,
                dropout=self.dropout_rate, activation="gelu"
            ))
            
        self.disable_coor_loss = args.disable_coor_loss
        self.disable_text_loss = args.disable_text_loss
        self.disable_multilabel_loss = args.disable_multilabel_loss

    def _encode_with_mask3d(self, batch: dict):
        paths = batch.get("mask3d_feature_path", None)
        instance_ids = batch.get("instance_ids", None)
        if paths is None or instance_ids is None:
            raise ValueError("mask3d_feature_root is set but batch missing mask3d_feature_path/instance_ids")
        if isinstance(paths, str):
            paths = [paths]
        inst_tensor = torch.as_tensor(instance_ids)
        if inst_tensor.dim() == 1:
            inst_tensor = inst_tensor.unsqueeze(0)
        B, max_context = inst_tensor.shape
        obj_feats = []
        scannet_labels = (
            torch.full((B, max_context), -1, dtype=torch.long, device=self.device)
            if self.use_scannet200_obj_cls and self.scannet_num_classes is not None
            else None
        )
        for b in range(B):
            fpath = paths[b]
            try:
                data = torch.load(fpath, map_location=self.device)
                queries = data.get("object_queries", None)
            except Exception:
                queries = None
                data = {}
            if queries is None:
                obj_feats.append(torch.zeros(max_context, self.object_dim, device=self.device))
                continue
            if isinstance(queries, np.ndarray):
                queries = torch.from_numpy(queries)
            queries = queries.to(self.device)
            # Project to object_dim and apply lightweight self-attention adapter.
            feats_b = torch.zeros(max_context, queries.shape[1], device=self.device)
            mapping = data.get("gt_to_query_map", {}) or {}
            inst_cls = data.get("gt_instance_classes", {}) or {}
            for j in range(max_context):
                inst_id = int(inst_tensor[b, j].item())
                if inst_id < 0:
                    continue
                q_idx = mapping.get(inst_id, None)
                if q_idx is None:
                    continue
                if 0 <= int(q_idx) < int(queries.shape[0]):
                    feats_b[j] = queries[int(q_idx)]
                    if scannet_labels is not None:
                        cls_id = inst_cls.get(inst_id, None)
                        if cls_id is not None:
                            try:
                                cls_id_int = int(cls_id)
                            except Exception:
                                cls_id_int = None
                            if cls_id_int is not None and self.scannet_id_to_contig is not None:
                                mapped = self.scannet_id_to_contig.get(cls_id_int, -1)
                                if 0 <= int(mapped) < int(self.scannet_num_classes):
                                    scannet_labels[b, j] = int(mapped)
            feats_b = self.mask3d_proj_in(feats_b)
            obj_feats.append(feats_b)
        obj_feats = torch.stack(obj_feats, dim=0)  # [B, max_context, object_dim]
        obj_feats = self.mask3d_adapter(obj_feats)
        if scannet_labels is not None:
            batch["scannet_class_labels"] = scannet_labels
        return obj_feats, scannet_labels

    @torch.no_grad()
    def aug_input(self, input_points, box_infos, rel_coors):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N,P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device)
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).to(self.device)
        
        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0,self.rotate_number,(B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]).to(self.device)[None].repeat(B,1,1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B,N*P,3), rotate_matrix).reshape(B,N,P,3)
            bxyz = torch.matmul(bxyz.reshape(B,N,3), rotate_matrix).reshape(B,N,3)
            rel_coors = torch.matmul(rel_coors.reshape(B,self.order_len*N,3), rotate_matrix.double()).reshape(B,self.order_len,N,3)
 
        # multi-view
        bsize = box_infos[:,:,-1:]
        boxs=[]
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize],dim=-1))
        boxs=torch.stack(boxs,dim=1)

        return input_points, boxs, rel_coors

    def compute_basic_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = 0.0
        # Original Vigor object classification loss (607 classes).
        if not self.use_scannet200_obj_cls and CLASS_LOGITS is not None:
            obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        # Optional ScanNet200-based object classification loss.
        if self.use_scannet200_obj_cls and SCANNET_CLASS_LOGITS is not None and "scannet_class_labels" in batch:
            labels = batch["scannet_class_labels"]
            valid = labels >= 0
            if valid.any():
                logits_flat = SCANNET_CLASS_LOGITS[valid]
                labels_flat = labels[valid]
                obj_clf_loss = self.scannet_obj_logits_loss(logits_flat, labels_flat)
            else:
                obj_clf_loss = torch.tensor(0.0, device=LOGITS.device)

        lang_clf_loss = 0
        if not self.disable_text_loss:
            if not self.lang_multilabel:
                lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
            else:
                lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['anchor_ind'])

        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss

    def forward(self, batch: dict, epoch=None):
        TOTAL_LOSS = 0
        # Robust device inference: DataParallel replicas keep parameters on their local device.
        # Avoid next(self.parameters()) which can raise StopIteration in edge cases.
        self.device = self.obj_feature_mapping[0].weight.device

        ## rotation augmentation and multi_view generation
        obj_points, boxs, batch['rel_coors'] = self.aug_input(batch['objects'], batch['box_info'], batch['rel_coors'])

        B,N,P = obj_points.shape[:3] # torch.Size([24, 52, 1024, 6])
        
        ## obj_encoding
        scannet_labels = None
        if self.use_mask3d_features:
            objects_features, scannet_labels = self._encode_with_mask3d(batch)  # [B, N, object_dim]
        else:
            objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack) # torch.Size([B, N, object_dim])
        obj_feats = self.obj_feature_mapping(objects_features) # torch.Size([B, N, inner_dim])
        box_infos = self.box_feature_mapping(boxs.float())
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1).squeeze() + box_infos # torch.Size([24, 4, 52, 768])
        if len(obj_infos.shape) == 3:
            assert self.view_number == 1
            obj_infos = obj_infos.unsqueeze(1).repeat(1, self.view_number, 1, 1)
            
        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]

        # <LOSS>: lang_cls
        lang_features = lang_infos[:,0]
        LANG_LOGITS = self.language_clf(lang_infos[:,0])
        mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)

        # start feature encoding
        # NOTE: To support DataParallel, `order_tokens` can be either:
        #   - 2D: [B*order_len, L] (legacy), or
        #   - 3D: [B, order_len, L] (DP-safe).
        order_tokens = {k: v.to(self.device) for k, v in batch["order_tokens"].items()}
        order_ids = order_tokens["input_ids"]
        if order_ids.dim() == 3:
            B_ot, O, L = order_ids.shape
            order_tokens_flat = {k: v.reshape(B_ot * O, L) for k, v in order_tokens.items()}
            mentioned_obj_lang_infos = self.language_encoder(**order_tokens_flat)[0]
            mentioned_obj_lang_infos = mentioned_obj_lang_infos.reshape(B_ot, O, L, -1)
        else:
            mentioned_obj_lang_infos = self.language_encoder(**order_tokens)[0]
            mentioned_obj_lang_infos = mentioned_obj_lang_infos.reshape(B, self.order_len, order_ids.size(1), -1)

        if self.use_step_markers and self.use_step_slot_only:
            try:
                if mentioned_obj_lang_infos.size(2) >= 2:
                    mentioned_obj_lang_infos = mentioned_obj_lang_infos[:, :, 1:2, :]
            except Exception:
                pass

        cat_infos = obj_infos.reshape(B*self.view_number, -1, self.inner_dim) # torch.Size([96, 52, 768])

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            # In DataParallel, each replica runs on its own device. Keep a per-replica
            # cached copy of class_name_tokens on the correct device to avoid
            # cuda:0 vs cuda:1 mismatches.
            if (self._class_name_tokens_cached_device is None) or (self._class_name_tokens_cached_device != self.device):
                if isinstance(self.class_name_tokens, dict):
                    self.class_name_tokens = {k: v.to(self.device) for k, v in self.class_name_tokens.items()}
                else:
                    # transformers BatchEncoding behaves like a mapping
                    self.class_name_tokens = {k: v.to(self.device) for k, v in self.class_name_tokens.items()}
                self._class_name_tokens_cached_device = self.device

            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:,0] # torch.Size([n_classes, 768])
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B*N,-1), label_lang_infos.permute(1,0)).reshape(B,N,-1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B*N,-1)).reshape(B,N,-1) # torch.Size([24, 52, 525])        

        SCANNET_CLASS_LOGITS = None
        if self.use_scannet200_obj_cls and self.scannet_obj_clf is not None:
            SCANNET_CLASS_LOGITS = self.scannet_obj_clf(obj_feats.reshape(B*N,-1)).reshape(B, N, -1)

        for i in range(self.order_len):            
            mask = batch['pred_class_mask'][:, i, :].unsqueeze(1).unsqueeze(3).repeat(1, self.view_number, 1, self.inner_dim)
            mask = mask.reshape(B*self.view_number, -1, self.inner_dim)

            masked_obj_infos = cat_infos * mask
            mentioned_features = mentioned_obj_lang_infos[:, i, :, :].unsqueeze(1).repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)

            cat_infos = self.refer_encoder[i](
                cat_infos.transpose(0, 1),
                masked_obj_infos.transpose(0, 1),
                mem_infos.transpose(0, 1),
                mentioned_features.transpose(0, 1),
            ) # torch.Size([96, 52, 768])

            if self.multilabel_pretraining:
                if not self.disable_multilabel_loss:
                    TB_MULTILABEL_LOGITS = self.feat_to_multilabel_clf(cat_infos).squeeze() # torch.Size([96, 52])
                    clf_ans = batch['ordered_multilabel_gt'][:, i, :].unsqueeze(1).repeat(1, self.view_number, 1).reshape(B*self.view_number, -1)
                    TB_MULTILABEL_LOSS = self.ml_feature_constraint_loss(TB_MULTILABEL_LOGITS, clf_ans.float())
                    TOTAL_LOSS += TB_MULTILABEL_LOSS * 0.5
                if not self.disable_coor_loss:
                    COOR_REG = self.feat_to_coor_reg(cat_infos).squeeze()
                    reg_ans = batch['rel_coors'][:, i, :, :].unsqueeze(1).repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, batch['rel_coors'].shape[-1]).squeeze()
                    COOR_REG_LOSS = self.coor_reg_loss(COOR_REG, reg_ans.float())
                    TOTAL_LOSS += COOR_REG_LOSS * 0.5

        ## multi-modal_fusion
        out_feats = cat_infos.reshape(B, self.view_number, -1, self.inner_dim)

        ## view_aggregation
        if self.aggregate_type=='avg':
            agg_feats = (out_feats / self.view_number).sum(dim=1) # torch.Size([24, 52, 768])
        elif self.aggregate_type=='avgmax':
            agg_feats = (out_feats / self.view_number).sum(dim=1) + out_feats.max(dim=1).values
        else:
            agg_feats = out_feats.max(dim=1).values

        # <LOSS>: ref_cls
        LOGITS = self.object_language_clf(agg_feats).squeeze(-1)
        BASIC_LOSS = self.compute_basic_loss(
            batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS=SCANNET_CLASS_LOGITS
        )
        TOTAL_LOSS += BASIC_LOSS
        
        # Return scannet_labels explicitly so DataParallel callers can compute metrics/loss
        # without relying on in-place mutation of `batch`.
        return TOTAL_LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS, scannet_labels
