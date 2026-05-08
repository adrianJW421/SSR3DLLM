import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
from .utils import get_siamese_features, get_mlp_head
import math
try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import BertModel, BertConfig
from referit3d.models import MLP
import yaml

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

        self.use_mask3d_features = getattr(args, "mask3d_feature_root", None) not in [None, ""]
        if self.use_mask3d_features:
            self.mask3d_feature_root = Path(args.mask3d_feature_root)
            in_dim = getattr(args, "mask3d_feature_dim", self.object_dim)
            self.mask3d_proj_in = nn.Linear(in_dim, self.object_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.object_dim,
                nhead=self.decoder_nhead_num,
                dim_feedforward=self.object_dim * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            # 两层轻量自注意力，用于在上下文对象间调制 Mask3D 特征。
            self.mask3d_adapter = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.object_encoder = None  # 不使用 PointNet++
        else:
            self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                            sa_n_samples=[[32], [32], [None]],
                                            sa_radii=[[0.2], [0.4], [None]],
                                            sa_mlps=[[[3, 64, 64, 128]],
                                                    [[128, 128, 128, 256]],
                                                    [[256, 256, self.object_dim, self.object_dim]]])

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        # Optional: "step-slot" mode. We freeze BERT and only train the embedding rows
        # corresponding to <stepK> tokens. This mimics STAMP-style training where
        # grounding loss only updates a few dedicated slots.
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
        if args.lang_multilabel:
            self.anchor_clf = get_mlp_head(self.inner_dim, self.inner_dim, 485, dropout=self.dropout_rate)
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

        self.use_mask3d_dino_features = str(
            os.environ.get(
                "VIGOR_MASK3D_DINO_ENABLE",
                "1" if str(os.environ.get("VIGOR_MASK3D_DINO_SAMPLE_CACHE_ROOT", "")).strip() else "0",
            )
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.mask3d_dino_alpha = float(str(os.environ.get("VIGOR_MASK3D_DINO_ALPHA", "1.0")).strip())
        self.mask3d_dino_feature_dim = int(str(os.environ.get("VIGOR_MASK3D_DINO_FEATURE_DIM", "1024")).strip())
        if self.use_mask3d_dino_features:
            self.mask3d_dino_feature_mapping = nn.Sequential(
                nn.Linear(self.mask3d_dino_feature_dim, self.inner_dim),
                nn.LayerNorm(self.inner_dim),
            )
        else:
            self.mask3d_dino_feature_mapping = None

        self.class_name_tokens = class_name_tokens

        self.lang_multilabel = args.lang_multilabel
        self.multilabel_pretraining = args.multilabel_pretraining
        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        if self.lang_multilabel:
            self.anchor_logits_loss = nn.BCEWithLogitsLoss()
        if self.multilabel_pretraining:
            self.ml_feature_constraint_loss = nn.CrossEntropyLoss()
            self.feat_to_multilabel_clf = get_mlp_head(self.inner_dim, self.inner_dim, 1, dropout=self.dropout_rate)

        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Optional ScanNet200-based object classification head / loss.
        # Also optionally use ScanNet200 label space for the text classification head.
        self.scannet_obj_clf = None
        self.scannet_obj_logits_loss = None
        self.scannet_num_classes = None
        self.scannet_id_to_contig = None
        self.use_scannet200_text_cls = (
            bool(self.use_scannet200_obj_cls)
            and str(os.environ.get("VIGOR_TEXT_CLS_SCANNET200", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        )
        if self.use_scannet200_obj_cls:
            try:
                repo_root = Path(__file__).resolve().parents[4]  # .../SSR3DLLM
                label_db_path = repo_root.parent / "label_database.yaml"
                if not label_db_path.exists():
                    # Also try inside repo_root directly
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
                        # ScanNet200 label ids are sparse (max id can be > 1000). Remap to
                        # a contiguous [0..199] space for stable training/metrics.
                        self.scannet_id_to_contig = {int(k): i for i, k in enumerate(keys)}
                        self.scannet_num_classes = len(keys)
                        self.scannet_obj_clf = MLP(
                            self.inner_dim,
                            [self.object_dim, self.object_dim, self.scannet_num_classes],
                            dropout_rate=self.dropout_rate,
                        )
                        # Use -1 as ignore_index for missing labels.
                        self.scannet_obj_logits_loss = nn.CrossEntropyLoss(ignore_index=-1)
            except Exception:
                # If anything fails, fall back to original behaviour.
                self.use_scannet200_obj_cls = False
                self.use_scannet200_text_cls = False

        # If requested, switch the language/text classification head to ScanNet200 label space.
        # This makes the text head consistent with Mask3D/ScanNet200 taxonomy and avoids
        # dataset-dependent class counts (e.g. 524/607) for `language_clf`.
        if self.use_scannet200_text_cls and (self.scannet_num_classes is not None):
            try:
                self.language_clf = get_mlp_head(
                    self.inner_dim, self.inner_dim, int(self.scannet_num_classes), dropout=self.dropout_rate
                )
                # Use -1 ignore_index because not all GT instance labels may map cleanly to ScanNet200.
                self.lang_logits_loss = nn.CrossEntropyLoss(ignore_index=-1)
                print(
                    f"[Vigor] using ScanNet200 text-clf head: n={int(self.scannet_num_classes)}",
                    flush=True,
                )
            except Exception:
                self.use_scannet200_text_cls = False

        self.order_len = args.order_len

        self.refer_encoder = nn.ModuleList()
        for _ in range(self.order_len):
            self.refer_encoder.append(RefEcoderLayer(
                self.inner_dim, n_heads=self.decoder_nhead_num, dim_feedforward=2048,
                dropout=self.dropout_rate, activation="gelu"
            ))

        # Optional: adaptive halting over the fixed `order_len` steps.
        # Enabled via env var `VIGOR_ADAPTIVE_HALT=1`. We keep this head always present so it
        # can be trained, but allow loading older checkpoints with `VIGOR_STRICT_LOAD=0`.
        self.halt_head = nn.Linear(self.inner_dim, 1)

        self.disable_text_loss = args.disable_text_loss
        self.disable_multilabel_loss = args.disable_multilabel_loss

    def _encode_with_mask3d(self, batch: dict):
        """
        Use pre-extracted Mask3D per-scene features instead of PointNet++.
        Expects batch to contain:
          - mask3d_feature_path: list/tuple of length B with scene feature paths
          - instance_ids: tensor/list [B, max_context] with instance ids for each context slot
        Feature file should contain:
          - object_queries: [num_queries, feat_dim]
          - gt_to_query_map: dict mapping instance_id -> query idx (optional)
        """
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
                # fallback zeros
                in_dim = self.mask3d_proj_in.in_features
                obj_feats.append(torch.zeros(max_context, in_dim, device=self.device))
                continue
            if isinstance(queries, np.ndarray):
                queries = torch.from_numpy(queries)
            queries = queries.to(self.device)
            in_dim = queries.shape[1]
            feats_b = torch.zeros(max_context, in_dim, device=self.device)
            mapping = data.get("gt_to_query_map", {}) or {}
            inst_cls = data.get("gt_instance_classes", {}) or {}
            for j in range(max_context):
                inst_id = int(inst_tensor[b, j].item())
                if inst_id < 0:
                    continue
                lookup_id = inst_id
                q_idx = mapping.get(lookup_id, None)
                if q_idx is None:
                    # Common mismatch: referit3d uses 0-based object_id while ScanNet
                    # instance ids in some preprocessing/codepaths are 1-based.
                    # If the checkpoint/feature file is 1-based, allow a safe +1 fallback.
                    if lookup_id >= 0 and (lookup_id + 1) in mapping and 0 not in mapping and 1 in mapping:
                        lookup_id = lookup_id + 1
                        q_idx = mapping.get(lookup_id, None)
                if q_idx is None:
                    continue
                if 0 <= q_idx < queries.shape[0]:
                    feats_b[j] = queries[q_idx]
                    if scannet_labels is not None:
                        cls_id = inst_cls.get(lookup_id, None)
                        if cls_id is not None and self.scannet_id_to_contig is not None:
                            try:
                                cls_id_int = int(cls_id)
                            except Exception:
                                cls_id_int = None
                            if cls_id_int is not None:
                                mapped = self.scannet_id_to_contig.get(cls_id_int, -1)
                                if 0 <= int(mapped) < int(self.scannet_num_classes):
                                    scannet_labels[b, j] = int(mapped)
            obj_feats.append(feats_b)
        obj_feats = torch.stack(obj_feats, dim=0)  # [B, max_context, in_dim]
        obj_feats = self.mask3d_proj_in(obj_feats)
        obj_feats = self.mask3d_adapter(obj_feats)  # transformer is batch_first
        # Some checkpoints use `object_dim != inner_dim`. In PointNet++ mode we always
        # map object features via `obj_feature_mapping` to `inner_dim`; do the same for
        # Mask3D-backed features when needed so downstream fusion (e.g. + box features)
        # stays well-defined.
        try:
            if obj_feats.shape[-1] != self.inner_dim:
                obj_feats = self.obj_feature_mapping(obj_feats)
        except Exception:
            pass
        if scannet_labels is not None:
            # Attach ScanNet200 class labels to batch for loss computation (single-GPU case).
            batch["scannet_class_labels"] = scannet_labels
        return obj_feats, scannet_labels

    def _encode_with_mask3d_queries(self, object_queries: torch.Tensor) -> torch.Tensor:
        """
        Encode in-memory Mask3D object queries instead of loading from disk.

        Args:
            object_queries: Tensor of shape [B, N_ctx, feat_dim]
        Returns:
            obj_feats: Tensor of shape [B, N_ctx, object_dim]
        """
        if object_queries is None or not torch.is_tensor(object_queries):
            raise ValueError("mask3d_object_queries must be a Tensor [B,N,feat_dim]")
        oq = object_queries.to(self.device)
        if oq.dim() != 3:
            raise ValueError(f"mask3d_object_queries must be 3D [B,N,D], got {tuple(oq.shape)}")
        # Project to Vigor's object_dim and apply the same adapter used in file-backed Mask3D mode.
        oq = self.mask3d_proj_in(oq)
        try:
            oq = self.mask3d_adapter(oq)
        except Exception:
            # Adapter can be absent in older checkpoints/configs; keep projected features.
            pass
        # See `_encode_with_mask3d`: if object_dim != inner_dim, map features before fusion.
        try:
            if oq.shape[-1] != self.inner_dim:
                oq = self.obj_feature_mapping(oq)
        except Exception:
            pass
        return oq

    def _encode_mask3d_dino_features(self, batch: dict) -> torch.Tensor | None:
        if not self.use_mask3d_dino_features:
            return None
        if "mask3d_dino_features" not in batch:
            raise RuntimeError("VIGOR_MASK3D_DINO_ENABLE=1 requires batch['mask3d_dino_features']")
        dino = batch["mask3d_dino_features"]
        if not torch.is_tensor(dino):
            raise TypeError(f"mask3d_dino_features must be a Tensor [B,N,D], got {type(dino)}")
        dino = dino.to(self.device, dtype=torch.float32)
        if dino.dim() != 3:
            raise RuntimeError(f"mask3d_dino_features must be [B,N,D], got {tuple(dino.shape)}")
        if int(dino.shape[-1]) != int(self.mask3d_dino_feature_dim):
            raise RuntimeError(
                f"mask3d_dino_features dim mismatch: got D={int(dino.shape[-1])} "
                f"expected D={int(self.mask3d_dino_feature_dim)}"
            )
        valid = batch.get("mask3d_dino_valid_mask", None)
        if valid is None:
            valid = torch.ones(dino.shape[:2], device=self.device, dtype=torch.bool)
        else:
            if not torch.is_tensor(valid):
                raise TypeError(f"mask3d_dino_valid_mask must be a Tensor [B,N], got {type(valid)}")
            valid = valid.to(self.device, dtype=torch.bool)
        if valid.dim() != 2 or int(valid.shape[0]) != int(dino.shape[0]) or int(valid.shape[1]) != int(dino.shape[1]):
            raise RuntimeError(
                f"mask3d_dino_valid_mask must be [B,N] aligned with features, "
                f"got mask={tuple(valid.shape)} feats={tuple(dino.shape)}"
            )
        dino = F.normalize(dino, dim=-1)
        dino_infos = self.mask3d_dino_feature_mapping(dino)
        return dino_infos * valid.to(dtype=dino_infos.dtype).unsqueeze(-1)

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
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
        return input_points, boxs

    @torch.no_grad()
    def _multiview_rotate_box_info(self, box_info: torch.Tensor) -> torch.Tensor:
        """
        Build multi-view rotated `box_info` to match `aug_input(...)[1]` in eval mode.

        Args:
            box_info: [B, N, 4] (cx,cy,cz,volume)
        Returns:
            boxs: [B, V, N, 4] where V=self.view_number
        """
        box_info = torch.as_tensor(box_info, device=self.device, dtype=torch.float32)
        if box_info.dim() != 3 or box_info.size(-1) != 4:
            raise RuntimeError(f"[Vigor] box_info must be [B,N,4], got {tuple(box_info.shape)}")
        B, N = int(box_info.size(0)), int(box_info.size(1))
        bxyz = box_info[:, :, :3]
        bsize = box_info[:, :, -1:].contiguous()
        view_theta_arr = torch.as_tensor(
            [i * 2.0 * np.pi / float(self.view_number) for i in range(int(self.view_number))],
            device=self.device,
            dtype=torch.float32,
        )
        boxs = []
        for theta in view_theta_arr:
            rotate_matrix = torch.as_tensor(
                [
                    [math.cos(float(theta)), -math.sin(float(theta)), 0.0],
                    [math.sin(float(theta)), math.cos(float(theta)), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            boxs.append(torch.cat([rxyz, bsize], dim=-1))
        return torch.stack(boxs, dim=1)

    def compute_basic_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, ANCHOR_LOGITS=None, SCANNET_CLASS_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = 0.0
        # Original Vigor object classification loss (607 classes).
        if not self.use_scannet200_obj_cls and CLASS_LOGITS is not None:
            obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        # Optional ScanNet200-based object classification loss.
        if self.use_scannet200_obj_cls and SCANNET_CLASS_LOGITS is not None and "scannet_class_labels" in batch:
            labels = batch['scannet_class_labels']
            valid = labels >= 0
            if valid.any():
                # Flatten only valid positions: [M, C] and [M]
                logits_flat = SCANNET_CLASS_LOGITS[valid]
                labels_flat = labels[valid]
                obj_clf_loss = self.scannet_obj_logits_loss(logits_flat, labels_flat)
            else:
                obj_clf_loss = torch.tensor(0.0, device=LOGITS.device)
        lang_clf_loss = 0
        if not self.disable_text_loss:
            # 在 DataParallel 下若 tokens 未被正确切分，LANG_LOGITS 可能重复，形状大于 target_class。
            if LANG_LOGITS.shape[0] != batch['target_class'].shape[0]:
                LANG_LOGITS = LANG_LOGITS[:batch['target_class'].shape[0]]
            # Optionally override target_class with ScanNet200-contiguous label (per-target) when available.
            target_cls = batch.get('target_class', None)
            if self.use_scannet200_text_cls and ('scannet_class_labels' in batch):
                try:
                    labels = batch['scannet_class_labels']
                    if torch.is_tensor(labels) and torch.is_tensor(batch.get('target_pos', None)):
                        B = int(batch['target_pos'].shape[0])
                        idx = torch.arange(B, device=labels.device)
                        tgt = labels[idx, batch['target_pos'].long().to(labels.device)]
                        target_cls = tgt
                        # Propagate for downstream meters (single-GPU case).
                        batch['target_class'] = target_cls
                except Exception:
                    pass
            lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, target_cls)
            if ANCHOR_LOGITS is not None:
                lang_clf_loss += self.anchor_logits_loss(ANCHOR_LOGITS, batch['anchor_ind'])
        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss

    def forward(self, batch: dict, epoch=None):
        TOTAL_LOSS = 0
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id
        self.device = self.obj_feature_mapping[0].weight.device

        # Inference / runtime mode: allow passing in-memory Mask3D queries directly.
        # This is used by SSR3DLLM when routing "<geom>" to a pretrained Vigor listener.
        use_inmemory_mask3d = (
            self.use_mask3d_features
            and ("mask3d_object_queries" in batch)
            and (batch.get("mask3d_object_queries", None) is not None)
        )
        dino_infos = self._encode_mask3d_dino_features(batch)

        # obj_encoding
        scannet_labels = None
        if use_inmemory_mask3d:
            obj_feats = self._encode_with_mask3d_queries(batch["mask3d_object_queries"])  # [B,N,object_dim]
            boxs = batch.get("box_info", None)
            if boxs is None:
                boxs = torch.zeros((obj_feats.shape[0], obj_feats.shape[1], 4), device=self.device)
            else:
                boxs = torch.as_tensor(boxs, device=self.device)
            B, N = obj_feats.shape[:2]
            # In in-memory Mask3D mode we don't have point clouds to call `aug_input()`,
            # but the pretrained listener expects *multi-view rotated* box features.
            # Enable by env to keep backward compatibility.
            enable_mv = str(os.environ.get("SSR3DLLM_VIGOR_INMEMORY_BOX_MULTIVIEW", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if enable_mv and int(self.view_number) > 1:
                boxs_mv = self._multiview_rotate_box_info(boxs)  # [B,V,N,4]
                box_infos = self.box_feature_mapping(boxs_mv.float())  # [B,V,N,D]
                # Defensive: ensure object and box dims match for fusion.
                if obj_feats.shape[-1] != box_infos.shape[-1]:
                    if obj_feats.shape[-1] == self.object_dim and box_infos.shape[-1] == self.inner_dim:
                        obj_feats = self.obj_feature_mapping(obj_feats)
                    else:
                        raise RuntimeError(
                            f"[Vigor] feature dim mismatch: obj_feats={tuple(obj_feats.shape)} "
                            f"box_infos={tuple(box_infos.shape)} object_dim={self.object_dim} inner_dim={self.inner_dim}"
                        )
                obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos
            else:
                # Backward-compatible: replicate box features across views.
                box_infos = self.box_feature_mapping(boxs.float())  # [B,N,D]
                if obj_feats.shape[-1] != box_infos.shape[-1]:
                    if obj_feats.shape[-1] == self.object_dim and box_infos.shape[-1] == self.inner_dim:
                        obj_feats = self.obj_feature_mapping(obj_feats)
                    else:
                        raise RuntimeError(
                            f"[Vigor] feature dim mismatch: obj_feats={tuple(obj_feats.shape)} "
                            f"box_infos={tuple(box_infos.shape)} object_dim={self.object_dim} inner_dim={self.inner_dim}"
                        )
                obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos[:, None].repeat(
                    1, self.view_number, 1, 1
                )
        else:
            ## rotation augmentation and multi_view generation
            obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])

            B, N, P = obj_points.shape[:3]

            if self.use_mask3d_features:
                obj_feats, scannet_labels = self._encode_with_mask3d(batch)  # [B, max_context, object_dim]
            else:
                objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack) # torch.Size([24, 52, 768])
                obj_feats = self.obj_feature_mapping(objects_features) # torch.Size([24, 52, 768])
            box_infos = self.box_feature_mapping(boxs.float())
            obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1).squeeze() + box_infos # torch.Size([24, 4, 52, 768])
        if len(obj_infos.shape) == 3:
            assert self.view_number == 1
            obj_infos = obj_infos.unsqueeze(1).repeat(1, self.view_number, 1, 1)
        if dino_infos is not None:
            if (
                int(dino_infos.shape[0]) != int(obj_infos.shape[0])
                or int(dino_infos.shape[1]) != int(obj_infos.shape[2])
                or int(dino_infos.shape[2]) != int(obj_infos.shape[3])
            ):
                raise RuntimeError(
                    f"mask3d_dino_features must align with obj_infos, got "
                    f"dino={tuple(dino_infos.shape)} obj_infos={tuple(obj_infos.shape)}"
                )
            obj_infos = obj_infos + float(self.mask3d_dino_alpha) * dino_infos[:, None].repeat(
                1, self.view_number, 1, 1
            )
        ## language_encoding
        # Option A (default): encode `lang_tokens` with BERT (Vigor original).
        # Option B: directly provide precomputed `lang_embeds` to bypass BERT,
        #           e.g. projected LLM hidden states or soft memory tokens.
        lang_embeds = batch.get("lang_embeds", None)
        if lang_embeds is not None:
            lang_infos = torch.as_tensor(lang_embeds, device=self.device)
            if lang_infos.dim() != 3:
                raise RuntimeError(f"[Vigor] lang_embeds must be [B,L,D], got {tuple(lang_infos.shape)}")
            if int(lang_infos.size(0)) != int(B):
                raise RuntimeError(
                    f"[Vigor] lang_embeds batch mismatch: got B={int(lang_infos.size(0))} expected {int(B)}"
                )
            if int(lang_infos.size(-1)) != int(self.inner_dim):
                raise RuntimeError(
                    f"[Vigor] lang_embeds dim mismatch: got D={int(lang_infos.size(-1))} expected {int(self.inner_dim)}"
                )
        else:
            if "lang_tokens" not in batch:
                raise RuntimeError("[Vigor] batch must contain either lang_tokens or lang_embeds")
            lang_tokens = {k: v.to(self.device) for k, v in batch["lang_tokens"].items()}
            lang_infos = self.language_encoder(**lang_tokens)[0]

        # <LOSS>: lang_cls
        lang_features = lang_infos[:,0]

        LANG_LOGITS = self.language_clf(lang_infos[:,0])
        if self.lang_multilabel:
            ANCHOR_LOGITS = self.anchor_clf(lang_infos[:,0])
        mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)

        # start feature encoding
        # Option A (default): encode `order_tokens` with BERT (Vigor original).
        # Option B: directly provide precomputed order embeddings via `order_embeds`,
        #           e.g. projected LLM hidden states for <stepK> tokens.
        order_embeds = batch.get("order_embeds", None)
        if order_embeds is not None:
            mentioned_obj_lang_infos = torch.as_tensor(order_embeds, device=self.device)
            # Accept:
            #   - [B, order_len, D]   -> treat as single token per step
            #   - [B, order_len, L, D]
            if mentioned_obj_lang_infos.dim() == 3:
                mentioned_obj_lang_infos = mentioned_obj_lang_infos.unsqueeze(2)
            if mentioned_obj_lang_infos.dim() != 4:
                raise RuntimeError(
                    f"[Vigor] order_embeds must be [B,order_len,D] or [B,order_len,L,D], "
                    f"got {tuple(mentioned_obj_lang_infos.shape)}"
                )
            if int(mentioned_obj_lang_infos.size(1)) != int(self.order_len):
                raise RuntimeError(
                    f"[Vigor] order_embeds order_len mismatch: got {int(mentioned_obj_lang_infos.size(1))} "
                    f"expected {int(self.order_len)}"
                )
        else:
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

        # In step-marker mode, optionally keep only the <stepK> slot hidden state (pos=1)
        # as the per-step "mentioned_features" input to the refer encoder layers.
        # This forces the model to route step guidance through the dedicated step tokens.
        if self.use_step_markers and self.use_step_slot_only:
            try:
                if mentioned_obj_lang_infos.size(2) >= 2:
                    mentioned_obj_lang_infos = mentioned_obj_lang_infos[:, :, 1:2, :]
            except Exception:
                pass

        adaptive_halt = str(os.environ.get("VIGOR_ADAPTIVE_HALT", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        halt_logits = None
        halt_weights = None
        if adaptive_halt:
            # Use per-step order text [CLS] embedding to score which step to "halt" at.
            # mentioned_obj_lang_infos: [B, order_len, L, D]
            try:
                cls_feats = mentioned_obj_lang_infos[:, :, 0, :]  # [B, order_len, D]
                halt_logits = self.halt_head(cls_feats).squeeze(-1)  # [B, order_len]
                tau = float(str(os.environ.get("VIGOR_HALT_TAU", "1.0")).strip())
                if not np.isfinite(tau) or tau <= 0:
                    tau = 1.0
                halt_weights = torch.softmax(halt_logits / tau, dim=1)  # [B, order_len]
            except Exception:
                adaptive_halt = False
                halt_logits = None
                halt_weights = None
        
        cat_infos = obj_infos.reshape(B*self.view_number, -1, self.inner_dim) # torch.Size([96, 52, 768])

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            class_tokens = {k: v.to(self.device) for k, v in self.class_name_tokens.items()}
            label_lang_infos = self.language_encoder(**class_tokens)[0][:,0] # torch.Size([525, 768])
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B*N,-1), label_lang_infos.permute(1,0)).reshape(B,N,-1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B*N,-1)).reshape(B,N,-1) # torch.Size([24, 52, 525])

        SCANNET_CLASS_LOGITS = None
        if self.use_scannet200_obj_cls and self.scannet_obj_clf is not None:
            SCANNET_CLASS_LOGITS = self.scannet_obj_clf(obj_feats.reshape(B*N,-1)).reshape(B, N, -1)

        # Expose per-step multilabel logits (view-aggregated) for offline distillation.
        # Export tools read `model.last_tb_multilabel_logits_steps` after forward().
        tb_multilabel_steps = []
        # Also expose per-step *referential* logits (object_language_clf after view aggregation)
        # so downstream students can distill an object-level distribution at each step.
        ref_logits_steps = []
        ref_logits_steps_raw = []

        # Variable-length chains (paper setting): ignore padded steps by a validity mask.
        # - Enable via env `VIGOR_VARLEN_CHAIN=1`.
        # - Mask source priority: batch["order_valid_mask"] -> batch["ori_order_len"].
        # - Optional speed: `VIGOR_VARLEN_EARLY_STOP=1` loops only up to max valid len in the batch.
        varlen_enabled = str(os.environ.get("VIGOR_VARLEN_CHAIN", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        varlen_early_stop = str(os.environ.get("VIGOR_VARLEN_EARLY_STOP", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        order_valid_mask = batch.get("order_valid_mask", None)
        if varlen_enabled and (order_valid_mask is None):
            ori_len = batch.get("ori_order_len", None)
            if ori_len is not None:
                try:
                    ori_len_t = torch.as_tensor(ori_len, device=self.device).long().view(-1)
                    steps = torch.arange(int(self.order_len), device=self.device).view(1, -1)
                    order_valid_mask = (steps < ori_len_t.view(-1, 1)).to(dtype=torch.float32)
                except Exception:
                    order_valid_mask = None

        loop_steps = int(self.order_len)
        if varlen_enabled and varlen_early_stop and (order_valid_mask is not None):
            try:
                loop_steps = int(order_valid_mask.sum(dim=1).max().item())
                loop_steps = max(1, min(int(self.order_len), loop_steps))
            except Exception:
                loop_steps = int(self.order_len)

        for i in range(loop_steps):
            mask = batch['pred_class_mask'][:, i, :].unsqueeze(1).unsqueeze(3).repeat(1, self.view_number, 1, self.inner_dim)
            mask = mask.reshape(B*self.view_number, -1, self.inner_dim)

            cat_infos_prev = cat_infos
            masked_obj_infos = cat_infos_prev * mask
            mentioned_features = mentioned_obj_lang_infos[:, i, :, :].unsqueeze(1).repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)
            cat_infos_new = self.refer_encoder[i](
                cat_infos_prev.transpose(0, 1),
                masked_obj_infos.transpose(0, 1),
                mem_infos.transpose(0, 1),
                mentioned_features.transpose(0, 1),
            ) # torch.Size([96, 52, 768])
            if varlen_enabled and (order_valid_mask is not None):
                try:
                    m = order_valid_mask[:, i].to(device=cat_infos_new.device, dtype=cat_infos_new.dtype).view(B, 1, 1, 1)
                    m = m.repeat(1, self.view_number, 1, 1).reshape(B * self.view_number, 1, 1)
                    cat_infos = (1.0 - m) * cat_infos_prev + m * cat_infos_new
                except Exception:
                    cat_infos = cat_infos_new
            else:
                cat_infos = cat_infos_new
            # Save per-step referential logits (view-aggregated): [B, N_ctx]
            step_logits = None
            step_feats = cat_infos.reshape(B, self.view_number, -1, self.inner_dim)  # [B,V,N,D]
            if self.aggregate_type == 'avg':
                step_agg = (step_feats / self.view_number).sum(dim=1)
            elif self.aggregate_type == 'avgmax':
                step_agg = (step_feats / self.view_number).sum(dim=1) + step_feats.max(dim=1).values
            else:
                step_agg = step_feats.max(dim=1).values
            step_logits = self.object_language_clf(step_agg).squeeze(-1)  # [B,N]
            ref_logits_steps_raw.append(step_logits)
            ref_logits_steps.append(step_logits.detach())
            if self.multilabel_pretraining:
                if not self.disable_multilabel_loss:
                    TB_MULTILABEL_LOGITS = self.feat_to_multilabel_clf(cat_infos).squeeze() # torch.Size([96, 52])
                    ans = batch['ordered_multilabel_gt'][:, i, :].unsqueeze(1).repeat(1, self.view_number, 1).reshape(B*self.view_number, -1)
                    TB_MULTILABEL_LOSS = self.ml_feature_constraint_loss(TB_MULTILABEL_LOGITS, ans.float())
                    TOTAL_LOSS += TB_MULTILABEL_LOSS * 0.5
                    # Save view-aggregated logits for this step: [B, N_ctx]
                    try:
                        tb = TB_MULTILABEL_LOGITS
                        if tb.dim() == 1:
                            tb = tb.unsqueeze(0)
                        tb = tb.reshape(B, self.view_number, -1)
                        tb_multilabel_steps.append(tb.mean(dim=1).detach())
                    except Exception:
                        pass

        # Attach per-step logits for exporters (may be empty if disabled).
        self.last_tb_multilabel_logits_steps = tb_multilabel_steps
        self.last_ref_logits_steps = ref_logits_steps

        ## multi-modal_fusion
        out_feats = cat_infos.reshape(B, self.view_number, -1, self.inner_dim) # torch.Size([24, 4, 52, 768])
        
        ## view_aggregation
        refer_feat = out_feats
        if self.aggregate_type=='avg':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) # torch.Size([24, 52, 768])
        elif self.aggregate_type=='avgmax':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
        else:
            agg_feats = refer_feat.max(dim=1).values

        # <LOSS>: ref_cls (or inference logits)
        LOGITS_LAST = self.object_language_clf(agg_feats).squeeze(-1)
        LOGITS = LOGITS_LAST
        if adaptive_halt and (halt_weights is not None) and (len(ref_logits_steps_raw) == self.order_len):
            try:
                step_logits_stack = torch.stack(ref_logits_steps_raw, dim=1)  # [B, order_len, N]
                LOGITS = (halt_weights.unsqueeze(-1) * step_logits_stack).sum(dim=1)
            except Exception:
                LOGITS = LOGITS_LAST

        # Expose for debugging / downstream runtime (optional).
        self.last_halt_logits = halt_logits.detach() if torch.is_tensor(halt_logits) else None
        self.last_halt_weights = halt_weights.detach() if torch.is_tensor(halt_weights) else None

        if not batch.get("inference", False):
            if self.lang_multilabel:
                BASIC_LOSS = self.compute_basic_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, ANCHOR_LOGITS, SCANNET_CLASS_LOGITS)
            else:
                BASIC_LOSS = self.compute_basic_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS=SCANNET_CLASS_LOGITS)
            TOTAL_LOSS += BASIC_LOSS

            # Optional adaptive-halting losses.
            if adaptive_halt and (halt_logits is not None) and (len(ref_logits_steps_raw) == self.order_len):
                try:
                    ori_len = batch.get("ori_order_len", None)
                    if ori_len is not None:
                        stop_idx = torch.as_tensor(ori_len, device=self.device).long().view(-1) - 1
                        stop_idx = stop_idx.clamp(min=0, max=int(self.order_len - 1))
                        halt_loss = F.cross_entropy(halt_logits, stop_idx)

                        # Step-level alignment: ensure the oracle step's logits can predict the target.
                        step_logits_stack = torch.stack(ref_logits_steps_raw, dim=1)  # [B, order_len, N]
                        batch_idx = torch.arange(B, device=self.device)
                        oracle_logits = step_logits_stack[batch_idx, stop_idx]  # [B, N]
                        step_ref_loss = F.cross_entropy(oracle_logits, batch["target_pos"])

                        halt_w = float(str(os.environ.get("VIGOR_HALT_LOSS_W", "0.1")).strip())
                        step_w = float(str(os.environ.get("VIGOR_STEP_REF_LOSS_W", "0.05")).strip())
                        if not np.isfinite(halt_w):
                            halt_w = 0.1
                        if not np.isfinite(step_w):
                            step_w = 0.05
                        TOTAL_LOSS += (halt_w * halt_loss) + (step_w * step_ref_loss)
                except Exception:
                    pass

        return TOTAL_LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS, SCANNET_CLASS_LOGITS, scannet_labels
