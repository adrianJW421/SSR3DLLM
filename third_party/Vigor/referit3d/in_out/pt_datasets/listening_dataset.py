import numpy as np
import json
import ast
import os
from random import sample 
from torch.utils.data import Dataset
from functools import partial
import torch

from .utils import dataset_to_dataloader, max_io_workers
from .utils import check_segmented_object_order, sample_scan_object, pad_samples
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from .utils import ScannetDatasetConfig
from ...data_generation.nr3d import decode_stimulus_string
from pathlib import Path


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, multilabel_pretraining=False, lang_multilabel=False, 
                 cascading=False, order_len=4, mask3d_feature_root: str | None = None):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1 # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError

        self.mask3d_feature_root = Path(mask3d_feature_root) if mask3d_feature_root else None
        dino_cache_root = str(os.environ.get("VIGOR_MASK3D_DINO_SAMPLE_CACHE_ROOT", "")).strip()
        self.mask3d_dino_sample_cache_root = Path(dino_cache_root) if dino_cache_root else None
        self._mask3d_scene_cache: dict[str, dict | None] = {}

        # Original Vigor uses BUTD/PCNet object classification results to build
        # pred_class_mask for the referential-order decoder. When using Mask3D
        # features, that file is in a different label space; prefer Mask3D
        # pred_classes from the exported per-scene feature files.
        self.cls_results = None
        if self.mask3d_feature_root is None:
            ref3d_root = Path(__file__).resolve().parents[2]  # .../Vigor/referit3d
            cls_path = ref3d_root / 'data' / 'butd_pcnet_cls_results.json'
            with cls_path.open('r') as fid:
                self.cls_results = json.load(fid)

        self.scannetconfig_butd = ScannetDatasetConfig('butd')
        self.multilabel_pretraining = multilabel_pretraining
        self.lang_multilabel = lang_multilabel
        self.cascading = cascading
        self.order_len = order_len
        self._debug_printed = False

    @staticmethod
    def _norm_name(name: str) -> str:
        return str(name).strip().lower().replace("_", " ")

    def _load_mask3d_scene(self, scan_id: str) -> dict | None:
        if not self.mask3d_feature_root:
            return None
        if scan_id in self._mask3d_scene_cache:
            return self._mask3d_scene_cache[scan_id]
        path = self.mask3d_feature_root / f"{scan_id}.pt"
        if not path.exists():
            self._mask3d_scene_cache[scan_id] = None
            return None
        try:
            full = torch.load(path, map_location="cpu")
        except Exception:
            full = None
        if isinstance(full, dict):
            feat = {
                "gt_to_query_map": full.get("gt_to_query_map") or {},
                "pred_class_names": full.get("pred_class_names"),
                "pred_classes": full.get("pred_classes"),
                # Optional predicted boxes from Mask3D full-res masks (no GT bboxes).
                # - pred_aabb: [Q,6] (minx,miny,minz,maxx,maxy,maxz)
                # - pred_box_info: [Q,4] (cx,cy,cz,volume)
                "pred_aabb": full.get("pred_aabb"),
                "pred_box_info": full.get("pred_box_info"),
            }
        else:
            feat = None
        self._mask3d_scene_cache[scan_id] = feat
        return feat

    @staticmethod
    def _normalize_query_map(raw_map, source: str) -> dict[int, int]:
        if raw_map is None:
            return {}
        if not isinstance(raw_map, dict):
            raise TypeError(f"gt_to_query_map must be a dict in {source}, got {type(raw_map)}")
        return {int(k): int(v) for k, v in raw_map.items()}

    def _sample_cache_relpath_from_row(self, ref, scan_id: str) -> Path:
        rel = None
        if "mask3d_sample_cache_relpath" in ref:
            rel = ref["mask3d_sample_cache_relpath"]
        if (rel is None or (isinstance(rel, float) and np.isnan(rel)) or str(rel).strip() == "") and "mask3d_sample_cache_path" in ref:
            rel = ref["mask3d_sample_cache_path"]
        if rel is None or (isinstance(rel, float) and np.isnan(rel)) or str(rel).strip() == "":
            raise RuntimeError(f"sample-level cache root is set but row for scene {scan_id} has no sample cache path")
        return Path(str(rel).strip())

    def _load_mask3d_dino_sample(self, ref, scan_id: str) -> dict | None:
        if self.mask3d_dino_sample_cache_root is None:
            return None
        raw_path = self._sample_cache_relpath_from_row(ref, scan_id)
        path = raw_path if raw_path.is_absolute() else self.mask3d_dino_sample_cache_root / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"sample-level Mask3D-DINO cache not found for scene {scan_id}: {path}")
        full = torch.load(path, map_location="cpu")
        if not isinstance(full, dict):
            raise RuntimeError(f"sample-level Mask3D-DINO cache must be a dict: {path}")
        if "proposal_dino_features" not in full:
            raise RuntimeError(f"Mask3D-DINO cache missing proposal_dino_features: {path}")
        full["_sample_cache_path"] = str(path)
        return full

    def _get_mask3d_pred_name(self, scan_id: str, inst_id: int, fallback: str) -> str:
        mode = str(os.environ.get("VIGOR_MASK3D_PRED_NAME_MODE", "gt")).strip().lower()
        # Mode semantics:
        # - "gt" (default): fallback to GT instance_label (legacy behavior).
        # - "unknown": never use GT fallback; return "unknown" when prediction is missing.
        # - "error": never use GT fallback; raise if prediction is missing.
        if mode in {"gt_fallback", "fallback"}:
            mode = "gt"

        def _on_missing(reason: str) -> str:
            if mode == "gt":
                return self._norm_name(fallback)
            if mode in {"unknown", "none", "no_gt"}:
                return "unknown"
            if mode in {"error", "strict", "raise"}:
                raise ValueError(
                    f"Mask3D predicted class missing (mode={mode}): scan_id={scan_id} inst_id={inst_id} reason={reason}"
                )
            # Safety: unknown mode -> preserve legacy.
            return self._norm_name(fallback)

        feat = self._load_mask3d_scene(scan_id)
        if not feat:
            return _on_missing("no_mask3d_feat")
        gt_map = feat.get("gt_to_query_map") or {}
        try:
            qidx = gt_map.get(int(inst_id), None)
        except Exception:
            qidx = None
        if qidx is None:
            return _on_missing("no_gt_to_query_map")
        pred_names = feat.get("pred_class_names")
        if isinstance(pred_names, list) and 0 <= int(qidx) < len(pred_names):
            name = self._norm_name(pred_names[int(qidx)])
            if name and name != "unknown":
                return name
            return _on_missing("pred_class_names_unknown")
        pred_classes = feat.get("pred_classes")
        if isinstance(pred_classes, torch.Tensor) and pred_classes.ndim == 1 and 0 <= int(qidx) < int(pred_classes.shape[0]):
            cid = int(pred_classes[int(qidx)].item())
            if cid >= 0:
                return f"id_{cid}"
            return _on_missing("pred_classes_negative")
        return _on_missing("pred_classes_missing")

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)
        is_nr3d = ref['dataset'] == 'nr3d'

        LLM_info = dict()
        LLM_info['target_object'] = ref['target_object']
        LLM_info['anchor_objects'] = ast.literal_eval(ref['anchor_objects'])
        LLM_info['referential_order'] = ast.literal_eval(ref['referential_order'])
        LLM_info['referential_order'] = [word.strip('*') for word in LLM_info['referential_order']]

        return scan, target, tokens, is_nr3d, scan_id, LLM_info

    def prepare_distractors(self, scan, target):
        # Optional: use *all* objects in the scene (no random clutter sampling),
        # so context contains every ScanNet GT object proposal.
        # NOTE: this requires `max_context_size` to be large enough; otherwise we
        # will still truncate to avoid shape errors.
        use_all = str(os.environ.get("VIGOR_USE_ALL_OBJECTS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        if use_all:
            distractors = [o for o in scan.three_d_objects if o != target]
            # Safety: keep tensor shapes bounded.
            if len(distractors) > self.max_distractors:
                sid = getattr(scan, "scan_id", None)
                if sid is None:
                    try:
                        sid = getattr(scan, "scan_id", "unknown")
                    except Exception:
                        sid = "unknown"
                if not getattr(self, "_warned_truncate_all_objects", False):
                    self._warned_truncate_all_objects = True
                    print(
                        "[Vigor][warn] VIGOR_USE_ALL_OBJECTS=1 but scene has more objects than max_distractors; "
                        f"truncating: scan_id={sid} n_scene_objects={len(scan.three_d_objects)} "
                        f"max_context_size={self.max_context_size} (max_distractors={self.max_distractors})"
                    )
                distractors = distractors[: self.max_distractors]
            np.random.shuffle(distractors)
            return distractors

        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors
    
    def prepare_distractors_ours(self, scan, target, scan_id):
        target_label = target.instance_label
        already_included = {target_label}
        distractors = []
        pred = []
        if self.cls_results is None:
            raise RuntimeError("cls_results is not available (Mask3D mode should not call prepare_distractors_ours).")
        for i, o in enumerate(scan.three_d_objects):
            if o.instance_label == target_label and (o != target):
                distractors.append(o)
                pred.append(self.cls_results[scan_id][i])
        clutter = []
        pred2 = []
        for i, o in enumerate(scan.three_d_objects):
            if o.instance_label not in already_included:
                clutter.append(o)
                pred2.append(self.cls_results[scan_id][i])
        
        temp = list(zip(clutter, pred2))
        np.random.shuffle(temp)
        clutter, pred2 = zip(*temp)
        clutter, pred2 = list(clutter), list(pred2)

        distractors.extend(clutter)
        pred.extend(pred2)

        distractors = distractors[:self.max_distractors]
        pred = pred[:self.max_distractors]

        temp = list(zip(distractors, pred))
        np.random.shuffle(temp)
        distractors, pred = zip(*temp)
        distractors, pred = list(distractors), list(pred)   

        target_pred = self.cls_results[scan_id][target.object_id]     

        return distractors, pred, target_pred

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, scan_id, LLM_info = self.get_reference_data(index)
        ref = self.references.loc[index]
        dino_feat = self._load_mask3d_dino_sample(ref, scan_id) if self.mask3d_dino_sample_cache_root is not None else None
        # Optional: multi-target GT set (M3DRef). Stored as a stringified list in CSV.
        target_ids = None
        try:
            if "target_ids" in ref and isinstance(ref["target_ids"], str) and ref["target_ids"].strip():
                target_ids = ast.literal_eval(ref["target_ids"])
        except Exception:
            target_ids = None
        if isinstance(target_ids, (list, tuple)):
            tmp = []
            for x in target_ids:
                try:
                    tmp.append(int(x))
                except Exception:
                    continue
            # de-dup
            seen = set()
            target_ids = [x for x in tmp if (x not in seen and not seen.add(x))]
        else:
            target_ids = None

        # Make a context of distractors.
        # - In Mask3D mode, do NOT rely on BUTD/PCNet classification results.
        # - In original mode, keep Vigor's cls_results-driven selection.
        pred_box = None
        target_pred = None
        if self.mask3d_feature_root is not None:
            # Optional: use the official ReferIt3D "stimulus" candidate set, i.e.
            # only the same-class objects listed in stimulus_id (no extra random clutter).
            # This aligns the candidate construction with `stimulus_id` parsing instead
            # of the default "same-class + random clutter" sampling.
            context_mode = str(os.environ.get("VIGOR_CONTEXT_MODE", "sampled")).strip().lower()
            if context_mode in {"stimulus", "official", "stimulus_only"}:
                try:
                    stim = str(self.references.loc[index]["stimulus_id"])
                    _, _, _, _, distractors_ids = decode_stimulus_string(stim)
                except Exception:
                    distractors_ids = []
                context = []
                for did in distractors_ids:
                    try:
                        did = int(did)
                    except Exception:
                        continue
                    # Safety: object_ids are expected to match indices.
                    if 0 <= did < len(scan.three_d_objects):
                        o = scan.three_d_objects[did]
                        if o != target:
                            context.append(o)
                # Safety: ensure bounded shapes (still uses max_context_size).
                if len(context) > self.max_distractors:
                    context = context[: self.max_distractors]
                np.random.shuffle(context)
            else:
                context = self.prepare_distractors(scan, target)
        else:
            context, pred_box, target_pred = self.prepare_distractors_ours(scan, target, scan_id)
            if target_pred == -1:
                target_pred = 325  # 325 for butd id
            pred_box = [i if i != -1 else 325 for i in pred_box]

        # Add target object into list of context
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)
        if pred_box is not None:
            pred_box.insert(target_pos, target_pred)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])
        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        res['scan_id'] = scan_id
        # box_info:
        # - default (original Vigor): GT bbox center+volume from ScanNet objects
        # - optional (Mask3D-Vigor pred-box mode): predicted AABB-derived (cx,cy,cz,volume)
        use_pred_box = str(os.environ.get("VIGOR_USE_PRED_BOX_INFO", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        box_info = np.zeros((self.max_context_size, 4), dtype=np.float32)
        box_corners = np.zeros((self.max_context_size, 8, 3), dtype=np.float32)
        # Always keep GT corners for evaluation (ScanRefer/M3DRef Acc@IoU expects pred vs GT).
        gt_box_corners = np.zeros((self.max_context_size, 8, 3), dtype=np.float32)
        box_info_center = np.zeros((self.max_context_size, 3), dtype=np.float32)
        if self.mask3d_feature_root is not None and use_pred_box:
            feat = self._load_mask3d_scene(scan_id)
            gt_map = (feat.get("gt_to_query_map") if isinstance(feat, dict) else None) or {}
            pred_aabb = feat.get("pred_aabb") if isinstance(feat, dict) else None
            pred_box = feat.get("pred_box_info") if isinstance(feat, dict) else None

            if pred_aabb is not None:
                try:
                    pred_aabb = torch.as_tensor(pred_aabb, dtype=torch.float32).cpu().numpy()
                except Exception:
                    pred_aabb = None
            if pred_box is not None:
                try:
                    pred_box = torch.as_tensor(pred_box, dtype=torch.float32).cpu().numpy()
                except Exception:
                    pred_box = None

            for j, o in enumerate(context):
                if j >= self.max_context_size:
                    break
                inst_id = int(o.object_id)
                qidx = gt_map.get(inst_id, None)
                if qidx is None:
                    # Common mismatch: referit3d uses 0-based object_id while some
                    # Mask3D/ScanNet mappings are stored 1-based (instance_id starts at 1).
                    if inst_id >= 0 and (inst_id + 1) in gt_map and 0 not in gt_map and 1 in gt_map:
                        qidx = gt_map.get(inst_id + 1, None)
                try:
                    qidx = int(qidx) if qidx is not None else None
                except Exception:
                    qidx = None
                if qidx is None:
                    continue
                if pred_box is not None and 0 <= qidx < int(pred_box.shape[0]):
                    box_info[j, :] = pred_box[qidx, :]
                    box_info_center[j, :] = pred_box[qidx, :3]
                if pred_aabb is not None and 0 <= qidx < int(pred_aabb.shape[0]) and int(pred_aabb.shape[1]) == 6:
                    mn = pred_aabb[qidx, 0:3]
                    mx = pred_aabb[qidx, 3:6]
                    # 8 corners of AABB
                    box_corners[j, :, :] = np.array(
                        [
                            [mn[0], mn[1], mn[2]],
                            [mn[0], mn[1], mx[2]],
                            [mn[0], mx[1], mn[2]],
                            [mn[0], mx[1], mx[2]],
                            [mx[0], mn[1], mn[2]],
                            [mx[0], mn[1], mx[2]],
                            [mx[0], mx[1], mn[2]],
                            [mx[0], mx[1], mx[2]],
                        ],
                        dtype=np.float32,
                    )
            # GT corners for all valid context objects.
            gt_box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        else:
            box_info[:len(context),0] = [o.get_bbox().cx for o in context]
            box_info[:len(context),1] = [o.get_bbox().cy for o in context]
            box_info[:len(context),2] = [o.get_bbox().cz for o in context]
            box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
            box_corners[:len(context)] = [o.get_bbox().corners for o in context]
            gt_box_corners[:len(context)] = [o.get_bbox().corners for o in context]
            box_info_center[:len(context)] = [o.get_bbox().center() for o in context]

        if (
            not self._debug_printed
            and str(os.environ.get("VIGOR_DEBUG_DATA", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        ):
            self._debug_printed = True
            feat_path = str(self.mask3d_feature_root / f"{scan_id}.pt") if self.mask3d_feature_root else None
            feat = self._load_mask3d_scene(scan_id) if self.mask3d_feature_root else None
            has_pred_box = isinstance(feat, dict) and feat.get("pred_box_info") is not None
            has_pred_aabb = isinstance(feat, dict) and feat.get("pred_aabb") is not None
            print(
                "[Vigor][data_debug] "
                f"scan_id={scan_id} "
                f"mask3d_feature_root={str(self.mask3d_feature_root) if self.mask3d_feature_root else None} "
                f"mask3d_feature_path={feat_path} "
                f"use_pred_box={use_pred_box} "
                f"has_pred_box_info={has_pred_box} "
                f"has_pred_aabb={has_pred_aabb} "
                f"context_size={len(context)} target_object_id={int(target.object_id)}"
            )
            try:
                vols = box_info[: len(context), 3]
                print(
                    f"[Vigor][data_debug] box_info(volume) "
                    f"min={float(np.min(vols)):.6f} max={float(np.max(vols)):.6f} mean={float(np.mean(vols)):.6f}"
                )
            except Exception:
                pass
        res['objects'] = pad_samples(samples, self.max_context_size)
        # 记录实例 ID，便于基于 Mask3D 预提取特征进行映射。
        res['instance_ids'] = np.array([o.object_id for o in context] + [-1] * (self.max_context_size - len(context)), dtype=np.int64)
        # 若指定了 Mask3D 预提取特征目录，记录该 scene 的特征路径。
        if self.mask3d_feature_root:
            res['mask3d_feature_path'] = str(self.mask3d_feature_root / f"{scan_id}.pt")
        if dino_feat is not None:
            dino_src = torch.as_tensor(dino_feat["proposal_dino_features"], dtype=torch.float32)
            if dino_src.ndim != 2:
                raise RuntimeError(
                    f"proposal_dino_features must be [Q,D], got {tuple(dino_src.shape)} "
                    f"from {dino_feat.get('_sample_cache_path')}"
                )
            dino_valid_src = dino_feat.get("proposal_dino_valid_mask")
            if dino_valid_src is None:
                dino_valid_src = torch.ones((dino_src.shape[0],), dtype=torch.bool)
            else:
                dino_valid_src = torch.as_tensor(dino_valid_src, dtype=torch.bool)
            if dino_valid_src.ndim != 1 or int(dino_valid_src.shape[0]) != int(dino_src.shape[0]):
                raise RuntimeError(
                    f"proposal_dino_valid_mask must be [Q], got {tuple(dino_valid_src.shape)} "
                    f"for features {tuple(dino_src.shape)}"
                )
            raw_map = dino_feat.get("gt_to_query_map", None)
            if raw_map is None:
                base_feat = self._load_mask3d_scene(scan_id)
                if not isinstance(base_feat, dict):
                    raise RuntimeError(f"Mask3D-DINO alignment requires gt_to_query_map for scene {scan_id}")
                raw_map = base_feat.get("gt_to_query_map", None)
            gt_map = self._normalize_query_map(raw_map, str(dino_feat.get("_sample_cache_path", scan_id)))
            if not gt_map:
                raise RuntimeError(f"Mask3D-DINO alignment got empty gt_to_query_map for scene {scan_id}")
            aligned_dino = torch.zeros((self.max_context_size, dino_src.shape[-1]), dtype=torch.float32)
            aligned_dino_valid = torch.zeros((self.max_context_size,), dtype=torch.bool)
            for j, o in enumerate(context):
                if j >= self.max_context_size:
                    break
                qidx = gt_map.get(int(o.object_id), None)
                if qidx is None:
                    continue
                if not (0 <= int(qidx) < int(dino_src.shape[0])):
                    raise IndexError(
                        f"Mask3D-DINO qidx out of range for scene={scan_id} "
                        f"object_id={int(o.object_id)} qidx={int(qidx)} Q={int(dino_src.shape[0])}"
                    )
                if bool(dino_valid_src[int(qidx)].item()):
                    aligned_dino[j] = dino_src[int(qidx)]
                    aligned_dino_valid[j] = True
            if not bool(aligned_dino_valid.any().item()):
                raise RuntimeError(f"Mask3D-DINO did not align to any context object for scene {scan_id}")
            res['mask3d_dino_features'] = aligned_dino
            res['mask3d_dino_valid_mask'] = aligned_dino_valid
        res['center_coors'] = box_info_center
        res['corner_coors'] = box_corners
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)
        # get object mask
        obj_existance = np.zeros((self.max_context_size, 1))
        obj_existance[:len(context),0] = 1
        res['obj_mask'] = obj_existance
        res['context_size'] = len(samples)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        if self.mask3d_feature_root is not None:
            pred_class_labels = [
                self._get_mask3d_pred_name(scan_id, o.object_id, o.instance_label) for o in context
            ]
        else:
            pred_class_labels = [self.scannetconfig_butd.class2type[pred_box[i]] for i in range(len(context))]

        order = LLM_info['referential_order']
        
        # pad order
        if order == []:
            tmp = list(set(self.scannetconfig_butd.type2class.keys()).difference(set([o.instance_label for o in context])))
            order = sample(tmp, 1) # this will lead to a all-zero mask
        while len(order) > self.order_len:
            del order[0]
        # `ori_order_len` is the *effective* number of steps before padding (after truncation).
        # When enabling adaptive halting, this value serves as a stop-target (index = ori_len-1).
        res['ori_order_len'] = len(order)

        adaptive_halt = str(os.environ.get("VIGOR_ADAPTIVE_HALT", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        varlen_chain = str(os.environ.get("VIGOR_VARLEN_CHAIN", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        stop_token = str(os.environ.get("VIGOR_STOP_TOKEN", "<STOP>")).strip() or "<STOP>"
        # Paper-style variable-length chains (STOP + validity mask) require that the *prefix*
        # of length `ori_order_len` corresponds to the true chain steps.
        # Therefore, when `VIGOR_VARLEN_CHAIN=1`, we avoid the legacy "repeat pattern" padding
        # (e.g. len=2 -> [a,a,b,b]) and instead pad by repeating the last step (len=2 -> [a,b,b,b]).
        if adaptive_halt or varlen_chain:
            # For adaptive halting, avoid the legacy "repeat pattern" padding (e.g. [a,a,b,b]),
            # which makes step index supervision ambiguous. Instead, keep unique steps in order
            # and pad by repeating the last step (e.g. [a,b,b,b]).
            if len(order) == 0:
                order = ["unknown"]
            if len(order) < self.order_len:
                if varlen_chain:
                    # Paper-style: pad with STOP markers so the remaining slots are explicitly "end-of-trace".
                    order = order + [stop_token] * (self.order_len - len(order))
                else:
                    # Adaptive-halt (legacy): repeat last step.
                    order = order + [order[-1]] * (self.order_len - len(order))
        else:
            # Legacy Vigor padding: repeats early items to reach `order_len`.
            if self.order_len == 5:
                if len(order) == 1:
                    order *= self.order_len
                elif len(order) == 2:
                    order = [order[0]] * 2 + [order[1]] * 3
                elif len(order) == 3:
                    order = [order[0]] * 1 + [order[1]] * 1 + [order[2]] * 3
                elif len(order) == 4:
                    order.append(order[-1])
            if self.order_len == 6:
                if len(order) == 1:
                    order *= self.order_len
                elif len(order) == 2:
                    order = [order[0]] * 3 + [order[1]] * 3
                elif len(order) == 3:
                    order = [order[0]] * 2 + [order[1]] * 2 + [order[2]] * 2
                elif len(order) == 4:
                    order = [order[0]] * 1 + [order[1]] * 1 + [order[2]] * 1 + [order[3]] * 3
                elif len(order) == 5:
                    order.append(order[-1])
            if self.order_len == 4:
                if len(order) == 1:
                    order *= self.order_len
                elif len(order) == 2:
                    order = [order[0]] * 2 + [order[1]] * 2
                elif len(order) == 3:
                    order.append(order[-1])
            elif self.order_len == 3:
                if len(order) == 1:
                    order *= self.order_len
                elif len(order) == 2:
                    order = [order[0]] * 1 + [order[1]] * 2
            elif self.order_len == 2:
                if len(order) == 1:
                    order *= self.order_len
            elif self.order_len == 1:
                pass

        # Map STOP markers to the pad label id so downstream code always sees valid class ids.
        pad_name = "pad"
        mapped = []
        for i in order:
            if str(i).strip() == stop_token:
                mapped.append(self.scannetconfig_butd.type2class.get(pad_name, 0))
            else:
                mapped.append(self.scannetconfig_butd.type2class[i])
        res['order_labels'] = np.array(mapped)

        if self.multilabel_pretraining:
            ordered_multilabel_gt = []
            for i, obj in enumerate(order):
                mask = np.zeros(self.max_context_size, dtype=bool)
                if not self.cascading:
                    mask[:len(context)] = [obj == o.instance_label for o in context]
                else:
                    mask[:len(context)] = [o.instance_label in order[i:] for o in context]
                ordered_multilabel_gt.append(mask)
            ordered_multilabel_gt = np.stack(ordered_multilabel_gt, axis=0).astype(int)
            res['ordered_multilabel_gt'] = ordered_multilabel_gt

        pred_class_mask = []
        def _is_stop(x: str) -> bool:
            try:
                return str(x).strip() == stop_token
            except Exception:
                return False

        for i, obj in enumerate(order):
            mask = np.zeros(self.max_context_size, dtype=bool)
            if not self.cascading:
                if _is_stop(obj):
                    all_obj = None
                else:
                    all_obj = {self._norm_name(obj)}
            else:
                all_obj = {self._norm_name(x) for x in order[i:] if (not _is_stop(x))}
            if all_obj is None:
                # STOP step: do not constrain candidates (it will be ignored by validity mask anyway).
                mask[:len(context)] = True
            else:
                mask[:len(context)] = [self._norm_name(pred_class_labels[k]) in all_obj for k in range(len(context))]

            pred_class_mask.append(mask)
        pred_class_mask = np.stack(pred_class_mask, axis=0)
 
        if self.lang_multilabel:
            anchor_ind = np.zeros(485) # ignore the pad class (525)
            anchor_order = set(order)
            anchor_order.discard(order[-1])
            for i in anchor_order:
                if _is_stop(i):
                    continue
                if i not in self.scannetconfig_butd.type2class:
                    continue
                anchor_ind[self.scannetconfig_butd.type2class[i]] = 1
            res['anchor_ind'] = anchor_ind

        res['target_object'] = LLM_info['target_object']
        res['pred_class_mask'] = pred_class_mask

        cascaded_order = []
        if self.cascading:
            # NOTE: In varlen/STOP mode, `order` is padded with explicit STOP markers.
            # When building cascaded step text (suffix string), we must NOT leak STOP into
            # valid steps (otherwise "teacher" step phrases become STOP-contaminated).
            # Keep STOP only for truly invalid/padded steps.
            stop_token = str(os.environ.get("VIGOR_STOP_TOKEN", "<STOP>")).strip() or "<STOP>"
            try:
                ori_len_eff = int(res.get("ori_order_len", len(order)))
            except Exception:
                ori_len_eff = len(order)
            varlen_chain = str(os.environ.get("VIGOR_VARLEN_CHAIN", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            for i in range(len(order)):
                if str(order[i]).strip() == stop_token:
                    cascaded_order.append(stop_token)
                    continue
                suffix = order[i : (ori_len_eff if varlen_chain else len(order))]
                sub_order = list(dict.fromkeys(suffix))
                sub_order = [x for x in sub_order if str(x).strip() != stop_token]
                if not sub_order:
                    cascaded_order.append(stop_token)
                else:
                    cascaded_order.append(", ".join(sub_order))
            res['referential_order'] = cascaded_order
        else:
            res['referential_order'] = order

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['box_info'] = box_info
        res['box_corners'] = box_corners
        res['gt_box_corners'] = gt_box_corners

        # Multi-target mask for strict M3DRef F1 evaluation.
        if target_ids is not None:
            mt = np.zeros((self.max_context_size,), dtype=np.float32)
            for j, o in enumerate(context):
                if j >= self.max_context_size:
                    break
                try:
                    if int(o.object_id) in target_ids:
                        mt[j] = 1.0
                except Exception:
                    continue
            res["multi_target_mask"] = mt
            res["multi_target_count"] = int(np.sum(mt[: len(context)]))

        if self.visualization:
            # For ReferIt3D analysis, Vigor historically stored up to 6 distractor positions.
            # With full-scene contexts enabled (e.g. VIGOR_USE_ALL_OBJECTS=1), the number of
            # same-class distractors can exceed 6, so keep the fixed-size buffer but avoid OOB.
            distrators_pos = np.zeros((6), dtype=np.int64)  # legacy visualization buffer
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    if j < int(distrators_pos.shape[0]):
                        distrators_pos[j] = k
                        j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            # External datasets (e.g. ScanRefer/M3DRef) can be single-target by design.
            # Vigor's original ReferIt3D evaluation focuses on "multiple distractors" cases
            # and filters out unique-target utterances. Allow disabling this behavior via env var.
            keep_unique = str(os.environ.get("VIGOR_ALLOW_UNIQUE_TEST", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
            if keep_unique:
                print("[Vigor][data] VIGOR_ALLOW_UNIQUE_TEST=1: keeping single-target test utterances.")
            else:
                def multiple_targets_utterance(x):
                    _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                    return len(distractors_ids) > 0

                def _coerce_bool_mask(mask_like, n_rows: int) -> np.ndarray:
                    arr = np.asarray(mask_like)
                    if arr.ndim == 0:
                        arr = np.repeat(bool(arr), int(n_rows))
                    else:
                        arr = arr.reshape(-1)
                    if arr.shape[0] != int(n_rows):
                        raise ValueError(
                            "unexpected mask shape: got {} for n_rows={}".format(arr.shape, n_rows)
                        )
                    # Robust to object-dtype masks from different pandas behaviors.
                    return np.asarray([bool(x) for x in arr], dtype=np.bool_)

                n_before = len(d_set)
                multiple_targets_mask = _coerce_bool_mask(
                    d_set.apply(multiple_targets_utterance, axis=1), n_before
                )
                d_set = d_set.loc[multiple_targets_mask]
                d_set.reset_index(drop=True, inplace=True)
                print("length of dataset before removing non multiple test utterances {}".format(n_before))
                print(
                    "removed {} utterances from the test set that don't have multiple distractors".format(
                        int(np.count_nonzero(~multiple_targets_mask))
                    )
                )
                print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

                # Safety: avoid ambiguous truth value when the split is empty.
                if len(d_set) > 0:
                    remain_mask = _coerce_bool_mask(
                        d_set.apply(multiple_targets_utterance, axis=1), len(d_set)
                    )
                    assert int(np.count_nonzero(~remain_mask)) == 0

        feat_root = getattr(args, "mask3d_feature_root", None)
        if split == "test":
            feat_root = getattr(args, "mask3d_feature_root_test", None) or feat_root

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   lang_multilabel=args.lang_multilabel,
                                   multilabel_pretraining=args.multilabel_pretraining,
                                   cascading=args.cascading,
                                   order_len=args.order_len,
                                   mask3d_feature_root=feat_root)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
