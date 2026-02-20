import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform, shuffle
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from random import randrange
import json
import os
import csv

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

import torch

from baseline.dataset.datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
    CLASS_LABELS_200,
    CLASS_LABELS_20,
    VALID_CLASS_IDS_200,
)
from .utils import read_axis_align_matrix, concatenate_texts_with_separator

from .language_info import lang_info_data, grounding_data
from .data_aug import *


def _infer_repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in start.parents:
        if (p / "config.py").exists():
            return p
    return start.parents[0]


def _resolve_langdata_path(filename: str) -> Path:
    """
    Resolve `data/langdata/<filename>` robustly.

    The original code used cwd-relative paths like `./data/langdata/...`, which
    breaks when scripts are launched from a different working directory or when
    `data/` is mounted externally (e.g. via symlink on a server).
    """
    name = str(filename).strip()
    if not name:
        raise FileNotFoundError("empty langdata filename")

    env_root = (
        os.environ.get("SSR3DLLM_LANGDATA_ROOT", "").strip()
        or os.environ.get("GROUNDED3DLLM_LANGDATA_ROOT", "").strip()
    )
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    repo_root = _infer_repo_root()
    candidates.append(repo_root / "data" / "langdata")
    candidates.append(Path.cwd() / "data" / "langdata")

    tried: list[str] = []
    for root in candidates:
        try:
            p = root / name
            tried.append(str(p))
            if p.is_file():
                return p
        except Exception:
            continue

    raise FileNotFoundError(
        "Missing langdata json: "
        + name
        + "\nTried:\n  - "
        + "\n  - ".join(tried)
        + "\nTip: create a `data/langdata/` folder under the repo root (or symlink it), "
        + "or set `SSR3DLLM_LANGDATA_ROOT` to a directory containing these files."
    )


class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        task="instance_segmentation",
        filter_out_classes=[],
        label_offset=0,
        is_elastic_distortion=True,
        lang_query=False,
        positive_lang_query_ratio=0.5,
        lang_max_token_length=256,
        num_concat_texts=4,
        bert_path="./bert-base-uncased",
        lang_data_conf='',
        rel3d_max_per_scene=8,
        sample_class_labels=False,
        axis_align_coord=False,
        filter_scene00=False,
    ):
        assert task in [
            "instance_segmentation",
        ], "unknown task"

        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.sample_class_labels = sample_class_labels

        self.lang_query = lang_query
        self.positive_lang_query_ratio = positive_lang_query_ratio
        self.num_concat_texts = num_concat_texts
        self.axis_align_coord = axis_align_coord

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
            self.color_map[255] = (255, 255, 255)
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_raw_coordinates = add_raw_coordinates
        self.lang_data_conf = lang_data_conf
        self.filter_scene00 = filter_scene00
        self.rel3d_max_per_scene = rel3d_max_per_scene

        # e.g. REL3D_SEM_WEAKEN_ENABLE=1 REL3D_SEM_WEAKEN_PROB=0.3 REL3D_SEM_WEAKEN_TARGETS="scanrefer,m3dref"
        sem_weaken_enable = os.environ.get("REL3D_SEM_WEAKEN_ENABLE", "1").lower()
        self.semantic_weaken_enable = sem_weaken_enable in {"1", "true", "yes", "on"}
        self.semantic_weaken_prob = float(os.environ.get("REL3D_SEM_WEAKEN_PROB", "0.0"))
        self.semantic_weaken_targets = [
            t for t in os.environ.get("REL3D_SEM_WEAKEN_TARGETS", "scanrefer,m3dref").split(",") if t
        ]

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            db_yaml = database_path / f"{mode}_database.yaml"
            if not db_yaml.exists():
                raise FileNotFoundError(
                    "Missing dataset split database yaml.\n"
                    f"  required: {db_yaml}\n"
                    f"  dataset: {self.dataset_name}\n"
                    f"  mode: {mode}\n"
                    f"  data_dir: {database_path}\n"
                    f"  cwd: {Path.cwd()}\n"
                    "Expected split files under data_dir:\n"
                    "  - train_database.yaml\n"
                    "  - validation_database.yaml\n"
                    "  - (optional) test_database.yaml\n"
                    "Fix options:\n"
                    "  1) set GROUNDED3DLLM_DATA_ROOT/DEFAULT_DATA_ROOT to your dataset root\n"
                    "  2) or symlink data/processed/scannet200 -> $SCANNET200_ROOT"
                )
            self._data.extend(
                self._load_yaml(database_path / f"{mode}_database.yaml")
            )
        labels = self._load_yaml(Path(label_db_filepath))

        if self.filter_scene00:
            scanrefer_path = './data/langdata/scanrefer/ScanRefer_filtered_full_withroot_addeval.json'
            with open(scanrefer_path) as f:
                scanrefer_source = json.load(f)
            scanrefer_scene_ids = set(
                np.unique([i['scene_id'] for i in scanrefer_source]))

            self._data = [i for i in self._data if i['instance_gt_filepath'].split(
                '/')[-1][:-4] in scanrefer_scene_ids]

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            raise ValueError(
                "pass mean and std as tuple of tuples, or as an .yaml file"
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.scene_ids = set([self.data[i]['instance_gt_filepath'].split(
            '/')[-1][:-4] for i in range(len(self.data))])

        self.max_points_per_scene = int(os.environ.get("REL3D_MAX_POINTS_PER_SCENE", "0"))
        self._path_resolve_cache: dict[str, str] = {}

        self.lang_max_token_length = lang_max_token_length
        if self.num_concat_texts > 0:
            from transformers import AutoTokenizer, BertConfig
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_path, model_max_length=self.lang_max_token_length)

        self.relation_lang_dict = {}
        self.enable_relation_qa = os.environ.get("REL3D_ENABLE_RELATION_QA", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.has_relation_factory = hasattr(lang_info_data, "from_relation_sample")
        #   REL3D_REL_JSON_PATH=/ssd/.../rel3d_relations.json
        #
        #   - REL3D_REL_JSON_PATH_TRAIN / REL3D_REL_JSON_PATH_EVAL
        #   - REL3D_EXTRA_REL_JSON_PATH_TRAIN / REL3D_EXTRA_REL_JSON_PATH_EVAL
        rel_json_default = "datasets/qwen3_32b_gen/training_data_v1/scanrefer_scanqa_m3dref_embodied_plan_global_scene_grounded_scene.json"
        rel_json_fallback = os.environ.get("REL3D_REL_JSON_PATH", rel_json_default)
        if str(self.mode).lower() == "train":
            rel_json = os.environ.get("REL3D_REL_JSON_PATH_TRAIN", rel_json_fallback)
            extra_rel_json = os.environ.get(
                "REL3D_EXTRA_REL_JSON_PATH_TRAIN",
                os.environ.get("REL3D_EXTRA_REL_JSON_PATH", ""),
            )
        else:
            rel_json = os.environ.get("REL3D_REL_JSON_PATH_EVAL", rel_json_fallback)
            extra_rel_json = os.environ.get(
                "REL3D_EXTRA_REL_JSON_PATH_EVAL",
                os.environ.get("REL3D_EXTRA_REL_JSON_PATH", ""),
            )
        rel_anno_path = Path(rel_json)
        extra_path = Path(extra_rel_json) if extra_rel_json else None

        def _load_rel_json(path: Path) -> dict:
            # NOTE: cache must be split-aware; otherwise a train-created cache (filtered to train scene_ids)
            # will make validation/test think "no relations exist" and yield rel3dref_total=0.
            cache_path = path.with_suffix(f".{self.dataset_name}.{self.mode}.pkl")
            disable_cache = os.environ.get("REL3D_DISABLE_CACHE", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            rel_dict = {}
            if (not disable_cache) and cache_path.exists():
                try:
                    import pickle

                    with open(cache_path, "rb") as f:
                        rel_dict = pickle.load(f)
                except Exception:
                    rel_dict = {}
            if rel_dict:
                return rel_dict
            with open(path, "r") as f:
                entries = json.load(f)
            tmp = {}
            for item in entries:
                sid = item.get("scene_id")
                if sid not in self.scene_ids:
                    continue
                tmp.setdefault(sid, []).append(item)
            rel_dict = tmp
            if not disable_cache:
                try:
                    import pickle

                    with open(cache_path, "wb") as f:
                        pickle.dump(rel_dict, f)
                except Exception:
                    pass
            return rel_dict

        should_try_load_rel = (
            self.enable_relation_qa
            and self.has_relation_factory
            and (
                rel_anno_path.exists()
                or (extra_path is not None and extra_path.exists())
            )
        )

        if should_try_load_rel:
            merged = {}

            def _merge(d: dict) -> None:
                for sid, items in d.items():
                    merged.setdefault(sid, []).extend(items)

            loaded_paths = []
            if rel_anno_path.exists():
                loaded_paths.append(rel_anno_path)
                _merge(_load_rel_json(rel_anno_path))

            if extra_path is not None and extra_path.exists() and extra_path != rel_anno_path:
                loaded_paths.append(extra_path)
                _merge(_load_rel_json(extra_path))

            self.relation_lang_dict = merged

            # Debug-friendly summary (rank0 only): helps confirm train/val/test all see relations.
            try:
                rank = int(os.environ.get("RANK", "0"))
            except Exception:
                rank = 0
            if rank == 0:
                num_rels = sum(len(v) for v in self.relation_lang_dict.values())
                print(
                    f"[Rel3D][{self.mode}] Loaded relation annotations from "
                    f"{', '.join(str(p) for p in loaded_paths) if loaded_paths else '(none)'} "
                    f"for {len(self.relation_lang_dict)} scenes, entries={num_rels}."
                )
                # Debug: print per-source counts to verify merge works across datasets.
                try:
                    from collections import Counter

                    src_counter = Counter()
                    for items in self.relation_lang_dict.values():
                        for it in items:
                            src = it.get("source_dataset", "unknown")
                            src_counter[str(src)] += 1
                    print(f"[Rel3D][{self.mode}] Relation sources summary: {dict(src_counter)}")
                except Exception:
                    pass
                if not rel_anno_path.exists():
                    print(
                        f"[Rel3D][{self.mode}] NOTE: main rel JSON missing, "
                        f"using EXTRA only: {extra_path}"
                    )

            # Make smoke validation robust:
            # when `limit_val_batches` is small (e.g. 2), it's easy to pick only scenes
            # without any rel3d annotations and get `rel3dref_total=0` in eval logs.
            # Reorder non-train datasets so that scenes with rel3d entries appear first.
            if str(self.mode).lower() != "train" and self.relation_lang_dict:
                # Prefer scenes that have at least one "chain-definable" relation
                # (i.e., has anchor gt id). This makes `limit_val_batches` smoke tests
                # much less likely to report `rel3dref_total=0`.
                allowed_sources_raw = os.environ.get("SSR3DLLM_GEOM_REL_SOURCES", "").strip()
                allowed_sources = set()
                if allowed_sources_raw:
                    allowed_sources = {
                        s.strip().lower() for s in allowed_sources_raw.split(",") if s.strip()
                    }

                def _is_chain_rel(rel_item: dict) -> bool:
                    try:
                        if allowed_sources:
                            src = str(rel_item.get("source_dataset", "")).lower()
                            if src not in allowed_sources:
                                return False
                        if rel_item.get("anchor_id_source") != "direct_annotation":
                            return False
                        return isinstance(rel_item.get("anchor_object_gt_id", None), int)
                    except Exception:
                        return False

                chain_scene_set = {
                    sid for sid, items in self.relation_lang_dict.items()
                    if any(_is_chain_rel(it) for it in items)
                }
                rel_scene_set = set(self.relation_lang_dict.keys())

                def _scene_id_of(item: dict) -> str:
                    try:
                        return Path(str(item.get("instance_gt_filepath", ""))).stem
                    except Exception:
                        return ""

                try:
                    # Stable sort: chain scenes first, then any relation scenes, then the rest.
                    self._data.sort(
                        key=lambda it: (
                            0
                            if _scene_id_of(it) in chain_scene_set
                            else (1 if _scene_id_of(it) in rel_scene_set else 2)
                        )
                    )
                    if rank == 0:
                        print(
                            f"[Rel3D][{self.mode}] Reordered dataset to prioritize chain scenes "
                            f"(chain_scenes={len(chain_scene_set)}/{len(self._data)}, "
                            f"rel_scenes={len(rel_scene_set)}/{len(self._data)})."
                        )
                except Exception:
                    pass
        else:
            try:
                rank = int(os.environ.get("RANK", "0"))
            except Exception:
                rank = 0
            if rank == 0 and self.enable_relation_qa and self.has_relation_factory:
                print(
                    f"[Rel3D][{self.mode}] relation_qa enabled but missing data | "
                    f"rel_anno_exists={rel_anno_path.exists()} path={rel_anno_path} "
                    f"extra_exists={(extra_path.exists() if extra_path is not None else False)} "
                    f"extra_path={extra_path if extra_path is not None else ''}"
                )

        if self.dataset_name == 'scannet':
            self.dataset_class_labels = CLASS_LABELS_20
        elif self.dataset_name == 'scannet200':
            self.dataset_class_labels = CLASS_LABELS_200
        else:
            raise NotImplementedError

        assert 'noscanrefer' in lang_data_conf or 'scanrefer' in lang_data_conf
        # Use exact token membership instead of substring checks, so `noscanrefer`
        # won't accidentally enable ScanRefer loading.
        _lang_sources = []
        for k in lang_data_conf.split('+'):
            k = k.split(',')[0]
            assert k in ['scanrefer', 'm3dref', 'groundedscenecaption', 'scan2cap', 'scanqa', 'objdesc',
                         'scenedesc', '3dllm', 'alpaca', 'none', 'embodieddialog', 'embodiedplan', "globalscenecap", "noscanrefer",
                         "referit3d"]
            _lang_sources.append(k)
        lang_sources = set([k for k in _lang_sources if k])

        if self.lang_query > 0:
            self.multi_lang_source = []
            # Filter out degenerate referential grounding entries (missing text / missing target ids)
            # to avoid downstream "[grounding_steps] missing target_gt_id" skips during eval.
            def _valid_ref_entry(x):
                try:
                    if not isinstance(x.get('description', None), str) or not x['description'].strip():
                        return False
                    obj = x.get('object_ids', None)
                    if not isinstance(obj, list) or len(obj) == 0:
                        return False
                    if not isinstance(obj[0], list) or len(obj[0]) == 0:
                        return False
                    return True
                except Exception:
                    return False
            if 'scanrefer' in lang_sources:
                with open(_resolve_langdata_path('scanrefer_format.json')) as f:
                    scanrefer_source = json.load(f)
                scanrefer_source = [
                    i for i in scanrefer_source if i['scene_id'] in self.scene_ids]
                scanrefer_source = [i for i in scanrefer_source if _valid_ref_entry(i)]
                self.multi_lang_source.extend(scanrefer_source)
                print(
                    f'[{self.mode}] Added ScanRefer Database: {len(scanrefer_source)}')

            if 'referit3d' in lang_sources:
                def _load_referit3d_csv(path: str, *, tag: str) -> list:
                    p = Path(path)
                    cache_path = p.with_suffix(f".{self.dataset_name}.{self.mode}.{tag}.pkl")
                    disable_cache = os.environ.get("REFERIT3D_DISABLE_CACHE", "0").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    }
                    if (not disable_cache) and cache_path.exists():
                        try:
                            import pickle
                            with open(cache_path, "rb") as f:
                                return pickle.load(f)
                        except Exception:
                            pass

                    out = []
                    with open(path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            scene_id = str(row.get("scan_id", "")).strip()
                            if not scene_id or scene_id not in self.scene_ids:
                                continue
                            utterance = str(row.get("utterance", "")).strip()
                            if not utterance:
                                continue
                            try:
                                target_id = int(row.get("target_id"))
                            except Exception:
                                continue
                            instance_type = str(row.get("instance_type", "object")).strip() or "object"

                            # Best-effort phrase span: mark the first occurrence of instance_type;
                            # fall back to the full utterance to keep `all_phrases_positions` valid.
                            s = utterance.lower()
                            ptn = instance_type.lower().strip()
                            start = s.find(ptn) if ptn else -1
                            if start >= 0:
                                span = [int(start), int(start + len(ptn))]
                            else:
                                span = [0, int(len(utterance))]

                            ann_id = row.get("stimulus_id", None) or row.get("assignmentid", None) or ""
                            if ann_id is None:
                                ann_id = ""
                            out.append(
                                {
                                    "scene_id": scene_id,
                                    "object_name": instance_type,
                                    "ann_id": str(ann_id),
                                    "description": utterance,
                                    "all_phrases": [instance_type],
                                    "all_phrases_positions": [span],
                                    "eval_type": "single",
                                    "lang_type": f"referit3d:{tag}",
                                    "object_ids": [[target_id]],
                                }
                            )

                    if not disable_cache:
                        try:
                            import pickle
                            with open(cache_path, "wb") as f:
                                pickle.dump(out, f)
                        except Exception:
                            pass
                    return out

                nr3d_csv = os.environ.get("REFERIT3D_NR3D_CSV", "datasets/nr3d_sr3d/nr3d.csv")
                sr3d_csv = os.environ.get("REFERIT3D_SR3D_CSV", "datasets/nr3d_sr3d/sr3d.csv")
                referit_source = []
                if Path(nr3d_csv).exists():
                    referit_source.extend(_load_referit3d_csv(nr3d_csv, tag="nr3d"))
                if Path(sr3d_csv).exists():
                    referit_source.extend(_load_referit3d_csv(sr3d_csv, tag="sr3d"))
                referit_source = [i for i in referit_source if _valid_ref_entry(i)]
                self.multi_lang_source.extend(referit_source)
                print(f'[{self.mode}] Added ReferIt3D (nr3d+sr3d) Database: {len(referit_source)}')

            if 'm3dref' in lang_sources:
                with open(_resolve_langdata_path('m3dref_format.json')) as f:
                    m3dref_source = json.load(f)
                m3dref_source = [
                    i for i in m3dref_source if i['scene_id'] in self.scene_ids]
                # Multi3DRef json may contain entries with empty object_ids (≈10%).
                # These cannot provide target_gt_id and will be skipped by SSR3DLLM strict eval.
                m3dref_source = [i for i in m3dref_source if _valid_ref_entry(i)]
                self.multi_lang_source.extend(m3dref_source)
                print(
                    f'[{self.mode}] Added Multi3DRef Database: {len(m3dref_source)}')

            if 'groundedscenecaption' in lang_sources and self.mode == 'train':
                with open(_resolve_langdata_path('groundedscenecaption_format.json')) as f:
                    GroundedSceneCaption_source = json.load(f)
                GroundedSceneCaption_source = [
                    i for i in GroundedSceneCaption_source if i['scene_id'] in self.scene_ids]
                self.multi_lang_source.extend(GroundedSceneCaption_source)
                print(
                    f'[{self.mode}] Added Grounded Scene Caption Database: {len(GroundedSceneCaption_source)}')

            self.multi_lang_source = [
                i for i in self.multi_lang_source if i['scene_id'] in self.scene_ids]
            print(
                f'Total lang sources ({self.mode} mode): {len(self.multi_lang_source)}')
            print(
                '----------------------------------------------------------------------')

            # collect to dict
            self.multi_lang_dict = {}
            for i in self.multi_lang_source:
                if not i['scene_id'] in self.multi_lang_dict:
                    self.multi_lang_dict[i['scene_id']] = [i]
                else:
                    self.multi_lang_dict[i['scene_id']].append(i)
            assert set(self.multi_lang_dict.keys()).issubset(self.scene_ids)

            # ----------------------------- Instruction following data ------------------------
            instruction_following_sources = []

            if 'scanqa' in lang_sources:
                with open(_resolve_langdata_path('scanqa_format.json')) as f:
                    scanqa_lang_source = json.load(f)
                scanqa_lang_source = [
                    i for i in scanqa_lang_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added ScanQA Database: {len(scanqa_lang_source)}')
                instruction_following_sources.extend(scanqa_lang_source)

            if 'objdesc' in lang_sources:
                with open(_resolve_langdata_path('objectdescription_format.json')) as f:
                    objectdescription_source = json.load(f)
                objectdescription_source = [
                    i for i in objectdescription_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added Object Description dataset {len(objectdescription_source)}.')
                instruction_following_sources.extend(objectdescription_source)

            if 'scenedesc' in lang_sources:
                # load from grounded scene caption dataset
                with open(_resolve_langdata_path('groundedscenecaption_format.json')) as f:
                    scenedesc_source = json.load(f)

                # scenedesc_source = scene_description_v1 + scene_description_v2
                scenedesc_source = [
                    i for i in scenedesc_source if i['scene_id'] in self.scene_ids]
                for i, lang in enumerate(scenedesc_source):
                    qa_dict = dict(
                        scene_id=lang['scene_id'],
                        answer=lang['description'],
                        object_ids=lang['object_ids'],
                        all_phrases_positions=lang['all_phrases_positions'],
                        lang_type='scenedesc:v3',
                        # question is generated online
                    )
                    scenedesc_source[i] = qa_dict
                print(
                    f'[{self.mode}] Added Scene Description Database: {len(scenedesc_source)}.')
                instruction_following_sources.extend(scenedesc_source)

            if 'scan2cap' in lang_sources:
                with open(_resolve_langdata_path('scanrefer_format.json')) as f:
                    scan2cap_source = json.load(f)
                scan2cap_source = [
                    i for i in scan2cap_source if i['scene_id'] in self.scene_ids]

                for i, cap in enumerate(scan2cap_source):
                    scene_id = cap['scene_id']
                    cap['lang_type'] = 'scan2cap:' + cap['eval_type']
                    qa_dict = dict(
                        scene_id=cap['scene_id'],
                        answer=cap['description'],
                        object_ids=cap['object_ids'],
                        lang_type=cap['lang_type'],
                        all_phrases_positions=cap['all_phrases_positions']
                    )
                    scan2cap_source[i] = qa_dict

                print(
                    f'[{self.mode}] Added scan2cap(ScanRefer) Database: {len(scan2cap_source)}')
                instruction_following_sources.extend(scan2cap_source)

            if '3dllm' in lang_sources:
                with open(_resolve_langdata_path('3dllm_format.json')) as f:
                    data_3dllm_source = json.load(f)
                data_3dllm_source = [
                    i for i in data_3dllm_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added 3D LLM dataset {len(data_3dllm_source)}.')
                instruction_following_sources.extend(data_3dllm_source)

            if 'embodiedplan' in lang_sources:
                with open(_resolve_langdata_path('embodiedplan_format.json')) as f:
                    embodiedplan_source = json.load(f)
                embodiedplan_source = [
                    i for i in embodiedplan_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added Embodied Planning dataset {len(embodiedplan_source)}.')
                instruction_following_sources.extend(embodiedplan_source)

            if 'embodieddialog' in lang_sources:
                with open(_resolve_langdata_path('embodieddialog_format.json')) as f:
                    embodieddialog_source = json.load(f)
                embodieddialog_source = [
                    i for i in embodieddialog_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added Embodied Dialog dataset {len(embodieddialog_source)}.')
                instruction_following_sources.extend(embodieddialog_source)

            if 'globalscenecap' in lang_sources:
                with open(_resolve_langdata_path('global_scene_cap_format.json')) as f:
                    global_scene_caption_source = json.load(f)
                global_scene_caption_source = [
                    i for i in global_scene_caption_source if i['scene_id'] in self.scene_ids]
                print(
                    f'[{self.mode}] Added Global Caption dataset {len(global_scene_caption_source)}.')
                instruction_following_sources.extend(
                    global_scene_caption_source)

            self.instruction_lang_dict = {}
            for i in instruction_following_sources:
                if not i['scene_id'] in self.instruction_lang_dict:
                    self.instruction_lang_dict[i['scene_id']] = [i]
                else:
                    self.instruction_lang_dict[i['scene_id']].append(i)

            if len(instruction_following_sources) > 0:
                print(
                    f'Total Instruction QA sources ({self.mode} mode): {len(instruction_following_sources)}')
                print(
                    '----------------------------------------------------------------------')

        # sample numbers for each instruction dataset
        max_sample_lang_type_count = {
            'scanqa': 10,
            'objdesc': 10,
            'scenedesc': 0,
            'scan2cap': 10,
            '3dllm': 0,
            'embodiedplan': 0,
            'embodieddialog': 0,
            "globalscenecap": 0,
        }
        for k in lang_data_conf.split('+'):
            if ',' in k:
                lang_type, sample_num = k.split(',')
                max_sample_lang_type_count[lang_type] = int(sample_num)
        self.max_sample_lang_type_count = max_sample_lang_type_count

        # avoid empty training
        if 'nocls' in self.lang_data_conf and self.mode == 'train':
            self._data = [i for i in self._data if i['instance_gt_filepath'].split(
                '/')[-1][:-4] in self.multi_lang_dict]

        print('---------------------------------------------------------------------')
        print(f'{self.mode} scenes: {len(self._data)}')
        print('---------------------------------------------------------------------')

        # ------------------- Pure instruction following -------------------------
        self.alpaca_source = []
        if 'alpaca' in self.lang_data_conf and self.mode == 'train':
            with open(_resolve_langdata_path("alpaca_data.json"), 'r') as f:
                alpaca_source = json.load(f)
            print(f'[{self.mode}] Added Alpaca dataset {len(alpaca_source)}.')
            self.alpaca_source = alpaca_source

    def _weaken_description(self, text: str, object_name: Optional[str], all_phrases: Optional[List[str]]) -> str:
        """
        """
        placeholder_candidates = ["object", "item", "thing"]
        placeholder = placeholder_candidates[int(random() * len(placeholder_candidates))]

        if not isinstance(text, str) or not text:
            return text

        target_phrase = None
        if isinstance(object_name, str) and object_name.strip():
            target_phrase = object_name.strip()
        elif isinstance(all_phrases, list) and all_phrases:
            cand = all_phrases[0]
            if isinstance(cand, str) and cand.strip():
                target_phrase = cand.strip()

        if not target_phrase:
            return text

        lower_text = text.lower()
        lower_target = target_phrase.lower()
        idx = lower_text.find(lower_target)
        if idx == -1:
            return text

        return text[:idx] + placeholder + text[idx + len(target_phrase):]

    def __len__(self):
        return len(self.data)

    def _resolve_dataset_path(self, path: str | os.PathLike | None) -> str | None:
        """
        Resolve dataset file paths across machines.

        Background: scannet preprocessing YAMLs sometimes store absolute filepaths
        from a previous machine/repo clone. After moving the repo, those paths can
        break and DataLoader workers may crash with FileNotFoundError.

        This helper tries a few deterministic remaps:
        1) Use the original path if it exists.
        2) Remap paths containing `scannet_temp/` or `data/` to the *current* repo root.
        3) Remap by re-rooting at each configured `data_dir` (scannet200 root) using the
           suffix after the `scannet200/` segment.
        4) Optional explicit prefix rewrite via env vars SSR3DLLM_PATH_REMAP_FROM/TO.
        """
        if path is None:
            return None
        raw = str(path)
        if not raw:
            return None

        cache = getattr(self, "_path_resolve_cache", None)
        if isinstance(cache, dict) and raw in cache:
            return cache[raw]

        p = Path(raw)
        if p.exists():
            out = str(p)
            if isinstance(cache, dict):
                cache[raw] = out
            return out

        # Relative path: resolve against current working directory.
        if not p.is_absolute():
            cand = (Path.cwd() / p).resolve()
            if cand.exists():
                out = str(cand)
                if isinstance(cache, dict):
                    cache[raw] = out
                return out

        # Repo-relative remap: preserve suffix after scannet_temp/ or data/.
        try:
            repo_root = _infer_repo_root()
            for marker in ("scannet_temp", "data"):
                if marker in p.parts:
                    i = p.parts.index(marker)
                    cand = repo_root / marker / Path(*p.parts[i + 1 :])
                    if cand.exists():
                        out = str(cand)
                        if isinstance(cache, dict):
                            cache[raw] = out
                        return out
        except Exception:
            pass

        # Re-root at configured dataset roots by suffix after `scannet200/`.
        try:
            if "scannet200" in p.parts:
                i = p.parts.index("scannet200")
                suffix = Path(*p.parts[i + 1 :])
                for base in (self.data_dir or []):
                    base_p = Path(str(base)).resolve()
                    cand = base_p / suffix
                    if cand.exists():
                        out = str(cand)
                        if isinstance(cache, dict):
                            cache[raw] = out
                        return out
        except Exception:
            pass

        # Explicit prefix rewrite (last resort).
        remap_from = os.environ.get("SSR3DLLM_PATH_REMAP_FROM", "").strip()
        remap_to = os.environ.get("SSR3DLLM_PATH_REMAP_TO", "").strip()
        if remap_from and remap_to and raw.startswith(remap_from):
            cand = remap_to + raw[len(remap_from) :]
            if Path(cand).exists():
                if isinstance(cache, dict):
                    cache[raw] = cand
                return cand

        if isinstance(cache, dict):
            cache[raw] = raw
        return raw

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)

        filepath = self._resolve_dataset_path(self.data[idx].get("filepath"))
        if filepath is None:
            raise FileNotFoundError(f"[SemanticSegmentationDataset] missing filepath for idx={idx}")
        points = np.load(filepath)
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        inst_path = self._resolve_dataset_path(self.data[idx].get("instance_gt_filepath"))
        if inst_path:
            scene_id = Path(inst_path).stem
        else:
            # Fallback: infer from the points filename.
            scene_id = Path(filepath).stem

        if self.axis_align_coord:  # axis align matrix for detection boxes
            # ScanNet raw scan metadata file (contains `axisAlignment = ...`).
            # Default path matches historical repo layout; allow overriding via env
            # to support custom datasets locations (e.g., scannet_temp downloads).
            raw_scans_dir = os.environ.get("SSR3DLLM_RAWSCANNET_SCANS_DIR", "").strip()
            repo_root = _infer_repo_root()
            candidates = []
            if raw_scans_dir:
                candidates.append(Path(raw_scans_dir))
            candidates.extend([
                Path("./data/rawscannet/scans"),
                repo_root / "data" / "rawscannet" / "scans",
                repo_root / "scannet_temp" / "scannet" / "scans",
                repo_root / "scannet_temp" / "scannet" / "scans" / "scans",  # tolerate one extra level
            ])
            axis_align_path = None
            for base in candidates:
                try:
                    cand = (base / scene_id / f"{scene_id}.txt").resolve()
                except Exception:
                    continue
                if cand.exists():
                    axis_align_path = str(cand)
                    break
            if axis_align_path is None:
                tried = [str((Path(b) / scene_id / f"{scene_id}.txt")) for b in candidates]
                raise FileNotFoundError(
                    "[SemanticSegmentationDataset] axis alignment file not found. "
                    "Set SSR3DLLM_RAWSCANNET_SCANS_DIR to your ScanNet `scans/` directory. "
                    f"scene_id={scene_id} tried={tried}"
                )
            axis_align_matrix = read_axis_align_matrix(axis_align_path)
            assert np.all(np.fabs(axis_align_matrix[3, :3]) < 1e-8)
            # same to mesh.transform
            coordinates = coordinates @ axis_align_matrix[:3,
                                                          :3].T + axis_align_matrix[:3, 3:4].T

        coordinates -= coordinates.mean(0)

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates += (
                np.random.uniform(coordinates.min(0), coordinates.max(0))
                / 2
            )

            for i in (0, 1):  # flip x,y planes
                if np.random.rand() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            aug = self.volume_augmentations(  # scale, rotate the scene
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )

            if np.random.rand() < 0.95:
                if float(self.is_elastic_distortion) > 0.:
                    for granularity, magnitude in ((0.2, 0.4 * float(self.is_elastic_distortion)), (0.8, 1.6 * float(self.is_elastic_distortion))):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )

            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))
        # labels: [num_points, 3] # class, instance, segments

        extra_groundings = grounding_data()

        # concatenate detection labels to text
        if self.num_concat_texts > 0 and ((not 'nocls' in self.lang_data_conf) or (not self.mode == 'train')):
            if (not self.sample_class_labels or self.mode != 'train'):
                text_class_labels = list(deepcopy(self.dataset_class_labels))
                for cls_id, class_label in enumerate(text_class_labels):
                    if cls_id in self.filter_out_classes:
                        continue
                    extra_groundings.add_detection(class_label, gt_insts=np.unique(
                        labels[(labels[:, 0] == cls_id), 1]).tolist())
            else:
                text_class_labels = list(deepcopy(self.dataset_class_labels))

                positive_cls_id_sets = set(np.unique(labels[:, 0]))
                negative_cls_id_sets = np.asarray(
                    list(set(np.arange(len(self.dataset_class_labels))) - positive_cls_id_sets))
                np.random.shuffle(negative_cls_id_sets)
                negative_cls_id_sets = negative_cls_id_sets[:int(
                    len(positive_cls_id_sets) * (np.random.rand() * 2.))]

                # positive labels:
                for cls_id in positive_cls_id_sets:
                    if not (0 <= cls_id < len(text_class_labels)):
                        continue  # 255 / -1 ignore
                    if cls_id in self.filter_out_classes:
                        continue  # continue rather concat
                    class_label = text_class_labels[cls_id]

                    extra_groundings.add_detection(class_label, gt_insts=np.unique(
                        labels[(labels[:, 0] == cls_id), 1]).tolist())

                # negative labels:
                for cls_id in negative_cls_id_sets:
                    if not (0 <= cls_id < len(text_class_labels)):
                        continue  # 255 / -1 ignore
                    if cls_id in self.filter_out_classes:
                        continue  # continue rather concat
                    class_label = text_class_labels[cls_id]

                    extra_groundings.add_detection(class_label, gt_insts=[])

        if self.lang_query:
            if self.mode == 'train':
                positive_lang_query = min(int(self.lang_query * self.positive_lang_query_ratio), len(
                    self.multi_lang_dict[scene_id]) if scene_id in self.multi_lang_dict else 0)
                negative_lang_query = min(self.lang_query - positive_lang_query, int(
                    positive_lang_query * (1-self.positive_lang_query_ratio)))
            else:
                positive_lang_query = len(
                    self.multi_lang_dict[scene_id]) if scene_id in self.multi_lang_dict else 0
                negative_lang_query = 0  # avoid empty list

            pos_idx = []
            if scene_id in self.multi_lang_dict:  # if there are caption for scene_id
                pos_idx = np.arange(len(self.multi_lang_dict[scene_id]))
            if len(pos_idx) > 0 and self.mode == 'train':
                pos_idx = np.random.choice(
                    pos_idx, positive_lang_query, replace=False)
            for select_idx in pos_idx:
                lang_entry = self.multi_lang_dict[scene_id][select_idx]
                assert 'description' in lang_entry

                # filter out some ignore classes like wall, floor
                if lang_entry['lang_type'].split(':')[0] != 'groundedscenecaption':
                    # groundedscenecaption has filtered before
                    filter_out_flag = False
                    # all other sentence-level uses the same instances ids
                    for inst_id in lang_entry['object_ids'][0]:
                        inst_rows = labels[labels[:, 1] == inst_id]
                        if inst_rows.shape[0] == 0:
                            filter_out_flag = True
                            break
                        if inst_rows[0, 0] in self.filter_out_classes:
                            filter_out_flag = True
                            break
                        if inst_rows[0, 0] == self.ignore_label:
                            filter_out_flag = True
                            break
                    if filter_out_flag:
                        continue

                grounding_text = lang_entry['description']
                if self.mode == 'train' and self.semantic_weaken_enable and self.semantic_weaken_prob > 0.0:
                    base_type = lang_entry['lang_type'].split(':')[0]
                    if base_type in self.semantic_weaken_targets and random() < self.semantic_weaken_prob:
                        grounding_text = self._weaken_description(
                            grounding_text,
                            lang_entry.get('object_name'),
                            lang_entry.get('all_phrases'),
                        )

                extra_groundings.add_grounding(
                    grounding_text=grounding_text,
                    gt_insts=lang_entry['object_ids'],
                    positives=lang_entry['all_phrases_positions'],
                    grounding_type=lang_entry['lang_type']
                )

            # random sample negatives from left
            if negative_lang_query > 0 and len(self.multi_lang_source) > 0:
                neg_idx = []
                for select_idx in range(len(self.multi_lang_source)):
                    if self.multi_lang_source[select_idx]['scene_id'] == scene_id:
                        continue
                    if 'description' not in self.multi_lang_source[select_idx]:
                        continue
                    neg_idx.append(select_idx)
                neg_idx = np.asarray(neg_idx)
                neg_idx = np.random.choice(neg_idx, min(
                    negative_lang_query, len(neg_idx)), replace=False)
                for select_idx in neg_idx:
                    extra_groundings.add_grounding(
                        grounding_text=self.multi_lang_source[select_idx]['description'],
                        gt_insts=[
                            []] * len(self.multi_lang_source[select_idx]['all_phrases_positions']),
                        positives=self.multi_lang_source[select_idx]['all_phrases_positions'],
                        grounding_type=self.multi_lang_source[select_idx]['lang_type'],
                    )

            if self.mode == 'train':
                # When language sources are disabled (e.g. noscanrefer) and detection
                # concatenation is off, `extra_groundings` may be empty.
                if len(extra_groundings.types) > 1:
                    extra_groundings.shuffle_grounding()

        #
        if self.mode != 'train' and self.lang_query and len(extra_groundings.types) == 0:
            pass

        if self.num_concat_texts > 0:
            extra_groundings.concat_multi_grounding(
                tokenizer=self.tokenizer, max_batch_tokens=self.lang_max_token_length, max_tokens=min(
                    512, self.lang_max_token_length),
                num_concat_texts=self.num_concat_texts if self.mode == 'train' else 48,
            )

            if self.mode != 'train':
                if len(extra_groundings.concat_types) < len(extra_groundings.types):
                    print(
                        f'Some langauges are missing as the language clip (16 x 256) during eval: raw has {len(extra_groundings.types)} but get {len(extra_groundings.concat_types)}')

        # scene QA
        instruction_lang_info = []
        if self.lang_query and scene_id in self.instruction_lang_dict and ('scanqa' in self.lang_data_conf or
                                                                           'objdesc' in self.lang_data_conf or 'scenedesc' in self.lang_data_conf or 'scan2cap' in self.lang_data_conf):
            if self.mode == 'train':
                from utils.sample_utils import sample_by_type

                lang_type_with_index = np.asarray([(d['lang_type'].split(':')[0], i) for i, d in enumerate(
                    self.instruction_lang_dict[scene_id])], dtype=object)
                sampled_lang_type_with_index = sample_by_type(
                    lang_type_with_index, self.max_sample_lang_type_count)
                sampled_index = sampled_lang_type_with_index[:, 1]
            else:
                sampled_index = range(
                    len(self.instruction_lang_dict[scene_id]))

            for select_idx in sampled_index:
                instruction_item = self.instruction_lang_dict[scene_id][select_idx]

                if self.mode != 'train':
                    if np.random.rand() > 0.05:  # random select some for inference (No benchmark) to accelerate
                        if 'scenedesc' in instruction_item['lang_type'] or \
                            '3dllm' in instruction_item['lang_type'] or \
                            'embodieddialog' in instruction_item['lang_type'] or \
                            'embodiedplan' in instruction_item['lang_type'] or \
                            'globalscenecap' in instruction_item['lang_type']:
                            continue
                        
                instruction_lang_info.append(
                    lang_info_data.from_instruction_following(
                        instruction_item,
                        train_mode=(self.mode == 'train')
                    ))

        if (
            self.lang_query
            and self.enable_relation_qa
            and self.has_relation_factory
            and scene_id in self.relation_lang_dict
        ):
            rel_items = self.relation_lang_dict[scene_id]
            max_rel = int(self.rel3d_max_per_scene) if self.rel3d_max_per_scene is not None else 8
            if max_rel > 0 and len(rel_items) > max_rel:
                if self.mode == "train":
                    rel_indices = np.random.choice(len(rel_items), max_rel, replace=False)
                else:
                    # validation/test: deterministic sampling per scene to keep evaluation stable.
                    seed = (abs(hash(scene_id)) % (2**32))
                    rng = np.random.RandomState(seed)
                    rel_indices = rng.choice(len(rel_items), max_rel, replace=False)
            else:
                rel_indices = range(len(rel_items))
            for rid in rel_indices:
                rel_item = rel_items[rid]
                if not rel_item.get("is_positive", True):
                    continue
                try:
                    instruction_lang_info.append(
                        lang_info_data.from_relation_sample(rel_item)
                    )
                except Exception as exc:
                    print(f"[Rel3D] Skip relation sample for scene {scene_id} due to error: {exc}")

        # full text
        if self.mode == 'train' and self.max_sample_lang_type_count.get("alpaca", 0):
            alpaca_data_sampled = sample(
                self.alpaca_source, self.max_sample_lang_type_count.get("alpaca", 0))
            for instruction_item in alpaca_data_sampled:
                instruction_item['lang_type'] = 'alpaca'
                instruction_lang_info.append(lang_info_data.from_instruction_following(
                    instruction_item,
                ))

        # --------------- ASSERTATION --------------------
        for instruction_info in instruction_lang_info:
            assert len(instruction_info.inst_ids_answer) == len(
                instruction_info.positives_answer)
            assert len(instruction_info.inst_ids_question) == len(
                instruction_info.positives_question)
            # -------- print positives -------------
            # for beg, end in instruction_info.positives_question:
            #     if instruction_info.question[beg:end] not in ['object', 'objects']:
            #         print(instruction_info.question[beg:end])
            # for beg, end in instruction_info.positives_answer:
            #     if instruction_info.answer[beg:end] not in ['object', 'objects']:
            #         print(instruction_info.answer[beg:end])

        if self.mode == "train" and self.max_points_per_scene > 0:
            num_points = coordinates.shape[0]
            if num_points > self.max_points_per_scene:
                idx_keep = np.random.choice(num_points, self.max_points_per_scene, replace=False)
                coordinates = coordinates[idx_keep]
                color = color[idx_keep]
                labels = labels[idx_keep]
                if normals is not None:
                    normals = normals[idx_keep]

        features = color
        if self.add_normals and normals is not None:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        if self.data[idx]["raw_filepath"].split("/")[-2] in [
            "scene0636_00",
            "scene0154_00",
        ]:
            return self.__getitem__(0)

        return [
            coordinates,
            features,
            labels,
            self.data[idx]["raw_filepath"].split("/")[-2],
            raw_color,
            raw_normals,
            raw_coordinates,
            idx,
            extra_groundings,
            instruction_lang_info
        ]

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.safe_load(f)
        return file

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            if label not in self.color_map:
                print(
                    f'WARNING: Found label {label}, temperally changed it to 255')
                label = 255
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    # in ScanNet-200, label = label - 1:  0->255, 1->0, 2->1, 3->2
    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    # in ScanNet-200, label = label + 1: 0->1, 1->2, 2->3
    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped
