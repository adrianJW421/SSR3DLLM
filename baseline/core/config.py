# config.py

import os
import re
import torch
import yaml  # Requires PyYAML: pip install pyyaml
from pathlib import Path
from copy import deepcopy
from collections.abc import MutableMapping
from importlib import import_module

# --------------------------------------------------------------------------
# Default data root. Update this if your data is not under `data/` in the release repo.
# Example: DEFAULT_DATA_ROOT = "/mnt/datasets/Grounded3DLLM/data"
# --------------------------------------------------------------------------
def _infer_repo_root() -> Path:
        """
        Resolve the release repo root.

        This file lives under `baseline/core/`, so the expected root is two levels up.
        Keep a small fallback search to tolerate minor layout tweaks.
        """
        here = Path(__file__).resolve()
        expected = here.parents[2]
        if (expected / "baseline").is_dir() and (expected / "scripts").is_dir():
                return expected
        for parent in here.parents:
                if (parent / "baseline").is_dir() and (parent / "scripts").is_dir():
                        return parent
        return here.parent


REPO_ROOT = _infer_repo_root()
#
# NOTE:
# - Some environments do not have legacy external mount points, and package-name clashes may occur.
# - Keep path probing robust so utility scripts can run directly across machines.
#   1) Prefer GROUNDED3DLLM_DATA_ROOT / DEFAULT_DATA_ROOT when provided.
#   2) Otherwise try repo-local `data/` and `scannet_temp/`.
#   3) Avoid machine-specific absolute defaults in the release config.
_env_data_root = os.environ.get("GROUNDED3DLLM_DATA_ROOT", "").strip() or os.environ.get("DEFAULT_DATA_ROOT", "").strip()
_default_candidates = []
if _env_data_root:
        _default_candidates.append(Path(_env_data_root))
_default_candidates.append(REPO_ROOT / "data")
_default_candidates.append(REPO_ROOT / "scannet_temp")

# Pick the first candidate that looks like a valid root for processed/scannet200.
DEFAULT_DATA_ROOT = None
for _cand in _default_candidates:
        try:
                if (_cand / "processed" / "scannet200").exists() or (_cand / "scannet200").exists():
                        DEFAULT_DATA_ROOT = _cand
                        break
        except Exception:
                continue
if DEFAULT_DATA_ROOT is None:
        # Prefer an explicit env var, otherwise default to `<repo_root>/data`.
        # Do not error at import-time if the directory is missing; training/eval will
        # surface a clearer message when the dataset is actually accessed.
        DEFAULT_DATA_ROOT = Path(_env_data_root) if _env_data_root else (REPO_ROOT / "data")

# Support both layouts: prefer processed/scannet200, then fall back to scannet200.
_candidate_processed = DEFAULT_DATA_ROOT / "processed" / "scannet200"
if _candidate_processed.exists():
        SCANNET_PROCESSED_ROOT = _candidate_processed
else:
        fallback_processed = DEFAULT_DATA_ROOT / "scannet200"
        # Do not raise at import-time. Downstream dataset code should validate paths
        # when it actually needs to load data.
        SCANNET_PROCESSED_ROOT = fallback_processed


class AttrDict(MutableMapping):
        """
        Small dict wrapper with attribute access.
        Use either `cfg.foo` or `cfg["foo"]`.
        Nested dicts are wrapped automatically.
        """

        __slots__ = ("_storage",)

        def __init__(self, initial=None, **kwargs):
                object.__setattr__(self, "_storage", {})
                if initial:
                        for key, value in initial.items():
                                self[key] = value
                for key, value in kwargs.items():
                        self[key] = value

        def __getitem__(self, key):
                return self._storage[key]

        def __setitem__(self, key, value):
                self._storage[key] = self._wrap(value)

        def __delitem__(self, key):
                del self._storage[key]

        def __iter__(self):
                return iter(self._storage)

        def __len__(self):
                return len(self._storage)

        def __getattr__(self, key):
                if key.startswith("__") and key.endswith("__"):
                        raise AttributeError(key)
                if key in self._storage:
                        return self._storage[key]
                raise AttributeError(key)

        def __setattr__(self, key, value):
                if key.startswith("_"):
                        object.__setattr__(self, key, value)
                else:
                        self[key] = value

        def __contains__(self, key):
                return key in self._storage

        def get(self, key, default=None):
                return self._storage.get(key, default)

        def keys(self):
                return self._storage.keys()

        def items(self):
                return self._storage.items()

        def values(self):
                return self._storage.values()

        def copy(self):
                return AttrDict(self.to_dict())

        def to_dict(self):
                return {key: _convert_to_plain(value) for key, value in self._storage.items()}

        def _wrap(self, value):
                if isinstance(value, AttrDict):
                        return value
                if isinstance(value, MutableMapping):
                        return AttrDict(value)
                if isinstance(value, list):
                        return [self._wrap(v) for v in value]
                return value


def _convert_to_plain(obj):
        """
        Recursively convert config containers to native Python types.
        """
        if isinstance(obj, AttrDict):
                return {key: _convert_to_plain(val) for key, val in obj.items()}
        if isinstance(obj, MutableMapping):
                return {key: _convert_to_plain(val) for key, val in obj.items()}
        if isinstance(obj, list):
                return [_convert_to_plain(item) for item in obj]
        if isinstance(obj, torch.Tensor):
                return obj.clone()
        if hasattr(obj, "__dict__") and not isinstance(obj, (type, torch.dtype)):
                result = {}
                for key in dir(obj):
                        if key.startswith("__"):
                                continue
                        try:
                                value = getattr(obj, key)
                        except AttributeError:
                                continue
                        if callable(value):
                                continue
                        result[key] = _convert_to_plain(value)
                return result
        try:
                return deepcopy(obj)
        except Exception:
                return obj


def config_to_dict(config_obj):
        """Recursively convert a config object to a plain Python dict."""
        return _convert_to_plain(config_obj)


def instantiate(config_section, *args, recursive=True, **kwargs):
        """
        Minimal replacement for `hydra.utils.instantiate`.
        - Works with AttrDict or plain dict.
        - Can recursively instantiate nested `_target_` blocks.
        - Allows extra `*args` / `**kwargs` overrides.
        """
        if config_section is None:
                return None

        if isinstance(config_section, AttrDict):
                config_dict = config_section.to_dict()
        elif isinstance(config_section, MutableMapping):
                config_dict = {key: _convert_to_plain(value) for key, value in config_section.items()}
        else:
                config_dict = _convert_to_plain(config_section)
                if not isinstance(config_dict, dict) or "_target_" not in config_dict:
                        return config_dict

        if not isinstance(config_dict, dict):
                return config_dict

        recursive_flag = config_dict.pop("_recursive_", recursive)

        target = config_dict.pop("_target_", None)
        if target is None:
                raise ValueError("`_target_` field is required to instantiate an object.")

        def _resolve(value):
                if isinstance(value, dict):
                        if recursive_flag and "_target_" in value:
                                return instantiate(value)
                        return AttrDict({k: _resolve(v) for k, v in value.items()})
                if isinstance(value, list):
                        return [_resolve(item) for item in value]
                return value

        resolved_kwargs = {key: _resolve(val) for key, val in config_dict.items()}
        resolved_kwargs.update(kwargs)

        module_path, _, attribute = target.rpartition(".")
        if not module_path:
                raise ValueError(f"Invalid `_target_` path: {target}")
        module = import_module(module_path)
        callable_obj = getattr(module, attribute)

        return callable_obj(*args, **resolved_kwargs)


REFERENCE_PATTERN = re.compile(r"\${([^}]+)}")


def _ensure_attrdict(obj):
        if isinstance(obj, AttrDict):
                return obj
        if isinstance(obj, MutableMapping):
                return AttrDict(obj)
        return obj


def _get_value_by_path(cfg, path):
        current = cfg
        for key in path.split('.'):
                if isinstance(current, AttrDict):
                        current = current[key]
                elif isinstance(current, MutableMapping):
                        current = current[key]
                else:
                        current = getattr(current, key)
        return current


def _set_value_by_path(cfg, path, value):
        parts = path.split('.')
        current = cfg
        for key in parts[:-1]:
                if isinstance(current, AttrDict):
                        if key not in current:
                                current[key] = AttrDict()
                        current = current[key]
                elif isinstance(current, MutableMapping):
                        if key not in current:
                                current[key] = {}
                        current = current[key]
                else:
                        if not hasattr(current, key) or getattr(current, key) is None:
                                setattr(current, key, AttrDict())
                        current = getattr(current, key)
        last_key = parts[-1]
        if isinstance(current, AttrDict):
                current[last_key] = value
        elif isinstance(current, MutableMapping):
                current[last_key] = value
        else:
                setattr(current, last_key, value)


def apply_overrides(cfg, overrides):
        """
        Recursively merge overrides into cfg, including nested dicts and dotted keys.
        """
        if overrides is None:
                return cfg

        if not isinstance(overrides, MutableMapping):
                raise TypeError("Overrides must be a mapping.")

        for key, value in overrides.items():
                if isinstance(key, str) and '.' in key:
                        _set_value_by_path(cfg, key, value)
                        continue

                if isinstance(value, MutableMapping):
                        node = cfg.get(key, AttrDict())
                        node = _ensure_attrdict(node)
                        cfg[key] = node
                        apply_overrides(node, value)
                else:
                        cfg[key] = value
        return cfg


def _resolve_references(item, context):
        if isinstance(item, str):
                match = REFERENCE_PATTERN.fullmatch(item.strip())
                if match:
                        reference = match.group(1)
                        return deepcopy(_get_value_by_path(context, reference))
                return item
        if isinstance(item, list):
                return [_resolve_references(elem, context) for elem in item]
        if isinstance(item, MutableMapping):
                return {k: _resolve_references(v, context) for k, v in item.items()}
        return deepcopy(item)


def load_yaml_config(path, context=None):
        """
        Load YAML and resolve ${...} references.
        """
        with open(path, 'r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream) or {}
        context = context or task_config
        return _resolve_references(data, context)


def refresh_links(cfg):
        """
        Refresh dependent fields to preserve original Hydra-style link behavior.
        """
        cfg.general.save_dir = f"saved/{cfg.general.experiment_name}"
        cfg.data.task = cfg.general.task
        cfg.model.num_classes = cfg.general.num_targets
        cfg.model.train_on_segments = cfg.general.train_on_segments
        cfg.loss.num_classes = cfg.general.num_targets
        cfg.metrics.num_classes = cfg.data.num_labels
        cfg.metrics.ignore_label = cfg.data.ignore_label

        cfg.model.voxel_size = cfg.data.voxel_size

        feature_channels = 0
        if cfg.data.add_colors:
                feature_channels += 3
        if cfg.data.add_normals:
                feature_channels += 3
        if feature_channels == 0:
                feature_channels = cfg.data.in_channels or 1
        cfg.data.in_channels = feature_channels
        cfg.model.config['in_channels'] = feature_channels
        cfg.model.config.backbone['in_channels'] = feature_channels
        cfg.model.config.backbone['out_channels'] = cfg.data.num_labels
        cfg.model.lang_max_token_length = cfg.data.lang_max_token_length

        cfg.matcher.softmax_mode = cfg.model.softmax_mode
        cfg.matcher.language_mode = cfg.model.language_model
        if 'language_model' in cfg.matcher:
                del cfg.matcher['language_model']
        cfg.matcher.num_queries = cfg.model.num_queries
        cfg.loss.softmax_mode = cfg.model.softmax_mode
        cfg.loss.language_mode = cfg.model.language_model
        cfg.loss.num_points = cfg.matcher.num_points

        cfg.scheduler.scheduler['max_lr'] = cfg.optimizer.lr
        cfg.scheduler.scheduler['epochs'] = cfg.trainer.max_epochs

        if cfg.callbacks:
                cfg.callbacks[0]['dirpath'] = cfg.general.save_dir
        if cfg.logging:
                cfg.logging[0]['name'] = cfg.general.experiment_id
                cfg.logging[0]['version'] = cfg.general.version
                cfg.logging[0]['save_dir'] = cfg.general.save_dir

        cfg.data.train_dataloader['pin_memory'] = cfg.data.pin_memory
        cfg.data.train_dataloader['num_workers'] = cfg.data.num_workers
        cfg.data.train_dataloader['batch_size'] = cfg.data.batch_size

        cfg.data.validation_dataloader['pin_memory'] = cfg.data.pin_memory
        cfg.data.validation_dataloader['num_workers'] = cfg.data.num_workers
        cfg.data.validation_dataloader['batch_size'] = cfg.data.test_batch_size

        cfg.data.test_dataloader['pin_memory'] = cfg.data.pin_memory
        cfg.data.test_dataloader['num_workers'] = cfg.data.num_workers
        cfg.data.test_dataloader['batch_size'] = cfg.data.test_batch_size

        common_dataset_fields = {
                'ignore_label'             : cfg.data.ignore_label,
                'num_labels'               : cfg.data.num_labels,
                'add_raw_coordinates'      : cfg.data.add_raw_coordinates,
                'add_colors'               : cfg.data.add_colors,
                'add_normals'              : cfg.data.add_normals,
                'sample_class_labels'      : cfg.data.sample_class_labels,
                'axis_align_coord'         : cfg.model.axis_align_coord,
                'lang_data_conf'           : cfg.data.lang_data_conf,
                'lang_max_token_length'    : cfg.data.lang_max_token_length,
                'num_concat_texts'         : cfg.data.num_concat_texts,
                'bert_path'                : cfg.model.bert_path,
                'positive_lang_query_ratio': cfg.data.positive_lang_query_ratio,
                'lang_query'               : cfg.data.lang_query,
                'filter_scene00'           : cfg.general.filter_scene00,
        }
        # Optional field used by SSR3DLLM relation-QA injection.
        if 'rel3d_max_per_scene' in cfg.data:
                common_dataset_fields['rel3d_max_per_scene'] = cfg.data.rel3d_max_per_scene
        datasets = [
                cfg.data.train_dataset,
                cfg.data.validation_dataset,
                cfg.data.test_dataset,
        ]
        for dataset in datasets:
                for field, value in common_dataset_fields.items():
                        dataset[field] = value
        cfg.data.train_dataset.mode = cfg.data.train_mode
        cfg.data.validation_dataset.mode = cfg.data.validation_mode
        cfg.data.test_dataset.mode = cfg.data.test_mode

        common_collation_fields = {
                'ignore_label'          : cfg.data.ignore_label,
                'voxel_size'            : cfg.data.voxel_size,
                'task'                  : cfg.general.task,
                'ignore_class_threshold': cfg.general.ignore_class_threshold,
                'num_queries'           : cfg.model.num_queries,
                'bert_path'             : cfg.model.bert_path,
                'sample_class_labels'   : cfg.data.sample_class_labels,
        }
        collations = [
                cfg.data.train_collation,
                cfg.data.validation_collation,
                cfg.data.test_collation,
        ]
        for coll in collations:
                for field, value in common_collation_fields.items():
                        coll[field] = value

        cfg.data.train_collation.mode = cfg.data.train_mode
        cfg.data.train_collation.filter_out_classes = cfg.data.train_dataset.filter_out_classes
        cfg.data.train_collation.label_offset = cfg.data.train_dataset.label_offset

        cfg.data.validation_collation.mode = cfg.data.validation_mode
        cfg.data.validation_collation.filter_out_classes = cfg.data.validation_dataset.filter_out_classes
        cfg.data.validation_collation.label_offset = cfg.data.validation_dataset.label_offset

        cfg.data.test_collation.mode = cfg.data.test_mode
        cfg.data.test_collation.filter_out_classes = cfg.data.test_dataset.filter_out_classes
        cfg.data.test_collation.label_offset = cfg.data.test_dataset.label_offset


def clone_config():
        """
        Return a deep copy so edits do not mutate global task_config.
        """
        return AttrDict(config_to_dict(task_config))


class TaskConfig:
        """
        Main place for default training settings.
        The structure mirrors the old Hydra YAML layout under `conf/`.

        Structure:
        1. Nested classes define defaults.
        2. `task_config` is created at the bottom.
        3. Dependent fields are linked after construction.
        """

        # --------------------------------------------------------------------------
        # Nested class definitions: static defaults only.
        # --------------------------------------------------------------------------

        class GeneralConfig:
                train_mode: bool = True
                task: str = "instance_segmentation"
                seed: int = 1
                checkpoint: str = None
                train_on_segments: bool = True
                eval_on_segments: bool = True
                save_visualizations: bool = False
                visualization_point_size: int = 20
                decoder_id: int = -1
                export: bool = False
                use_dbscan: bool = False
                ignore_class_threshold: int = 100
                project_name: str = "scannet"
                experiment_name: str = "xxx"
                num_targets: int = 199
                dbscan_eps: float = 0.95
                dbscan_min_points: int = 1
                use_gt_proposals_for_llm: bool = False
                save_runtime_config: bool = True
                llm_config: str = "baseline/core/conf/llm/nollm.json"
                llm_data_config: str = "baseline/core/conf/llm/det10.json"
                filter_scene00: bool = False
                topk_per_image: int = 750
                # Filled in later during link resolution.
                save_dir: str = f"saved/xxx"
                experiment_id: str = "debug"
                version: int = 1
                gpus: int = 1
                timestamp: str = None

        class DataConfig:
                # Defaults from `conf/data/indoor.yaml`.
                train_mode: str = "train"
                validation_mode: str = "validation"
                test_mode: str = "validation"
                ignore_label: int = 255
                add_raw_coordinates: bool = True
                add_colors: bool = True
                add_normals: bool = False
                in_channels: int = 6  # 3 * (add_normals + add_colors + add_raw_coordinates)
                num_labels: int = 200
                pin_memory: bool = False
                num_workers: int = 4
                batch_size: int = 5
                test_batch_size: int = 1
                voxel_size: float = 0.02
                sample_class_labels: bool = False
                lang_data_conf: str = "scanrefer"
                num_concat_texts: int = 0
                lang_max_token_length: int = 0
                lang_query: int = 0
                positive_lang_query_ratio: float = 0.5
                # Filled from `general.task`.
                task: str = ""

                # Defaults from `conf/data/datasets/scannet200.yaml`.
                class TrainDatasetConfig:
                        _target_: str = "baseline.dataset.dataset_code.semseg.SemanticSegmentationDataset"
                        dataset_name: str = "scannet200"
                        data_dir: str = str(SCANNET_PROCESSED_ROOT)
                        image_augmentations_path: str = "baseline/core/conf/augmentation/albumentations_aug.yaml"
                        volume_augmentations_path: str = "baseline/core/conf/augmentation/volumentations_aug.yaml"
                        label_db_filepath: str = str(SCANNET_PROCESSED_ROOT / "label_database.yaml")
                        color_mean_std: str = str(SCANNET_PROCESSED_ROOT / "color_mean_std.yaml")
                        filter_out_classes: list = [0, 2]
                        label_offset: int = 2

                class ValidationDatasetConfig(TrainDatasetConfig):  # Inherits and overrides
                        image_augmentations_path: str = None
                        volume_augmentations_path: str = None

                class TestDatasetConfig(TrainDatasetConfig):  # Inherits and overrides
                        image_augmentations_path: str = None
                        volume_augmentations_path: str = None

                # Defaults from `conf/data/data_loaders/simple_loader.yaml`.
                class TrainDataloaderConfig:
                        _target_: str = "torch.utils.data.DataLoader"
                        shuffle: bool = True

                class ValidationDataloaderConfig:
                        _target_: str = "torch.utils.data.DataLoader"
                        shuffle: bool = False

                class TestDataloaderConfig:
                        _target_: str = "torch.utils.data.DataLoader"
                        shuffle: bool = False

                # Defaults from `conf/data/collation_functions/voxelize_collate.yaml`.
                class CollationConfig:
                        _target_: str = "baseline.dataset.dataset_code.utils.VoxelizeCollate"

        class ModelConfig:
                _target_: str = "models.Mask3D"
                _recursive_: bool = False
                hidden_dim: int = 128
                dim_feedforward: int = 1024
                num_queries: int = 100
                num_heads: int = 8
                num_decoders: int = 3
                dropout: float = 0.0
                pre_norm: bool = False
                use_level_embed: bool = False
                normalize_pos_enc: bool = True
                positional_encoding_type: str = "fourier"
                gauss_scale: float = 1.0
                hlevels: list = [0, 1, 2, 3]
                bert_path: str = "./pretrained/bert-base-uncased"
                axis_align_coord: bool = True
                language_model: bool = False
                softmax_mode: bool = True
                non_parametric_queries: bool = True
                random_normal: bool = False
                random_queries: bool = False
                sample_sizes: list = [200, 800, 3200, 12800, 51200]
                max_sample_size: bool = False
                shared_decoder: bool = True
                scatter_type: str = "mean"

                class BackboneConfig:
                        _target_: str = "models.Res16UNet34C"

                        class InnerConfig:
                                dialations: list = [1, 1, 1, 1]
                                conv1_kernel_size: int = 5
                                bn_momentum: float = 0.02

                        config = InnerConfig()
                        out_fpn: bool = True

                class ModelBackboneWrapper:
                        def __init__(self, backbone_cls):
                                self.backbone = backbone_cls()

                config = ModelBackboneWrapper(BackboneConfig)

        class OptimizerConfig:
                _target_: str = "torch.optim.AdamW"
                lr: float = 0.0001

        class SchedulerConfig:
                class SchedulerParams:
                        _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
                        steps_per_epoch: int = -1

                class PytorchLightningParams:
                        interval: str = "step"

                scheduler = SchedulerParams()
                pytorch_lightning_params = PytorchLightningParams()

        class MatcherConfig:
                _target_: str = "models.matcher.HungarianMatcher"
                cost_class: float = 2.0
                cost_mask: float = 5.0
                cost_dice: float = 2.0
                num_points: int = -1

        class LossConfig:
                _target_: str = "models.criterion.SetCriterion"
                eos_coef: float = 0.1
                losses: list = ["labels", "masks"]
                oversample_ratio: float = 3.0
                importance_sample_ratio: float = 0.75
                class_weights: int = -1

        class MetricsConfig:
                _target_: str = "models.metrics.ConfusionMatrix"
                num_classes: int = 200
                ignore_label: int = 255

        class TrainerConfig:
                max_epochs: int = 601
                min_epochs: int = 1
                deterministic: bool = False
                resume_from_checkpoint: str = None
                check_val_every_n_epoch: int = 50
                num_sanity_val_steps: int = 2

        callbacks = [
                {
                        '_target_'      : 'pytorch_lightning.callbacks.ModelCheckpoint',
                        'monitor'       : 'val_mean_ap_50',
                        'save_last'     : False,
                        'save_top_k'    : 0,
                        'mode'          : 'max',
                        'filename'      : "{epoch}-{val_mean_ap_50:.3f}",
                        'every_n_epochs': 1
                },
                {
                        '_target_': 'pytorch_lightning.callbacks.LearningRateMonitor'
                }
        ]

        logging = [
                {
                        '_target_': 'pytorch_lightning.loggers.TensorBoardLogger',
                }
        ]


# --------------------------------------------------------------------------
# Create global config instance
# --------------------------------------------------------------------------
task_config = TaskConfig()

# --------------------------------------------------------------------------
# Resolve instance links (replace `${...}` references)
# --------------------------------------------------------------------------

# Build nested config objects.
task_config.general = TaskConfig.GeneralConfig()
task_config.data = TaskConfig.DataConfig()
task_config.model = TaskConfig.ModelConfig()
task_config.optimizer = TaskConfig.OptimizerConfig()
task_config.scheduler = TaskConfig.SchedulerConfig()
task_config.matcher = TaskConfig.MatcherConfig()
task_config.loss = TaskConfig.LossConfig()
task_config.metrics = TaskConfig.MetricsConfig()
task_config.trainer = TaskConfig.TrainerConfig()

# Build nested objects under `data`.
task_config.data.train_dataset = TaskConfig.DataConfig.TrainDatasetConfig()
task_config.data.validation_dataset = TaskConfig.DataConfig.ValidationDatasetConfig()
task_config.data.test_dataset = TaskConfig.DataConfig.TestDatasetConfig()
task_config.data.train_dataloader = TaskConfig.DataConfig.TrainDataloaderConfig()
task_config.data.validation_dataloader = TaskConfig.DataConfig.ValidationDataloaderConfig()
task_config.data.test_dataloader = TaskConfig.DataConfig.TestDataloaderConfig()
task_config.data.train_collation = TaskConfig.DataConfig.CollationConfig()
task_config.data.validation_collation = TaskConfig.DataConfig.CollationConfig()
task_config.data.test_collation = TaskConfig.DataConfig.CollationConfig()

# Wire up dependent fields.

# Link general settings.
task_config.general.save_dir = f"saved/{task_config.general.experiment_name}"
task_config.data.task = task_config.general.task
task_config.model.num_classes = task_config.general.num_targets
task_config.model.train_on_segments = task_config.general.train_on_segments
task_config.loss.num_classes = task_config.general.num_targets

# Link data settings.
task_config.model.voxel_size = task_config.data.voxel_size
task_config.model.config.in_channels = task_config.data.in_channels
task_config.model.config.out_channels = task_config.data.num_labels
task_config.model.lang_max_token_length = task_config.data.lang_max_token_length

# Link model settings.
task_config.matcher.softmax_mode = task_config.model.softmax_mode
task_config.matcher.language_model = task_config.model.language_model
task_config.matcher.num_queries = task_config.model.num_queries
task_config.loss.softmax_mode = task_config.model.softmax_mode
task_config.loss.language_mode = task_config.model.language_model

# Link matcher settings.
task_config.loss.num_points = task_config.matcher.num_points

# Link optimizer and trainer settings.
task_config.scheduler.scheduler.max_lr = task_config.optimizer.lr
task_config.scheduler.scheduler.epochs = task_config.trainer.max_epochs

# Link save/log paths
task_config.callbacks[0]['dirpath'] = task_config.general.save_dir
task_config.logging[0]['name'] = task_config.general.experiment_id
task_config.logging[0]['version'] = task_config.general.version
task_config.logging[0]['save_dir'] = task_config.general.save_dir

# Fill dataloader options.
task_config.data.train_dataloader.pin_memory = task_config.data.pin_memory
task_config.data.train_dataloader.num_workers = task_config.data.num_workers
task_config.data.train_dataloader.batch_size = task_config.data.batch_size

task_config.data.validation_dataloader.pin_memory = task_config.data.pin_memory
task_config.data.validation_dataloader.num_workers = task_config.data.num_workers
task_config.data.validation_dataloader.batch_size = task_config.data.test_batch_size

task_config.data.test_dataloader.pin_memory = task_config.data.pin_memory
task_config.data.test_dataloader.num_workers = task_config.data.num_workers
task_config.data.test_dataloader.batch_size = task_config.data.test_batch_size

# Fill shared dataset/collation options.
_common_dataset_fields = {
        'ignore_label'         : task_config.data.ignore_label, 'num_labels': task_config.data.num_labels,
        'add_raw_coordinates'  : task_config.data.add_raw_coordinates, 'add_colors': task_config.data.add_colors,
        'add_normals'          : task_config.data.add_normals, 'sample_class_labels': task_config.data.sample_class_labels,
        'axis_align_coord'     : task_config.model.axis_align_coord, 'lang_data_conf': task_config.data.lang_data_conf,
        'lang_max_token_length': task_config.data.lang_max_token_length, 'num_concat_texts': task_config.data.num_concat_texts,
        'bert_path'            : task_config.model.bert_path, 'positive_lang_query_ratio': task_config.data.positive_lang_query_ratio,
        'lang_query'           : task_config.data.lang_query, 'filter_scene00': task_config.general.filter_scene00
}
_datasets_to_configure = [task_config.data.train_dataset, task_config.data.validation_dataset, task_config.data.test_dataset]
for _ds in _datasets_to_configure:
        for _field, _value in _common_dataset_fields.items():
                setattr(_ds, _field, _value)
task_config.data.train_dataset.mode = task_config.data.train_mode
task_config.data.validation_dataset.mode = task_config.data.validation_mode
task_config.data.test_dataset.mode = task_config.data.test_mode

_common_collation_fields = {
        'ignore_label'       : task_config.data.ignore_label, 'voxel_size': task_config.data.voxel_size,
        'task'               : task_config.general.task, 'ignore_class_threshold': task_config.general.ignore_class_threshold,
        'num_queries'        : task_config.model.num_queries, 'bert_path': task_config.model.bert_path,
        'sample_class_labels': task_config.data.sample_class_labels
}
_collations_to_configure = [task_config.data.train_collation, task_config.data.validation_collation, task_config.data.test_collation]
for _cl in _collations_to_configure:
        for _field, _value in _common_collation_fields.items():
                setattr(_cl, _field, _value)

task_config.data.train_collation.mode = task_config.data.train_mode
task_config.data.train_collation.filter_out_classes = task_config.data.train_dataset.filter_out_classes
task_config.data.train_collation.label_offset = task_config.data.train_dataset.label_offset

task_config.data.validation_collation.mode = task_config.data.validation_mode
task_config.data.validation_collation.filter_out_classes = task_config.data.validation_dataset.filter_out_classes
task_config.data.validation_collation.label_offset = task_config.data.validation_dataset.label_offset

task_config.data.test_collation.mode = task_config.data.test_mode
task_config.data.test_collation.filter_out_classes = task_config.data.test_dataset.filter_out_classes
task_config.data.test_collation.label_offset = task_config.data.test_dataset.label_offset

# --------------------------------------------------------------------------
# Convert config object to AttrDict for project-wide direct access
# --------------------------------------------------------------------------
task_config = AttrDict(config_to_dict(task_config))
refresh_links(task_config)

if __name__ == "__main__":
        """
        Run this script to validate generated config output.
        Steps:
        1. Run verify_hydra.py in the original project and save output:
           python verify_hydra.py > hydra_config_output.yaml
        2. Run this script and save output:
           python config.py > my_config_output.yaml
        3. Compare the two files with diff:
           diff hydra_config_output.yaml my_config_output.yaml
        """

        # Create a deep copy before conversion because some objects are shared by reference.
        config_instance_for_print = deepcopy(task_config)

        refresh_links(config_instance_for_print)

        # Convert dict-like entries in lists to plain dict.
        config_instance_for_print.callbacks = [config_to_dict(cb) if not isinstance(cb, dict) else cb for cb in config_instance_for_print.callbacks]
        config_instance_for_print.logging = [config_to_dict(lg) if not isinstance(lg, dict) else lg for lg in config_instance_for_print.logging]

        # Convert the whole config object to a plain dict.
        config_dict = config_to_dict(config_instance_for_print)

        # Remove Hydra-specific entries absent in this release config.
        if 'hydra' in config_dict:
                del config_dict['hydra']

        # Print as YAML output.
        print("# This is the configuration generated from config.py")
        print(yaml.dump(config_dict, sort_keys=False, default_flow_style=False))
