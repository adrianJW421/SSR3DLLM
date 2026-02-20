#!/usr/bin/env python3

"""
Baseline inference interface for Grounded 3D-LLM.

This module mirrors the official evaluation pipeline used in
`baseline/core/trainer/trainer.py` so that predictions obtained here match the
results produced by the project's Lightning evaluation loops.

Usage example
-------------

```python
from baseline.api.baseline_interface import Grounded3DLLMBaselineInterface

interface = Grounded3DLLMBaselineInterface(
    checkpoint="saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt",
    data_split="validation",
)

prediction = interface.predict_scene("scene0000_00")
print(prediction.keys())
```
"""

from __future__ import annotations

import copy
import gc
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
from pdb import set_trace as stop

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

from config import (
        clone_config,
        apply_overrides,
        load_yaml_config,
        refresh_links,
        instantiate,
)
from models.matcher import HungarianMatcher
from baseline.core.trainer.trainer import ModelingGrounded3DLLM
from utils.utils import load_checkpoint_with_missing_or_exsessive_keys
from models.LLM.llama_utils import grounded_3d_llm_data, extract_decoder_hidden_states
from transformers import LlamaForCausalLM

LOGGER = logging.getLogger(__name__)


class Grounded3DLLMBaselineInterface:
        """
        Lightweight wrapper around the official trainer that exposes
        deterministic, per-scene inference APIs.
        """

        def __init__(
                        self,
                        checkpoint: Optional[str],
                        data_split: str = "validation",
                        data_config: str = "baseline/core/conf/data/indoor_dialog.yaml",
                        model_config: str = "baseline/core/conf/model/mask3d_lang.yaml",
                        trainer_config: Optional[str] = "baseline/core/conf/trainer/trainer50.yaml",
                        experiment_name: str = "baseline_interface",
                        project_name: str = "scannet200",
                        device: Optional[str] = None,
                        extra_overrides: Optional[Dict] = None,
                        llm_config: Optional[str] = "baseline/core/conf/llm/tiny_vicuna_len512_bs4.json",
                        llm_data_config: Optional[str] = "baseline/core/conf/llm/det10.json",
                        topk_per_image: Optional[int] = 750,
        ) -> None:
                """
                Parameters
                ----------
                checkpoint:
                    Path to the `.ckpt` file to load. If ``None`` the model runs with
                    random weights (useful for debugging only).
                data_split:
                    Which split to use. Must be one of ``{"train", "validation", "test"}``.
                data_config:
                    YAML file describing the dataset configuration (matches the hydra
                    config that used to live in ``conf/data``).
                model_config:
                    YAML file describing the model backbone / decoder configuration.
                trainer_config:
                    Optional YAML file with trainer hyper-parameters. If ``None`` the
                    defaults from ``config.py`` are used.
                experiment_name:
                    Experiment name that controls where intermediate artifacts are
                    written (under ``saved/<experiment_name>``).
                project_name:
                    High level project grouping, kept for parity with the original
                    configuration.
                device:
                    Torch device string. Defaults to ``"cuda"`` when available.
                extra_overrides:
                    Optional nested dictionary passed to :func:`apply_overrides` after
                    the standard overrides are applied. This mirrors Hydra CLI overrides.
                """

                if data_split not in {"train", "validation", "test"}:
                        raise ValueError(f"Unsupported data split: {data_split}")

                self._device = torch.device(
                        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
                )

                cfg = clone_config()

                # ------------------------------------------------------------------
                # Apply base overrides so the config matches our inference scenario.
                # ------------------------------------------------------------------
                general_overrides = {
                        "train_mode"         : False,
                        "gpus"               : 1 if self._device.type == "cuda" else 0,
                        "experiment_name"    : experiment_name,
                        "project_name"       : project_name,
                        "checkpoint"         : str(checkpoint) if checkpoint is not None else None,
                        "save_visualizations": False,
                        "use_dbscan"         : False,
                }
                # Optional override to avoid writing into a broken/remote `saved/` symlink.
                # This is especially useful for offline feature export on clusters.
                save_dir_env = os.environ.get("GROUNDED3DLLM_SAVE_DIR", "").strip()
                if save_dir_env:
                        general_overrides["save_dir"] = save_dir_env
                self._user_general_overrides = {
                        key: value for key, value in general_overrides.items() if value is not None
                }
                if llm_config is not None:
                        general_overrides["llm_config"] = llm_config
                if llm_data_config is not None:
                        general_overrides["llm_data_config"] = llm_data_config
                if topk_per_image is not None:
                        general_overrides["topk_per_image"] = topk_per_image

                base_overrides = {
                        "general": general_overrides,
                }
                apply_overrides(cfg, base_overrides)

                if data_config:
                        apply_overrides(cfg.data, load_yaml_config(data_config, context=cfg))
                if model_config:
                        apply_overrides(cfg.model, load_yaml_config(model_config, context=cfg))
                if trainer_config:
                        apply_overrides(cfg.trainer, load_yaml_config(trainer_config, context=cfg))

                if extra_overrides:
                        apply_overrides(cfg, extra_overrides)

                # Ensure `data.test_mode` matches the requested split.
                # Some data YAMLs set a default (e.g. validation) which would otherwise
                # override the split requested by export/eval scripts.
                apply_overrides(cfg, {"data": {"test_mode": data_split}})
                refresh_links(cfg)
                self.config = cfg

                # ------------------------------------------------------------------
                # Instantiate model + load checkpoint on the requested device.
                # ------------------------------------------------------------------
                self.model = ModelingGrounded3DLLM(cfg).to(self._device)
                self.model.eval()

                if cfg.general.checkpoint:
                        _, self.model = load_checkpoint_with_missing_or_exsessive_keys(cfg, self.model)
                        self.model.eval()
                        self._enforce_general_overrides()

                # ------------------------------------------------------------------
                # Select dataset / collate objects aligned with the requested split.
                # ------------------------------------------------------------------
                if data_split == "train":
                        # In inference mode we may not instantiate `train_dataset` to keep
                        # memory bounded. In that case, `validation_dataset` is typically an
                        # alias to the instantiated dataset (controlled by `data.test_mode`).
                        if self.model.train_dataset is None:
                                LOGGER.warning(
                                        "train_dataset is not instantiated (train_mode=%s); "
                                        "falling back to validation_dataset (test_mode=%s).",
                                        getattr(self.config.general, "train_mode", None),
                                        getattr(self.config.data, "test_mode", None),
                                )
                                self.dataset = self.model.validation_dataset
                                self._collate_fn = instantiate(cfg.data.validation_collation)
                        else:
                                self.dataset = self.model.train_dataset
                                self._collate_fn = instantiate(cfg.data.train_collation)
                elif data_split == "validation":
                        self.dataset = self.model.validation_dataset
                        self._collate_fn = instantiate(cfg.data.validation_collation)
                else:  # "test"
                        self.dataset = self.model.test_dataset
                        self._collate_fn = instantiate(cfg.data.test_collation)

                if self.dataset is None:
                        raise RuntimeError(f"Dataset for split '{data_split}' could not be instantiated.")

                self.scene_index = self._build_scene_index(self.dataset.data)
                LOGGER.info(
                        "Interface initialised | split=%s | scenes=%d | device=%s",
                        data_split,
                        len(self.scene_index),
                        self._device,
                )

        # ----------------------------------------------------------------------
        # Public API
        # ----------------------------------------------------------------------
        @torch.no_grad()
        def predict_scene(self, scene_id: str) -> Dict[str, object]:
                """
                Run a forward pass for a single scene and return the predictions.

                The returned dictionary matches the structure produced during the
                official evaluation loop (see ``baseline/core/trainer/trainer.py``).
                """
                indices = self.scene_index.get(scene_id)
                if not indices:
                        raise KeyError(f"Scene '{scene_id}' not present in the {self.dataset.mode} split.")

                # The dataset should only contain one entry per scene for validation/test.
                idx = indices[0]
                sample = self.dataset[idx]
                batch = self._collate_fn([sample])
                raw_data, target, file_names = batch

                target = self._move_target_to_device(target, self._device)
                batch = (raw_data, target, file_names)

                file_name = file_names[0]
                # Clear previous predictions to avoid mixing results from multiple scenes.
                self.model.preds = {}
                self.model.bbox_preds = {}
                self.model.bbox_gt = {}

                try:
                        _ = self.model.eval_step(batch, batch_idx=0)
                except RuntimeError as exc:
                        if "NO_LANGUAGE_QUERIES" in str(exc):
                                raise RuntimeError(f"NO_LANGUAGE_QUERIES::{scene_id}")
                        raise

                if file_name not in self.model.preds:
                        raise RuntimeError(
                                f"Model did not populate predictions for scene '{file_name}'. "
                                "Please ensure the checkpoint and configuration are valid."
                        )

                prediction = dict(self.model.preds[file_name])  # shallow copy
                prediction["file_name"] = file_name

                if file_name in self.model.bbox_preds:
                        prediction["bbox_preds"] = self.model.bbox_preds[file_name]

                return prediction

        @torch.no_grad()
        def batched_predict(self, scene_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
                """
                Convenience wrapper to evaluate multiple scenes sequentially.
                """
                outputs: Dict[str, Dict[str, object]] = {}
                for scene_id in scene_ids:
                        outputs[scene_id] = self.predict_scene(scene_id)
                return outputs

        def collect_metrics(self) -> Optional[Dict[str, object]]:
                """
                Pull the metrics accumulated by the underlying LightningModule.
                Mirrors what `test_epoch_end` prints/logs during the official run.
                """
                try:
                        if getattr(self.model, "preds", None):
                                ap_results = self.model.eval_instance_epoch_end(
                                        self.model.preds,
                                        getattr(self.model, "bbox_preds", {}),
                                        getattr(self.model, "bbox_gt", {}),
                                )
                                return {k: v for k, v in ap_results.items() if k.startswith("val_mean")}
                except Exception as exc:  # pragma: no cover
                        LOGGER.warning("Failed to collect metrics from interface run: %s", exc)
                return None

        # ----------------------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------------------
        @staticmethod
        def _build_scene_index(dataset_entries: List[Dict]) -> Dict[str, List[int]]:
                """
                Build a map from scene_id to dataset indices for fast lookups.
                """
                index: Dict[str, List[int]] = {}
                for idx, entry in enumerate(dataset_entries):
                        scene = Path(entry["instance_gt_filepath"]).stem
                        index.setdefault(scene, []).append(idx)
                return index

        def _enforce_general_overrides(self) -> None:
                if not hasattr(self, "_user_general_overrides"):
                        return
                for key, value in self._user_general_overrides.items():
                        setattr(self.config.general, key, value)
                refresh_links(self.config)

        @staticmethod
        def _move_target_to_device(
                        target: Sequence[Dict[str, object]],
                        device: torch.device,
        ) -> List[Dict[str, object]]:
                def _move(obj: object) -> object:
                        if isinstance(obj, torch.Tensor):
                                return obj.to(device=device, non_blocking=True)
                        if isinstance(obj, list):
                                return [_move(item) for item in obj]
                        if isinstance(obj, tuple):
                                return tuple(_move(item) for item in obj)
                        if isinstance(obj, dict):
                                return {k: _move(v) for k, v in obj.items()}
                        return obj

                return [_move(entry) for entry in target]


class NpyColumnIndex:
        """Utility describing the column layout of the ScanNet `.npy` dumps."""

        COORDS = slice(0, 3)
        COLORS = slice(3, 6)
        NORMALS = slice(6, 9)
        SEGMENT_ID = 9
        SEMANTIC_LABEL = 10
        INSTANCE_LABEL = 11


class BaselineModelAPI:
        """
        Compatibility layer that exposes the historical `BaselineModelAPI`
        functionality on top of :class:`Grounded3DLLMBaselineInterface`.

        The training / evaluation utilities in the original research repo (not shipped in this release)
        were written against the original API.  Rather than rewriting every caller
        we emulate the previous surface area (scene feature extraction, GT ->
        query-id mapping, prompt -> Top-K matching) while delegating model
        initialisation and config handling to the new interface.
        """

        def __init__(self, config: Dict[str, Any]):
                self.config = config

                # ------------------------------------------------------------------
                # Resolve paths / defaults
                # ------------------------------------------------------------------
                self.scannet_root = Path(
                        config.get("scannet_processed_root", "datasets/scannet200")
                ).expanduser()
                requested_split = config.get("split_type", "validation")
                split_aliases = {
                        "train"     : "train",
                        "training"  : "train",
                        "val"       : "validation",
                        "valid"     : "validation",
                        "validation": "validation",
                        "test"      : "test",
                        "testing"   : "test",
                }
                normalized_split = split_aliases.get(str(requested_split).lower())
                if normalized_split is None:
                        raise ValueError(f"Unsupported data split: {requested_split}")
                self.split_type = normalized_split

                hydra_root = Path(config.get("config_path", "baseline/core/conf")).expanduser()
                def_path = lambda rel: hydra_root / rel
                checkpoint = config.get("checkpoint")
                data_config = config.get("data_config", def_path("data/indoor_dialog.yaml"))
                model_config = config.get("model_config", def_path("model/mask3d_lang.yaml"))
                trainer_config = config.get("trainer_config", def_path("trainer/trainer50.yaml"))
                llm_config = config.get("llm_config", def_path("llm/tiny_vicuna_len512_bs4.json"))
                llm_data_config = config.get("llm_data_config", def_path("llm/det10.json"))

                experiment_name = config.get("experiment_name", "baseline_api")
                project_name = config.get("project_name", "scannet200")
                topk_per_image = config.get("topk_per_image", 750)
                extra_overrides = config.get("extra_overrides", None)

                # ------------------------------------------------------------------
                # Bootstrap the official interface and grab the underlying trainer.
                # ------------------------------------------------------------------
                self.interface = Grounded3DLLMBaselineInterface(
                        checkpoint=checkpoint,
                        data_split=self.split_type,
                        data_config=str(data_config),
                        model_config=str(model_config),
                        trainer_config=str(trainer_config),
                        experiment_name=experiment_name,
                        project_name=project_name,
                        device=config.get("device"),
                        extra_overrides=extra_overrides,
                        llm_config=str(llm_config),
                        llm_data_config=str(llm_data_config),
                        topk_per_image=topk_per_image,
                )

                self.trainer = self.interface.model
                self.detector = self.trainer.model
                if getattr(self.detector, "train_on_segments", False):
                        LOGGER.warning("Disabling train_on_segments for standalone inference.")
                        self.detector.train_on_segments = False
                self.llama_model = getattr(self.trainer, "llama_model", None)
                self.llama_tokenizer = getattr(self.trainer, "llama_tokenizer", None)
                (
                        self._llama_for_causal_lm,
                        self._llama_core,
                        self._llama_prompts_owner,
                ) = self._resolve_llama_components()
                self.device = self.interface._device
                self._cpu_detector: Optional[torch.nn.Module] = None

                self._voxel_size = float(self.interface.config.model.voxel_size)

                # Internal caches to avoid recomputing heavy forwards.
                self._scene_forward_cache: Dict[str, Dict[str, Any]] = {}
                def _to_bool(val: Any) -> bool:
                        if isinstance(val, bool):
                                return val
                        if isinstance(val, (int, float)):
                                return bool(val)
                        if isinstance(val, str):
                                return val not in {"0", "false", "False", "", "no", "No"}
                        return False
                prefer_semantic_ref = config.get("prefer_semantic_ref", None)
                if prefer_semantic_ref is None:
                        prefer_semantic_ref = os.environ.get("BASELINE_PREFER_SEM_REF", "0")
                self._prefer_semantic_ref = _to_bool(prefer_semantic_ref)
                LOGGER.info(
                        "BaselineModelAPI init | split=%s | llama_model=%s | llama_core=%s | has_instance2embed=%s",
                        self.split_type,
                        type(self.llama_model).__name__ if self.llama_model else None,
                        type(self._llama_core).__name__ if self._llama_core else None,
                        bool(self._llama_core and hasattr(self._llama_core, "instance2embed")),
                )

        def _resolve_llama_components(self):
                if self.llama_model is None:
                        return None, None, None

                llama_for_causal_lm = None
                prompts_owner = None
                llama_core = None

                base_wrapper = getattr(self.llama_model, "base_model", None)
                if base_wrapper is not None:
                        llama_for_causal_lm = getattr(base_wrapper, "model", None)

                if llama_for_causal_lm is None:
                        llama_for_causal_lm = getattr(self.llama_model, "model", None)

                if llama_for_causal_lm is not None:
                        core_candidate = getattr(llama_for_causal_lm, "model", None)
                        if hasattr(core_candidate, "instance2embed"):
                                llama_core = core_candidate
                        elif core_candidate is not None and hasattr(core_candidate, "model"):
                                inner_candidate = getattr(core_candidate, "model", None)
                                if hasattr(inner_candidate, "instance2embed"):
                                        llama_core = inner_candidate
                        if hasattr(llama_for_causal_lm, "prompts"):
                                prompts_owner = llama_for_causal_lm

                if llama_core is None:
                        direct_candidate = getattr(self.llama_model, "model", None)
                        if hasattr(direct_candidate, "instance2embed"):
                                llama_core = direct_candidate
                        elif direct_candidate is not None and hasattr(direct_candidate, "model"):
                                inner_candidate = getattr(direct_candidate, "model", None)
                                if hasattr(inner_candidate, "instance2embed"):
                                        llama_core = inner_candidate

                if prompts_owner is None:
                        if hasattr(self.llama_model, "prompts"):
                                prompts_owner = self.llama_model
                        elif hasattr(base_wrapper, "prompts"):
                                prompts_owner = base_wrapper

                return llama_for_causal_lm, llama_core, prompts_owner

        def _get_detector_for_device(self, device: torch.device) -> torch.nn.Module:
                if device.type == self.device.type:
                        return self.detector
                if device.type == "cpu":
                        if self._cpu_detector is None:
                                LOGGER.warning(
                                        "Initialising CPU fallback detector – inference will be slow but ensures coverage when GPU SparseTensor fails."
                                )
                                self._cpu_detector = copy.deepcopy(self.detector).to(device)
                                self._cpu_detector.eval()
                        return self._cpu_detector
                raise ValueError(f"Unsupported fallback device requested: {device}")

        # ------------------------------------------------------------------
        # Public helpers used by the training scripts
        # ------------------------------------------------------------------
        def get_best_match_for_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
                """
                Compute the GT -> query mapping for ``scene_id`` and return the
                associated visual features (object queries + sampled coordinates).
                """
                forward = self._run_full_forward(scene_id)
                if forward is None:
                        return None

                gt_map = forward.get("gt_to_query_map")
                if not gt_map:
                        return None

                return {
                        "gt_to_query_map"    : gt_map,
                        "object_queries"     : forward["object_queries"].cpu(),
                        "sampled_coords"     : forward["sampled_coords"].cpu()
                        if isinstance(forward["sampled_coords"], torch.Tensor)
                        else forward["sampled_coords"],
                        "pred_classes"       : forward["pred_classes"].cpu(),
                        "pred_scores"        : forward["pred_scores"].cpu(),
                        "gt_instance_classes": forward.get("gt_instance_classes"),
                }

        def get_scene_data_for_verification(self, scene_id: str) -> Optional[Dict[str, Any]]:
                """
                Return all tensors required by verification / reranking scripts:
                object queries, sampled coordinates and full-resolution prediction masks.
                """
                forward = self._run_full_forward(scene_id)
                if forward is None:
                        return None

                return {
                        "object_queries": forward["object_queries"].cpu(),
                        "sampled_coords": forward["sampled_coords"].cpu()
                        if isinstance(forward["sampled_coords"], torch.Tensor)
                        else forward["sampled_coords"],
                        "pred_masks"    : forward["pred_masks_full_res"],
                        "pred_classes"  : forward["pred_classes"].cpu(),
                        "pred_scores"   : forward["pred_scores"].cpu(),
                }

        # The light-weight prompt matching helpers are almost identical to the
        # historical implementation and therefore kept verbatim.
        @torch.no_grad()
        def get_topk_query_ids_for_prompt(
                        self, scene_id: str, prompt: str, k: int = 5
        ) -> Optional[List[Tuple[int, float]]]:
                features = self._get_vision_features(scene_id)
                if not features:
                        return None
                return self.get_topk_query_ids_for_prompt_light(prompt, features["object_queries"], k)

        @torch.no_grad()
        def get_topk_query_ids_for_prompt_light(
                        self,
                        prompt: str,
                        object_queries: torch.Tensor,
                        match_queries: Optional[torch.Tensor] = None,
                        k: int = 10,
                        print_generation: bool = False,
                        original_description: Optional[str] = None,
                        prefer_semantic_ref: Optional[bool] = None,
                        allowed_query_ids: Optional[Sequence[int]] = None,
        ) -> Optional[List[Tuple[int, float]]]:
                if self.llama_model is None or self.llama_tokenizer is None:
                        LOGGER.warning("LLM components are not initialised – cannot run prompt matching.")
                        return None

                prompt_for_generate = prompt.strip()
                # The baseline grounding prompt expects a literal suffix " (with grounding)".
                # Without the leading space (e.g., "couch(with grounding)"), tokenization changes and
                # the model often fails to reliably emit a usable <ref>.
                grounding_suffix = os.environ.get("REFERIT_GROUNDING_SUFFIX", "(with grounding)").strip()
                if grounding_suffix:
                        # Normalize existing suffix to include a leading whitespace.
                        normalized_suffix = grounding_suffix
                        if not normalized_suffix.startswith(" "):
                                normalized_suffix = " " + normalized_suffix
                        if prompt_for_generate.endswith(grounding_suffix) and not prompt_for_generate.endswith(normalized_suffix):
                                prompt_for_generate = prompt_for_generate[: -len(grounding_suffix)] + normalized_suffix
                        elif not prompt_for_generate.endswith(grounding_suffix) and not prompt_for_generate.endswith(normalized_suffix):
                                prompt_for_generate += normalized_suffix
                        # Some ReferIt3D CSVs store utterances wrapped in quotes, e.g.,
                        # "'The lamp between the beds.'" which becomes "'...'" + " (with grounding)".
                        # Strip a single pair of wrapping quotes while keeping the suffix.
                        for q in ("'", '"'):
                                if prompt_for_generate.startswith(q) and prompt_for_generate.endswith(q + normalized_suffix):
                                        prompt_for_generate = prompt_for_generate[1 : -(len(q + normalized_suffix))] + normalized_suffix
                                        break

                temp_instance = grounded_3d_llm_data(
                        input_text=prompt_for_generate, instance_feature=object_queries, eval_type="chat"
                )
                llama_for_causal_lm = self._llama_for_causal_lm
                llama_core = self._llama_core
                if llama_for_causal_lm is None or llama_core is None:
                        (
                                self._llama_for_causal_lm,
                                self._llama_core,
                                self._llama_prompts_owner,
                        ) = self._resolve_llama_components()
                        llama_for_causal_lm = self._llama_for_causal_lm
                        if self._llama_core is None:
                                (
                                        self._llama_for_causal_lm,
                                        self._llama_core,
                                        self._llama_prompts_owner,
                                ) = self._resolve_llama_components()
                        llama_core = self._llama_core
                if llama_for_causal_lm is None or llama_core is None:
                        LOGGER.warning("LLM core modules missing – cannot run prompt matching.")
                        return None
                if not hasattr(llama_core, "instance2embed"):
                        LOGGER.warning("LLM core has no instance2embed – cannot run prompt matching.")
                        return None
                prompts_source = self._llama_prompts_owner or llama_for_causal_lm
                prompts = getattr(prompts_source, "prompts", None)
                if prompts is None:
                        LOGGER.warning("LLM prompter has no prompts attribute – cannot run prompt matching.")
                        return None
                input_ids, _ = temp_instance.build_input_from_segments(
                        tokenizer=self.llama_tokenizer,
                        prompts=prompts,
                        input_text=prompt_for_generate,
                        inference=True,
                )

                visual_embeds = llama_core.instance2embed(object_queries.to(torch.bfloat16).to(self.device))
                eos_embed = llama_core.embed_tokens(
                        torch.tensor([self.llama_tokenizer.eos_token_id], device=self.device)
                )
                text_embeds = llama_core.embed_tokens(input_ids.to(self.device))
                inputs_embeds = torch.cat([visual_embeds, eos_embed, text_embeds], dim=0).unsqueeze(0)
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

                # Ensure deterministic inference (disable dropout etc.) during prompt matching.
                was_training = bool(getattr(self.llama_model, "training", False))
                if was_training:
                        self.llama_model.eval()
                try:
                        # Align generation-time decoding params with `models/LLM/LLama3d.py:LLama3dForCausalLM.evaluate`
                        # to avoid degenerate repetitions (e.g., emitting many consecutive <ref> tokens).
                        try:
                                max_new = int(os.environ.get("G3DLLM_REF_MAX_NEW_TOKENS", "150"))
                        except Exception:
                                max_new = 150
                        try:
                                top_p = float(os.environ.get("G3DLLM_REF_TOP_P", "1.0"))
                        except Exception:
                                top_p = 1.0
                        try:
                                rep_pen = float(os.environ.get("G3DLLM_REF_REPETITION_PENALTY", "1.2"))
                        except Exception:
                                rep_pen = 1.2
                        try:
                                len_pen = float(os.environ.get("G3DLLM_REF_LENGTH_PENALTY", "1.0"))
                        except Exception:
                                len_pen = 1.0
                        try:
                                num_beams = int(getattr(self.llama_model, "beam_size", 1) or 1)
                        except Exception:
                                num_beams = 1
                        num_beams = max(1, int(num_beams))

                        common_params = dict(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                max_new_tokens=int(max_new),
                                output_hidden_states=True,
                                output_scores=True,
                                return_dict_in_generate=True,
                                eos_token_id=self.llama_tokenizer.eos_token_id,
                                pad_token_id=self.llama_tokenizer.eos_token_id,
                                num_beams=num_beams,
                                do_sample=False,
                                min_length=1,
                                top_p=float(top_p),
                                repetition_penalty=float(rep_pen),
                                length_penalty=float(len_pen),
                        )

                        # Follow trainer behavior: use autocast on CUDA to match numerics and speed.
                        if str(getattr(self.device, "type", "")).lower() == "cuda":
                                with torch.autocast("cuda"):
                                        outputs = self.llama_model.generate(**common_params)
                        else:
                                outputs = self.llama_model.generate(**common_params)
                finally:
                        if was_training:
                                self.llama_model.train()

                num_generated_tokens = len(outputs.scores)
                if num_generated_tokens == 0:
                        return None

                newly_generated_ids = outputs.sequences[0][-num_generated_tokens:]
                # Decode-aligned hidden states for the selected output sequence.
                # IMPORTANT: with beam search, `outputs.hidden_states` is not trivial to index correctly.
                # Use the same helper as the trainer (`extract_decoder_hidden_states`) to obtain the
                # hidden representations aligned with `outputs.sequences`.
                decoded_hs = None
                try:
                        decoded_hs = extract_decoder_hidden_states(outputs)
                except Exception as exc:
                        LOGGER.warning("Failed to extract decoder hidden states; falling back to raw hidden_states: %s", exc)
                        decoded_hs = None

                def _get_decoded_token_feature(generated_pos: int) -> Optional[torch.Tensor]:
                        """
                        Return the last-layer hidden state vector for the token at position `generated_pos`
                        within `newly_generated_ids` (0-based).
                        """
                        if generated_pos < 0 or generated_pos >= num_generated_tokens:
                                return None
                        if decoded_hs is None:
                                return None
                        try:
                                # Map generated-pos to absolute position in `outputs.sequences[0]`.
                                # `extract_decoder_hidden_states` returns hidden states aligned to `sequences[1:]`
                                # (it uses `seqlen = sequences.shape[1] - 1`) and then prepends a dummy zero
                                # vector so that the returned tensor aligns 1:1 with `outputs.sequences` indices.
                                seq_len = int(outputs.sequences.shape[1])
                                abs_pos = int(seq_len - num_generated_tokens + generated_pos)
                                hs_pos = abs_pos
                                if torch.is_tensor(decoded_hs):
                                        # [B, seqlen, H]
                                        if decoded_hs.ndim != 3:
                                                return None
                                        hs_pos = max(0, min(hs_pos, int(decoded_hs.shape[1] - 1)))
                                        return decoded_hs[0, hs_pos, :]
                                # Some versions may return a list/tuple per-sample.
                                if isinstance(decoded_hs, (list, tuple)) and len(decoded_hs) > 0:
                                        t0 = decoded_hs[0]
                                        if not torch.is_tensor(t0) or t0.ndim != 3:
                                                return None
                                        hs_pos = max(0, min(hs_pos, int(t0.shape[1] - 1)))
                                        return t0[0, hs_pos, :]
                        except Exception:
                                return None
                        return None
                if print_generation:
                        generated = self.llama_tokenizer.decode(newly_generated_ids, skip_special_tokens=False)
                        tokens = self.llama_tokenizer.convert_ids_to_tokens(newly_generated_ids.tolist())
                        ref_id = getattr(self.llama_tokenizer, "ref_token_id", None)
                        ref_tok = (
                                self.llama_tokenizer.convert_ids_to_tokens([int(ref_id)])[0]
                                if ref_id is not None
                                else None
                        )
                        contains_ref = False
                        if ref_id is not None:
                                contains_ref = bool((newly_generated_ids == int(ref_id)).any().item())
                        preview = " ".join(tokens[:50])
                        print(f"[LLM Generation] {prompt_for_generate} -> {generated.replace(chr(10), ' ')}")
                        print(
                                f"[LLM Tokens] ref_token_id={ref_id} token={ref_tok} "
                                f"contains_ref={contains_ref} tokens[:50]={preview}"
                        )
                ref_positions = (newly_generated_ids == self.llama_tokenizer.ref_token_id).nonzero(as_tuple=True)[0]
                if len(ref_positions) == 0:
                        return None

                if prefer_semantic_ref is None:
                        prefer_semantic_ref = self._prefer_semantic_ref

                selected_ref_pos = ref_positions[0].item()
                selected_prev_tok = None
                if prefer_semantic_ref:
                        # Prefer a <ref> that follows a semantic token (avoid <p>, </p>, stopwords, etc.).
                        special_ids = {
                                int(getattr(self.llama_tokenizer, "ref_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "gs_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "ge_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "inref_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "eos_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "bos_token_id", -1)),
                                int(getattr(self.llama_tokenizer, "pad_token_id", -1)),
                        }
                        stopwords = {
                                "the", "a", "an", "of", "to", "in", "on", "at", "for", "from",
                                "near", "far", "farthest", "closest", "close", "next", "beside",
                                "between", "left", "right", "front", "back", "above", "below",
                                "with", "and", "or", "is", "are", "was", "were", "that", "this",
                        }
                        for ref_pos in ref_positions.tolist():
                                prev_idx = ref_pos - 1
                                if prev_idx < 0:
                                        continue
                                prev_id = int(newly_generated_ids[prev_idx].item())
                                if prev_id in special_ids:
                                        continue
                                prev_tok = self.llama_tokenizer.convert_ids_to_tokens([prev_id])[0]
                                # Heuristic: prefer word-start tokens (SentencePiece uses ▁ prefix).
                                # This avoids selecting subword fragments such as "ton" in "<ref> ton <ref>".
                                if not prev_tok.startswith("▁"):
                                        continue
                                cleaned = prev_tok.lstrip("▁").strip()
                                cleaned_lower = cleaned.lower()
                                if not cleaned_lower:
                                        continue
                                if cleaned_lower in stopwords:
                                        continue
                                if not any(ch.isalpha() for ch in cleaned_lower):
                                        continue
                                selected_ref_pos = ref_pos
                                selected_prev_tok = prev_tok
                                break
                if selected_prev_tok is None:
                        prev_idx = selected_ref_pos - 1
                        if prev_idx >= 0:
                                prev_id = int(newly_generated_ids[prev_idx].item())
                                selected_prev_tok = self.llama_tokenizer.convert_ids_to_tokens([prev_id])[0]

                # When many <ref> tokens are emitted, the earliest <ref> may be a boilerplate token
                # (e.g., "<p> </p> <ref> ...") with weak grounding signal. Optionally select the <ref>
                # position whose hidden state yields the strongest query similarity (does not use GT).
                ref_pick_mode = os.environ.get("REFERIT_REF_PICK_MODE", "prev_word").strip().lower()
                try:
                        ref_pick_max = int(os.environ.get("REFERIT_REF_PICK_MAX", "8"))
                except Exception:
                        ref_pick_max = 8
                ref_pick_max = max(1, ref_pick_max)

                # Feature source for matching:
                # - "ref" : use the hidden state at the generated <ref> token position (recommended).
                # - "prev": use the hidden state at the token immediately before <ref> (legacy/diagnostic).
                ref_feature_source = os.environ.get("REFERIT_REF_FEATURE_SOURCE", "ref").strip().lower()
                def _feature_pos_from_ref_pos(ref_pos: int) -> int:
                        return int(ref_pos - 1) if ref_feature_source == "prev" else int(ref_pos)

                hidden_state_index = _feature_pos_from_ref_pos(int(selected_ref_pos))

                if ref_pick_mode == "best_score" and len(ref_positions) > 1:
                        ref_positions_list = ref_positions.tolist()
                        candidates = ref_positions_list[-ref_pick_max:]
                        best_score = None
                        best_ref_pos = None
                        best_prev_tok = None
                        for ref_pos in candidates:
                                feat_pos = _feature_pos_from_ref_pos(int(ref_pos))
                                if feat_pos < 0 or feat_pos >= num_generated_tokens:
                                        continue
                                try:
                                        feat = _get_decoded_token_feature(int(feat_pos))
                                except Exception:
                                        continue
                                if feat is None:
                                        continue
                                top1_scores, top1_indices = self._match_feature_to_queries_topk(
                                        feat,
                                        match_queries if match_queries is not None else object_queries,
                                        k=1,
                                        allowed_query_ids=allowed_query_ids,
                                )
                                if top1_scores is None or top1_indices is None:
                                        continue
                                score0 = float(top1_scores[0].item()) if hasattr(top1_scores[0], "item") else float(top1_scores[0])
                                if best_score is None or score0 > best_score:
                                        best_score = score0
                                        best_ref_pos = int(ref_pos)
                                        prev_idx = int(ref_pos) - 1
                                        if prev_idx >= 0:
                                                try:
                                                        prev_id = int(newly_generated_ids[prev_idx].item())
                                                        best_prev_tok = self.llama_tokenizer.convert_ids_to_tokens([prev_id])[0]
                                                except Exception:
                                                        best_prev_tok = None
                        if best_ref_pos is not None:
                                selected_ref_pos = best_ref_pos
                                selected_prev_tok = best_prev_tok or selected_prev_tok
                                hidden_state_index = _feature_pos_from_ref_pos(int(selected_ref_pos))

                if hidden_state_index < 0 or hidden_state_index >= num_generated_tokens:
                        return None
                if print_generation:
                        feature_tok_id = int(newly_generated_ids[hidden_state_index].item())
                        feature_tok = self.llama_tokenizer.convert_ids_to_tokens([feature_tok_id])[0]
                        print(
                                f"[LLM RefPick] mode={ref_pick_mode} feature_source={ref_feature_source} "
                                f"num_ref={int(len(ref_positions))} "
                                f"allowed_q={0 if allowed_query_ids is None else len(list(allowed_query_ids))}"
                        )
                        print(
                                f"[LLM Ref] ref_idx={selected_ref_pos} feature_pos={hidden_state_index} "
                                f"feature_tok_id={feature_tok_id} feature_tok={feature_tok} "
                                f"prev_tok={selected_prev_tok}"
                        )

                ref_feature = _get_decoded_token_feature(int(hidden_state_index))
                if ref_feature is None:
                        return None
                topk_scores, topk_indices = self._match_feature_to_queries_topk(
                        ref_feature,
                        match_queries if match_queries is not None else object_queries,
                        k,
                        allowed_query_ids=allowed_query_ids,
                )
                if topk_scores is None:
                        return None
                return list(zip(topk_indices.tolist(), topk_scores.tolist()))

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------
        def _resolve_scene_file(self, scene_id: str) -> Optional[Path]:
                normalized = scene_id.replace("scene", "")
                # Allow scannet_root to be either the dataset root or an already split subdir
                roots_to_search = [self.scannet_root]
                if self.scannet_root.name in {"train", "validation", "test"}:
                        roots_to_search.append(self.scannet_root.parent)
                candidates = []
                for root in roots_to_search:
                        candidates.extend(
                                [root / split / f"{normalized}.npy" for split in ("train", "validation", "test")]
                                + [root / split / f"{scene_id}.npy" for split in ("train", "validation", "test")]
                                + [root / f"{normalized}.npy", root / f"{scene_id}.npy"]
                        )
                for candidate in candidates:
                        if candidate.exists():
                                return candidate
                return None

        def _load_scene_points(self, scene_id: str) -> Optional[np.ndarray]:
                path = self._resolve_scene_file(scene_id)
                if path is None:
                        LOGGER.warning("Unable to locate .npy file for scene %s under %s", scene_id, self.scannet_root)
                        return None
                return np.load(path)

        def _run_full_forward(self, scene_id: str) -> Optional[Dict[str, Any]]:
                if scene_id in self._scene_forward_cache:
                        return self._scene_forward_cache[scene_id]

                points = self._load_scene_points(scene_id)
                if points is None or points.shape[1] <= NpyColumnIndex.INSTANCE_LABEL:
                        return None

                coordinates = points[:, NpyColumnIndex.COORDS].astype(np.float32)
                features = (points[:, NpyColumnIndex.COLORS] / 127.5) - 1.0
                instance_labels = points[:, NpyColumnIndex.INSTANCE_LABEL].astype(np.int32)
                semantic_labels = points[:, NpyColumnIndex.SEMANTIC_LABEL].astype(np.int32)

                quantized_coords = np.floor(coordinates / self._voxel_size)
                coords_tensor = torch.from_numpy(quantized_coords).contiguous()
                _, unique_map, inverse_map = ME.utils.sparse_quantize(
                        coordinates=coords_tensor,
                        return_index=True,
                        return_inverse=True,
                )

                if len(unique_map) == 0:
                        LOGGER.warning("Scene %s collapsed during voxelisation.", scene_id)
                        return None

                coords_with_batch = torch.cat(
                        [torch.zeros(len(unique_map), 1, dtype=torch.int), torch.from_numpy(quantized_coords[unique_map])],
                        dim=1,
                ).contiguous().int()
                me_features = torch.from_numpy(features[unique_map]).float().contiguous()

                if coords_with_batch.shape[0] != me_features.shape[0]:
                        raise ValueError(
                                f"Coordinate/feature length mismatch for scene {scene_id}: "
                                f"{coords_with_batch.shape} vs {me_features.shape}"
                        )
                if not torch.isfinite(me_features).all():
                        raise ValueError(f"Non-finite features detected for scene {scene_id}")

                def _build_sparse_tensor(feats: torch.Tensor, coords: torch.Tensor, device: torch.device) -> ME.SparseTensor:
                        return ME.SparseTensor(
                                features=feats.to(device=device, non_blocking=True),
                                coordinates=coords.to(device=device, non_blocking=True),
                                device=device,
                        )

                tensor_device = self.device
                active_detector = self.detector
                try:
                        sparse_tensor = _build_sparse_tensor(me_features, coords_with_batch, tensor_device)
                except RuntimeError as exc:
                        LOGGER.error(
                                "SparseTensor construction failed on %s for %s (features=%s, coords=%s): %s",
                                tensor_device,
                                scene_id,
                                tuple(me_features.shape),
                                tuple(coords_with_batch.shape),
                                exc,
                        )
                        cpu_device = torch.device("cpu")
                        cpu_tensor: Optional[ME.SparseTensor] = None
                        try:
                                cpu_tensor = _build_sparse_tensor(me_features, coords_with_batch, cpu_device)
                                LOGGER.warning(
                                        "SparseTensor succeeded on CPU after failing on %s — indicates GPU kernel mismatch for scene %s.",
                                        tensor_device,
                                        scene_id,
                                )
                        except RuntimeError as cpu_exc:
                                LOGGER.error(
                                        "SparseTensor also failed on CPU for %s: %s",
                                        scene_id,
                                        cpu_exc,
                                )

                        if cpu_tensor is not None:
                                sparse_tensor = cpu_tensor
                                tensor_device = cpu_device
                                active_detector = self._get_detector_for_device(cpu_device)
                                LOGGER.warning(
                                        "Falling back to CPU inference for scene %s; expect slower processing.",
                                        scene_id,
                                )
                        else:
                                limit = min(50000, me_features.shape[0])
                                truncated_feats = me_features[:limit].contiguous()
                                truncated_coords = coords_with_batch[:limit].contiguous()
                                sparse_tensor = _build_sparse_tensor(truncated_feats, truncated_coords, self.device)
                raw_coordinates = sparse_tensor.C[:, 1:].float() * self._voxel_size

                with torch.no_grad():
                        vision_output = active_detector(x=sparse_tensor, raw_coordinates=raw_coordinates, is_eval=True)

                pred_masks_sparse = (vision_output["pred_masks"][0] > 0.).detach().cpu()
                object_queries = vision_output["queries_hidden_state"][0].detach().cpu()
                queries_normalized_embed = vision_output.get("queries_normalized_embed")
                if isinstance(queries_normalized_embed, list):
                        queries_normalized_embed = queries_normalized_embed[-1]
                if queries_normalized_embed is not None:
                        try:
                                queries_normalized_embed = queries_normalized_embed[0].detach().cpu()
                        except Exception:
                                queries_normalized_embed = None
                sampled_coords_raw = vision_output.get("sampled_coords")
                if isinstance(sampled_coords_raw, list):
                        sampled_coords_raw = sampled_coords_raw[-1]
                if sampled_coords_raw is None:
                        sampled_coords = torch.zeros(object_queries.shape[0], 3)
                else:
                        sampled_coords = torch.as_tensor(sampled_coords_raw[0]).detach().cpu()

                pred_logits = vision_output.get("pred_logits")
                if isinstance(pred_logits, list):
                        pred_logits = pred_logits[-1]
                if pred_logits is not None:
                        pred_logits = pred_logits[0].detach().cpu()
                        pred_probs = torch.sigmoid(pred_logits)
                        pred_scores, raw_classes = torch.max(pred_probs, dim=-1)
                        raw_classes_np = raw_classes.numpy()
                        if hasattr(self.interface.dataset, "_remap_model_output"):
                                remapped = self.interface.dataset._remap_model_output(raw_classes_np)
                                pred_classes = torch.from_numpy(remapped).long()
                        else:
                                pred_classes = raw_classes.long()
                else:
                        pred_scores = torch.zeros(object_queries.shape[0])
                        pred_classes = torch.full((object_queries.shape[0],), -1, dtype=torch.long)

                # Map sparse masks back to full resolution.
                num_points = points.shape[0]
                if pred_masks_sparse.shape[0] != len(unique_map):
                        pred_masks_sparse = pred_masks_sparse.transpose(0, 1)
                sparse_masks = torch.zeros((len(unique_map), pred_masks_sparse.shape[1]), dtype=torch.float32)
                sparse_masks[:] = pred_masks_sparse

                # Map sparse masks back to full resolution.
                # - Legacy behavior: only assign masks on `unique_map` indices.
                #   This underestimates masks (zeros for the remaining points).
                # - Fixed behavior (opt-in): use `inverse_map` so every point gets
                #   its voxel mask value.
                #
                # Keep the fix behind an env flag so historical exports remain reproducible.
                use_fullres_fix = os.environ.get("GROUNDED3DLLM_FULLRES_MASK_FIX", "0") == "1"
                if use_fullres_fix:
                        full_res_masks = sparse_masks.float()[inverse_map.detach().cpu()]
                else:
                        full_res_masks = torch.zeros((num_points, sparse_masks.shape[1]), dtype=torch.float32)
                        full_res_masks[unique_map.detach().cpu()] = sparse_masks.float()
                full_res_masks_np = full_res_masks.numpy()

                temp_forward = {
                        "pred_masks_full_res": full_res_masks_np,
                        "instance_labels"    : instance_labels,
                }
                gt_map, gt_instance_classes = self._compute_gt_query_mapping(temp_forward, semantic_labels)

                cache = {
                        "scene_id"           : scene_id,
                        "file_name"          : scene_id,
                        "points"             : points,
                        "instance_labels"    : instance_labels,
                        "inverse_map"        : inverse_map.numpy(),
                        "object_queries"     : object_queries,
                        "sampled_coords"     : sampled_coords,
                        "pred_masks_sparse"  : sparse_masks,
                        "pred_masks_full_res": full_res_masks_np,
                        "pred_classes"       : pred_classes,
                        "pred_scores"        : pred_scores,
                        "gt_to_query_map"    : gt_map,
                        "gt_instance_classes": gt_instance_classes,
                        "queries_normalized_embed": queries_normalized_embed,
                }

                self._scene_forward_cache[scene_id] = cache

                # Free intermediate tensors
                del sparse_tensor, vision_output
                gc.collect()
                if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                return cache

        def _compute_gt_query_mapping(
                        self,
                        forward: Dict[str, Any],
                        semantic_labels: Optional[np.ndarray] = None,
        ) -> Tuple[Dict[int, int], Dict[int, int]]:
                pred_masks = torch.as_tensor(forward["pred_masks_full_res"], device=self.device, dtype=torch.float32)
                gt_ids = np.unique(forward["instance_labels"])

                gt_masks = []
                final_ids = []
                for inst_id in gt_ids:
                        if inst_id < 1:
                                continue
                        mask_full = (forward["instance_labels"] == inst_id).astype(np.float32)
                        mask_tensor = torch.from_numpy(mask_full)
                        if mask_tensor.sum() == 0:
                                continue
                        gt_masks.append(mask_tensor)
                        final_ids.append(int(inst_id))

                if not gt_masks:
                        return {}, {}

                gt_tensor = torch.stack(gt_masks).to(self.device)
                intersection = pred_masks.T @ gt_tensor.T
                area_pred = pred_masks.sum(dim=0).unsqueeze(1)  # [num_queries, 1]
                area_gt = gt_tensor.sum(dim=1).unsqueeze(0)  # [1, num_gt]
                union = area_pred + area_gt - intersection

                iou_matrix = (intersection / (union + 1e-8)).clamp(min=0.0, max=1.0)

                matcher = HungarianMatcher(
                        cost_class=0.0,
                        cost_mask=1.0,
                        cost_dice=1.0,
                        num_points=-1,
                        num_queries=pred_masks.shape[1],
                        language_mode=False,
                        softmax_mode=False,
                ).to(self.device)

                scene_label = forward.get("scene_id") or forward.get("file_name") or "unknown"
                num_pred_points, num_queries = pred_masks.shape
                num_gt = gt_tensor.shape[0]
                gt_points = gt_tensor.shape[1] if gt_tensor.dim() > 1 else 0
                # LOGGER.info(
                #     "Hungarian sanity | scene=%s | pred_masks_shape=%s | gt_tensor_shape=%s | pred_points=%d | gt_points=%d | num_queries=%d | num_gt=%d | matcher.num_points=%s",
                #     scene_label,
                #     tuple(pred_masks.shape),
                #     tuple(gt_tensor.shape),
                #     num_pred_points,
                #     gt_points,
                #     num_queries,
                #     num_gt,
                #     matcher.num_points,
                # )

                outputs = {
                        "pred_logits": torch.zeros(
                                1, pred_masks.shape[1], 1, device=self.device, dtype=torch.float32
                        ),
                        "pred_masks" : pred_masks.unsqueeze(0),
                }
                targets = [
                        {
                                "labels": torch.zeros(gt_tensor.shape[0], dtype=torch.long, device=self.device),
                                "masks" : gt_tensor,
                        }
                ]

                try:
                        indices = matcher(outputs, targets, mask_type="masks")
                        query_indices, gt_indices = indices[0]
                        mapping: Dict[int, int] = {}
                        for q_idx, gt_idx in zip(query_indices.tolist(), gt_indices.tolist()):
                                mapping[final_ids[gt_idx]] = int(q_idx)
                except Exception as exc:
                        LOGGER.warning("Hungarian matcher failed (%s); falling back to IoU argmax.", exc)
                        _, best_indices = torch.max(iou_matrix, dim=0)
                        mapping = {}
                        for idx, gt_id in enumerate(final_ids):
                                mapping[gt_id] = int(best_indices[idx].item())
                instance_classes: Dict[int, int] = {}
                if semantic_labels is not None:
                        for inst_id in final_ids:
                                mask = forward["instance_labels"] == inst_id
                                if not np.any(mask):
                                        continue
                                classes = semantic_labels[mask]
                                if classes.size == 0:
                                        continue
                                counts = np.bincount(classes.astype(np.int64))
                                if counts.size == 0:
                                        continue
                                instance_classes[int(inst_id)] = int(np.argmax(counts))
                return mapping, instance_classes

        def _get_vision_features(self, scene_id: str) -> Optional[Dict[str, torch.Tensor]]:
                forward = self._run_full_forward(scene_id)
                if forward is None:
                        return None
                return {
                        "object_queries": forward["object_queries"],
                        "sampled_coords": forward["sampled_coords"],
                }

        def _match_feature_to_queries_topk(
                        self,
                        ref_feature: torch.Tensor,
                        object_queries: torch.Tensor,
                        k: int,
                        allowed_query_ids: Optional[Sequence[int]] = None,
        ):
                try:
                        llama_core = self._llama_core
                        if llama_core is None or not hasattr(llama_core, "hidden_state2query"):
                                LOGGER.warning("LLM core has no hidden_state2query – cannot run feature matching.")
                                return None, None
                        # Keep everything on the same device as hidden_state2query
                        target_device = None
                        target_dtype = None
                        try:
                                p0 = next(llama_core.hidden_state2query.parameters())
                                target_device = p0.device
                                target_dtype = p0.dtype
                        except StopIteration:
                                target_device = ref_feature.device
                                target_dtype = ref_feature.dtype
                        if target_dtype is None:
                                target_dtype = torch.float32
                        ref_feature = ref_feature.to(target_device)
                        object_queries = object_queries.to(target_device)
                        # Autocast during generation can yield bf16/half hidden states; ensure
                        # dtype matches the projection head (typically float32) to avoid
                        # `expected scalar type Float but found BFloat16`.
                        ref_feature = ref_feature.to(dtype=target_dtype)
                        object_queries = object_queries.to(dtype=target_dtype)
                        ref_query_feature = llama_core.hidden_state2query(ref_feature)
                        similarities = F.cosine_similarity(ref_query_feature.unsqueeze(0), object_queries)
                        if allowed_query_ids is not None:
                                allowed = torch.zeros_like(similarities, dtype=torch.bool)
                                for qid in allowed_query_ids:
                                        qi = int(qid)
                                        if 0 <= qi < int(allowed.shape[0]):
                                                allowed[qi] = True
                                similarities = similarities.masked_fill(~allowed, -1e9)
                        if allowed_query_ids is not None:
                                allowed_n = int(allowed.sum().item())
                                if allowed_n <= 0:
                                        return None, None
                                k = min(k, allowed_n)
                        k = min(k, similarities.shape[0])
                        return torch.topk(similarities, k=k, dim=0)
                except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.error("Failed to match ref feature to queries: %s", exc)
                        return None, None


__all__ = ["Grounded3DLLMBaselineInterface", "BaselineModelAPI"]
