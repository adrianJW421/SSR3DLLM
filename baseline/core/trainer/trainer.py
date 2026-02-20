import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]  # release repo root
sys.path.insert(0, str(repo_root))               # Add repo root first so local packages take precedence.
sys.path.insert(0, str(repo_root / "src"))       # Also add src/ if the project keeps source code there.

import gc
import statistics
import shutil
import os
import os.path as osp
import math
from loguru import logger
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from datetime import datetime

import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from hashlib import md5
import glob
import pickle
import json

from baseline.dataset.dataset_code.language_info import lang_info_data

from baseline.dataset.datasets.scannet200.scannet200_splits import (
        HEAD_CATS_SCANNET_200,
        TAIL_CATS_SCANNET_200,
        COMMON_CATS_SCANNET_200,
        VALID_CLASS_IDS_200_VALIDATION,
)
from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_200
try:
        # Optional: used only for instance segmentation AP evaluation.
        # Keep import-time robust so `--help` and baseline interfaces work even if
        # the benchmark helpers are not vendored into the release snapshot.
        from benchmark.evaluate_semantic_instance import evaluate  # type: ignore
except Exception:  # pragma: no cover
        evaluate = None  # type: ignore

from models.metrics import IoU
from models.metrics.evaluate_LLM import eval_llm_iou_score
from models.misc import get_batch_aabb_pair_ious, logical_or_sum, get_evenly_distributed_colors, fix_seed, print_grad_status
from utils.votenet_utils.eval_det import eval_det

from models.metrics.utils import eval_seg_model, collect_grounding_score
from transformers import AutoConfig
from config import instantiate
from typing import Dict, Optional

# SSR3DLLM: optional geometry head for LLM instance queries.
from models.geom_head_llm_adapter import SSR3DLLMGeomHeadForLLM  # type: ignore


def _ssr3dllm_env_flag(name: str, default: str = "0") -> bool:
        try:
                v = os.environ.get(name, default)
                return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
                return False


def _ssr3dllm_env_int(name: str, default: int) -> int:
        try:
                return int(str(os.environ.get(name, str(default))).strip())
        except Exception:
                return int(default)


def _ssr3dllm_proc_status_mb(keys: "list[str]") -> str:
        """
        Best-effort memory snapshot from `/proc/self/status` (Linux only).
        Returns a compact string like: `VmRSS=1234MB VmHWM=1500MB`.
        """
        try:
                p = "/proc/self/status"
                if not osp.exists(p):
                        return ""
                txt = Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
                kv = {}
                for line in txt:
                        if ":" not in line:
                                continue
                        k, v = line.split(":", 1)
                        k = k.strip()
                        if k in keys:
                                kv[k] = v.strip()
                parts = []
                for k in keys:
                        v = kv.get(k)
                        if not v:
                                continue
                        # v like: "123456 kB"
                        toks = v.split()
                        if len(toks) >= 2 and toks[1].lower() == "kb":
                                try:
                                        mb = float(toks[0]) / 1024.0
                                        parts.append(f"{k}={mb:.1f}MB")
                                        continue
                                except Exception:
                                        pass
                        parts.append(f"{k}={v}")
                return " ".join(parts)
        except Exception:
                return ""


def _ssr3dllm_disk_usage_mb(path: str) -> str:
        try:
                u = shutil.disk_usage(path)
                gb = 1024.0 ** 3
                return f"disk_free={u.free/gb:.1f}GB disk_used={u.used/gb:.1f}GB"
        except Exception:
                return ""


def _ssr3dllm_gpu_mem_mb(device: torch.device | None = None) -> str:
        try:
                if not torch.cuda.is_available():
                        return ""
                if device is None:
                        device = torch.device("cuda", torch.cuda.current_device())
                idx = device.index if device.index is not None else torch.cuda.current_device()
                gb = 1024.0 ** 3
                alloc = torch.cuda.memory_allocated(idx) / gb
                reserved = torch.cuda.memory_reserved(idx) / gb
                max_alloc = torch.cuda.max_memory_allocated(idx) / gb
                max_reserved = torch.cuda.max_memory_reserved(idx) / gb
                return f"cuda_alloc={alloc:.2f}GB cuda_resv={reserved:.2f}GB cuda_max_alloc={max_alloc:.2f}GB cuda_max_resv={max_reserved:.2f}GB"
        except Exception:
                return ""


def _ssr3dllm_log_resource(tag: str, *, save_dir: str = "", tmp_dir: str = "", device: torch.device | None = None, extra: str = "") -> None:
        parts = []
        mem = _ssr3dllm_proc_status_mb(["VmRSS", "VmHWM", "VmSize"])
        if mem:
                parts.append(mem)
        gpu = _ssr3dllm_gpu_mem_mb(device)
        if gpu:
                parts.append(gpu)
        if save_dir:
                du = _ssr3dllm_disk_usage_mb(save_dir)
                if du:
                        parts.append(f"save_dir({save_dir}) {du}")
        if tmp_dir:
                du = _ssr3dllm_disk_usage_mb(tmp_dir)
                if du:
                        parts.append(f"tmp_dir({tmp_dir}) {du}")
        if extra:
                parts.append(extra)
        if parts:
                logger.warning(f"[SSR3DLLM][resource] {tag} | " + " | ".join(parts))


class ModelingGrounded3DLLM(pl.LightningModule):
        def __init__(self, config):
                super().__init__()

                self.decoder_id = config.general.decoder_id

                if config.model.train_on_segments:
                        self.mask_type = "segment_mask"
                else:
                        self.mask_type = "masks"

                self.eval_on_segments = config.general.eval_on_segments
                self.config = config
                self.max_eval_queries = getattr(self.config.general, "max_eval_queries", 0)
                self.eval_lang_type_limits = {}
                lang_conf = getattr(self.config.data, "lang_data_conf", "")
                for token in lang_conf.split('+'):
                        if ',' in token:
                                lang_type, sample_num = token.split(',')
                                try:
                                        self.eval_lang_type_limits[lang_type] = int(sample_num)
                                except ValueError:
                                        continue
                if not getattr(self.config.data, "sample_class_labels", False):
                        self.eval_lang_type_limits.setdefault("detection", 0)

                # ================= temporary folders for saved results (multi-gpu) ================
                self.tmpdir = osp.join(
                        './.dist_test', md5(self.config.general.save_dir.encode()).hexdigest())
                os.makedirs(self.tmpdir, exist_ok=True)
                for i in glob.glob(self.tmpdir + '/*'):
                        os.remove(i)

                self.save_hyperparameters()

                # ============== Prepare llama model ==============
                self.init_llama_model()

                # ================= initialize the segmentation network ================
                self.model = instantiate(config.model)

                # ===== Rel3D-LLM: optional relation-field features for LLM =====
                self.use_rel3d_hist_for_llm = getattr(
                        config.general, "use_rel3d_hist_for_llm", False
                )
                if self.use_rel3d_hist_for_llm:
                        num_dirs = int(getattr(config.model, "rel3d_num_dirs", 6))
                        hidden_dim = int(getattr(self.model, "mask_dim", 128))
                        self.rel3d_hist_proj = nn.Sequential(
                                nn.Linear(num_dirs, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                        )
                        self.rel3d_global_weight = float(getattr(config.general, "rel3d_global_weight", 1.0))
                        self.rel3d_anchor_weight = float(getattr(config.general, "rel3d_anchor_weight", 1.0))
                        if self.global_rank == 0:
                                print(
                                        f"[Rel3D] LLM will consume relation histograms: "
                                        f"use_rel3d_geom={getattr(config.model, 'use_rel3d_geom', False)}, "
                                        f"num_dirs={num_dirs}, hidden_dim={hidden_dim}, "
                                        f"global_weight={self.rel3d_global_weight}, "
                                        f"anchor_weight={self.rel3d_anchor_weight}"
                                )

                # Complete SSR3DLLM init: geometry head, losses, datasets, etc.
                # This was previously part of __init__ but is kept as a helper for readability.
                self._init_ssr3dllm_and_losses(config)

        def _init_ssr3dllm_and_losses(self, config):
                def _flag(name: str, default: str = "0") -> bool:
                        v = os.environ.get(name, default)
                        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

                # ===== SSR3DLLM: optional geometry head for LLM =====
                self.enable_ssr3dllm_geom = bool(
                        getattr(self.config.general, "enable_ssr3dllm_geom", False)
                )
                self.ssr3dllm_geom_weight = float(
                        getattr(self.config.general, "ssr3dllm_geom_weight", 1.0)
                )
                self.ssr3dllm_ref_loss_weight = float(
                        getattr(self.config.general, "ssr3dllm_ref_loss_weight", 1.0)
                )
                self.ssr3dllm_anchor_loss_weight = float(
                        getattr(self.config.general, "ssr3dllm_anchor_loss_weight", 1.0)
                )
                self.ssr3dllm_relcls_loss_weight = float(
                        getattr(self.config.general, "ssr3dllm_relcls_loss_weight", 1.0)
                )
                self.ssr3dllm_chain_loss_weight = float(
                        getattr(self.config.general, "ssr3dllm_chain_loss_weight", 1.0)
                )
                # SSR3DLLM: optional offline teacher-logits distillation (Vigor).
                self.ssr3dllm_distill_vigor_weight = float(
                        getattr(self.config.general, "ssr3dllm_distill_vigor_weight", 0.0)
                )
                self.ssr3dllm_distill_temperature = float(
                        getattr(self.config.general, "ssr3dllm_distill_temperature", 1.0)
                )
                if self.enable_ssr3dllm_geom:
                        hidden_dim = int(getattr(self.model, "mask_dim", 128))
                        self.ssr3dllm_geom_head = SSR3DLLMGeomHeadForLLM(
                                hidden_dim=hidden_dim,
                                bert_model=getattr(self.config.general, "ssr3dllm_bert_model", "pretrained/bert-base-uncased"),
                        ).to(self.device)
                else:
                        self.ssr3dllm_geom_head = None

                self.ssr3dllm_geom_only = bool(
                        getattr(self.config.general, "ssr3dllm_geom_only", False)
                )
                if self.ssr3dllm_geom_only:
                        for _, param in self.model.named_parameters():
                                param.requires_grad = False
                        if hasattr(self, "llama_model") and self.llama_model is not None:
                                for _, param in self.llama_model.named_parameters():
                                        param.requires_grad = False

                if self.llama_config.llm_only:
                        # froze seg model
                        for name, param in self.model.named_parameters():
                                param.requires_grad = False

                # Step-token SFT default: keep the CLASP-tuned Mask3D query representation stable.
                # This helps preserve the pretrained vision-query semantics while letting only:
                # - LLM <stepK> token rows
                # - SSR3DLLM geometry chain head
                # learn to cooperate.
                if self._is_step_token_sft() and _flag("SSR3DLLM_FREEZE_MASK3D_LANG", "0"):
                        for _, param in self.model.named_parameters():
                                param.requires_grad = False

                # loss
                self.ignore_label = config.data.ignore_label

                matcher = instantiate(
                        config.matcher,
                )
                weight_dict = {
                        "loss_ce"  : matcher.cost_class,
                        "loss_mask": matcher.cost_mask,
                        "loss_dice": matcher.cost_dice,
                }

                aux_weight_dict = {}
                for i in range(self.model.num_levels * self.model.num_decoders):
                        aux_weight_dict.update(
                                {k + f"_{i}": v for k, v in weight_dict.items()}
                        )
                weight_dict.update(aux_weight_dict)

                self.preds = dict()
                self.bbox_preds = dict()
                self.bbox_gt = dict()

                self.criterion = instantiate(
                        config.loss, matcher=matcher, weight_dict=weight_dict,
                )
                if self.ssr3dllm_geom_only:
                        self.criterion.weight_dict = {}

                # metrics
                self.confusion = instantiate(config.metrics)
                self.iou = IoU()
                # misc
                self.labels_info = dict()

                # Datasets can be extremely large (e.g. `indoor_dialog.yaml` mixes many
                # instruction-following sources). In `--mode test`, Lightning will only
                # run `test_dataloader()`, but the old eager init still loaded
                # train/val/test datasets, which can OOM and get the process SIGKILL-ed
                # without a Python traceback (observed as logs stopping right after
                # `LOCAL_RANK: ...`).
                #
                # To keep evaluation stable and memory-bounded, instantiate only what we
                # need for the current run mode:
                # - train mode: train + val + test
                # - test  mode: test only, and alias `validation_dataset` to it because
                #   many helpers historically reference `validation_dataset` for label
                #   metadata / remap utilities.
                self.train_dataset = None
                self.validation_dataset = None
                self.test_dataset = None

                is_train_mode = bool(getattr(self.config.general, "train_mode", True))
                if is_train_mode:
                        self.train_dataset = instantiate(self.config.data.train_dataset)
                        self.validation_dataset = instantiate(self.config.data.validation_dataset)
                        self.test_dataset = instantiate(self.config.data.test_dataset)
                else:
                        self.test_dataset = instantiate(self.config.data.test_dataset)
                        self.validation_dataset = self.test_dataset

                # Prefer label_info from the training dataset when available; otherwise
                # fall back to the (aliased) validation/test dataset.
                try:
                        if self.train_dataset is not None:
                                self.labels_info = self.train_dataset.label_info
                        elif self.validation_dataset is not None:
                                self.labels_info = self.validation_dataset.label_info
                        else:
                                self.labels_info = {}
                except Exception:  # pragma: no cover - defensive for eval stability
                        self.labels_info = {}

                self.automatic_optimization = False  # mannual step

                # ===== Rel3D-LLM: optional query-role and teacher heads (auxiliary losses) =====
                self.enable_rel3d_role_loss = bool(
                        getattr(self.config.general, "enable_rel3d_role_loss", False)
                )
                self.rel3d_role_loss_weight = float(
                        getattr(self.config.general, "rel3d_role_loss_weight", 0.0)
                )
                if self.enable_rel3d_role_loss:
                        hidden_dim = int(getattr(self.model, "mask_dim", 128))
                        self.rel3d_role_head = nn.Linear(hidden_dim, 3)

        def load_state_dict(self, state_dict, strict: bool = True):
                """
                Lightning resume/load safety:

                Some SSR3DLLM submodules are created lazily (e.g. the Vigor step projection
                `ssr3dllm_geom_head._vigor_step_proj`). If a checkpoint was saved after the
                lazy module was materialized, its weights will exist in the checkpoint, but a
                fresh model instance (before the first forward) may not have the submodule
                registered yet (still `None`). In strict loading this shows up as
                "Unexpected key(s) in state_dict".

                We pre-create such lazy modules based on checkpoint shapes before delegating
                to the standard loader, so stage-to-stage resume works reliably.
                """

                try:
                        w_key = "ssr3dllm_geom_head._vigor_step_proj.weight"
                        if w_key in state_dict and getattr(self, "ssr3dllm_geom_head", None) is not None:
                                geom_head = getattr(self, "ssr3dllm_geom_head")
                                if getattr(geom_head, "_vigor_step_proj", None) is None:
                                        w = state_dict[w_key]
                                        out_dim = int(w.shape[0])
                                        try:
                                                dev = next(self.parameters()).device
                                        except StopIteration:  # pragma: no cover
                                                dev = torch.device("cpu")
                                        try:
                                                _ = geom_head._get_vigor_step_proj(inner_dim=out_dim, device=dev)
                                        except Exception:  # pragma: no cover - best effort fallback
                                                geom_head._vigor_step_proj = nn.Linear(int(w.shape[1]), out_dim, bias=True).to(device=dev)
                except Exception:  # pragma: no cover - keep resume robust
                        pass

                return super().load_state_dict(state_dict, strict=strict)

        def _is_step_token_sft(self) -> bool:
                """
                Heuristic flag for the "LLM step-token" experiment:
                - LLM is enabled (teacher-forcing is used)
                - `<stepK>` tokens are enabled in the tokenizer
                - rel3dref answers are set to step-token sequences
                - legacy `<ref>` grounding is disabled
                """

                def _flag(name: str) -> bool:
                        v = os.environ.get(name, "0")
                        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

                return (
                        bool(getattr(self, "llama_config", None) and getattr(self.llama_config, "enable_llm", False))
                        and _flag("SSR3DLLM_STEP_TOKENS")
                        and _flag("SSR3DLLM_REL3D_OUTPUT_STEPS")
                        and _flag("SSR3DLLM_DISABLE_LLM_GROUNDING")
                )

        def _ssr3dllm_order_mode(self) -> str:
                return str(os.environ.get("SSR3DLLM_ORDER_MODE", "")).strip().lower()

        def _ssr3dllm_should_force_stepslot_output(self, lang_type: str) -> bool:
                """
                When `SSR3DLLM_ORDER_MODE=slots`, force the LLM output text to be only `<stepK>` tokens
                for selected prefixes so we can extract step-token hidden states in a single forward
                (without generating/encoding textual chains).
                """
                if self._ssr3dllm_order_mode() != "slots":
                        return False
                # Safer default: do NOT force output text unless explicitly enabled.
                # (For SSR3DLLM capability-preservation joint SFT, we want to keep normal LM targets.)
                force_output = str(os.environ.get("SSR3DLLM_ORDER_FORCE_OUTPUT", "0")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                }
                if not force_output:
                        return False
                if not isinstance(lang_type, str) or not lang_type:
                        return False
                prefixes_raw = str(os.environ.get("SSR3DLLM_ORDER_PREFIXES", "rel3dref,scanrefer,m3dref")).strip()
                prefixes = {p.strip() for p in prefixes_raw.split(",") if p.strip()}
                prefix = lang_type.split(":")[0]
                return (not prefixes) or (prefix in prefixes)

        def _ssr3dllm_stepslot_output_text(self) -> str:
                try:
                        order_len = int(str(os.environ.get("SSR3DLLM_ORDER_MAX_LEN", "")).strip() or "0")
                except Exception:
                        order_len = 0
                if order_len <= 0:
                        try:
                                order_len = int(str(os.environ.get("SSR3DLLM_STEP_ORDER_LEN", "")).strip() or "4")
                        except Exception:
                                order_len = 4
                order_len = max(1, int(order_len))
                return " ".join([f"<step{i}>" for i in range(1, int(order_len) + 1)]).strip()

        def _ssr3dllm_build_output_texts(self, batch_lang_infos: list) -> list:
                out = []
                forced = 0
                forced_prefixes = set()
                forced_text = self._ssr3dllm_stepslot_output_text()
                for li in batch_lang_infos:
                        lt = getattr(li, "lang_type", "")
                        if self._ssr3dllm_should_force_stepslot_output(lt):
                                forced += 1
                                try:
                                        forced_prefixes.add(str(lt).split(":")[0])
                                except Exception:
                                        pass
                                out.append(forced_text)
                        else:
                                out.append(getattr(li, "answer", ""))
                if forced > 0 and int(getattr(self, "global_rank", 0)) == 0 and not getattr(self, "_ssr3dllm_slots_output_warned", False):
                        setattr(self, "_ssr3dllm_slots_output_warned", True)
                        try:
                                print(
                                        f"[SSR3DLLM][order_slots] forcing output_text to step tokens for {forced} samples "
                                        f"(prefixes={sorted(list(forced_prefixes))}) output='{forced_text}'",
                                        flush=True,
                                )
                        except Exception:
                                pass
                return out

        def init_llama_model(self):
                save_folder_name = datetime.now().strftime(
                        "%m-%d-%H-%M-%S") if not self.config.general.timestamp else self.config.general.timestamp

                llama_config = AutoConfig.from_pretrained(
                        self.config.general.llm_config)
                llama_config.save_path = f"{self.config['general']['save_dir']}/{save_folder_name}"

                try:
                        self.data_to_load = json.load(
                                open(self.config.general.llm_data_config))
                except Exception as exc:  # pragma: no cover - best effort
                        print(f"[Rel3D] Warning: failed to load llm_data_config "
                              f"{self.config.general.llm_data_config}: {exc}")
                        self.data_to_load = {}
                # Debug: always print the effective sampling config (helps diagnose env overrides not taking effect).
                try:
                        if int(getattr(self, "global_rank", 0)) == 0:
                                path = getattr(self.config.general, "llm_data_config", None)
                                try:
                                        path_s = str(path)
                                except Exception:
                                        path_s = "<unavailable>"
                                keys = []
                                try:
                                        keys = sorted(list(self.data_to_load.keys())) if isinstance(self.data_to_load, dict) else []
                                except Exception:
                                        keys = []
                                print(
                                        f"[SSR3DLLM][llm_data_config] path={path_s} keys={keys} cfg={self.data_to_load}",
                                        flush=True,
                                )
                except Exception:
                        pass

                if llama_config.enable_llm:
                        print('*****************************************************************')
                        print(f'Using config: {self.config.general.llm_config}')
                        print(f'Using data config: {self.config.general.llm_data_config}')
                        assert llama_config.vicuna_version == llama_config.vicuna_version, "conflict model"
                        print('*****************************************************************')

                        if self.global_rank == 0:
                                llama_config.save_pretrained(
                                        f"{self.config['general']['save_dir']}")
                        if not llama_config.load_pretrain_weight:
                                logger.warning(
                                        f"llm pretrain weight is not loaded: do you need to debug or resume from last_epoch.ckpt !?"
                                )

                        os.makedirs(llama_config.save_path, exist_ok=True)
                        os.makedirs(f"{llama_config.save_path}/m3drefer", exist_ok=True)

                        assert not self.config.general.use_dbscan
                        # init tokenizer and add special tokens
                        from models.LLM.LLama3d import load_llama_model_and_tokenizer
                        self.llama_model, self.llama_tokenizer = load_llama_model_and_tokenizer(
                                llama_config)
                else:
                        print(" ====================== llm is disabled ===================")

                self.llama_config = llama_config

        def forward(
                        self, x, point2segment=None, raw_coordinates=None, extra_lang=None, is_eval=False
        ):
                x = self.model(
                        x,
                        point2segment,
                        raw_coordinates=raw_coordinates,
                        is_eval=is_eval,
                        extra_lang=extra_lang
                )
                return x

        def prepare_llm(self, output, extra_lang, assigner_indices=None, target=None, raw_data=None, file_names=None):
                batch_lang_infos = []
                batch_map_target_to_query = []

                batch_size = output['queries_hidden_state'].shape[0]
                # When `num_concat_texts=0` (e.g. rel3d step-token SFT), the collated
                # BatchLangData may have `batch_num_concat_texts=None`. Treat it as 0 for
                # each scene to keep downstream code working.
                batch_num_concat_texts = getattr(extra_lang, "batch_num_concat_texts", None)
                if batch_num_concat_texts is None:
                        batch_num_concat_texts = [0 for _ in range(batch_size)]
                batch_concat_texts = getattr(extra_lang, "batch_concat_texts", None) or []

                limits: Dict[str, int] = {}
                counters: Dict[str, int] = defaultdict(int)
                if not self.training:
                        limits = getattr(self, "eval_lang_type_limits", {})
                        counters = defaultdict(int, {k: 0 for k in limits})

                total_concat_texts = 0
                for bid in range(batch_size):
                        q_hidden_cached, q_norm_cached = None, None

                        n_concat = int(batch_num_concat_texts[bid]) if bid < len(batch_num_concat_texts) else 0
                        raw_texts_bid = []
                        # Prefer the per-raw-text strings provided by the collate fn.
                        # This avoids fragile reconstruction from concatenated texts (which can
                        # break when the separator changes or when raw texts contain ". ").
                        raw_texts_all = getattr(extra_lang, "raw_texts", None)
                        if (
                                isinstance(raw_texts_all, (list, tuple))
                                and bid < len(raw_texts_all)
                                and isinstance(raw_texts_all[bid], (list, tuple))
                        ):
                                try:
                                        raw_texts_bid = [str(x) for x in raw_texts_all[bid] if str(x).strip()]
                                except Exception:
                                        raw_texts_bid = []
                        # Backward-compatible fallback (best-effort): split the concatenated string.
                        if not raw_texts_bid and n_concat > 0:
                                raw_texts_bid = ''.join(
                                        batch_concat_texts[total_concat_texts: total_concat_texts + n_concat]
                                ).split('. ')[:-1]  # remove last empty ''
                        # Some training modes (e.g. rel3d-only step-token SFT) disable
                        # language-conditioned detection, so these fields can be missing/None.
                        flatten_pairs_all = getattr(extra_lang, "flatten_lang_token_inst_id_pairs", None)
                        raw_lang_types_all = getattr(extra_lang, "raw_lang_types", None)
                        # Per-raw-text dataset target id (original ScanNet instance id, BEFORE remap_inst_ids()).
                        # Used for teacher-forced grounding_steps lookup (ScanRefer/M3DRef).
                        raw_target_gt_ids_all = getattr(extra_lang, "raw_target_gt_ids", None)

                        if output['extra_queries'] is not None:
                                assert n_concat == output[
                                        'extra_queries']['embedded'].shape[0] // batch_size

                                # get the start and end token id for each sentence
                                each_lang_query_features = []
                                for concat_text_id, (concat_text_pos_ids, text_token_mask) in enumerate(zip(
                                                output['extra_queries']['position_ids'][total_concat_texts:
                                                total_concat_texts + n_concat],
                                                output['extra_queries']['text_token_mask'][total_concat_texts:
                                                total_concat_texts + n_concat]
                                )):
                                        each_concat_text_features = []
                                        start_i_of_pos_ids = 0
                                        while start_i_of_pos_ids < len(concat_text_pos_ids) and text_token_mask[start_i_of_pos_ids]:
                                                i = start_i_of_pos_ids + 1
                                                while i < len(concat_text_pos_ids) and text_token_mask[i] and concat_text_pos_ids[i - 1] < concat_text_pos_ids[i]:
                                                        i += 1
                                                each_concat_text_features.append(
                                                        output['extra_queries']['embedded'][total_concat_texts + concat_text_id, start_i_of_pos_ids:i])
                                                start_i_of_pos_ids = i
                                        each_lang_query_features.extend(
                                                each_concat_text_features[1:-1])  # drop first 0 and last 0
                                # get targets for each sentence (may be missing in rel3d-only modes)
                                if flatten_pairs_all is None or raw_lang_types_all is None:
                                        flatten_lang_token_inst_id_pair = []
                                        raw_lang_type = []
                                else:
                                        flatten_lang_token_inst_id_pair = flatten_pairs_all[bid] or []
                                        raw_lang_type = raw_lang_types_all[bid] or []
                                # If raw_text extraction is inconsistent, try a best-effort fallback and
                                # then align lengths to avoid index errors.
                                if raw_lang_type and len(raw_texts_bid) != len(raw_lang_type):
                                        if n_concat > 0:
                                                raw_texts_fallback = ''.join(
                                                        batch_concat_texts[total_concat_texts: total_concat_texts + n_concat]
                                                ).split('. ')[:-1]
                                                if len(raw_texts_fallback) == len(raw_lang_type):
                                                        raw_texts_bid = raw_texts_fallback
                                        # Last resort: pad/truncate.
                                        if len(raw_texts_bid) < len(raw_lang_type):
                                                raw_texts_bid = list(raw_texts_bid) + [""] * (len(raw_lang_type) - len(raw_texts_bid))
                                        elif len(raw_texts_bid) > len(raw_lang_type):
                                                raw_texts_bid = list(raw_texts_bid)[: len(raw_lang_type)]

                                if flatten_lang_token_inst_id_pair and raw_lang_type:
                                        # Some rare tokenization / separator mismatches can make the extracted
                                        # per-sentence features disagree with metadata lengths. Prefer a
                                        # best-effort alignment over crashing the entire run.
                                        n_pairs = len(flatten_lang_token_inst_id_pair)
                                        n_types = len(raw_lang_type)
                                        n_feats = len(each_lang_query_features)
                                        n_texts = len(raw_texts_bid)
                                        if not (n_pairs == n_types == n_feats == n_texts):
                                                if (
                                                        int(getattr(self, "global_rank", 0)) == 0
                                                        and not getattr(self, "_prepare_llm_len_mismatch_warned", False)
                                                ):
                                                        self._prepare_llm_len_mismatch_warned = True
                                                        try:
                                                                print(
                                                                        "[prepare_llm][WARN] length mismatch; truncating to min length: "
                                                                        f"pairs={n_pairs} types={n_types} feats={n_feats} texts={n_texts} "
                                                                        f"(scene_bid={bid}, n_concat={n_concat})",
                                                                        flush=True,
                                                                )
                                                        except Exception:
                                                                pass
                                                m = min(n_pairs, n_types, n_feats, n_texts)
                                                if m <= 0:
                                                        flatten_lang_token_inst_id_pair = []
                                                        raw_lang_type = []
                                                        each_lang_query_features = []
                                                        raw_texts_bid = []
                                                else:
                                                        flatten_lang_token_inst_id_pair = flatten_lang_token_inst_id_pair[:m]
                                                        raw_lang_type = raw_lang_type[:m]
                                                        each_lang_query_features = each_lang_query_features[:m]
                                                        raw_texts_bid = raw_texts_bid[:m]
                                else:
                                        # get targets for each sentence
                                        if flatten_pairs_all is None or raw_lang_types_all is None:
                                                flatten_lang_token_inst_id_pair = []
                                                raw_lang_type = []
                                        else:
                                                flatten_lang_token_inst_id_pair = flatten_pairs_all[bid] or []
                                                raw_lang_type = raw_lang_types_all[bid] or []
        
                                        each_lang_query_features = [None] * len(raw_lang_type)
                                        raw_texts_bid = raw_texts_bid + \
                                                        [None] * (len(raw_lang_type) -
                                                                  len(raw_texts_bid))
        
                                expected_n_raw = int(len(raw_lang_type)) if raw_lang_type is not None else 0

                                # Align per-raw-text target ids to this bid (may be missing in some modes).
                                raw_target_gt_ids_bid = []
                                if (
                                        isinstance(raw_target_gt_ids_all, (list, tuple))
                                        and bid < len(raw_target_gt_ids_all)
                                        and isinstance(raw_target_gt_ids_all[bid], (list, tuple))
                                ):
                                        raw_target_gt_ids_bid = list(raw_target_gt_ids_all[bid])
                                # Best-effort align.
                                if expected_n_raw > 0:
                                        if len(raw_target_gt_ids_bid) < expected_n_raw:
                                                raw_target_gt_ids_bid = raw_target_gt_ids_bid + [None] * (
                                                        expected_n_raw - len(raw_target_gt_ids_bid)
                                                )
                                        elif len(raw_target_gt_ids_bid) > expected_n_raw:
                                                raw_target_gt_ids_bid = raw_target_gt_ids_bid[:expected_n_raw]

                        # ---------- compute target to query mapping -------------
                        assert self.llama_config.valid_target_iou >= 0., 'The matched iou should be larger than 0.'
                        use_gt_llm = getattr(self.config.general, "use_gt_proposals_for_llm", False)
                        if self.model.train_on_segments:
                                pred_masks = (
                                        output["pred_masks"][bid]
                                        .detach()
                                        .cpu()[target[bid]["point2segment"].cpu()]
                                )  # map back to raw points
                        else:
                                pred_masks = (
                                        output["pred_masks"][bid]
                                        .detach()
                                        .cpu()
                                )

                        if use_gt_llm:
                                # Oracle / GT proposal mode for LLM: directly map each
                                # GT instance to its own index; consider all targets valid.
                                num_inst = target[bid]['masks'].shape[0]
                                map_target_to_query = torch.arange(num_inst, device='cpu')
                                gt_ious = torch.ones(num_inst, device='cpu')
                                valid_target = np.ones(num_inst, dtype=bool)
                                if not self.training:
                                        max_gt_iou = torch.ones(num_inst, device='cpu')
                                        max_gt_iou_query_id = map_target_to_query.clone()
                        elif not self.training:  # evaluation use box iou
                                target_mask = target[bid]['masks'].cpu().float()

                                # box iou
                                target_boxes = []
                                full_res_target_mask = target_mask[
                                        :,
                                        raw_data.inverse_maps[0],
                                ]
                                for mask in full_res_target_mask:
                                        gt_points = raw_data.full_res_coords[0][mask > 0.5]
                                        min_vals, max_vals = gt_points.min(axis=0), gt_points.max(axis=0)
                                        target_boxes.append([min_vals, max_vals])  # 2×3 numpy
                                if target_boxes:
                                        target_boxes_np = np.stack(target_boxes, axis=0)  # [N,2,3]
                                else:
                                        target_boxes_np = np.zeros((0, 2, 3), dtype=np.float32)
                                target_boxes = torch.from_numpy(target_boxes_np)

                                pred_boxes = []
                                full_res_pred_mask = pred_masks.T[
                                        :,
                                        raw_data.inverse_maps[0],
                                ]
                                for mask in full_res_pred_mask:
                                        gt_points = raw_data.full_res_coords[0][mask > 0.5]
                                        if gt_points.shape[0] > 0:
                                                min_vals, max_vals = gt_points.min(axis=0), gt_points.max(axis=0)
                                        else:
                                                min_vals, max_vals = np.zeros((3,)), np.zeros((3,))
                                        pred_boxes.append([min_vals, max_vals])
                                if pred_boxes:
                                        pred_boxes_np = np.stack(pred_boxes, axis=0)  # [M,2,3]
                                else:
                                        pred_boxes_np = np.zeros((0, 2, 3), dtype=np.float32)
                                pred_boxes = torch.from_numpy(pred_boxes_np)

                                box_iou = torch.zeros(
                                        (len(target_boxes), len(pred_boxes)), device='cpu')
                                for i, tb in enumerate(target_boxes):
                                        for j, pb in enumerate(pred_boxes):
                                                box_iou[i, j] = get_batch_aabb_pair_ious(
                                                        tb[None], pb[None])

                                map_target_to_query = box_iou.argmax(1)
                                gt_ious = box_iou.max(1)[0]

                                # following LL3DA, Scan2Cap, Vote2Cap-DETR++
                                max_query_iou, max_query_iou_gt_id = box_iou.max(
                                        dim=0)  # for nqueries
                                tmpiou = torch.zeros_like(box_iou)
                                tmpiou[max_query_iou_gt_id, torch.arange(
                                        tmpiou.shape[1])] = max_query_iou
                                max_gt_iou, max_gt_iou_query_id = tmpiou.max(
                                        dim=1)  # find the maximum gt

                                valid_target = np.ones_like(
                                        map_target_to_query, dtype=bool)
                        else:
                                # mask iou (not use)
                                inter = (target[bid]['masks'].cpu().float()
                                         @ (pred_masks > 0).float())
                                outer = logical_or_sum(
                                        target[bid]['masks'].cpu(), (pred_masks.T > 0))
                                iou = inter / (outer + 1e-8)
                                map_target_to_query = iou.argmax(1)
                                gt_ious = iou.max(1)[0]
                                valid_target = gt_ious >= self.llama_config.valid_target_iou

                        batch_map_target_to_query.append(
                                [map_target_to_query, valid_target])

                        q_hidden_cached, q_norm_cached = self._select_instance_features_for_llm(
                                output, bid, target=target
                        )

                        if self.training:
                                from utils.sample_utils import sample_by_type
                                max_sample_lang_type_count = dict(
                                        detection=self.data_to_load.get("detection", 0),
                                        scanrefer=self.data_to_load.get("scanrefer", 0),
                                        m3dref=self.data_to_load.get("m3dref", 0),
                                        rel3dref=self.data_to_load.get("rel3dref", 0),
                                        referit3d=self.data_to_load.get("referit3d", 0),
                                        groundedscenecaption=self.data_to_load.get("groundedscenecaption", 0),
                                )
                                if raw_lang_type:
                                        lang_type_with_index = np.asarray(
                                                [(d.split(':')[0], i) for i, d in enumerate(raw_lang_type)], dtype=object)
                                        sampled_lang_type_with_index = sample_by_type(
                                                lang_type_with_index, max_sample_lang_type_count)
                                        sampled_indices = sampled_lang_type_with_index[:, 1]
                                else:
                                        sampled_indices = []
                        else:
                                local_counters = counters
                                sampled_indices = []
                                # Optional: prioritize some prefixes during evaluation sampling.
                                # This is useful for fast sanity checks (e.g. always include scanrefer/m3dref
                                # within a small max_eval_queries budget).
                                #
                                # Env:
                                #   SSR3DLLM_EVAL_PREFIX_PRIORITY="scanrefer,m3dref"
                                idx_list = list(range(len(raw_lang_type)))
                                try:
                                        raw = str(os.environ.get("SSR3DLLM_EVAL_PREFIX_PRIORITY", "")).strip()
                                except Exception:
                                        raw = ""
                                if raw:
                                        pfx = [p.strip() for p in raw.split(",") if p.strip()]
                                        rank = {p: i for i, p in enumerate(pfx)}

                                        def _rk(ii: int) -> tuple:
                                                try:
                                                        lt = raw_lang_type[ii]
                                                        px = lt.split(":")[0] if isinstance(lt, str) else ""
                                                except Exception:
                                                        px = ""
                                                return (rank.get(px, 10 ** 9), ii)

                                        idx_list.sort(key=_rk)

                                for idx in idx_list:
                                        lang_type = raw_lang_type[idx]
                                        prefix = lang_type.split(':')[0]
                                        limit = limits.get(prefix)
                                        if limit is not None and local_counters[prefix] >= limit:
                                                continue
                                        sampled_indices.append(idx)
                                        if limit is not None:
                                                local_counters[prefix] += 1
                                        if self.max_eval_queries and len(sampled_indices) >= self.max_eval_queries:
                                                break
                                if self.max_eval_queries and len(sampled_indices) > self.max_eval_queries:
                                        sampled_indices = sampled_indices[:self.max_eval_queries]

                        for sample_idx in sampled_indices:
                                lang_token_inst_id_pair, lang_text, lang_type, lang_feat = flatten_lang_token_inst_id_pair[
                                        sample_idx], raw_texts_bid[sample_idx], raw_lang_type[sample_idx], each_lang_query_features[sample_idx]

                                lang_info = lang_info_data.from_grounding(
                                        raw_text=lang_text,
                                        lang_type=lang_type,
                                        lang_token_inst_id_pair=lang_token_inst_id_pair,
                                        map_target_to_query=map_target_to_query,
                                        valid_target=valid_target,
                                        support_counting=getattr(
                                                self.llama_config, "support_counting", False),
                                        count_instance=getattr(
                                                self.llama_config, "count_instance", True),
                                )
                                # from_grounding may return None for unsupported lang types.
                                if lang_info is None:
                                        continue
                                # Attach dataset-defined target_gt_id for teacher-forced grounding_steps lookup.
                                # IMPORTANT: this must be the original ScanNet instance id (pre remap_inst_ids()).
                                tgt_id = None
                                if sample_idx < len(raw_target_gt_ids_bid):
                                        try:
                                                v = raw_target_gt_ids_bid[sample_idx]
                                                tgt_id = int(v) if v is not None else None
                                        except Exception:
                                                tgt_id = None
                                lang_info.target_gt_id = tgt_id
                                # Keep the raw grounding text around for teacher-forced
                                # referential_order lookup (ScanRefer/M3DRef step4).
                                try:
                                        lang_info.raw_grounding_text = str(lang_text)
                                except Exception:
                                        lang_info.raw_grounding_text = None
                                lang_info.append_prompt_postfix()
                                # SSR3DLLM: mark grounding-style language items as geometry-triggered by injecting
                                # the routing token `<geom>` (for unified multi-task training with a shared checkpoint).
                                #
                                # Enable via:
                                #   SSR3DLLM_GROUNDING_ADD_GEOM_TOKEN=1
                                # Optionally restrict prefixes:
                                #   SSR3DLLM_GROUNDING_GEOM_PREFIXES=scanrefer,m3dref,referit3d
                                try:
                                        add_geom = str(
                                                os.environ.get("SSR3DLLM_GROUNDING_ADD_GEOM_TOKEN", "0")
                                        ).strip().lower() in {"1", "true", "yes", "y", "on"}
                                        if add_geom:
                                                try:
                                                        pfx = str(lang_info.lang_type).split(":")[0]
                                                except Exception:
                                                        pfx = ""
                                                pfx_raw = str(
                                                        os.environ.get(
                                                                "SSR3DLLM_GROUNDING_GEOM_PREFIXES",
                                                                "scanrefer,m3dref,referit3d",
                                                        )
                                                ).strip()
                                                pfx_set = {p.strip() for p in pfx_raw.split(",") if p.strip()}
                                                if (not pfx_set) or (pfx in pfx_set):
                                                        q = str(getattr(lang_info, "question", "") or "")
                                                        if "<geom>" not in q:
                                                                lang_info.question = f"<geom> {q}".strip()
                                                        setattr(lang_info, "use_geom_trigger", True)
                                except Exception:
                                        pass
                                if q_hidden_cached is not None and q_norm_cached is not None:
                                        lang_info.set_context_features(
                                                query_hidden_feature=q_hidden_cached,
                                                query_normalized_embed=q_norm_cached,
                                        )
                                lang_info.set_batch_idx(bid)
                                # SSR3DLLM/Vigor backend needs per-sample scene_id to load per-scene Mask3D
                                # exports (e.g., pred_box_info / gt_to_query_map). Attach it here to
                                # keep downstream geometry heads stateless.
                                if (
                                        file_names is not None
                                        and isinstance(file_names, (list, tuple))
                                        and bid < len(file_names)
                                        and isinstance(file_names[bid], str)
                                ):
                                        lang_info.scene_id = file_names[bid]
                                        # Optional: attach teacher-forced referential_order for
                                        # ScanRefer/M3DRef grounding samples, so the SSR3DLLM
                                        # stepslot adapter can produce llm_step_embeds.
                                        #
                                        # This is opt-in and controlled by:
                                        # - SSR3DLLM_GROUNDING_STEPS_JSON_{TRAIN,EVAL}
                                        # - SSR3DLLM_GROUNDING_STEPS_STRICT=1 to error on missing
                                        try:
                                                prefix = str(lang_info.lang_type).split(":")[0]
                                        except Exception:
                                                prefix = ""
                                        if prefix in {"scanrefer", "m3dref"}:
                                                online_order_mode = str(
                                                        os.environ.get("SSR3DLLM_GROUNDING_ORDER_ONLINE", "0")
                                                ).strip().lower()
                                                online_order_llm = online_order_mode in {"llm", "model", "gen", "generate"}
                                                steps_strict = str(
                                                        os.environ.get("SSR3DLLM_GROUNDING_STEPS_STRICT", "0")
                                                ).strip().lower() in {"1", "true", "yes", "on"}
                                                steps_skip_missing = str(
                                                        os.environ.get("SSR3DLLM_GROUNDING_STEPS_SKIP_MISSING", "0")
                                                ).strip().lower() in {"1", "true", "yes", "on"}
                                                raw_grounding_text = getattr(lang_info, "raw_grounding_text", "") or ""
                                                tgt_id = getattr(lang_info, "target_gt_id", None)
                                                try:
                                                        tgt_id = int(tgt_id) if tgt_id is not None else None
                                                except Exception:
                                                        tgt_id = None

                                                split = "train" if self.training else "eval"

                                                # In ScanRefer/M3DRef we optionally attach a teacher-forced referential
                                                # order (list[str]) so the Vigor stepslot pipeline can produce per-step
                                                # embeddings. Missing orders can cause:
                                                # - massive sample dropping
                                                # - per-rank filtering divergence → DDP hangs
                                                #
                                                # This block enforces a strict policy (if enabled).

                                                online_attempted = False
                                                online_ok = False
                                                online_fail_reason = None
                                                online_out_preview = None

                                                # ------------------------------------------------------------
                                                # SSR3DLLM: single-step fallback for ScanRefer/M3DRef (minimal probe)
                                                #
                                                # Motivation:
                                                # - ScanRefer/M3DRef do not always have reliable multi-step chains.
                                                # - For debugging "is the geom pipeline working at all?", we want a
                                                #   deterministic, GT-derived single-step chain to avoid massive skipping
                                                #   due to missing referential_order / steps_json coverage.
                                                #
                                                # Behavior:
                                                # - When enabled, we set `lang_info.rel_referential_order` to a fixed-length
                                                #   list of the *target semantic class name* (repeated) based on GT labels,
                                                #   i.e. ["chair","chair","chair","chair"] (order_len controlled by env).
                                                # - This avoids any heuristic parsing and does NOT require steps_json.
                                                #
                                                # Enable via:
                                                #   export SSR3DLLM_GROUNDING_SINGLE_STEP=1
                                                # Optionally restrict prefixes:
                                                #   export SSR3DLLM_GROUNDING_SINGLE_STEP_PREFIXES=scanrefer,m3dref
                                                # ------------------------------------------------------------
                                                single_step = str(os.environ.get("SSR3DLLM_GROUNDING_SINGLE_STEP", "0")).strip().lower() in {
                                                        "1",
                                                        "true",
                                                        "yes",
                                                        "on",
                                                }
                                                # Best-effort: if `target_gt_id` is missing (common for some m3dref collate paths),
                                                # fall back to the first `inst_ids_answer` entry so we can avoid mass skipping.
                                                if single_step and tgt_id is None:
                                                        try:
                                                                inst_ans = getattr(lang_info, "inst_ids_answer", None)
                                                                if isinstance(inst_ans, list) and inst_ans:
                                                                        first = inst_ans[0]
                                                                        if isinstance(first, (list, tuple)) and first:
                                                                                tgt_id = int(first[0])
                                                                        elif isinstance(first, (int, np.integer)):
                                                                                tgt_id = int(first)
                                                        except Exception:
                                                                pass
                                                        try:
                                                                if tgt_id is not None:
                                                                        setattr(lang_info, "target_gt_id", int(tgt_id))
                                                        except Exception:
                                                                pass
                                                if single_step and tgt_id is not None:
                                                        prefixes_env = str(
                                                                os.environ.get(
                                                                        "SSR3DLLM_GROUNDING_SINGLE_STEP_PREFIXES",
                                                                        "scanrefer,m3dref",
                                                                )
                                                        ).strip()
                                                        single_prefixes = {
                                                                p.strip() for p in prefixes_env.split(",") if p.strip()
                                                        }
                                                        if (not single_prefixes) or (prefix in single_prefixes):
                                                                try:
                                                                        import numpy as _np
                                                                        import torch as _torch

                                                                        order_len_env = os.environ.get("SSR3DLLM_STEP_ORDER_LEN", "").strip()
                                                                        try:
                                                                                _order_len = int(order_len_env) if order_len_env else 4
                                                                        except Exception:
                                                                                _order_len = 4
                                                                        _order_len = max(1, int(_order_len))

                                                                        labels_arr = None
                                                                        if isinstance(target, list) and bid < len(target) and isinstance(target[bid], dict):
                                                                                labels_arr = target[bid].get("labels", None)
                                                                        if isinstance(labels_arr, _torch.Tensor):
                                                                                labels_np = labels_arr.detach().cpu().numpy()
                                                                        else:
                                                                                labels_np = _np.asarray(labels_arr) if labels_arr is not None else None

                                                                        phrase = None
                                                                        used_tid = None
                                                                        used_mask_idx = None
                                                                        # Prefer the explicit `target_gt_id`, but note:
                                                                        # - `target[bid]["labels"]` is *per-instance* class ids (1D), not per-point labels.
                                                                        # - `target[bid]["instance_mapping"]` maps ORIGINAL instance ids -> mask row indices.
                                                                        # So we must resolve instance_id -> mask_row -> class_id -> class_name.
                                                                        cand_tids = []
                                                                        if tgt_id is not None:
                                                                                cand_tids.append(int(tgt_id))
                                                                        try:
                                                                                inst_ans = getattr(lang_info, "inst_ids_answer", None)
                                                                                if isinstance(inst_ans, list) and inst_ans:
                                                                                        first = inst_ans[0]
                                                                                        if isinstance(first, (list, tuple)):
                                                                                                for x in first:
                                                                                                        try:
                                                                                                                cand_tids.append(int(x))
                                                                                                        except Exception:
                                                                                                                pass
                                                                        except Exception:
                                                                                pass
                                                                        # Dedup while preserving order.
                                                                        seen = set()
                                                                        cand_tids2 = []
                                                                        for x in cand_tids:
                                                                                if x in seen:
                                                                                        continue
                                                                                seen.add(x)
                                                                                cand_tids2.append(x)

                                                                        try:
                                                                                td = None
                                                                                if isinstance(target, list) and bid < len(target) and isinstance(target[bid], dict):
                                                                                        td = target[bid]
                                                                                labels_vec = td.get("labels", None) if isinstance(td, dict) else None
                                                                                inst_mapping = td.get("instance_mapping", None) if isinstance(td, dict) else None
                                                                                orig_ids = td.get("orig_instance_ids", None) if isinstance(td, dict) else None

                                                                                if isinstance(labels_vec, _torch.Tensor):
                                                                                        labels_np = labels_vec.detach().cpu().numpy()
                                                                                else:
                                                                                        labels_np = _np.asarray(labels_vec) if labels_vec is not None else None

                                                                                if labels_np is not None and labels_np.ndim == 1:
                                                                                        # NOTE: `labels_np` is produced by `dataset_code.utils.get_instance_masks`:
                                                                                        #   l = clamp(label_id - label_offset, min=0)
                                                                                        # after filtering `filter_out_classes`. Therefore the class id space
                                                                                        # here is NOT the raw ScanNet200 index space, and must be remapped back
                                                                                        # to a human-readable name accordingly (otherwise many samples become
                                                                                        # `phrase='wall'` etc when label_offset=2 and wall/floor are filtered).
                                                                                        try:
                                                                                                from baseline.dataset.datasets.scannet200.scannet200_constants import CLASS_LABELS_200  # type: ignore
                                                                                        except Exception:
                                                                                                CLASS_LABELS_200 = []  # type: ignore

                                                                                        def _resolve_shifted_scannet200_class_name(shifted_cid: int) -> str | None:
                                                                                                try:
                                                                                                        cache = getattr(self, "_ssr3dllm_shifted_scannet200_class_cache", None)
                                                                                                except Exception:
                                                                                                        cache = None

                                                                                                # Prefer the dataset's own config (it controls the shift logic).
                                                                                                ds = None
                                                                                                for _cand in (
                                                                                                        getattr(self, "validation_dataset", None),
                                                                                                        getattr(self, "train_dataset", None),
                                                                                                        getattr(self, "test_dataset", None),
                                                                                                ):
                                                                                                        if _cand is not None:
                                                                                                                ds = _cand
                                                                                                                break
                                                                                                try:
                                                                                                        label_offset = int(getattr(ds, "label_offset", 0) or 0)
                                                                                                except Exception:
                                                                                                        label_offset = 0
                                                                                                try:
                                                                                                        filter_out = getattr(ds, "filter_out_classes", None)
                                                                                                except Exception:
                                                                                                        filter_out = None
                                                                                                try:
                                                                                                        filter_out_set = {int(x) for x in (filter_out or [])}
                                                                                                except Exception:
                                                                                                        filter_out_set = set()

                                                                                                cache_key = (label_offset, tuple(sorted(filter_out_set)))
                                                                                                if isinstance(cache, dict) and cache_key in cache:
                                                                                                        mapping = cache[cache_key]
                                                                                                else:
                                                                                                        mapping = {}
                                                                                                        if CLASS_LABELS_200:
                                                                                                                for label_id, name in enumerate(CLASS_LABELS_200):
                                                                                                                        if label_id in filter_out_set:
                                                                                                                                continue
                                                                                                                        shifted = max(int(label_id) - int(label_offset), 0)
                                                                                                                        # If collisions exist due to clamp(), keep the first mapping
                                                                                                                        # (filter_out_classes typically makes it unique).
                                                                                                                        if shifted not in mapping:
                                                                                                                                mapping[shifted] = str(name)
                                                                                                        if not isinstance(cache, dict):
                                                                                                                cache = {}
                                                                                                        cache[cache_key] = mapping
                                                                                                        try:
                                                                                                                setattr(self, "_ssr3dllm_shifted_scannet200_class_cache", cache)
                                                                                                        except Exception:
                                                                                                                pass

                                                                                                name = mapping.get(int(shifted_cid), None)
                                                                                                if not name:
                                                                                                        return None
                                                                                                return str(name).replace("_", " ").strip()
                                                                                        for tid in cand_tids2:
                                                                                                midx = None
                                                                                                if isinstance(inst_mapping, dict):
                                                                                                        try:
                                                                                                                midx = inst_mapping.get(int(tid), None)
                                                                                                        except Exception:
                                                                                                                midx = None
                                                                                                if midx is None and isinstance(orig_ids, list):
                                                                                                        try:
                                                                                                                midx = int(orig_ids.index(int(tid)))
                                                                                                        except Exception:
                                                                                                                midx = None
                                                                                                if midx is None:
                                                                                                        continue
                                                                                                try:
                                                                                                        midx_i = int(midx)
                                                                                                except Exception:
                                                                                                        continue
                                                                                                if not (0 <= midx_i < int(labels_np.shape[0])):
                                                                                                        continue
                                                                                                cid = int(labels_np[midx_i])
                                                                                                phrase = _resolve_shifted_scannet200_class_name(cid)
                                                                                                if phrase:
                                                                                                        used_tid = int(tid)
                                                                                                        used_mask_idx = int(midx_i)
                                                                                                        break
                                                                        except Exception:
                                                                                phrase = None

                                                                        if phrase:
                                                                                lang_info.rel_referential_order = [phrase] * int(_order_len)
                                                                                setattr(lang_info, "rel_referential_order_source", "single_step_gt_class")
                                                                                # Oracle effective chain length for VarLen-STOP masking:
                                                                                # ScanRefer/M3DRef are evaluated/trained in single-step mode.
                                                                                try:
                                                                                        setattr(lang_info, "ori_order_len", 1)
                                                                                except Exception:
                                                                                        pass
                                                                                # Also export the target semantic class for downstream
                                                                                # step-slot supervision / debugging (best-effort).
                                                                                try:
                                                                                        setattr(lang_info, "ssr3dllm_target_class_name", str(phrase))
                                                                                except Exception:
                                                                                        pass
                                                                                try:
                                                                                        cid200 = int(CLASS_LABELS_200.index(str(phrase)))
                                                                                        setattr(lang_info, "ssr3dllm_target_class_id200", cid200)
                                                                                except Exception:
                                                                                        pass
                                                                                try:
                                                                                        n0 = int(getattr(self, "_ssr3dllm_single_step_debug_n", 0) or 0)
                                                                                except Exception:
                                                                                        n0 = 0
                                                                                try:
                                                                                        max_n = int(os.environ.get("SSR3DLLM_GROUNDING_SINGLE_STEP_DEBUG_MAX", "5").strip() or "5")
                                                                                except Exception:
                                                                                        max_n = 5
                                                                                if max_n > 0 and n0 < max_n:
                                                                                        try:
                                                                                                logger.warning(
                                                                                                        "[SSR3DLLM][grounding_steps][single_step] "
                                                                                                        "scene={} prefix={} target_gt_id={} used_inst_id={} used_mask_idx={} phrase='{}' order_len={} (debug {}/{})",
                                                                                                        getattr(lang_info, "scene_id", None),
                                                                                                        prefix,
                                                                                                        tgt_id,
                                                                                                        used_tid,
                                                                                                        used_mask_idx,
                                                                                                        phrase,
                                                                                                        int(_order_len),
                                                                                                        n0 + 1,
                                                                                                        max_n,
                                                                                                )
                                                                                        except Exception:
                                                                                                pass
                                                                                        try:
                                                                                                setattr(self, "_ssr3dllm_single_step_debug_n", n0 + 1)
                                                                                        except Exception:
                                                                                                pass
                                                                except Exception:
                                                                        # If anything goes wrong, fall back to the normal lookup/online path.
                                                                        pass

                                                if (not self.training) and online_order_llm:
                                                        online_attempted = True
                                                        try:
                                                                import json as _json
                                                                import re as _re

                                                                order_len_env = os.environ.get("SSR3DLLM_STEP_ORDER_LEN", "").strip()
                                                                try:
                                                                        order_len = int(order_len_env) if order_len_env else 4
                                                                except Exception:
                                                                        order_len = 4
                                                                order_len = max(1, int(order_len))

                                                                max_new_env = os.environ.get("SSR3DLLM_GROUNDING_ORDER_MAX_NEW_TOKENS", "").strip()
                                                                try:
                                                                        max_new = int(max_new_env) if max_new_env else 64
                                                                except Exception:
                                                                        max_new = 64
                                                                max_new = max(8, int(max_new))

                                                                prompt = (
                                                                        f"<geom> {str(raw_grounding_text).strip()}\n"
                                                                        "You MUST output a referential chain in the exact format below.\n"
                                                                        f"Rules:\n"
                                                                        f"1) Start from <step1> (do NOT skip to <step2>/<step3>).\n"
                                                                        f"2) Output EXACTLY {order_len} steps.\n"
                                                                        f"3) If you have fewer than {order_len} distinct steps, repeat the LAST step phrase to pad.\n"
                                                                        f"4) Output ONLY the chain text, no extra words.\n"
                                                                        f"Format:\n"
                                                                        f"phrase <step1> phrase <step2> ... phrase <step{order_len}>\n"
                                                                )
                                                                # IMPORTANT: if someone enabled SSR3DLLM_ROUTE_GEOM_VIGOR=1,
                                                                # `LLama3d.evaluate()` will bypass LLM generation whenever it sees
                                                                # "<geom>", returning a dummy "<geom>" output. That breaks online
                                                                # order parsing. Force-disable routing for this internal call.
                                                                _old_route = os.environ.get("SSR3DLLM_ROUTE_GEOM_VIGOR", None)
                                                                os.environ["SSR3DLLM_ROUTE_GEOM_VIGOR"] = "0"
                                                                # Speed: default to greedy decoding for online order generation.
                                                                _old_beam = getattr(self.llama_model, "beam_size", None)
                                                                try:
                                                                        _beam_raw = os.environ.get("SSR3DLLM_GROUNDING_ORDER_NUM_BEAMS", "").strip()
                                                                        _beam = int(_beam_raw) if _beam_raw else 1
                                                                except Exception:
                                                                        _beam = 1
                                                                try:
                                                                        if hasattr(self.llama_model, "beam_size"):
                                                                                setattr(self.llama_model, "beam_size", max(1, int(_beam)))
                                                                except Exception:
                                                                        pass
                                                                out_text = self.llama_model.evaluate(
                                                                        input_text_list=[prompt],
                                                                        batch_instance_queries_hidden_state=[q_hidden_cached],
                                                                        batch_instance_queries_normalized_embed=[q_norm_cached],
                                                                        batch_eval_types=["chat:geom_order"],
                                                                        batch_gt_inst_ids=[None],
                                                                        max_new_tokens=int(max_new),
                                                                        use_mini_batch=False,
                                                                        text_only_output=True,
                                                                )
                                                                if _old_route is None:
                                                                        os.environ.pop("SSR3DLLM_ROUTE_GEOM_VIGOR", None)
                                                                else:
                                                                        os.environ["SSR3DLLM_ROUTE_GEOM_VIGOR"] = str(_old_route)
                                                                try:
                                                                        if _old_beam is not None and hasattr(self.llama_model, "beam_size"):
                                                                                setattr(self.llama_model, "beam_size", _old_beam)
                                                                except Exception:
                                                                        pass
                                                                out_text_s = str(out_text or "").strip()
                                                                try:
                                                                        online_out_preview = out_text_s.replace("\n", " ").strip()
                                                                        if online_out_preview and len(online_out_preview) > 240:
                                                                                online_out_preview = online_out_preview[:240] + "..."
                                                                except Exception:
                                                                        online_out_preview = None

                                                                order_list = None
                                                                try:
                                                                        j0 = out_text_s.find("{")
                                                                        j1 = out_text_s.rfind("}")
                                                                        if j0 >= 0 and j1 > j0:
                                                                                obj = _json.loads(out_text_s[j0 : j1 + 1])
                                                                                ro = obj.get("referential_order", None) if isinstance(obj, dict) else None
                                                                                if isinstance(ro, list):
                                                                                        order_list = [str(x).strip() for x in ro if str(x).strip()]
                                                                except Exception:
                                                                        order_list = None

                                                                if not order_list:
                                                                        # NOTE: regex must match "<step3>" etc. Use \d (digit), not literal "\d".
                                                                        steps = list(_re.finditer(r"<step\d+>", out_text_s))
                                                                        if steps:
                                                                                chunks = _re.split(r"<step\d+>", out_text_s)
                                                                                order_list = []
                                                                                for c in chunks[: len(steps)]:
                                                                                        s = str(c).replace("\n", " ").strip()
                                                                                        # Strip common chat wrappers.
                                                                                        s = s.replace("<s>", " ").replace("</s>", " ").strip()
                                                                                        s = _re.sub(r"^\s*assistant\s*:\s*", "", s, flags=_re.IGNORECASE).strip()
                                                                                        s = s.strip(" ,;:.!?")
                                                                                        if s:
                                                                                                order_list.append(s)

                                                                if not order_list:
                                                                        raise RuntimeError("empty/invalid generated order")

                                                                # Enforce fixed-length order list (pad with last phrase).
                                                                if len(order_list) < int(order_len):
                                                                        order_list = list(order_list) + [order_list[-1]] * (int(order_len) - len(order_list))
                                                                elif len(order_list) > int(order_len):
                                                                        order_list = list(order_list)[: int(order_len)]

                                                                lang_info.rel_referential_order = order_list
                                                                online_ok = True
                                                        except Exception:
                                                                # Do NOT fall back to any heuristic.
                                                                # Missing referential_order will be handled by the strict/skip policy below.
                                                                try:
                                                                        online_fail_reason = "online_llm_failed"
                                                                except Exception:
                                                                        online_fail_reason = None

                                                if not isinstance(getattr(lang_info, "rel_referential_order", None), list):
                                                        order = None
                                                        if tgt_id is not None:
                                                                try:
                                                                        from utils.grounding_steps_map import (  # type: ignore
                                                                                load_grounding_steps_map,
                                                                                lookup_referential_order,
                                                                        )
                                                                        order = lookup_referential_order(
                                                                                split=split,
                                                                                scene_id=lang_info.scene_id,
                                                                                target_gt_id=tgt_id,
                                                                                raw_text=raw_grounding_text,
                                                                        )
                                                                except Exception as e:
                                                                        raise RuntimeError(
                                                                                f"[SSR3DLLM][grounding_steps] failed to lookup referential_order "
                                                                                f"(split={'train' if self.training else 'eval'} prefix={prefix} "
                                                                                f"scene={lang_info.scene_id} target_gt_id={tgt_id}): {e}"
                                                                        ) from e
                                                                if order:
                                                                        lang_info.rel_referential_order = order

                                                # Oracle chain length policy for unified SSR3DLLM grounding:
                                                # - ScanRefer/M3DRef are always treated as single-step grounding (L=1),
                                                #   independent of whether a multi-step order is available.
                                                try:
                                                        oracle_l1 = str(
                                                                os.environ.get("SSR3DLLM_ORACLE_LEN_SCANREFER_M3DREF", "1")
                                                        ).strip().lower() in {"1", "true", "yes", "y", "on"}
                                                        if oracle_l1 and prefix in {"scanrefer", "m3dref"}:
                                                                setattr(lang_info, "ori_order_len", 1)
                                                except Exception:
                                                        pass

                                                # Strict mode: require a list-valued referential order AND a valid target id.
                                                if steps_strict and not isinstance(getattr(lang_info, "rel_referential_order", None), list):
                                                        # target id missing: cannot lookup or build a consistent key.
                                                        if tgt_id is None:
                                                                if steps_skip_missing:
                                                                        printed = int(
                                                                                getattr(self, "_grounding_steps_missing_printed", 0)
                                                                        )
                                                                        if printed < 5:
                                                                                logger.warning(
                                                                                        "[SSR3DLLM][grounding_steps] missing target_gt_id; "
                                                                                        "skip this sample (strict=1, skip_missing=1). "
                                                                                        f"prefix={prefix} scene={lang_info.scene_id}"
                                                                                )
                                                                                setattr(
                                                                                        self,
                                                                                        "_grounding_steps_missing_printed",
                                                                                        printed + 1,
                                                                                )
                                                                        continue
                                                                raise RuntimeError(
                                                                        "[SSR3DLLM][grounding_steps] missing target_gt_id for "
                                                                        f"prefix={prefix} scene={lang_info.scene_id}. "
                                                                        "Check that extra_lang.raw_target_gt_ids is passed through the "
                                                                        "collate pipeline (dataset_code/language_info.py → dataset_code/utils.py → trainer)."
                                                                )

                                                        raw_text_norm = " ".join(str(raw_grounding_text).strip().lower().split())
                                                        raw_text_preview = str(raw_grounding_text).replace("\n", " ").strip()
                                                        if len(raw_text_preview) > 200:
                                                                raw_text_preview = raw_text_preview[:200] + "..."

                                                        if steps_skip_missing:
                                                                steps_map = load_grounding_steps_map(split)
                                                                if not steps_map:
                                                                        raise RuntimeError(
                                                                                "[SSR3DLLM][grounding_steps] steps JSON is not loaded "
                                                                                f"(split={'train' if self.training else 'eval'}). "
                                                                                "Set SSR3DLLM_GROUNDING_STEPS_JSON_{TRAIN,EVAL} (or "
                                                                                "SSR3DLLM_GROUNDING_STEPS_JSON) to a valid JSON file."
                                                                        )

                                                                total = int(
                                                                        getattr(self, "_grounding_steps_missing_total", 0)
                                                                ) + 1
                                                                setattr(self, "_grounding_steps_missing_total", total)
                                                                by_prefix = getattr(
                                                                        self, "_grounding_steps_missing_by_prefix", None
                                                                )
                                                                if not isinstance(by_prefix, dict):
                                                                        by_prefix = {}
                                                                by_prefix[prefix] = int(by_prefix.get(prefix, 0)) + 1
                                                                setattr(self, "_grounding_steps_missing_by_prefix", by_prefix)

                                                                printed = int(
                                                                        getattr(self, "_grounding_steps_missing_printed", 0)
                                                                )
                                                                if printed < 5:
                                                                        online_diag = ""
                                                                        if online_attempted:
                                                                                online_diag = (
                                                                                        f" online_order_mode={online_order_mode} "
                                                                                        f"online_ok={int(bool(online_ok))}"
                                                                                )
                                                                                if online_fail_reason:
                                                                                        online_diag += f" online_fail_reason={online_fail_reason}"
                                                                                if online_out_preview:
                                                                                        online_diag += f" online_out={online_out_preview!r}"

                                                                        steps_diag = ""
                                                                        try:
                                                                                from utils.grounding_steps_map import (  # type: ignore
                                                                                        _json_path_for_split,
                                                                                        load_grounding_steps_map,
                                                                                )
                                                                                _steps_path = _json_path_for_split(split)
                                                                                _steps_size = None
                                                                                try:
                                                                                        if _steps_path and os.path.exists(_steps_path):
                                                                                                _steps_size = int(os.path.getsize(_steps_path))
                                                                                except Exception:
                                                                                        _steps_size = None
                                                                                _steps_map_size = None
                                                                                try:
                                                                                        _steps_map_size = int(len(load_grounding_steps_map(split)))
                                                                                except Exception:
                                                                                        _steps_map_size = None
                                                                                steps_diag = (
                                                                                        f" steps_json={_steps_path!r}"
                                                                                        + (f" steps_bytes={_steps_size}" if _steps_size is not None else "")
                                                                                        + (f" steps_map_size={_steps_map_size}" if _steps_map_size is not None else "")
                                                                                )
                                                                        except Exception:
                                                                                steps_diag = ""

                                                                        logger.warning(
                                                                                "[SSR3DLLM][grounding_steps] missing referential_order; "
                                                                                "skip this sample (strict=1, skip_missing=1). "
                                                                                f"prefix={prefix} scene={lang_info.scene_id} target_gt_id={tgt_id} "
                                                                                f"raw_text_norm={raw_text_norm!r} raw_text={raw_text_preview!r}"
                                                                                + steps_diag
                                                                                + online_diag
                                                                        )
                                                                        setattr(
                                                                                self,
                                                                                "_grounding_steps_missing_printed",
                                                                                printed + 1,
                                                                        )
                                                                continue

                                                        raise RuntimeError(
                                                                "[SSR3DLLM][grounding_steps] missing referential_order for "
                                                                f"prefix={prefix} scene={lang_info.scene_id} target_gt_id={tgt_id}. "
                                                                f"raw_text_norm={raw_text_norm!r} raw_text={raw_text_preview!r} "
                                                                "Set SSR3DLLM_GROUNDING_STEPS_JSON_{TRAIN,EVAL} to a JSON that "
                                                                "covers these samples, or disable strict mode."
                                                        )
                                batch_lang_infos.append(lang_info)

                        if len(raw_data.extra_qa[bid]) > 0:
                                for i, lang_info in enumerate(raw_data.extra_qa[bid]):
                                        if not self.training:
                                                prefix = lang_info.lang_type.split(':')[0]
                                                limit = limits.get(prefix)
                                                if limit is not None and counters[prefix] >= limit:
                                                        continue
                                        if q_hidden_cached is not None and q_norm_cached is not None:
                                                lang_info.set_context_features(
                                                        query_hidden_feature=q_hidden_cached,
                                                        query_normalized_embed=q_norm_cached,
                                                )

                                        try:
                                                if ('scan2cap' in lang_info.lang_type or 'objdesc' in lang_info.lang_type) and not self.training:
                                                        mapping = max_gt_iou_query_id
                                                else:
                                                        mapping = map_target_to_query

                                                lang_info.query_ids_question = []
                                                lang_info.query_ids_answer = []
                                                for inst_ids in lang_info.inst_ids_question:
                                                        lang_info.query_ids_question.append(
                                                                mapping[inst_ids][valid_target[inst_ids]].tolist())
                                                for inst_ids in lang_info.inst_ids_answer:
                                                        lang_info.query_ids_answer.append(
                                                                mapping[inst_ids][valid_target[inst_ids]].tolist())
                                        except Exception as e:
                                                logger.error(f"Failed to map language info {lang_info.lang_type}: {e}")
                                                raise

                                        self._apply_anchor_rel3d_for_lang_info(
                                                lang_info=lang_info,
                                                output=output,
                                                bid=bid,
                                        )

                                        lang_info.append_prompt_postfix()
                                        lang_info.set_batch_idx(bid)
                                        if (
                                                file_names is not None
                                                and isinstance(file_names, (list, tuple))
                                                and bid < len(file_names)
                                                and isinstance(file_names[bid], str)
                                        ):
                                                lang_info.scene_id = file_names[bid]
                                        if not self.training:
                                                lang_info.set_max_gt_iou(max_gt_iou)
                                                if limit is not None:
                                                        counters[prefix] += 1
                                        batch_lang_infos.append(lang_info)

                        total_concat_texts += n_concat

                if not batch_lang_infos:
                        if not self.training:
                                print("warning: NO_LANGUAGE_QUERIES in this batch, skip LLM for this batch.")
                        return [], []

                # statistics
                all_eval_type = [i.split(':')[0]
                                 for i in [i.lang_type for i in batch_lang_infos]]
                print(
                        f'Data statistics ([{"train" if self.training else "val/test"}] batch_size={batch_size}): {dict(Counter(all_eval_type))}')

                return batch_lang_infos, batch_map_target_to_query

        def _select_instance_features_for_llm(self, output, bid: int, target=None):
                """
                Helper to choose which instance-level features are exposed to the LLM.

                By default, we use the standard Mask3DLang query features
                (queries_hidden_state / queries_normalized_embed). When
                `general.use_gt_proposals_for_llm` is enabled, this method is the
                single entry point to later swap in GT-based proposal features.
                """
                use_gt = getattr(self.config.general, "use_gt_proposals_for_llm", False)
                # Oracle / GT-proposal mode: aggregate per-instance features from
                # the per-point mask features using GT instance masks.
                if use_gt and target is not None and len(target) > bid and "masks" in target[bid]:
                        mask_features = output.get("mask_features", None)
                        if mask_features is None:
                                # Fallback to query-based features if mask features are unavailable.
                                if self.global_rank == 0 and self.training:
                                        print(
                                                "[OracleProposal] Warning: use_gt_proposals_for_llm=True "
                                                "but mask_features are missing; falling back to query features."
                                        )
                                q_hidden = output["queries_hidden_state"][bid]
                                q_norm = output["queries_normalized_embed"][bid]
                                return self._maybe_apply_rel3d_hist(output, bid, q_hidden), q_norm

                        # mask_features.decomposed_features[bid]: [P, C]
                        per_point_feat = mask_features.decomposed_features[bid]  # on same device as queries
                        inst_masks = target[bid]["masks"].to(per_point_feat.device).float()  # [I, P]
                        if inst_masks.numel() == 0:
                                q_hidden = output["queries_hidden_state"][bid]
                                q_norm = output["queries_normalized_embed"][bid]
                                return self._maybe_apply_rel3d_hist(output, bid, q_hidden), q_norm

                        # Simple mean pooling over GT instance masks to obtain [I, C] features.
                        denom = inst_masks.sum(dim=1, keepdim=True) + 1e-8
                        inst_feat = (inst_masks @ per_point_feat) / denom  # [I, C]
                        inst_norm = inst_feat / (inst_feat.norm(dim=1, keepdim=True) + 1e-8)

                        if self.global_rank == 0 and self.training and bid == 0:
                                print(
                                        f"[OracleProposal] Using GT-based proposals for LLM: "
                                        f"{inst_feat.shape[0]} instances, dim={inst_feat.shape[1]}."
                                )
                        return inst_feat, inst_norm

                q_hidden = output["queries_hidden_state"][bid]
                q_norm = output["queries_normalized_embed"][bid]

                # When routing geometry to a pretrained Vigor listener, we must feed the
                # raw Mask3D query features that Vigor was trained on. Any extra geometry
                # enhancement (rel3d_hist / anchor-aware fields / geom injection) will
                # shift the feature distribution and can destroy Vigor performance.
                geom_backend = os.environ.get("SSR3DLLM_GEOM_BACKEND", "").strip().lower()
                if geom_backend == "vigor":
                        return q_hidden, q_norm

                # SSR3DLLM: optional continuous relation field enhancement.
                if getattr(self, "enable_ssr3dllm_geom", False):
                        inject_geom = os.environ.get("SSR3DLLM_GEOM_INJECT_TO_LLM", "0").strip().lower() in {
                                "1",
                                "true",
                                "yes",
                                "y",
                                "on",
                        }
                        if not inject_geom:
                                q_hidden = self._maybe_apply_rel3d_hist(output, bid, q_hidden)
                                return q_hidden, q_norm
                        coords = output.get("sampled_coords", None)
                        if coords is not None:
                                # sampled_coords is stored as numpy array on CPU in Mask3DLang;
                                # convert to a torch tensor on the same device as queries.
                                if isinstance(coords, np.ndarray):
                                        coords_tensor = torch.from_numpy(coords)
                                else:
                                        coords_tensor = coords
                                if coords_tensor.dim() == 3 and coords_tensor.size(0) > bid:
                                        coords_bid = coords_tensor[bid]
                                        # Ensure [Q,3] shape.
                                        if coords_bid.dim() == 2 and coords_bid.size(0) == q_hidden.size(0):
                                                coords_bid = coords_bid.to(device=q_hidden.device, dtype=torch.float32)
                                                geom_delta = self.ssr3dllm_geom_head(q_hidden.detach(), coords_bid)
                                                q_hidden = q_hidden + self.ssr3dllm_geom_weight * geom_delta.detach()

                q_hidden = self._maybe_apply_rel3d_hist(output, bid, q_hidden)
                return q_hidden, q_norm

        def _maybe_apply_rel3d_hist(self, output, bid: int, q_hidden: torch.Tensor) -> torch.Tensor:
                """
                """
                if not getattr(self, "use_rel3d_hist_for_llm", False):
                        return q_hidden
                if getattr(self, "rel3d_global_weight", 0.0) == 0.0:
                        return q_hidden

                rel3d_hist = output.get("rel3d_hist", None)
                if rel3d_hist is None:
                        return q_hidden

                # rel3d_hist: [B, Q, R]，q_hidden: [Q, D]
                if not isinstance(rel3d_hist, torch.Tensor):
                        return q_hidden
                if rel3d_hist.dim() != 3 or rel3d_hist.size(0) <= bid:
                        return q_hidden

                hist_bid = rel3d_hist[bid]  # [Q, R]
                if hist_bid.size(0) != q_hidden.size(0):
                        return q_hidden

                geom_delta = self.rel3d_hist_proj(hist_bid.to(q_hidden.device))  # [Q, D]
                return q_hidden + self.rel3d_global_weight * geom_delta

        def _apply_anchor_rel3d_for_lang_info(self, lang_info, output, bid: int):
                """
                """
                # If we are using a pretrained Vigor listener as the geometry backend,
                # do NOT perturb query features with relation fields (Vigor expects raw Mask3D features).
                if os.environ.get("SSR3DLLM_GEOM_BACKEND", "").strip().lower() == "vigor":
                        return
                if not getattr(self, "use_rel3d_hist_for_llm", False):
                        return
                if not lang_info.lang_type.startswith("rel3dref"):
                        return
                if not getattr(self.model, "use_rel3d_geom", False):
                        return

                rel3d_hist = output.get("rel3d_hist", None)
                coords_np = output.get("sampled_coords", None)
                if not isinstance(rel3d_hist, torch.Tensor) or rel3d_hist.dim() != 3:
                        return
                if rel3d_hist.size(0) <= bid:
                        return
                if coords_np is None:
                        return

                q_hidden = getattr(lang_info, "query_hidden_feature", None)
                if q_hidden is None:
                        return
                if not isinstance(q_hidden, torch.Tensor):
                        return

                hist_bid = rel3d_hist[bid]  # [Q, R]
                Q, D = q_hidden.shape
                if hist_bid.size(0) != Q:
                    return

                anchor_q_ids = []
                for ids in getattr(lang_info, "query_ids_question", []) or []:
                        anchor_q_ids.extend(ids)
                anchor_q_ids = sorted(set(anchor_q_ids))
                if len(anchor_q_ids) == 0:
                        return

                coords_scene = coords_np[bid]
                try:
                        coords_scene = np.asarray(coords_scene)
                except Exception:
                        return
                if coords_scene.ndim != 2 or coords_scene.shape[0] != Q:
                        return
                coords = torch.from_numpy(coords_scene).to(q_hidden.device).float()  # [Q,3]

                anchor_idx = torch.tensor(anchor_q_ids, device=q_hidden.device, dtype=torch.long)
                if anchor_idx.numel() == 0:
                        return

                geom_anchor, _ = self.model._compute_anchor_rel3d_features(
                        coords.unsqueeze(0), [anchor_idx]
                )
                if geom_anchor is None:
                        return
                geom_anchor = geom_anchor.squeeze(0)  # [Q,D]
                beta = getattr(self, "rel3d_anchor_weight", 1.0)
                lang_info.query_hidden_feature = q_hidden + beta * geom_anchor

        def _compute_rel3d_role_loss(self, batch_lang_infos):
                """

                """
                if not getattr(self, "enable_rel3d_role_loss", False):
                        return None
                if not batch_lang_infos:
                        return None

                total_loss = 0.0
                count = 0

                for lang_info in batch_lang_infos:
                        if not lang_info.lang_type.startswith("rel3dref"):
                                continue

                        q_hidden = getattr(lang_info, "query_hidden_feature", None)
                        if q_hidden is None or not isinstance(q_hidden, torch.Tensor):
                                continue

                        Q, _ = q_hidden.shape
                        device = q_hidden.device
                        # 0: background, 1: anchor, 2: target
                        role_labels = torch.zeros(Q, dtype=torch.long, device=device)

                        q_ids_q = getattr(lang_info, "query_ids_question", None)
                        if q_ids_q:
                                for ids in q_ids_q:
                                        for qid in ids:
                                                if 0 <= qid < Q:
                                                        role_labels[qid] = 1

                        q_ids_a = getattr(lang_info, "query_ids_answer", None)
                        if q_ids_a:
                                for ids in q_ids_a:
                                        for qid in ids:
                                                if 0 <= qid < Q:
                                                        role_labels[qid] = 2

                        logits = self.rel3d_role_head(q_hidden)
                        loss_i = F.cross_entropy(logits, role_labels)
                        total_loss += loss_i
                        count += 1

                if count == 0:
                        return None
                return total_loss / count

        def training_step(self, batch, batch_idx):
                raw_data, target, file_names = batch

                optimizer = self.optimizers()
                optimizer.zero_grad()

                # DDP safety belt:
                # Some dataloader filters can produce an empty `target` on a subset of ranks
                # (e.g. proposal/box filtering, step-token strict filtering, etc.).
                # If one rank returns early while others perform backward, DDP will deadlock.
                #
                # We therefore skip the whole step on ALL ranks when ANY rank has empty targets.
                skip_on_empty_target = str(os.environ.get("SSR3DLLM_SKIP_STEP_ON_EMPTY_TARGET", "1")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                }
                if skip_on_empty_target:
                        has_targets = 1 if len(target) > 0 else 0
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                flag = torch.tensor(has_targets, device=self.device, dtype=torch.int32)
                                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                                all_have_targets = bool(int(flag.item()) == 1)
                        else:
                                all_have_targets = bool(has_targets == 1)
                        if not all_have_targets:
                                if getattr(self, "global_rank", 0) == 0:
                                        print("[SSR3DLLM][DDP] skip step: some rank has empty targets after filtering.")
                                self.log(
                                        "train_skip_empty_target_batch",
                                        1.0,
                                        on_step=True,
                                        on_epoch=False,
                                        prog_bar=True,
                                        logger=True,
                                        batch_size=max(len(target), 1),
                                )
                                return torch.zeros((), device=self.device, dtype=torch.float32)

                if len(target) == 0:
                        print("no targets")
                        return None

                raw_coordinates = None
                if self.config.data.add_raw_coordinates:
                        raw_coordinates = raw_data.features[:, -3:]
                        raw_data.features = raw_data.features[:, :-3]

                data = ME.SparseTensor(
                        coordinates=raw_data.coordinates,
                        features=raw_data.features,
                        device=self.device,
                )

                # SSR3DLLM: in geom-only pretraining we freeze the segmentation backbone and LLM.
                # To keep backbone stats stable (BN/dropout) and save memory, run the backbone
                # forward under `no_grad()` and in eval mode.
                if getattr(self, "ssr3dllm_geom_only", False):
                        self.model.eval()
                        with torch.no_grad():
                                output = self.forward(
                                        data,
                                        point2segment=[
                                                target[i]["point2segment"] for i in range(len(target))
                                        ],
                                        raw_coordinates=raw_coordinates,
                                        extra_lang=raw_data.extra_lang
                                )
                else:
                        output = self.forward(
                                data,
                                point2segment=[
                                        target[i]["point2segment"] for i in range(len(target))
                                ],
                                raw_coordinates=raw_coordinates,
                                extra_lang=raw_data.extra_lang
                        )

                output['raw_coordinates'] = raw_coordinates
                if len(raw_data.extra_lang) > 0:
                        output['extra_lang'] = raw_data.extra_lang
                        if 'aux_outputs' in output:
                                for aux_outputs in output["aux_outputs"]:
                                        aux_outputs['extra_lang'] = raw_data.extra_lang

                # In the "LLM step-token" experiment we don't train the segmentation/detection
                # objectives (they require language-conditioned `pred_logits`, which may be None
                # when `num_concat_texts=0`). We only keep the SSR3DLLM chain/LLM losses.
                losses = {}
                disable_detection_loss = _ssr3dllm_env_flag("SSR3DLLM_DISABLE_DETECTION_LOSS", "0")
                if (
                        (not disable_detection_loss)
                        and (not self._is_step_token_sft())
                        and (output.get("pred_logits", None) is not None)
                ):
                        losses = self.criterion(output, target, mask_type=self.mask_type)

                batch_lang_infos, batch_map_target_to_query = [], None
                if self.llama_config.enable_llm or getattr(self, "enable_rel3d_role_loss", False):
                        batch_lang_infos, batch_map_target_to_query = self.prepare_llm(
                                output,
                                raw_data.extra_lang,
                                None,
                                target,
                                raw_data,
                                file_names=file_names,
                        )

                llm_has_batch = False
                output_llm = {"lm_loss": None, "match_loss": None, "model_output": None}
                if self.llama_config.enable_llm and not getattr(self, "ssr3dllm_geom_only", False):
                        # Optional: exclude some prefixes from LM teacher-forcing loss, while still
                        # keeping them for geometry-chain losses (SSR3DLLM).
                        #
                        # This is useful when we want grounding tasks (e.g. scanrefer/m3dref) to train
                        # ONLY via the <geom>-routed S3G/SSR3DLLM pathway (stepslot + geom losses),
                        # without encouraging any template/chain text generation via LM loss.
                        #
                        # Env:
                        #   SSR3DLLM_LM_SKIP_PREFIXES="scanrefer,m3dref"
                        lm_lang_infos = list(batch_lang_infos)
                        skip_set = set()
                        try:
                                skip_raw = str(os.environ.get("SSR3DLLM_LM_SKIP_PREFIXES", "")).strip()
                        except Exception:
                                skip_raw = ""
                        if skip_raw:
                                skip_set = {p.strip() for p in skip_raw.split(",") if p.strip()}
                                if skip_set:
                                        lm_lang_infos = [
                                                i for i in batch_lang_infos
                                                if str(getattr(i, "lang_type", "")).split(":")[0] not in skip_set
                                        ]
                        if (
                                skip_set
                                and int(getattr(self, "global_rank", 0)) == 0
                                and not getattr(self, "_ssr3dllm_lm_skip_warned", False)
                        ):
                                setattr(self, "_ssr3dllm_lm_skip_warned", True)
                                try:
                                        print(
                                                f"[SSR3DLLM][lm_loss] skip_prefixes={sorted(list(skip_set))} "
                                                f"kept_for_lm={len(lm_lang_infos)}/{len(batch_lang_infos)}",
                                                flush=True,
                                        )
                                except Exception:
                                        pass

                        # If the filtered LM batch is empty, skip the LLM forward entirely.
                        # Geometry-side step-slot embeddings can be recomputed via `encode_stepslot_only`
                        # in the isolate-geom-lora routine when needed.
                        llm_has_batch = bool(lm_lang_infos)
                        if llm_has_batch:
                                batch_gt_inst_ids = [((i.batch_idx, i, i.max_gt_iou) if not self.training else (
                                        i.batch_idx, i)) for i in lm_lang_infos]
                                batch_input_texts = [i.question for i in lm_lang_infos]
                                batch_output_texts = self._ssr3dllm_build_output_texts(lm_lang_infos)
                                batch_eval_types = [i.lang_type for i in lm_lang_infos]
                                batch_instance_queries_hidden_state = [
                                        i.query_hidden_feature for i in lm_lang_infos]
                                batch_instance_queries_normalized_embed = [
                                        i.query_normalized_embed for i in lm_lang_infos]

                                output_llm = self.llama_model(batch_input_text_list=batch_input_texts,
                                                              batch_output_text_list=batch_output_texts,
                                                              batch_instance_queries_hidden_state=batch_instance_queries_hidden_state,
                                                              batch_instance_queries_normalized_embed=batch_instance_queries_normalized_embed,
                                                              batch_eval_types=batch_eval_types,
                                                              batch_gt_inst_ids=batch_gt_inst_ids,
                                                              )
                                output['output_llm'] = output_llm

                        if llm_has_batch:
                                # DDP safety belt:
                                # If ANY rank ends up with an empty LLM batch after internal filtering
                                # (see "warning: no valid batch content" in models/LLM/LLama3d.py),
                                # we skip this optimization step on ALL ranks to avoid:
                                # - rank crash (no grad graph) → other ranks hang
                                # - DDP deadlock when different ranks use different parameter subsets
                                #
                                # Controlled by env SSR3DLLM_SKIP_STEP_ON_EMPTY_LLM (default=1).
                                skip_on_empty_llm = str(os.environ.get("SSR3DLLM_SKIP_STEP_ON_EMPTY_LLM", "1")).strip().lower() in {
                                        "1",
                                        "true",
                                        "yes",
                                        "y",
                                        "on",
                                }
                                if skip_on_empty_llm:
                                        llm_has_output = 1 if output_llm.get("model_output", None) is not None else 0
                                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                                flag = torch.tensor(llm_has_output, device=self.device, dtype=torch.int32)
                                                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                                                llm_all_have_output = bool(int(flag.item()) == 1)
                                        else:
                                                llm_all_have_output = bool(llm_has_output == 1)
                                        if not llm_all_have_output:
                                                if getattr(self, "global_rank", 0) == 0:
                                                        print("[SSR3DLLM][DDP] skip step: some rank has empty LLM batch after filtering.")
                                                bs = len(target)
                                                self.log(
                                                        "train_skip_empty_llm_batch",
                                                        1.0,
                                                        on_step=True,
                                                        on_epoch=False,
                                                        prog_bar=True,
                                                        logger=True,
                                                        batch_size=bs,
                                                )
                                                return torch.zeros((), device=self.device, dtype=torch.float32)
                elif self.llama_config.enable_llm and getattr(self, "ssr3dllm_geom_only", False):
                        # SSR3DLLM geom-only: still run a frozen LLM forward (no-grad) so that
                        # `lang_info.llm_text_init` / `lang_info.llm_text_tokens` are populated
                        # for the geometry decoder (cross-attn to token sequence).
                        run_llm = os.environ.get("SSR3DLLM_GEOM_ONLY_RUN_LLM", "1").strip().lower() not in {
                                "",
                                "0",
                                "false",
                                "no",
                                "off",
                        }
                        if run_llm and batch_lang_infos:
                                rel_infos = []
                                for i in batch_lang_infos:
                                        lang_type = getattr(i, "lang_type", "")
                                        if not isinstance(lang_type, str):
                                                continue
                                        if lang_type.split(":")[0] == "rel3dref":
                                                rel_infos.append(i)
                                if rel_infos:
                                        rel_gt_inst_ids = [((i.batch_idx, i, getattr(i, "max_gt_iou", 0.0))) for i in rel_infos]
                                        rel_input_texts = [i.question for i in rel_infos]
                                        rel_output_texts = self._ssr3dllm_build_output_texts(rel_infos)
                                        rel_eval_types = [i.lang_type for i in rel_infos]
                                        rel_instance_queries_hidden_state = [i.query_hidden_feature for i in rel_infos]
                                        rel_instance_queries_normalized_embed = [i.query_normalized_embed for i in rel_infos]
                                        try:
                                                self.llama_model.eval()
                                                with torch.no_grad():
                                                        _ = self.llama_model(
                                                                batch_input_text_list=rel_input_texts,
                                                                batch_output_text_list=rel_output_texts,
                                                                batch_instance_queries_hidden_state=rel_instance_queries_hidden_state,
                                                                batch_instance_queries_normalized_embed=rel_instance_queries_normalized_embed,
                                                                batch_eval_types=rel_eval_types,
                                                                batch_gt_inst_ids=rel_gt_inst_ids,
                                                        )
                                        except Exception:
                                                pass

                if torch.cuda.is_available() and getattr(self, "global_rank", 0) == 0:
                        if batch_idx % 10 == 0:
                                alloc_gb = torch.cuda.memory_allocated() / 1e9
                                max_gb = torch.cuda.max_memory_allocated() / 1e9
                                print(f"[GPU-MEM] step={batch_idx} alloc={alloc_gb:.3f}GB max={max_gb:.3f}GB")

                for k in list(losses.keys()):
                        if k in self.criterion.weight_dict:
                                losses[k] *= self.criterion.weight_dict[k]
                        else:
                                losses.pop(k)

                if getattr(self, "enable_rel3d_role_loss", False):
                        rel3d_role_loss = self._compute_rel3d_role_loss(batch_lang_infos)
                        if rel3d_role_loss is not None and self.rel3d_role_loss_weight > 0.0:
                                losses["loss_rel3d_role"] = self.rel3d_role_loss_weight * rel3d_role_loss

                disable_lm_loss = _ssr3dllm_env_flag("SSR3DLLM_DISABLE_LM_LOSS", "0")
                if (
                        self.llama_config.enable_llm
                        and (not disable_lm_loss)
                        and (not getattr(self, "ssr3dllm_geom_only", False))
                        and bool(locals().get("llm_has_batch", False))
                ):
                        for k, v in (output_llm or {}).items():
                                if 'loss' in k:
                                        losses[k] = v
                        # print({k: f'{v.item():.3f}' for k, v in output_llm.items() if 'loss' in k})
                        # print({k: f'{v.item():.3f}' for k, v in losses.items()})

                # SSR3DLLM: optional rel3dref auxiliary losses based on
                # relation field and BERT text encoder.
                if self.enable_ssr3dllm_geom:
                        coords = output.get("sampled_coords", None)
                        if coords is not None and isinstance(self.ssr3dllm_geom_head, SSR3DLLMGeomHeadForLLM):
                                if isinstance(coords, np.ndarray):
                                        coords_tensor = torch.from_numpy(coords).to(self.device)
                                else:
                                        coords_tensor = coords.to(self.device)
                                geom_losses = self.ssr3dllm_geom_head.compute_rel3dref_losses_for_batch(
                                        batch_lang_infos=batch_lang_infos,
                                        sampled_coords=coords_tensor,
                                        device=self.device,
                                        w_ref=self.ssr3dllm_ref_loss_weight,
                                        w_anchor=self.ssr3dllm_anchor_loss_weight,
                                        w_relcls=self.ssr3dllm_relcls_loss_weight,
                                        w_chain=self.ssr3dllm_chain_loss_weight,
                                        w_distill_vigor=self.ssr3dllm_distill_vigor_weight,
                                        distill_temperature=self.ssr3dllm_distill_temperature,
                                )
                                for k, v in geom_losses.items():
                                        losses[k] = v
                                # Optional A0 step-slot supervision (single-pass slots + STOP).
                                # Enabled by: SSR3DLLM_ORDER_MODE=slots + SSR3DLLM_ORDER_LOSS_WEIGHT>0
                                try:
                                        slot_losses = self.ssr3dllm_geom_head.compute_stepslot_loss_for_batch(
                                                batch_lang_infos=batch_lang_infos,
                                                device=self.device,
                                        )
                                        for k, v in slot_losses.items():
                                                losses[k] = v
                                except Exception:
                                        pass

                logs = {
                        f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
                }

                for base_key in ["train_loss_ce", "train_loss_mask", "train_loss_dice"]:
                        if base_key in logs:
                                logs[f"{base_key}_raw"] = logs[base_key]
                                logs.pop(base_key)

                ce_vals = [v for k, v in logs.items() if "loss_ce" in k]
                mask_vals = [v for k, v in logs.items() if "loss_mask" in k]
                dice_vals = [v for k, v in logs.items() if "loss_dice" in k]
                logs["train_mean_loss_ce"] = statistics.mean(ce_vals) if ce_vals else 0.0
                logs["train_mean_loss_mask"] = statistics.mean(mask_vals) if mask_vals else 0.0
                logs["train_mean_loss_dice"] = statistics.mean(dice_vals) if dice_vals else 0.0
                step_sft = self._is_step_token_sft()

                if (
                        self.llama_config.enable_llm
                        and (not disable_lm_loss)
                        and (not getattr(self, "ssr3dllm_geom_only", False))
                        and bool(locals().get("llm_has_batch", False))
                ):
                        logs["train_mean_loss_lm"] = output_llm["lm_loss"]
                        logs["train_mean_loss_lm_match"] = output_llm["match_loss"]
                        batch_size = len(target)
                        self.log("lm_loss", output_llm["lm_loss"],
                                 on_step=True, on_epoch=True, logger=True, prog_bar=step_sft, batch_size=batch_size)
                        self.log("match_loss", output_llm["match_loss"],
                                 on_step=True, on_epoch=True, logger=True, prog_bar=step_sft, batch_size=batch_size)

                total_loss = None
                for k, v in losses.items():
                        # Avoid double-counting SSR3DLLM chain-loss components that are returned
                        # only for logging (see models/geom_head_llm_adapter.py).
                        if k.startswith("loss_ssr3d_chain_") and k != "loss_ssr3d_chain":
                                continue
                        if isinstance(v, torch.Tensor):
                                total_loss = v if total_loss is None else total_loss + v
                        elif isinstance(v, (float, int)):
                                t = torch.tensor(float(v), device=self.device, dtype=torch.float32)
                                total_loss = t if total_loss is None else total_loss + t

                # In geom-only pretraining (or rare corner cases), a batch may contain no
                # trainable loss terms (e.g. only detection samples while seg losses are disabled).
                # Ensure `total_loss` is a Tensor connected to a trainable parameter so DDP doesn't
                # deadlock and Lightning doesn't crash on `.backward()`.
                if total_loss is None or (
                        isinstance(total_loss, torch.Tensor)
                        and total_loss.grad_fn is None
                        and not total_loss.requires_grad
                ):
                        dummy_param = None
                        for p in self.parameters():
                                if getattr(p, "requires_grad", False):
                                        dummy_param = p
                                        break
                        if dummy_param is None:
                                return None
                        zero_loss = dummy_param.sum() * 0.0
                        total_loss = zero_loss if total_loss is None else (zero_loss + total_loss)
                batch_size = len(target)
                self.log("train_loss", total_loss,
                         on_step=True, on_epoch=True, prog_bar=True, logger=True,
                         batch_size=batch_size)
                if not step_sft:
                        self.log("train_loss_ce", logs.get("train_mean_loss_ce", 0.0),
                                 on_step=True, prog_bar=True, logger=True,
                                 batch_size=batch_size)
                        self.log("train_loss_mask", logs.get("train_mean_loss_mask", 0.0),
                                 on_step=True, prog_bar=True, logger=True,
                                 batch_size=batch_size)
                        self.log("train_loss_dice", logs.get("train_mean_loss_dice", 0.0),
                                 on_step=True, prog_bar=True, logger=True,
                                 batch_size=batch_size)
                for extra_name in [
                    "train_loss_ssr3d_ref",
                    "train_loss_ssr3d_anchor",
                    "train_loss_ssr3d_relcls",
                    "train_loss_ssr3d_chain",
                    "train_loss_ssr3d_chain_ref_ce",
                    "train_loss_ssr3d_chain_obj_ce",
                    "train_loss_ssr3d_chain_lang_ce",
                    "train_loss_ssr3d_distill_vigor",
                ]:
                        if extra_name in logs:
                                val = logs.pop(extra_name)
                                self.log(
                                        extra_name,
                                        val,
                                        on_step=True,
                                        prog_bar=True,
                                        logger=True,
                                        batch_size=batch_size,
                                )
                if "train_loss_rel3d_role" in logs:
                        self.log("train_loss_rel3d_role", logs["train_loss_rel3d_role"],
                                 on_step=True, prog_bar=True, logger=True,
                                 batch_size=batch_size)
                        logs["train_loss_rel3d_role_raw"] = logs["train_loss_rel3d_role"]
                        logs.pop("train_loss_rel3d_role")

                if step_sft:
                        # Hide legacy instance-seg loss components in logs for this experiment.
                        for k in list(logs.keys()):
                                if ("loss_ce" in k) or ("loss_mask" in k) or ("loss_dice" in k):
                                        logs.pop(k, None)
                        logs.pop("train_mean_loss_ce", None)
                        logs.pop("train_mean_loss_mask", None)
                        logs.pop("train_mean_loss_dice", None)

                self.log_dict(logs, batch_size=batch_size)

                # ------------------------------------------------------------
                # SSR3DLLM: gradient isolation between dialog/LM losses and
                # geometry (listener) losses.
                #
                # Goal (more stable):
                # - LM/dialog losses update LoRA (and other language params)
                # - Geometry losses update listener + (<stepK> rows / stepslot adapter)
                #   but DO NOT update LoRA
                #
                # Enable with: SSR3DLLM_ISOLATE_GEOM_LORA=1
                # Notes:
                # - This only affects gradients; forward behavior is unchanged.
                # - Best used with SSR3DLLM_TRAIN_STEP_ROWS=1 and
                #   SSR3DLLM_LLM_STEPSLOT_ADAPTER_TRAINABLE=1.
                # ------------------------------------------------------------
                isolate_geom_lora = str(os.environ.get("SSR3DLLM_ISOLATE_GEOM_LORA", "0")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                }

                if isolate_geom_lora:
                        geom_loss = None
                        non_geom_loss = None
                        for k, v in losses.items():
                                # Keep consistent with total_loss aggregation above.
                                if k.startswith("loss_ssr3d_chain_") and k != "loss_ssr3d_chain":
                                        continue
                                if not isinstance(v, torch.Tensor):
                                        continue
                                if k.startswith("loss_ssr3d_"):
                                        geom_loss = v if geom_loss is None else geom_loss + v
                                else:
                                        non_geom_loss = v if non_geom_loss is None else non_geom_loss + v

                        # DDP correctness:
                        # The isolate routine performs *two* backward passes (non-geom, then geom).
                        # If different ranks take different branches (e.g., some rank has no geom
                        # samples after filtering), DDP will deadlock. We therefore only run the
                        # 2-backward routine when ALL ranks agree the prerequisites are met.
                        use_isolate = bool(
                                geom_loss is not None
                                and non_geom_loss is not None
                                and hasattr(self, "llama_model")
                                and (self.llama_model is not None)
                        )
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                flag = torch.tensor(
                                        1 if use_isolate else 0,
                                        device=self.device,
                                        dtype=torch.int32,
                                )
                                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                                use_isolate = bool(int(flag.item()) == 1)

                        if not use_isolate:
                                # Fall back to a single backward to keep all ranks in sync.
                                self.manual_backward(total_loss)
                        else:
                                # IMPORTANT: avoid calling backward twice on the SAME autograd graph
                                # when gradient checkpointing is enabled (PyTorch will error in
                                # torch.utils.checkpoint if we reuse saved tensors).
                                #
                                # Strategy:
                                # 1) backward non-geom loss on graph A
                                # 2) recompute ONLY the geometry-side LLM step/lang embeddings + listener losses
                                #    on a fresh graph B, then backward geom loss
                                # 3) restore LoRA grads to what they were after (1) so geom loss does not update LoRA
                                #
                                # This keeps the intended gradient routing while remaining compatible with checkpointing.
                                self.manual_backward(non_geom_loss)

                                # Snapshot LoRA grads after non-geom backward.
                                lora_params = []
                                lora_grads = {}
                                try:
                                        if self.llama_model is not None:
                                                for n, p in self.llama_model.named_parameters():
                                                        if p is None or ("lora_" not in n):
                                                                continue
                                                        lora_params.append(p)
                                                        if p.grad is None:
                                                                lora_grads[id(p)] = None
                                                        else:
                                                                lora_grads[id(p)] = p.grad.detach().clone()

                                        # Detach CLASP-aligned query features so geom loss does NOT backprop
                                        # into Mask3D/Mask3DLang backbone (only listener + step tokens/adapter).
                                        for li in batch_lang_infos:
                                                qh = getattr(li, "query_hidden_feature", None)
                                                if isinstance(qh, torch.Tensor):
                                                        setattr(li, "query_hidden_feature", qh.detach())
                                                qn = getattr(li, "query_normalized_embed", None)
                                                if isinstance(qn, torch.Tensor):
                                                        setattr(li, "query_normalized_embed", qn.detach())
                                                # Clear old graph-A step/lang embeddings before recompute.
                                                if hasattr(li, "llm_step_embeds"):
                                                        setattr(li, "llm_step_embeds", None)
                                                if hasattr(li, "llm_lang_embeds"):
                                                        setattr(li, "llm_lang_embeds", None)

                                        # Recompute LLM step/lang embeddings only (fresh graph B).
                                        # NOTE: this function should ONLY populate llm_step_embeds / llm_lang_embeds
                                        # on lang_info objects and must NOT run full LM teacher-forcing losses.
                                        stepslot_encoder = None
                                        if self.llama_model is not None:
                                                stepslot_encoder = getattr(self.llama_model, "encode_stepslot_only", None)
                                                if stepslot_encoder is None:
                                                        # Common case: PEFT wraps the base model under `.base_model.model`.
                                                        base = getattr(self.llama_model, "base_model", None)
                                                        base_model = getattr(base, "model", None) if base is not None else None
                                                        stepslot_encoder = getattr(base_model, "encode_stepslot_only", None) if base_model is not None else None
                                                if stepslot_encoder is None:
                                                        # Last resort: handle potential wrapping (e.g., DataParallel-like `.module`).
                                                        mod = getattr(self.llama_model, "module", None)
                                                        stepslot_encoder = getattr(mod, "encode_stepslot_only", None) if mod is not None else None
                                        if stepslot_encoder is None:
                                                raise RuntimeError(
                                                        "[SSR3DLLM][isolate_geom_lora] missing encode_stepslot_only on llama_model. "
                                                        "Expected it on the base LLM (e.g., llama_model.base_model.model.encode_stepslot_only)."
                                                )
                                        stepslot_encoder(batch_lang_infos=batch_lang_infos)

                                        # Recompute geometry losses on fresh graph B.
                                        coords = output.get("sampled_coords", None)
                                        if coords is None:
                                                raise RuntimeError("[SSR3DLLM][isolate_geom_lora] missing output['sampled_coords'] for geom loss recompute.")
                                        if isinstance(coords, np.ndarray):
                                                coords_tensor = torch.from_numpy(coords).to(self.device)
                                        else:
                                                coords_tensor = coords.to(self.device)
                                        coords_tensor = coords_tensor.detach()

                                        geom_losses_b = self.ssr3dllm_geom_head.compute_rel3dref_losses_for_batch(
                                                batch_lang_infos=batch_lang_infos,
                                                sampled_coords=coords_tensor,
                                                device=self.device,
                                                w_ref=self.ssr3dllm_ref_loss_weight,
                                        w_anchor=self.ssr3dllm_anchor_loss_weight,
                                        w_relcls=self.ssr3dllm_relcls_loss_weight,
                                        w_chain=self.ssr3dllm_chain_loss_weight,
                                        w_distill_vigor=self.ssr3dllm_distill_vigor_weight,
                                        distill_temperature=self.ssr3dllm_distill_temperature,
                                        )
                                        geom_loss_b = None
                                        for k, v in geom_losses_b.items():
                                                if not isinstance(v, torch.Tensor):
                                                        continue
                                                if k.startswith("loss_ssr3d_chain_") and k != "loss_ssr3d_chain":
                                                        continue
                                                geom_loss_b = v if geom_loss_b is None else geom_loss_b + v
                                        # DDP correctness: if any rank ended up with no geom loss
                                        # (e.g., all geom samples filtered out), skip the geom backward
                                        # on ALL ranks to avoid deadlocks.
                                        do_geom_backward = bool(geom_loss_b is not None)
                                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                                flag_b = torch.tensor(
                                                        1 if do_geom_backward else 0,
                                                        device=self.device,
                                                        dtype=torch.int32,
                                                )
                                                torch.distributed.all_reduce(
                                                        flag_b, op=torch.distributed.ReduceOp.MIN
                                                )
                                                do_geom_backward = bool(int(flag_b.item()) == 1)
                                        if do_geom_backward:
                                                self.manual_backward(geom_loss_b)
                                        else:
                                                geom_loss_b = None
                                finally:
                                        # Restore LoRA grads to the snapshot (remove geom contribution).
                                        for p in lora_params:
                                                g = lora_grads.get(id(p), None)
                                                if g is None:
                                                        p.grad = None
                                                else:
                                                        if p.grad is None:
                                                                p.grad = g
                                                        else:
                                                                p.grad.detach().copy_(g)
                else:
                        self.manual_backward(total_loss)

                # clip gradients
                self.clip_gradients(optimizer, gradient_clip_val=0.1,
                                    gradient_clip_algorithm="norm")

                optimizer.step()

                lr_scheduler = self.lr_schedulers()
                if self.config.scheduler.pytorch_lightning_params.interval == 'step':
                        lr_scheduler.step()
                elif self.config.scheduler.pytorch_lightning_params.interval == 'epoch':
                        if self.trainer.is_last_batch:
                                lr_scheduler.step()
                else:
                        raise NotImplementedError
                # print('lr', lr_scheduler.get_lr())

                # `total_loss` is already a safe Tensor (never a Python int) and is what we
                # backpropagated above; return it for Lightning bookkeeping.
                return total_loss.detach()

        def validation_step(self, batch, batch_idx):
                return self.eval_step(batch, batch_idx)

        def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
                base_path = os.path.join(
                        self.config.general.save_dir,
                        "eval_output",
                        f"instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}",
                        f"decoder_{decoder_id}",
                )
                pred_mask_path = f"{base_path}/pred_mask"

                from pathlib import Path
                Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

                file_name = file_names
                with open(f"{base_path}/{file_name}.txt", "w") as fout:
                        real_id = -1
                        for instance_id in range(len(pred_classes)):
                                real_id += 1
                                pred_class = pred_classes[instance_id]
                                score = scores[instance_id]
                                mask = pred_masks[:, instance_id].astype("uint8")

                                if score > 1e-4:
                                        # reduce the export size a bit. I guess no performance difference
                                        np.savetxt(
                                                f"{pred_mask_path}/{file_name}_{real_id}.txt",
                                                mask,
                                                fmt="%d",
                                        )
                                        fout.write(
                                                f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                                        )

        def training_epoch_end(self, outputs):
                train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
                        outputs
                )
                results = {"train_loss_mean": train_loss}
                self.log_dict(results)

        def validation_epoch_end(self, outputs):
                self.test_epoch_end(outputs)

        def save_visualizations(
                        self,
                        target_full,
                        full_res_coords,
                        sorted_masks,
                        sort_classes,
                        file_name,
                        original_colors,
                        original_normals,
                        sort_scores_values,
                        point_size=20,
                        query_text=None,
                        query_mask=None,
                        query_mask_instance_coordscore=None,
                        gt_query_mask=None,
                        gt_ious=None,
                        max_num_of_queries=200,
                        max_num_of_instances=40,
        ):

                full_res_coords -= full_res_coords.mean(axis=0)

                gt_pcd_pos = []
                gt_pcd_normals = []
                gt_pcd_color = []
                gt_inst_pcd_color = []
                gt_boxes = []

                if "labels" in target_full:
                        instances_colors = torch.from_numpy(
                                np.vstack(
                                        get_evenly_distributed_colors(
                                                target_full["labels"].shape[0]
                                        )
                                )
                        )
                        for instance_counter, (label, mask) in enumerate(
                                        zip(target_full["labels"], target_full["masks"])
                        ):
                                if label == 255:
                                        continue

                                mask_tmp = mask.detach().cpu().numpy()
                                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                                if len(mask_coords) == 0:
                                        continue

                                gt_pcd_pos.append(mask_coords)
                                mask_coords_min = full_res_coords[
                                        mask_tmp.astype(bool), :
                                ].min(axis=0)
                                mask_coords_max = full_res_coords[
                                        mask_tmp.astype(bool), :
                                ].max(axis=0)
                                size = mask_coords_max - mask_coords_min
                                mask_coords_middle = mask_coords_min + size / 2

                                gt_boxes.append(
                                        {
                                                "position": mask_coords_middle,
                                                "size"    : size,
                                                "color"   : self.validation_dataset.map2color([label])[0],
                                        }
                                )

                                gt_pcd_color.append(
                                        self.validation_dataset.map2color([label]).repeat(
                                                gt_pcd_pos[-1].shape[0], 1
                                        )
                                )
                                gt_inst_pcd_color.append(
                                        instances_colors[instance_counter % len(instances_colors)]
                                        .unsqueeze(0)
                                        .repeat(gt_pcd_pos[-1].shape[0], 1)
                                )

                                gt_pcd_normals.append(
                                        original_normals[mask_tmp.astype(bool), :]
                                )

                        gt_pcd_pos = np.concatenate(gt_pcd_pos)
                        gt_pcd_normals = np.concatenate(gt_pcd_normals)
                        gt_pcd_color = np.concatenate(gt_pcd_color)
                        gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

                v = vis.Visualizer()

                v.add_points(
                        "RGB Input",
                        full_res_coords,
                        colors=original_colors,
                        normals=original_normals,
                        visible=True,
                        point_size=point_size,
                )

                if "labels" in target_full:
                        v.add_points(
                                "Semantics (GT)",
                                gt_pcd_pos,
                                colors=gt_pcd_color,
                                normals=gt_pcd_normals,
                                alpha=0.8,
                                visible=False,
                                point_size=point_size,
                        )
                        v.add_points(
                                "Instances (GT)",
                                gt_pcd_pos,
                                colors=gt_inst_pcd_color,
                                normals=gt_pcd_normals,
                                alpha=0.8,
                                visible=False,
                                point_size=point_size,
                        )

                pred_coords = []
                pred_normals = []
                pred_sem_color = []
                pred_inst_color = []

                if sorted_masks is not None:
                        for did in range(len(sorted_masks)):
                                instances_colors = torch.from_numpy(
                                        np.vstack(
                                                get_evenly_distributed_colors(
                                                        max(1, sorted_masks[did].shape[1])
                                                )
                                        )
                                )

                                for i in reversed(range(sorted_masks[did].shape[1])):
                                        coords = full_res_coords[
                                                sorted_masks[did][:, i].astype(bool), :
                                        ]

                                        mask_coords = full_res_coords[
                                                sorted_masks[did][:, i].astype(bool), :
                                        ]
                                        mask_normals = original_normals[
                                                sorted_masks[did][:, i].astype(bool), :
                                        ]

                                        label = sort_classes[did][i]

                                        if len(mask_coords) == 0:
                                                continue

                                        pred_coords.append(mask_coords)
                                        pred_normals.append(mask_normals)

                                        pred_sem_color.append(
                                                self.validation_dataset.map2color([label]).repeat(
                                                        mask_coords.shape[0], 1
                                                )
                                        )

                                        pred_inst_color.append(
                                                instances_colors[i % len(instances_colors)]
                                                .unsqueeze(0)
                                                .repeat(mask_coords.shape[0], 1)
                                        )

                                        # if sort_scores_values[did][i] > 0.1 and i < max_num_of_instances:
                                        #     lable2name = self.labels_info[label]["name"]
                                        #     v.add_points(
                                        #         f"Instance Label: {lable2name}",
                                        #         mask_coords,
                                        #         colors=np.concatenate([self.validation_dataset.map2color([label]).repeat(mask_coords.shape[0], 1)]),
                                        #         normals=mask_normals,
                                        #         visible=False,
                                        #         alpha=0.8,
                                        #         point_size=point_size,
                                        #     )

                                if len(pred_coords) > 0:
                                        pred_coords = np.concatenate(pred_coords)
                                        pred_normals = np.concatenate(pred_normals)
                                        pred_sem_color = np.concatenate(pred_sem_color)
                                        pred_inst_color = np.concatenate(pred_inst_color)

                                        v.add_points(
                                                "Semantics (Mask3D)",
                                                pred_coords,
                                                colors=pred_sem_color,
                                                normals=pred_normals,
                                                visible=False,
                                                alpha=0.8,
                                                point_size=point_size,
                                        )
                                        v.add_points(
                                                "Instances (Mask3D)",
                                                pred_coords,
                                                colors=pred_inst_color,
                                                normals=pred_normals,
                                                visible=False,
                                                alpha=0.8,
                                                point_size=point_size,
                                        )

                out_json = []
                valid_mask_count = 0
                if query_text is not None:
                        if isinstance(query_text, np.ndarray):
                                query_text = query_text.tolist()
                        for index, query in enumerate(query_text):
                                if not query_mask[index].any():
                                        continue
                                valid_mask_count += 1
                                if valid_mask_count > max_num_of_queries:
                                        break
                                # text_center = np.mean(mask_coords, axis=0)
                                use_color = np.array([255, 0, 0])[:, np.newaxis]
                                gt_use_color = np.array([0, 255, 0])[:, np.newaxis]

                                out_json.append({"text"           : query,
                                                 "name"           : f"Query text {index}",
                                                 "inst_coordscore": query_mask_instance_coordscore[index].tolist()
                                                 # "numberOfpoints":mask_coords.shape[0],
                                                 # "color":use_color.T.tolist()
                                                 }
                                                )

                                intersection = query_mask[index] & gt_query_mask[index]
                                union = query_mask[index] | gt_query_mask[index]
                                iou = intersection.sum() / (union.sum() + 1)

                                union_coords = full_res_coords[union.astype(bool), :]
                                union_normals = original_normals[union.astype(bool), :]
                                use_color = use_color.repeat(full_res_coords.shape[0], 1).T
                                use_color[gt_query_mask[index]] = gt_use_color.T
                                use_color[intersection] = np.asarray([[255, 255, 0]])
                                use_color = use_color[union.astype(bool)]

                                v.add_points(
                                        f"Query: text {index}",
                                        union_coords,
                                        colors=use_color,
                                        normals=union_normals,
                                        visible=False,
                                        alpha=0.8,
                                        point_size=point_size,
                                )

                        import json
                        from json import encoder
                        encoder.FLOAT_REPR = lambda o: format(o, '.2f')
                        os.makedirs(
                                f"{self.config['general']['save_dir']}/visualizations", exist_ok=True)
                        if len(out_json) > 0:
                                json.dump(out_json,
                                          open(
                                                  f"{self.config['general']['save_dir']}/visualizations/{file_name}query.json", 'w'),
                                          indent=4
                                          )

                v.save(
                        f"{self.config['general']['save_dir']}/visualizations/{file_name}"
                )

                # save each part as npy
                # for k1, v1 in v.elements.items():
                #     try:
                #         points = np.concatenate([v1.positions, v1.colors, v1.normals], axis=1)
                #         np.save(f"{self.config['general']['save_dir']}/visualizations/{file_name}/{k1}_pos_color_normal.npy", points)
                #     except Exception as e:
                #         print(e)

        def eval_step(self, batch, batch_idx):
                # For external-eval-only workflows (e.g. Vigor step-slot), skip the expensive
                # internal Mask3D/segmentation forward during validation/test.
                # We keep a single batch in Lightning so `*_epoch_end` hooks can run.
                if str(os.environ.get("SSR3DLLM_SKIP_INTERNAL_EVAL", "")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                }:
                        return {}
                raw_data, target, file_names = batch
                # Optional periodic resource snapshot to debug silent "Killed" failures on full eval.
                if _ssr3dllm_env_flag("SSR3DLLM_DEBUG_RESOURCE", "0") and int(getattr(self, "global_rank", 0)) == 0:
                        every = max(1, _ssr3dllm_env_int("SSR3DLLM_DEBUG_RESOURCE_EVERY", 50))
                        if int(batch_idx) % every == 0:
                                try:
                                        _ssr3dllm_log_resource(
                                                f"eval_step batch_idx={int(batch_idx)} file={file_names[0] if isinstance(file_names,(list,tuple)) and file_names else file_names}",
                                                save_dir=str(self.config.general.save_dir),
                                                tmp_dir=str(getattr(self, "tmpdir", "")),
                                                device=getattr(self, "device", None),
                                                extra=f"coords={int(len(getattr(raw_data, 'coordinates', [])))}",
                                        )
                                except Exception:
                                        pass
                inverse_maps = raw_data.inverse_maps
                target_full = raw_data.target_full
                original_colors = raw_data.original_colors
                data_idx = raw_data.idx
                original_normals = raw_data.original_normals
                original_coordinates = raw_data.original_coordinates

                if len(raw_data.coordinates) == 0:
                        return 0.0

                raw_coordinates = None
                if self.config.data.add_raw_coordinates:
                        raw_coordinates = raw_data.features[:, -3:]
                        raw_data.features = raw_data.features[:, :-3]

                if raw_coordinates.shape[0] == 0:
                        return 0.0

                data = ME.SparseTensor(
                        coordinates=raw_data.coordinates,
                        features=raw_data.features,
                        device=self.device,
                )

                output = self.forward(
                        data,
                        point2segment=[
                                target[i]["point2segment"] for i in range(len(target))
                        ],
                        raw_coordinates=raw_coordinates,
                        extra_lang=raw_data.extra_lang,
                        is_eval=True,
                )

                output['raw_coordinates'] = raw_coordinates

                if raw_data.extra_lang is not None:
                        output['extra_lang'] = raw_data.extra_lang
                        if 'aux_outputs' in output:
                                for aux_outputs in output["aux_outputs"]:
                                        aux_outputs['extra_lang'] = raw_data.extra_lang

                losses = {}
                if self.config.data.test_mode != "test":
                        if self.config.trainer.deterministic:
                                torch.use_deterministic_algorithms(False)

                        disable_detection_loss = _ssr3dllm_env_flag("SSR3DLLM_DISABLE_DETECTION_LOSS", "0")
                        if (not disable_detection_loss) and (not self._is_step_token_sft()) and (output.get("pred_logits", None) is not None):
                                losses = self.criterion(
                                        output, target, mask_type=self.mask_type
                                )

                        # for k in list(losses.keys()):
                        #     if k in self.criterion.weight_dict:
                        #         losses[k] *= self.criterion.weight_dict[k]
                        #     else:
                        #         # remove this loss if not specified in `weight_dict`
                        #         losses.pop(k)
                        if self.config.trainer.deterministic:
                                torch.use_deterministic_algorithms(True)

                # SSR3DLLM: during geom-only pretraining we skip LLM forward/evaluation to speed up
                # validation and avoid mixing in irrelevant LLM behavior/metrics.
                if self.llama_config.enable_llm and not getattr(self, "ssr3dllm_geom_only", False):
                        batch_lang_infos, batch_map_target_to_query = \
                                self.prepare_llm(output, raw_data.extra_lang,
                                                 None, target, raw_data, file_names=file_names)

                        # No language queries in this batch; skip LLM compute.
                        if len(batch_lang_infos) == 0:
                                total = sum(losses.values()) if len(losses) > 0 else 0.0
                                if not isinstance(total, torch.Tensor):
                                        total = torch.tensor(float(total), device=self.device)
                                return {"loss": total.detach()}

                        batch_gt_inst_ids = [((i.batch_idx, i, i.max_gt_iou) if not self.training else (
                                i.batch_idx, i)) for i in batch_lang_infos]
                        batch_input_texts = [i.question for i in batch_lang_infos]
                        batch_output_texts = [i.answer for i in batch_lang_infos]
                        batch_eval_types = [i.lang_type for i in batch_lang_infos]
                        batch_instance_queries_hidden_state = [
                                i.query_hidden_feature for i in batch_lang_infos]
                        batch_instance_queries_normalized_embed = [
                                i.query_normalized_embed for i in batch_lang_infos]

                        # SSR3DLLM: ensure validation/test rel3dref samples get the same LLM-derived
                        # `llm_text_init` as training (otherwise eval falls back to BERT and becomes
                        # inconsistent with the training path).
                        # Default on; disable via: `export SSR3DLLM_FORCE_LLM_TEXT_INIT_EVAL=0`.
                        force_llm_init = os.environ.get("SSR3DLLM_FORCE_LLM_TEXT_INIT_EVAL", "1").strip().lower()
                        if (
                                force_llm_init not in {"", "0", "false", "no", "off"}
                                and getattr(self, "enable_ssr3dllm_geom", False)
                                and isinstance(getattr(self, "ssr3dllm_geom_head", None), SSR3DLLMGeomHeadForLLM)
                        ):
                                # Make sure we are in eval/no-grad mode.
                                self.llama_model.eval()
                                # NOTE: `self.llama_model.evaluate()` does NOT run `LLama3d.model_forward`,
                                # therefore it will not populate `lang_info.llm_text_init`.
                                # For SSR3DLLM we want eval to be consistent with train, so we do a cheap
                                # teacher-forcing forward on the relevant language samples to fill
                                # `lang_info.llm_text_init` (detached) before running geom eval.
                                rel_infos = []
                                for i in batch_lang_infos:
                                        lang_type = getattr(i, "lang_type", "") or ""
                                        if not isinstance(lang_type, str):
                                                continue
                                        prefix = lang_type.split(":")[0]
                                        # rel3dref: explicit relation chain samples (rel3d JSON)
                                        # scanrefer/m3dref: grounding QA samples we also evaluate via geom head.
                                        if prefix in {"rel3dref", "scanrefer", "m3dref"}:
                                                rel_infos.append(i)
                                if len(rel_infos) > 0:
                                        rel_gt_inst_ids = [((i.batch_idx, i, getattr(i, "max_gt_iou", 0.0))) for i in rel_infos]
                                        rel_input_texts = [i.question for i in rel_infos]
                                        rel_output_texts = [i.answer for i in rel_infos]
                                        rel_eval_types = [i.lang_type for i in rel_infos]
                                        rel_instance_queries_hidden_state = [i.query_hidden_feature for i in rel_infos]
                                        rel_instance_queries_normalized_embed = [i.query_normalized_embed for i in rel_infos]
                                        # Run a cheap teacher-forcing forward to populate lang_info.llm_text_init.
                                        # This does not affect evaluation outputs, but keeps geom eval consistent.
                                        with torch.no_grad():
                                                _ = self.llama_model(
                                                        batch_input_text_list=rel_input_texts,
                                                        batch_output_text_list=rel_output_texts,
                                                        batch_instance_queries_hidden_state=rel_instance_queries_hidden_state,
                                                        batch_instance_queries_normalized_embed=rel_instance_queries_normalized_embed,
                                                        batch_eval_types=rel_eval_types,
                                                        batch_gt_inst_ids=rel_gt_inst_ids,
                                                )

                        save_features_for_demo = True
                        if save_features_for_demo:
                                saved_scene_feature = {}
                                scene_feat_dir = os.path.join(self.config.general.save_dir, "scene_features")
                                os.makedirs(scene_feat_dir, exist_ok=True)
                                for i, scene_id in enumerate(file_names):
                                        saved_scene_feature["instance_queries_hidden_state"] = batch_instance_queries_hidden_state[i]
                                        saved_scene_feature["instance_queries_normalized_embed"] = batch_instance_queries_normalized_embed[i]
                                        torch.save(
                                                saved_scene_feature,
                                                os.path.join(scene_feat_dir, f"{scene_id}.bin"),
                                        )

                        self.llama_model.eval()
                        out_json, llm_logits = self.llama_model.evaluate(input_text_list=batch_input_texts,
                                                                         batch_instance_queries_hidden_state=batch_instance_queries_hidden_state,
                                                                         batch_instance_queries_normalized_embed=batch_instance_queries_normalized_embed,
                                                                         use_mini_batch=True,
                                                                         mini_batch_size=self.llama_config.test_batch_size,
                                                                         batch_out_text=batch_output_texts,
                                                                         batch_eval_types=batch_eval_types,
                                                                         batch_gt_inst_ids=batch_gt_inst_ids,
                                                                         output_logits=True,
                                                                         )
                        # ======================== prepare llm det =============================
                        assert len(target) == 1
                        pred_inst_masks = (
                                        output["pred_masks"][0][target[0]["point2segment"].cpu()] > 0.).float().cpu().clone()
                        pred_inst_masks = self.get_full_res_mask(
                                pred_inst_masks, inverse_maps[0], target_full[0]['point2segment'])
                        pred_inst_masks = np.array(pred_inst_masks).astype(bool)

                        all_llm_dectection = []
                        assert len(file_names) == 1
                        last_test_type = -1
                        for iinstance in llm_logits:
                                # TODO: here we only test first token for one-to-many case
                                if last_test_type == iinstance[0]:
                                        continue
                                if iinstance[0] == 0:
                                        last_test_type = 0
                                if last_test_type < iinstance[0]:
                                        for _ in range(last_test_type + 1, iinstance[0]):
                                                all_llm_dectection.append(torch.zeros((1, 100)))
                                all_llm_dectection.append(iinstance[1].to("cpu"))
                                last_test_type = iinstance[0]
                        if len(all_llm_dectection) < 198:  # extend to 198
                                for _ in range(len(all_llm_dectection), 198):
                                        all_llm_dectection.append(torch.zeros((1, 100)))
                        all_llm_dectection = [torch.zeros((1, 100))] + [all_llm_dectection[0]] + [
                                torch.zeros((1, 100))] + all_llm_dectection[1:]
                        assert len(all_llm_dectection) == 200
                        all_llm_dectection = torch.vstack(all_llm_dectection).T
                        # from shape (100,200) to shape (100) (per-query)
                        get_max = torch.max(all_llm_dectection, dim=1)
                        np.savez_compressed(
                                f"{self.llama_config.save_path}/{file_names[0]}.npz",
                                pred_masks=pred_inst_masks,
                                pred_scores=get_max[0],
                                pred_classes=self.validation_dataset._remap_model_output(
                                        get_max[1]),
                        )
                        # ================================= end  =================================
                        for item, gt, evaluation_type, gt_ids in zip(out_json, batch_output_texts, batch_eval_types, batch_gt_inst_ids):
                                if item["gt"] == "NONE":
                                        item["gt"] = gt
                                item["type"] = evaluation_type
                elif self.llama_config.enable_llm and getattr(self, "ssr3dllm_geom_only", False):
                        # Geom-only validation: skip LLM evaluation, but still populate LLM token features
                        # for rel3dref samples to keep geometry decoder conditioning consistent.
                        run_llm = os.environ.get("SSR3DLLM_GEOM_ONLY_RUN_LLM", "1").strip().lower() not in {
                                "",
                                "0",
                                "false",
                                "no",
                                "off",
                        }
                        if run_llm:
                                batch_lang_infos, batch_map_target_to_query = \
                                        self.prepare_llm(output, raw_data.extra_lang,
                                                         None, target, raw_data, file_names=file_names)
                                rel_infos = []
                                for i in batch_lang_infos:
                                        lang_type = getattr(i, "lang_type", "")
                                        if isinstance(lang_type, str) and lang_type.split(":")[0] == "rel3dref":
                                                rel_infos.append(i)
                                if rel_infos:
                                        rel_gt_inst_ids = [((i.batch_idx, i, getattr(i, "max_gt_iou", 0.0))) for i in rel_infos]
                                        rel_input_texts = [i.question for i in rel_infos]
                                        rel_output_texts = [i.answer for i in rel_infos]
                                        rel_eval_types = [i.lang_type for i in rel_infos]
                                        rel_instance_queries_hidden_state = [i.query_hidden_feature for i in rel_infos]
                                        rel_instance_queries_normalized_embed = [i.query_normalized_embed for i in rel_infos]
                                        try:
                                                self.llama_model.eval()
                                                with torch.no_grad():
                                                        _ = self.llama_model(
                                                                batch_input_text_list=rel_input_texts,
                                                                batch_output_text_list=rel_output_texts,
                                                                batch_instance_queries_hidden_state=rel_instance_queries_hidden_state,
                                                                batch_instance_queries_normalized_embed=rel_instance_queries_normalized_embed,
                                                                batch_eval_types=rel_eval_types,
                                                                batch_gt_inst_ids=rel_gt_inst_ids,
                                                        )
                                        except Exception:
                                                pass

                # Step-token SFT / rel3d-only training can disable detection/segmentation heads,
                # making `pred_logits`/`pred_masks` unavailable. In that case, skip the legacy
                # instance-level evaluation (AP/IoU over masks) to avoid crashes during sanity check.
                if (not self._is_step_token_sft()) and (output.get("pred_logits", None) is not None):
                        self.eval_instance_step(
                                output,
                                target,
                                target_full,
                                inverse_maps,
                                file_names,
                                original_coordinates,
                                original_colors,
                                original_normals,
                                raw_coordinates,
                                data_idx,
                                extra_lang=raw_data.extra_lang
                        )

                if getattr(self, "enable_ssr3dllm_geom", False) and isinstance(
                        getattr(self, "ssr3dllm_geom_head", None), SSR3DLLMGeomHeadForLLM
                ):
                        coords = output.get("sampled_coords", None)
                        if coords is not None:
                                if isinstance(coords, np.ndarray):
                                        coords_tensor = torch.from_numpy(coords).to(self.device)
                                else:
                                        coords_tensor = coords.to(self.device)

                                if "batch_lang_infos" not in locals() or batch_lang_infos is None:
                                        batch_lang_infos, batch_map_target_to_query = self.prepare_llm(
                                                output, raw_data.extra_lang, None, target, raw_data, file_names=file_names
                                        )
                                elif "batch_map_target_to_query" not in locals() or batch_map_target_to_query is None:
                                        # We still need target->query mapping for optional Vigor debugging.
                                        _, batch_map_target_to_query = self.prepare_llm(
                                                output, raw_data.extra_lang, None, target, raw_data, file_names=file_names
                                        )
                                # Vigor backend: compute per-query full-res AABB box_info=[cx,cy,cz,volume]
                                # from Mask3D pred_masks + original_coordinates (matches mask3d-vigor inputs).
                                box_info_by_bid = None
                                valid_queries_by_bid = None
                                try:
                                        if os.environ.get("SSR3DLLM_GEOM_BACKEND", "decoder").strip().lower() == "vigor":
                                                box_list = []
                                                valid_list = []
                                                for bid in range(len(target)):
                                                        # Fallback: centers from Mask3D queries (volume placeholder).
                                                        Q_fallback = None
                                                        centers = None
                                                        if (
                                                                isinstance(coords_tensor, torch.Tensor)
                                                                and coords_tensor.dim() == 3
                                                                and coords_tensor.size(0) > bid
                                                        ):
                                                                Q_fallback = int(coords_tensor.size(1))
                                                                centers = coords_tensor[bid].detach().cpu().to(dtype=torch.float32)
                                                        # Try to infer Q from mask logits.
                                                        mask_logits = None
                                                        try:
                                                                if "pred_masks" in output:
                                                                        mask_logits = output["pred_masks"][bid]
                                                        except Exception:
                                                                mask_logits = None
                                                        if isinstance(mask_logits, torch.Tensor) and mask_logits.dim() == 2:
                                                                Q = int(mask_logits.size(1))
                                                        elif Q_fallback is not None:
                                                                Q = int(Q_fallback)
                                                        else:
                                                                # Last resort: infer Q from queries_hidden_state.
                                                                try:
                                                                        qh = output.get("queries_hidden_state", None)
                                                                        if (
                                                                                isinstance(qh, torch.Tensor)
                                                                                and qh.dim() == 3
                                                                                and qh.size(0) > bid
                                                                        ):
                                                                                Q = int(qh.size(1))
                                                                        else:
                                                                                continue
                                                                except Exception:
                                                                        continue

                                                        fallback_box_info = torch.zeros((Q, 4), dtype=torch.float32)
                                                        if isinstance(centers, torch.Tensor) and centers.dim() == 2 and centers.size(0) == Q:
                                                                fallback_box_info[:, :3] = centers
                                                        fallback_box_info[:, 3] = 1.0

                                                        # If anything required for full-res AABB is missing, use fallback.
                                                        if mask_logits is None or not isinstance(mask_logits, torch.Tensor):
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue
                                                        if "point2segment" not in target[bid] or target[bid]["point2segment"] is None:
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue
                                                        if (
                                                                bid >= len(target_full)
                                                                or "point2segment" not in target_full[bid]
                                                                or target_full[bid]["point2segment"] is None
                                                        ):
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue

                                                        # [num_segments, Q] -> [Nsparse, Q]
                                                        mask_logits_cpu = mask_logits.detach().cpu()
                                                        seg_idx = target[bid]["point2segment"].detach().cpu()
                                                        if seg_idx.numel() == 0:
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue
                                                        mask_pts_all = mask_logits_cpu[seg_idx]  # [Nsparse,Q]
                                                        mask_pts_all = (mask_pts_all > 0.0).float()

                                                        pred_full_all = self.get_full_res_mask(
                                                                mask_pts_all,
                                                                inverse_maps[bid],
                                                                target_full[bid]["point2segment"],
                                                        )
                                                        pred_full_all = (pred_full_all > 0.5)

                                                        coords_full = original_coordinates[bid]
                                                        if isinstance(coords_full, np.ndarray):
                                                                coords_full = torch.from_numpy(coords_full).to(dtype=torch.float32)
                                                        elif isinstance(coords_full, torch.Tensor):
                                                                coords_full = coords_full.detach().cpu().to(dtype=torch.float32)
                                                        else:
                                                                box_list.append(fallback_box_info)
                                                                continue

                                                        if coords_full.dim() != 2 or coords_full.size(1) != 3:
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue
                                                        if pred_full_all.dim() != 2 or pred_full_all.size(0) != coords_full.size(0):
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((Q,), dtype=torch.bool))
                                                                continue

                                                        Q = int(pred_full_all.size(1))
                                                        if Q != int(fallback_box_info.size(0)):
                                                                box_list.append(fallback_box_info)
                                                                valid_list.append(torch.ones((int(fallback_box_info.size(0)),), dtype=torch.bool))
                                                                continue

                                                        # Per-query validity: whether this query selects any full-res points.
                                                        valid_q = pred_full_all.any(dim=0).to(dtype=torch.bool)
                                                        box_info = fallback_box_info
                                                        for q in range(Q):
                                                                m = pred_full_all[:, q]
                                                                if m.numel() == 0 or (not bool(m.any())):
                                                                        continue
                                                                pts = coords_full[m]
                                                                if pts.numel() == 0:
                                                                        continue
                                                                minv = pts.min(dim=0).values
                                                                maxv = pts.max(dim=0).values
                                                                center = (minv + maxv) * 0.5
                                                                size = torch.clamp(maxv - minv, min=0.0)
                                                                vol = float((size[0] * size[1] * size[2]).item())
                                                                box_info[q, :3] = center
                                                                box_info[q, 3] = vol
                                                        box_list.append(box_info)
                                                        valid_list.append(valid_q)

                                                if box_list and all(isinstance(b, torch.Tensor) for b in box_list):
                                                        box_info_by_bid = torch.stack(box_list, dim=0)  # [B,Q,4]
                                                if valid_list and all(isinstance(v, torch.Tensor) for v in valid_list):
                                                        valid_queries_by_bid = torch.stack(valid_list, dim=0)  # [B,Q]
                                except Exception:
                                        box_info_by_bid = None
                                        valid_queries_by_bid = None
                                # Vigor backend: precompute per-query predicted class names (ScanNet200) for building
                                # Vigor's `pred_class_mask` in the runtime listener.
                                pred_class_names_by_bid = None
                                debug_predcls = os.environ.get("SSR3DLLM_DEBUG_VIGOR_PREDCLS", "0").strip().lower() in {
                                        "1",
                                        "true",
                                        "yes",
                                        "y",
                                        "on",
                                }
                                pred_logits = output.get("pred_logits", None)
                                # Mask3D / Mask3DLang may return:
                                #   - Tensor [B,Q,C]
                                #   - Tensor [L,B,Q,C] (decoder layers)
                                #   - list length B, each Tensor [Q,C]   (common in language-conditioned heads)
                                #   - list length L (decoder layers), each Tensor [B,Q,C] / list[B][Q,C]
                                if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
                                        # If this is a decoder-layer list, use last layer; otherwise keep it.
                                        if torch.is_tensor(pred_logits[-1]) or (
                                                isinstance(pred_logits[-1], (list, tuple)) and len(pred_logits[-1]) > 0
                                        ):
                                                pred_logits = pred_logits[-1]
                                # Some variants return [L,B,Q,C]; use the last layer.
                                if isinstance(pred_logits, torch.Tensor) and pred_logits.dim() == 4:
                                        pred_logits = pred_logits[-1]
                                # If per-bid list [Q,C], stack to [B,Q,C].
                                if (
                                        isinstance(pred_logits, (list, tuple))
                                        and len(pred_logits) > 0
                                        and all(isinstance(x, torch.Tensor) and x.dim() == 2 for x in pred_logits)
                                ):
                                        try:
                                                pred_logits = torch.stack(list(pred_logits), dim=0)
                                        except Exception:
                                                pass
                                if isinstance(pred_logits, torch.Tensor) and pred_logits.dim() == 3:
                                        B, Q, _ = pred_logits.shape
                                        C = int(pred_logits.size(-1))
                                        # Only treat pred_logits as ScanNet200 semantic logits when C matches.
                                        if C == len(CLASS_LABELS_200):
                                                logits_cpu = pred_logits.detach().cpu()
                                                cls_ids = torch.argmax(logits_cpu, dim=-1)  # [B,Q]
                                                names_all = []
                                                for bid in range(int(B)):
                                                        bid_cls = cls_ids[bid].tolist()
                                                        bid_names = [
                                                                str(CLASS_LABELS_200[int(c)]) if 0 <= int(c) < len(CLASS_LABELS_200) else "unknown"
                                                                for c in bid_cls
                                                        ]
                                                        names_all.append(bid_names)
                                                pred_class_names_by_bid = names_all
                                        else:
                                                if debug_predcls and int(getattr(self, "global_rank", 0)) == 0:
                                                        try:
                                                                print(
                                                                        "[SSR3DLLM][vigor_predcls][warn] "
                                                                        f"pred_logits last-dim={C} != scannet200({len(CLASS_LABELS_200)}); "
                                                                        "this head is likely token-conditioned, skip pred_class_names.",
                                                                        flush=True,
                                                                )
                                                        except Exception:
                                                                pass
                                else:
                                        if debug_predcls and int(getattr(self, "global_rank", 0)) == 0:
                                                try:
                                                        shape = tuple(pred_logits.shape) if isinstance(pred_logits, torch.Tensor) else type(pred_logits)
                                                        print(f"[SSR3DLLM][vigor_predcls][warn] pred_logits missing/unexpected: {shape}", flush=True)
                                                except Exception:
                                                        pass
                                # Fallback for Vigor: build per-query class names from GT instance labels.
                                # This mirrors Vigor's training-time behaviour where pred_class_mask can fall back
                                # to GT instance labels (when predicted class names are unavailable).
                                if pred_class_names_by_bid is None and isinstance(batch_map_target_to_query, list):
                                        try:
                                                from baseline.dataset.datasets.scannet200.scannet200_constants import (
                                                        CLASS_LABELS_200,
                                                        VALID_CLASS_IDS_200,
                                                )
                                                id_to_name = {int(cid): str(CLASS_LABELS_200[i]) for i, cid in enumerate(VALID_CLASS_IDS_200)}
                                        except Exception:
                                                id_to_name = {}
                                        names_all = []
                                        qh = output.get("queries_hidden_state", None)
                                        for bid in range(len(target)):
                                                # Determine Q
                                                Q = None
                                                if isinstance(qh, torch.Tensor) and qh.dim() == 3 and qh.size(0) > bid:
                                                        Q = int(qh.size(1))
                                                else:
                                                        try:
                                                                Q = int(output["pred_masks"][bid].shape[1])
                                                        except Exception:
                                                                Q = None
                                                if Q is None or Q <= 0:
                                                        names_all.append([])
                                                        continue
                                                pred_names = ["unknown"] * int(Q)
                                                if bid < len(batch_map_target_to_query):
                                                        mapping, valid = batch_map_target_to_query[bid]
                                                        try:
                                                                mapping_list = mapping.tolist() if isinstance(mapping, torch.Tensor) else list(mapping)
                                                        except Exception:
                                                                mapping_list = []
                                                        try:
                                                                valid_list = valid.tolist() if isinstance(valid, torch.Tensor) else list(valid)
                                                        except Exception:
                                                                valid_list = []
                                                        labels = None
                                                        if bid < len(target) and isinstance(target[bid], dict):
                                                                labels = target[bid].get("labels", None)
                                                        if isinstance(labels, torch.Tensor):
                                                                labels_list = labels.detach().cpu().tolist()
                                                        elif isinstance(labels, (list, tuple)):
                                                                labels_list = list(labels)
                                                        else:
                                                                labels_list = []
                                                        n_inst = min(len(mapping_list), len(valid_list), len(labels_list))
                                                        for inst_i in range(n_inst):
                                                                try:
                                                                        if not bool(valid_list[inst_i]):
                                                                                continue
                                                                except Exception:
                                                                        pass
                                                                try:
                                                                        q = int(mapping_list[inst_i])
                                                                except Exception:
                                                                        continue
                                                                if not (0 <= q < int(Q)):
                                                                        continue
                                                                try:
                                                                        cid = int(labels_list[inst_i])
                                                                except Exception:
                                                                        cid = -1
                                                                # Map either sparse ScanNet200 id or contiguous index to a string name.
                                                                name = id_to_name.get(cid, None)
                                                                if name is None and 0 <= cid < len(CLASS_LABELS_200):
                                                                        name = str(CLASS_LABELS_200[cid])
                                                                pred_names[q] = str(name) if name else "unknown"
                                                names_all.append(pred_names)
                                        if names_all and all(isinstance(x, list) and len(x) > 0 for x in names_all):
                                                pred_class_names_by_bid = names_all
                                geom_stats = self.ssr3dllm_geom_head.eval_rel3dref_for_batch(
                                        batch_lang_infos=batch_lang_infos,
                                        sampled_coords=coords_tensor,
                                        device=self.device,
                                        box_info_by_bid=box_info_by_bid,
                                        pred_class_names_by_bid=pred_class_names_by_bid,
                                        valid_queries_by_bid=valid_queries_by_bid,
                                )
                                losses["ssr3dllm_num_rel"] = torch.tensor(
                                        geom_stats.get("num_rel", 0.0),
                                        device=self.device,
                                        dtype=torch.float32,
                                )
                                losses["ssr3dllm_num_target_hit"] = torch.tensor(
                                        geom_stats.get("num_target_hit", 0.0),
                                        device=self.device,
                                        dtype=torch.float32,
                                )
                                losses["ssr3dllm_num_chain_hit"] = torch.tensor(
                                        geom_stats.get("num_chain_hit", 0.0),
                                        device=self.device,
                                        dtype=torch.float32,
                                )

                                # Optional: bbox IoU@0.25/0.5 on rel3dref samples, for quick monitoring.
                                # This is intentionally lightweight (no Lightning full test needed).
                                # Enable via: `export SSR3DLLM_EVAL_REL3D_IOU=1`.
                                #
                                # NOTE:
                                # The ScanRefer/M3DRef geom-grounding probe shares the same evaluation block below
                                # (it relies on the same per-scene `pred_masks` / `target_full` structures), so we
                                # also enter this block when geom-grounding eval is enabled.
                                eval_rel3d_iou = os.environ.get("SSR3DLLM_EVAL_REL3D_IOU", "0").strip().lower()
                                _eval_geom_grd_gate = os.environ.get("SSR3DLLM_EVAL_GEOM_GROUNDING", "0").strip().lower()
                                _disable_llm_grounding_gate = os.environ.get("SSR3DLLM_DISABLE_LLM_GROUNDING", "0").strip().lower()
                                if (
                                        eval_rel3d_iou not in {"", "0", "false", "no", "off"}
                                        or _eval_geom_grd_gate not in {"", "0", "false", "no", "off"}
                                        or _disable_llm_grounding_gate not in {"", "0", "false", "no", "off"}
                                ):
                                        iou_total = 0
                                        iou25_hit = 0
                                        iou50_hit = 0

                                        pred_items = self.ssr3dllm_geom_head.predict_rel3dref_for_batch(
                                                batch_lang_infos=batch_lang_infos,
                                                sampled_coords=coords_tensor,
                                                device=self.device,
                                        )
                                        if pred_items:
                                                # Pre-compute per-bid coords + masks only for the predicted query ids
                                                # that actually appear in this batch.
                                                by_bid: Dict[int, List[dict]] = defaultdict(list)
                                                for it in pred_items:
                                                        try:
                                                                by_bid[int(it.get("batch_idx", -1))].append(it)
                                                        except Exception:
                                                                continue

                                                for bid, items in by_bid.items():
                                                        if bid < 0 or bid >= len(target) or bid >= len(target_full):
                                                                continue
                                                        if "masks" not in target_full[bid]:
                                                                continue
                                                        if "point2segment" not in target[bid] or target[bid]["point2segment"] is None:
                                                                continue
                                                        if (
                                                                "point2segment" not in target_full[bid]
                                                                or target_full[bid]["point2segment"] is None
                                                        ):
                                                                continue

                                                        # Build a minimal set of query indices we need masks for.
                                                        num_gt_items = 0
                                                        q_pred_list = []
                                                        for it in items:
                                                                gt_inst = it.get("gt_inst_id", None)
                                                                if not isinstance(gt_inst, int):
                                                                        continue
                                                                num_gt_items += 1
                                                                q = it.get("pred_target_q", None)
                                                                if isinstance(q, int) and q >= 0:
                                                                        q_pred_list.append(q)
                                                        if num_gt_items <= 0:
                                                                continue

                                                        # Segment-level mask logits: [num_segments, Q]
                                                        mask_logits = output["pred_masks"][bid]
                                                        if mask_logits is None or not isinstance(mask_logits, torch.Tensor):
                                                                iou_total += num_gt_items
                                                                continue
                                                        Q = int(mask_logits.shape[1])

                                                        # Keep only valid query indices (<Q); STOP (=Q) is ignored for IoU.
                                                        q_pred_set = sorted({q for q in q_pred_list if 0 <= q < Q})
                                                        if not q_pred_set:
                                                                # All predictions are STOP/invalid for this bid.
                                                                iou_total += num_gt_items
                                                                continue
                                                        q_to_col = {q: i for i, q in enumerate(q_pred_set)}

                                                        # Per-sparse-point masks for selected queries: [Nsparse, K]
                                                        mask_pts = mask_logits[target[bid]["point2segment"].cpu()][:, q_pred_set]
                                                        mask_pts = (mask_pts > 0.0).float()
                                                        # Map to full-res points: [Nfull, K]
                                                        # IMPORTANT:
                                                        # For SSR3DLLM geom-grounding eval we compare masks against
                                                        # `original_coordinates` (point-level). If `self.eval_on_segments`
                                                        # is True, `get_full_res_mask()` would otherwise scatter to segments
                                                        # and cause a shape mismatch (leading to all samples skipped and
                                                        # metrics staying at 0). Force point-level masks here.
                                                        pred_full = self.get_full_res_mask(
                                                                mask_pts,
                                                                inverse_maps[bid],
                                                                target_full[bid]["point2segment"],
                                                                is_heatmap=True,
                                                        )
                                                        pred_full = (pred_full > 0.5)

                                                        # Full-res coords used for bbox; must align with target_full masks.
                                                        coords_full = original_coordinates[bid]
                                                        if isinstance(coords_full, np.ndarray):
                                                                coords_full = torch.from_numpy(coords_full).to(dtype=torch.float32)
                                                        elif isinstance(coords_full, torch.Tensor):
                                                                coords_full = coords_full.detach().cpu().to(dtype=torch.float32)
                                                        else:
                                                                iou_total += num_gt_items
                                                                continue

                                                        # Sanity: ensure point dimension matches.
                                                        if coords_full.dim() != 2 or coords_full.size(1) != 3:
                                                                iou_total += num_gt_items
                                                                continue
                                                        if pred_full.dim() != 2 or pred_full.size(0) != coords_full.size(0):
                                                                iou_total += num_gt_items
                                                                continue

                                                        gt_masks = target_full[bid]["masks"].detach().cpu().to(dtype=torch.bool)
                                                        inst_mapping = target_full[bid].get("instance_mapping", None)
                                                        for it in items:
                                                                pred_q = it.get("pred_target_q", None)
                                                                gt_inst = it.get("gt_inst_id", None)
                                                                if gt_inst is None or not isinstance(gt_inst, int):
                                                                        continue
                                                                if pred_q is None or not isinstance(pred_q, int):
                                                                        iou_total += 1
                                                                        continue
                                                                if pred_q < 0 or pred_q >= Q:
                                                                        iou_total += 1
                                                                        continue
                                                                if pred_q not in q_to_col:
                                                                        iou_total += 1
                                                                        continue

                                                                # `gt_inst_id` refers to original ScanNet instance id.
                                                                # But `target_full[bid]["masks"]` is indexed by a remapped
                                                                # 0..K-1 order. Use the stored mapping when needed.
                                                                gt_mask_idx = None
                                                                if 0 <= gt_inst < int(gt_masks.shape[0]):
                                                                        gt_mask_idx = gt_inst
                                                                elif isinstance(inst_mapping, dict):
                                                                        try:
                                                                                gt_mask_idx = int(inst_mapping.get(int(gt_inst), -1))
                                                                        except Exception:
                                                                                gt_mask_idx = -1
                                                                if gt_mask_idx is None or gt_mask_idx < 0 or gt_mask_idx >= int(gt_masks.shape[0]):
                                                                        iou_total += 1
                                                                        continue

                                                                pred_mask = pred_full[:, q_to_col[pred_q]].to(dtype=torch.bool)
                                                                gt_mask = gt_masks[gt_mask_idx]
                                                                if pred_mask.numel() != gt_mask.numel():
                                                                        iou_total += 1
                                                                        continue
                                                                if pred_mask.sum().item() == 0 or gt_mask.sum().item() == 0:
                                                                        # Treat empty mask as IoU=0 (counted, but no hit).
                                                                        iou_total += 1
                                                                        continue

                                                                pred_pts = coords_full[pred_mask]
                                                                gt_pts = coords_full[gt_mask]
                                                                if pred_pts.numel() == 0 or gt_pts.numel() == 0:
                                                                        iou_total += 1
                                                                        continue
                                                                pred_box = torch.stack(
                                                                        [pred_pts.min(dim=0).values, pred_pts.max(dim=0).values],
                                                                        dim=0,
                                                                ).unsqueeze(0)  # [1,2,3]
                                                                gt_box = torch.stack(
                                                                        [gt_pts.min(dim=0).values, gt_pts.max(dim=0).values],
                                                                        dim=0,
                                                                ).unsqueeze(0)
                                                                iou = float(get_batch_aabb_pair_ious(pred_box, gt_box)[0].item())
                                                                iou_total += 1
                                                                if iou >= 0.25:
                                                                        iou25_hit += 1
                                                                if iou >= 0.50:
                                                                        iou50_hit += 1

                                        losses["ssr3dllm_num_iou_total"] = torch.tensor(
                                                float(iou_total), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_iou25_hit"] = torch.tensor(
                                                float(iou25_hit), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_iou50_hit"] = torch.tensor(
                                                float(iou50_hit), device=self.device, dtype=torch.float32
                                        )

                                # SSR3DLLM: optional "geom-chain grounding eval" for existing grounding datasets
                                # (ScanRefer / M3DRef). This computes IoU@0.25/0.5 by selecting the target query
                                # from SSR3DLLM's geometry decoder, then evaluating against GT instance masks.
                                #
                                # Enable via:
                                #   export SSR3DLLM_EVAL_GEOM_GROUNDING=1
                                # Or implicitly when baseline LLM grounding is disabled:
                                #   export SSR3DLLM_DISABLE_LLM_GROUNDING=1
                                        # Default: keep the original behaviour (require explicit "<geom>" trigger).
                                        # When probing geom grounding (or when LLM grounding is disabled), we can
                                        # relax this gating to avoid "total=0" / NaN metrics.
                                        require_geom_trigger = True
                                        eval_geom_grd = os.environ.get("SSR3DLLM_EVAL_GEOM_GROUNDING", "0").strip().lower()
                                        disable_llm_grounding = os.environ.get("SSR3DLLM_DISABLE_LLM_GROUNDING", "0").strip().lower()
                                        if (
                                                eval_geom_grd not in {"", "0", "false", "no", "off"}
                                                or disable_llm_grounding not in {"", "0", "false", "no", "off"}
                                        ):
                                                # For eval-time probing, default to evaluating all ScanRefer/M3DRef
                                                # items (even without "<geom>") to avoid "total=0" / NaN metrics.
                                                # Users can force gating via SSR3DLLM_EVAL_GEOM_REQUIRE_TRIGGER=1.
                                                require_geom_trigger = (
                                                        os.environ.get("SSR3DLLM_EVAL_GEOM_REQUIRE_TRIGGER", "0").strip().lower()
                                                        not in {"", "0", "false", "no", "off"}
                                                )
                                                # If baseline LLM grounding is disabled, never require "<geom>".
                                                if disable_llm_grounding not in {"", "0", "false", "no", "off"}:
                                                        require_geom_trigger = False

                                        # Expected denominators: count ScanRefer/M3DRef language items even if
                                        # prediction/IoU computation fails, so metrics never become NaN due to
                                        # scan_total/m3d_total staying at 0.
                                        scan_total_expected = 0
                                        m3d_total_expected = 0
                                        try:
                                                for li in batch_lang_infos:
                                                        lt = getattr(li, "lang_type", "") or ""
                                                        if not isinstance(lt, str):
                                                                continue
                                                        prefix = lt.split(":")[0]
                                                        if prefix not in {"scanrefer", "m3dref"}:
                                                                continue
                                                        if require_geom_trigger:
                                                                qtext = getattr(li, "question", "") or ""
                                                                use_geom = getattr(li, "use_geom_trigger", False)
                                                                if ("<geom>" not in str(qtext)) and (not bool(use_geom)):
                                                                        continue
                                                        if prefix == "scanrefer":
                                                                scan_total_expected += 1
                                                        else:
                                                                m3d_total_expected += 1
                                        except Exception:
                                                scan_total_expected = 0
                                                m3d_total_expected = 0

                                        scan_total = 0
                                        scan_mask25 = 0
                                        scan_mask50 = 0
                                        scan_bbox25 = 0
                                        scan_bbox50 = 0

                                        m3d_total = 0
                                        m3d_mask25 = 0
                                        m3d_mask50 = 0
                                        m3d_bbox25 = 0
                                        m3d_bbox50 = 0
                                        m3d_bbox_f1_25_sum = 0.0
                                        m3d_bbox_f1_50_sum = 0.0

                                        # If using Vigor backend for geometry grounding, compute a per-query
                                        # full-res AABB box_info=[cx,cy,cz,volume] to match mask3d-vigor inputs.
                                        box_info_by_bid = None
                                        try:
                                                if os.environ.get("SSR3DLLM_GEOM_BACKEND", "decoder").strip().lower() == "vigor":
                                                        box_list = []
                                                        for bid in range(len(target)):
                                                                # Fallback: centers from Mask3D queries (volume placeholder).
                                                                Q_fallback = None
                                                                if isinstance(coords_tensor, torch.Tensor) and coords_tensor.dim() == 3 and coords_tensor.size(0) > bid:
                                                                        Q_fallback = int(coords_tensor.size(1))
                                                                        centers = coords_tensor[bid].detach().cpu().to(dtype=torch.float32)
                                                                else:
                                                                        centers = None

                                                                # Try to infer Q from mask logits.
                                                                mask_logits = None
                                                                try:
                                                                        if "pred_masks" in output:
                                                                                mask_logits = output["pred_masks"][bid]
                                                                except Exception:
                                                                        mask_logits = None
                                                                if isinstance(mask_logits, torch.Tensor) and mask_logits.dim() == 2:
                                                                        Q = int(mask_logits.size(1))
                                                                elif Q_fallback is not None:
                                                                        Q = int(Q_fallback)
                                                                else:
                                                                        continue

                                                                fallback_box_info = torch.zeros((Q, 4), dtype=torch.float32)
                                                                if isinstance(centers, torch.Tensor) and centers.dim() == 2 and centers.size(0) == Q and centers.size(1) == 3:
                                                                        fallback_box_info[:, :3] = centers
                                                                fallback_box_info[:, 3] = 1.0

                                                                # If anything required for full-res AABB is missing, use fallback.
                                                                if mask_logits is None or not isinstance(mask_logits, torch.Tensor):
                                                                        box_list.append(fallback_box_info)
                                                                        continue
                                                                if "point2segment" not in target[bid] or target[bid]["point2segment"] is None:
                                                                        box_list.append(fallback_box_info)
                                                                        continue
                                                                if (
                                                                        bid >= len(target_full)
                                                                        or "point2segment" not in target_full[bid]
                                                                        or target_full[bid]["point2segment"] is None
                                                                ):
                                                                        box_list.append(fallback_box_info)
                                                                        continue

                                                                # [num_segments, Q] -> [Nsparse, Q]
                                                                mask_logits_cpu = mask_logits.detach().cpu()
                                                                seg_idx = target[bid]["point2segment"].detach().cpu()
                                                                if seg_idx.numel() == 0:
                                                                        box_list.append(fallback_box_info)
                                                                        continue
                                                                mask_pts_all = mask_logits_cpu[seg_idx]  # [Nsparse,Q]
                                                                mask_pts_all = (mask_pts_all > 0.0).float()

                                                                pred_full_all = self.get_full_res_mask(
                                                                        mask_pts_all,
                                                                        inverse_maps[bid],
                                                                        target_full[bid]["point2segment"],
                                                                )
                                                                pred_full_all = (pred_full_all > 0.5)

                                                                coords_full = original_coordinates[bid]
                                                                if isinstance(coords_full, np.ndarray):
                                                                        coords_full = torch.from_numpy(coords_full).to(dtype=torch.float32)
                                                                elif isinstance(coords_full, torch.Tensor):
                                                                        coords_full = coords_full.detach().cpu().to(dtype=torch.float32)
                                                                else:
                                                                        box_list.append(fallback_box_info)
                                                                        continue

                                                                if coords_full.dim() != 2 or coords_full.size(1) != 3:
                                                                        box_list.append(fallback_box_info)
                                                                        continue
                                                                if pred_full_all.dim() != 2 or pred_full_all.size(0) != coords_full.size(0):
                                                                        box_list.append(fallback_box_info)
                                                                        continue

                                                                Q = int(pred_full_all.size(1))
                                                                if Q != int(fallback_box_info.size(0)):
                                                                        box_list.append(fallback_box_info)
                                                                        continue

                                                                box_info = fallback_box_info
                                                                for q in range(Q):
                                                                        m = pred_full_all[:, q]
                                                                        if m.numel() == 0 or (not bool(m.any())):
                                                                                continue
                                                                        pts = coords_full[m]
                                                                        if pts.numel() == 0:
                                                                                continue
                                                                        minv = pts.min(dim=0).values
                                                                        maxv = pts.max(dim=0).values
                                                                        center = (minv + maxv) * 0.5
                                                                        size = torch.clamp(maxv - minv, min=0.0)
                                                                        vol = float((size[0] * size[1] * size[2]).item())
                                                                        box_info[q, :3] = center
                                                                        box_info[q, 3] = vol
                                                                box_list.append(box_info)

                                                        if box_list and all(isinstance(b, torch.Tensor) for b in box_list):
                                                                box_info_by_bid = torch.stack(box_list, dim=0)  # [B,Q,4]
                                        except Exception:
                                                box_info_by_bid = None

                                        # SSR3DLLM: probe geom-chain grounding on ScanRefer/M3DRef by selecting a single
                                        # Mask3D query via the geometry head and computing mask/bbox IoU against GT masks.
                                        dbg_geom = os.environ.get("SSR3DLLM_DEBUG_GEOM_GROUNDING", "0").strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "on",
                                        }
                                        try:
                                                dbg_max = int(os.environ.get("SSR3DLLM_DEBUG_GEOM_GROUNDING_MAX", "5").strip())
                                        except Exception:
                                                dbg_max = 5
                                        dbg_max = max(0, int(dbg_max))
                                        dbg_printed = 0
                                        dbg_empty = os.environ.get("SSR3DLLM_DEBUG_GEOM_GROUNDING_EMPTY", "0").strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "on",
                                        }
                                        try:
                                                dbg_empty_max = int(
                                                        os.environ.get("SSR3DLLM_DEBUG_GEOM_GROUNDING_EMPTY_MAX", "20").strip()
                                                )
                                        except Exception:
                                                dbg_empty_max = 20
                                        dbg_empty_max = max(0, int(dbg_empty_max))
                                        dbg_empty_printed = 0

                                        skip_all = defaultdict(int)
                                        grd_items = []
                                        if getattr(self, "ssr3dllm_geom_head", None) is None:
                                                skip_all["no_geom_head"] += int(scan_total_expected + m3d_total_expected)
                                        else:
                                                # Attach GT target class name for oracle chain probes (best-effort).
                                                # This is consumed by `predict_geom_target_for_batch()` when
                                                # `SSR3DLLM_GEOM_CHAIN_MODE=oracle_*`.
                                                try:
                                                        geom_chain_mode = str(
                                                                os.environ.get("SSR3DLLM_GEOM_CHAIN_MODE", "bypass")
                                                        ).strip().lower()
                                                        if geom_chain_mode in {"oracle_sameclass", "oracle_gtanchors"}:
                                                                try:
                                                                        from baseline.dataset.datasets.scannet200.scannet200_constants import (
                                                                                CLASS_LABELS_200,
                                                                                VALID_CLASS_IDS_200,
                                                                        )
                                                                        id_to_name = {
                                                                                int(cid): str(CLASS_LABELS_200[i])
                                                                                for i, cid in enumerate(VALID_CLASS_IDS_200)
                                                                        }
                                                                except Exception:
                                                                        CLASS_LABELS_200 = []  # type: ignore
                                                                        id_to_name = {}

                                                                def _get_gt_class_name(li) -> str:
                                                                        bid0 = getattr(li, "batch_idx", None)
                                                                        if bid0 is None or not (0 <= int(bid0) < len(target_full)):
                                                                                return ""
                                                                        labels = None
                                                                        try:
                                                                                labels = target_full[int(bid0)].get("labels", None)
                                                                        except Exception:
                                                                                labels = None
                                                                        if labels is None:
                                                                                return ""
                                                                        if isinstance(labels, torch.Tensor):
                                                                                labels_list = labels.detach().cpu().tolist()
                                                                        elif isinstance(labels, (list, tuple)):
                                                                                labels_list = list(labels)
                                                                        else:
                                                                                return ""

                                                                        inst_ids = getattr(li, "inst_ids_answer", None)
                                                                        inst0 = None
                                                                        if isinstance(inst_ids, list) and inst_ids:
                                                                                first = inst_ids[0]
                                                                                if isinstance(first, list) and first:
                                                                                        inst0 = first[0]
                                                                                elif isinstance(first, (int, np.integer)):
                                                                                        inst0 = first
                                                                        if inst0 is None:
                                                                                return ""
                                                                        try:
                                                                                inst0 = int(inst0)
                                                                        except Exception:
                                                                                return ""
                                                                        if not (0 <= int(inst0) < len(labels_list)):
                                                                                return ""
                                                                        try:
                                                                                lid = int(labels_list[int(inst0)])
                                                                        except Exception:
                                                                                return ""
                                                                        # If already contiguous [0..199], map directly.
                                                                        if 0 <= lid < len(CLASS_LABELS_200):
                                                                                return str(CLASS_LABELS_200[lid])
                                                                        # Otherwise treat as ScanNet200 raw id.
                                                                        return str(id_to_name.get(int(lid), ""))

                                                                for li in batch_lang_infos:
                                                                        lt = getattr(li, "lang_type", "") or ""
                                                                        if not isinstance(lt, str):
                                                                                continue
                                                                        pfx = lt.split(":")[0]
                                                                        if pfx not in {"scanrefer", "m3dref"}:
                                                                                continue
                                                                        try:
                                                                                name = _get_gt_class_name(li)
                                                                                if name:
                                                                                        setattr(li, "ssr3dllm_target_class_name", name)
                                                                        except Exception:
                                                                                continue
                                                except Exception:
                                                        pass

                                                try:
                                                        grd_items = self.ssr3dllm_geom_head.predict_geom_target_for_batch(
                                                                batch_lang_infos=batch_lang_infos,
                                                                sampled_coords=coords_tensor,
                                                                device=self.device,
                                                                lang_prefixes=("scanrefer", "m3dref"),
                                                                require_geom_trigger=require_geom_trigger,
                                                                box_info_by_bid=box_info_by_bid,
                                                                pred_class_names_by_bid=pred_class_names_by_bid,
                                                                valid_queries_by_bid=valid_queries_by_bid,
                                                        )
                                                except Exception as exc:
                                                        grd_items = []
                                                        skip_all["predict_exception"] += int(scan_total_expected + m3d_total_expected)
                                                        if dbg_geom and int(getattr(self, "global_rank", 0)) == 0:
                                                                try:
                                                                        logger.warning(f"[SSR3DLLM][geom_grd][debug] predict exception: {exc}")
                                                                except Exception:
                                                                        pass

                                        if dbg_geom and int(getattr(self, "global_rank", 0)) == 0:
                                                try:
                                                        by_prefix_dbg = defaultdict(int)
                                                        for _it in (grd_items or []):
                                                                lt = _it.get("lang_type", "")
                                                                pfx = str(lt).split(":")[0] if isinstance(lt, str) else ""
                                                                by_prefix_dbg[pfx] += 1
                                                        logger.warning(
                                                                f"[SSR3DLLM][geom_grd][debug] grd_items={0 if grd_items is None else len(grd_items)} "
                                                                f"by_prefix={dict(by_prefix_dbg)} require_trigger={int(bool(require_geom_trigger))} "
                                                                f"expected={{'scanrefer':{int(scan_total_expected)},'m3dref':{int(m3d_total_expected)}}} "
                                                                f"eval_on_segments={int(bool(getattr(self, 'eval_on_segments', False)))}"
                                                        )
                                                except Exception:
                                                        pass

                                        def _parse_int(v):
                                                try:
                                                        if isinstance(v, (int, np.integer)):
                                                                return int(v)
                                                        if torch.is_tensor(v) and v.numel() == 1:
                                                                return int(v.item())
                                                        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                                                                return int(v.strip())
                                                except Exception:
                                                        return None
                                                return None

                                        if grd_items:
                                                by_bid_grd = defaultdict(list)
                                                for it in grd_items:
                                                        try:
                                                                by_bid_grd[int(it.get("batch_idx", -1))].append(it)
                                                        except Exception:
                                                                skip_all["bad_batch_idx"] += 1
                                                                continue

                                                for bid, items in by_bid_grd.items():
                                                        if bid < 0 or bid >= len(target) or bid >= len(target_full):
                                                                skip_all["bad_bid"] += len(items)
                                                                continue
                                                        if "masks" not in target_full[bid]:
                                                                skip_all["missing_target_full_masks"] += len(items)
                                                                continue
                                                        if "point2segment" not in target[bid] or target[bid]["point2segment"] is None:
                                                                skip_all["missing_target_point2segment"] += len(items)
                                                                continue
                                                        if (
                                                                "point2segment" not in target_full[bid]
                                                                or target_full[bid]["point2segment"] is None
                                                        ):
                                                                skip_all["missing_target_full_point2segment"] += len(items)
                                                                continue

                                                        # Count denominators per language item (treat failures as miss).
                                                        for it in items:
                                                                lang_type = it.get("lang_type", "") or ""
                                                                prefix = str(lang_type).split(":")[0] if isinstance(lang_type, str) else ""
                                                                if prefix == "scanrefer":
                                                                        scan_total += 1
                                                                elif prefix == "m3dref":
                                                                        m3d_total += 1

                                                        mask_logits = output.get("pred_masks", None)
                                                        if not isinstance(mask_logits, (list, tuple)) or bid >= len(mask_logits):
                                                                skip_all["missing_pred_masks"] += len(items)
                                                                continue
                                                        mask_logits = mask_logits[bid]
                                                        if mask_logits is None or not isinstance(mask_logits, torch.Tensor):
                                                                skip_all["bad_pred_masks_tensor"] += len(items)
                                                                continue
                                                        if mask_logits.dim() != 2:
                                                                skip_all["bad_pred_masks_shape"] += len(items)
                                                                continue
                                                        Q = int(mask_logits.shape[1])

                                                        # Collect predicted query ids in this bid.
                                                        q_pred_set = []
                                                        pred_q_by_item = []
                                                        for it in items:
                                                                pred_q = _parse_int(it.get("pred_target_q", None))
                                                                pred_q_by_item.append(pred_q)
                                                                if pred_q is None:
                                                                        continue
                                                                if 0 <= int(pred_q) < int(Q):
                                                                        q_pred_set.append(int(pred_q))
                                                        q_pred_set = sorted(set(q_pred_set))
                                                        if not q_pred_set:
                                                                skip_all["no_valid_pred_q"] += len(items)
                                                                continue
                                                        q_to_col = {q: i for i, q in enumerate(q_pred_set)}

                                                        # Segment-level logits -> sparse-point masks -> full-res point masks.
                                                        try:
                                                                mask_pts_raw = mask_logits[target[bid]["point2segment"].cpu()][:, q_pred_set]
                                                                mask_pts = (mask_pts_raw > 0.0).float()
                                                        except Exception:
                                                                skip_all["mask_pts_build_failed"] += len(items)
                                                                continue
                                                        pred_full_heat = self.get_full_res_mask(
                                                                mask_pts,
                                                                inverse_maps[bid],
                                                                target_full[bid]["point2segment"],
                                                                is_heatmap=True,
                                                        )
                                                        pred_full = (pred_full_heat > 0.5)

                                                        coords_full = original_coordinates[bid]
                                                        if isinstance(coords_full, np.ndarray):
                                                                coords_full = torch.from_numpy(coords_full).to(dtype=torch.float32)
                                                        elif isinstance(coords_full, torch.Tensor):
                                                                coords_full = coords_full.detach().cpu().to(dtype=torch.float32)
                                                        else:
                                                                skip_all["coords_full_type_bad"] += len(items)
                                                                continue
                                                        if coords_full.dim() != 2 or coords_full.size(1) != 3:
                                                                skip_all["bad_coords_full_shape"] += len(items)
                                                                continue
                                                        if pred_full.dim() != 2 or pred_full.size(0) != coords_full.size(0):
                                                                skip_all["pred_full_shape_mismatch"] += len(items)
                                                                continue

                                                        gt_masks_all = target_full[bid]["masks"]
                                                        if isinstance(gt_masks_all, torch.Tensor):
                                                                gt_masks_all = gt_masks_all.detach().cpu().to(dtype=torch.bool)
                                                        else:
                                                                skip_all["gt_masks_type_bad"] += len(items)
                                                                continue
                                                        inst_mapping = target_full[bid].get("instance_mapping", None)
                                                        # Normalize GT masks to point-level if they are stored at segment-level.
                                                        try:
                                                                if gt_masks_all.dim() == 2 and gt_masks_all.size(1) != coords_full.size(0):
                                                                        p2s = target_full[bid].get("point2segment", None)
                                                                        if torch.is_tensor(p2s) and p2s.numel() == coords_full.size(0):
                                                                                p2s = p2s.detach().cpu().long().view(-1)
                                                                                if int(gt_masks_all.size(1)) > int(p2s.max().item()):
                                                                                        gt_masks_all = gt_masks_all[:, p2s]
                                                        except Exception:
                                                                pass

                                                        def _map_gt_id_to_mask_idx(gid):
                                                                gid_int = _parse_int(gid)
                                                                if gid_int is None:
                                                                        return None
                                                                if 0 <= int(gid_int) < int(gt_masks_all.shape[0]):
                                                                        return int(gid_int)
                                                                if isinstance(inst_mapping, dict):
                                                                        try:
                                                                                mapped = inst_mapping.get(int(gid_int), None)
                                                                                mapped_int = _parse_int(mapped)
                                                                                if mapped_int is not None and 0 <= int(mapped_int) < int(gt_masks_all.shape[0]):
                                                                                        return int(mapped_int)
                                                                        except Exception:
                                                                                pass
                                                                # Fallback: treat gid as target-space remapped index -> original id -> full mapping.
                                                                try:
                                                                        orig_ids = target[bid].get("orig_instance_ids", None)
                                                                        if isinstance(orig_ids, list) and 0 <= int(gid_int) < len(orig_ids) and isinstance(inst_mapping, dict):
                                                                                oid = _parse_int(orig_ids[int(gid_int)])
                                                                                if oid is not None:
                                                                                        mapped = inst_mapping.get(int(oid), None)
                                                                                        mapped_int = _parse_int(mapped)
                                                                                        if mapped_int is not None and 0 <= int(mapped_int) < int(gt_masks_all.shape[0]):
                                                                                                return int(mapped_int)
                                                                except Exception:
                                                                        pass
                                                                return None

                                                        for it in items:
                                                                lang_type = it.get("lang_type", "") or ""
                                                                prefix = str(lang_type).split(":")[0] if isinstance(lang_type, str) else ""
                                                                if prefix not in {"scanrefer", "m3dref"}:
                                                                        skip_all["unknown_prefix"] += 1
                                                                        continue

                                                                pred_q = _parse_int(it.get("pred_target_q", None))
                                                                if pred_q is None or int(pred_q) not in q_to_col:
                                                                        skip_all["pred_q_invalid"] += 1
                                                                        continue

                                                                gt_ids_raw = it.get("gt_inst_ids", None)
                                                                if not isinstance(gt_ids_raw, list) or len(gt_ids_raw) == 0:
                                                                        skip_all["missing_gt_inst_ids"] += 1
                                                                        continue
                                                                gt_ids = []
                                                                for x in gt_ids_raw:
                                                                        if isinstance(x, (list, tuple)) and len(x) == 1:
                                                                                x = x[0]
                                                                        mid = _map_gt_id_to_mask_idx(x)
                                                                        if mid is not None:
                                                                                gt_ids.append(int(mid))
                                                                gt_ids = sorted(set([int(x) for x in gt_ids if 0 <= int(x) < int(gt_masks_all.shape[0])]))
                                                                if not gt_ids:
                                                                        skip_all["gt_ids_empty_after_map"] += 1
                                                                        if dbg_geom and int(getattr(self, "global_rank", 0)) == 0 and dbg_printed < dbg_max:
                                                                                dbg_printed += 1
                                                                                try:
                                                                                        logger.warning(
                                                                                                f"[SSR3DLLM][geom_grd][debug] gt_ids_empty_after_map "
                                                                                                f"scene={file_names[bid] if isinstance(file_names,(list,tuple)) and bid < len(file_names) else ''} "
                                                                                                f"prefix={prefix} gt_ids_raw={gt_ids_raw} gt_masks={tuple(gt_masks_all.shape)} inst_mapping={'Y' if isinstance(inst_mapping,dict) else 'N'}"
                                                                                        )
                                                                                except Exception:
                                                                                        pass
                                                                        continue

                                                                pred_mask = pred_full[:, q_to_col[int(pred_q)]].to(dtype=torch.bool)
                                                                if (
                                                                        dbg_empty
                                                                        and int(getattr(self, "global_rank", 0)) == 0
                                                                        and dbg_empty_printed < dbg_empty_max
                                                                        and int(pred_mask.sum().item()) == 0
                                                                ):
                                                                        dbg_empty_printed += 1
                                                                        try:
                                                                                col = int(q_to_col[int(pred_q)])
                                                                                seg_logits = mask_logits[:, int(pred_q)].detach().float().cpu()
                                                                                seg_max = float(seg_logits.max().item()) if seg_logits.numel() else float("nan")
                                                                                seg_min = float(seg_logits.min().item()) if seg_logits.numel() else float("nan")
                                                                                seg_mean = float(seg_logits.mean().item()) if seg_logits.numel() else float("nan")
                                                                                seg_pos = int((seg_logits > 0.0).sum().item()) if seg_logits.numel() else 0

                                                                                sp = mask_pts_raw[:, col].detach().float().cpu()
                                                                                sp_max = float(sp.max().item()) if sp.numel() else float("nan")
                                                                                sp_min = float(sp.min().item()) if sp.numel() else float("nan")
                                                                                sp_mean = float(sp.mean().item()) if sp.numel() else float("nan")
                                                                                sp_pos = int((sp > 0.0).sum().item()) if sp.numel() else 0

                                                                                full = pred_full_heat[:, col].detach().float().cpu()
                                                                                full_max = float(full.max().item()) if full.numel() else float("nan")
                                                                                full_mean = float(full.mean().item()) if full.numel() else float("nan")
                                                                                full_pos0 = int((full > 0.0).sum().item()) if full.numel() else 0
                                                                                full_pos05 = int((full > 0.5).sum().item()) if full.numel() else 0

                                                                                if (seg_pos == 0) and (sp_pos == 0):
                                                                                        diag = "MODEL_ALL_NONPOS"
                                                                                elif (sp_pos > 0) and (full_pos05 == 0) and (full_pos0 > 0):
                                                                                        diag = "THRESH_TOO_STRICT_OR_SMOOTHED"
                                                                                elif (sp_pos > 0) and (full_pos0 == 0):
                                                                                        diag = "FULL_MASK_MAPPING_ZERO"
                                                                                elif (seg_pos > 0) and (sp_pos == 0):
                                                                                        diag = "POINT2SEGMENT_INDEX_MISMATCH"
                                                                                else:
                                                                                        diag = "OTHER"

                                                                                _mode = it.get("geom_chain_mode", "")
                                                                                _text = it.get("geom_text", "")
                                                                                if not isinstance(_mode, str):
                                                                                        _mode = str(_mode)
                                                                                if not isinstance(_text, str):
                                                                                        _text = str(_text)
                                                                                _text = _text.replace("\n", " ").strip()
                                                                                if len(_text) > 120:
                                                                                        _text = _text[:120] + "..."
                                                                                logger.warning(
                                                                                        f"[SSR3DLLM][geom_grd][empty_mask] DIAG={diag} "
                                                                                        f"scene={file_names[bid] if isinstance(file_names,(list,tuple)) and bid < len(file_names) else ''} "
                                                                                        f"prefix={prefix} mode={_mode} pred_q={int(pred_q)} "
                                                                                        f"seg_logits[max={seg_max:.3f} min={seg_min:.3f} mean={seg_mean:.3f} pos={seg_pos}] "
                                                                                        f"sparse_pts[max={sp_max:.3f} min={sp_min:.3f} mean={sp_mean:.3f} pos={sp_pos}] "
                                                                                        f"full_heat[max={full_max:.3f} mean={full_mean:.3f} pos0={full_pos0} pos05={full_pos05}] "
                                                                                        f"text='{_text}'"
                                                                                )
                                                                        except Exception:
                                                                                pass
                                                                gt_mask_union = gt_masks_all[gt_ids].any(dim=0)
                                                                if pred_mask.numel() != gt_mask_union.numel():
                                                                        skip_all["mask_numel_mismatch"] += 1
                                                                        continue
                                                                inter = (pred_mask & gt_mask_union).sum().item()
                                                                outer = (pred_mask | gt_mask_union).sum().item()
                                                                if outer <= 0:
                                                                        skip_all["outer_le_0"] += 1
                                                                        continue
                                                                mask_iou = float(inter) / float(outer + 1e-8)

                                                                bbox_iou = 0.0
                                                                if pred_mask.sum().item() > 0 and gt_mask_union.sum().item() > 0:
                                                                        pred_pts = coords_full[pred_mask]
                                                                        gt_pts = coords_full[gt_mask_union]
                                                                        if pred_pts.numel() > 0 and gt_pts.numel() > 0:
                                                                                pred_box = torch.stack(
                                                                                        [pred_pts.min(dim=0).values, pred_pts.max(dim=0).values],
                                                                                        dim=0,
                                                                                ).unsqueeze(0)
                                                                                gt_box = torch.stack(
                                                                                        [gt_pts.min(dim=0).values, gt_pts.max(dim=0).values],
                                                                                        dim=0,
                                                                                ).unsqueeze(0)
                                                                                bbox_iou = float(get_batch_aabb_pair_ious(pred_box, gt_box)[0].item())

                                                                if dbg_geom and int(getattr(self, "global_rank", 0)) == 0 and dbg_printed < dbg_max:
                                                                        dbg_printed += 1
                                                                        try:
                                                                                _mode = it.get("geom_chain_mode", "")
                                                                                _text = it.get("geom_text", "")
                                                                                if not isinstance(_mode, str):
                                                                                        _mode = str(_mode)
                                                                                if not isinstance(_text, str):
                                                                                        _text = str(_text)
                                                                                _text = _text.replace("\n", " ").strip()
                                                                                if len(_text) > 120:
                                                                                        _text = _text[:120] + "..."
                                                                                logger.warning(
                                                                                        f"[SSR3DLLM][geom_grd][debug] scene={file_names[bid] if isinstance(file_names,(list,tuple)) and bid < len(file_names) else ''} "
                                                                                        f"prefix={prefix} mode={_mode} pred_q={int(pred_q)} gt_ids={gt_ids} "
                                                                                        f"pred_pts={int(pred_mask.sum().item())} gt_pts={int(gt_mask_union.sum().item())} "
                                                                                        f"mask_iou={mask_iou:.4f} bbox_iou={bbox_iou:.4f} text='{_text}'"
                                                                                )
                                                                        except Exception:
                                                                                pass

                                                                if prefix == "scanrefer":
                                                                        if mask_iou >= 0.25:
                                                                                scan_mask25 += 1
                                                                        if mask_iou >= 0.50:
                                                                                scan_mask50 += 1
                                                                        if bbox_iou >= 0.25:
                                                                                scan_bbox25 += 1
                                                                        if bbox_iou >= 0.50:
                                                                                scan_bbox50 += 1
                                                                elif prefix == "m3dref":
                                                                        if mask_iou >= 0.25:
                                                                                m3d_mask25 += 1
                                                                        if mask_iou >= 0.50:
                                                                                m3d_mask50 += 1
                                                                        if bbox_iou >= 0.25:
                                                                                m3d_bbox25 += 1
                                                                        if bbox_iou >= 0.50:
                                                                                m3d_bbox50 += 1

                                                                        max_iou = 0.0
                                                                        if pred_mask.sum().item() > 0:
                                                                                pred_pts = coords_full[pred_mask]
                                                                                if pred_pts.numel() > 0:
                                                                                        pred_box = torch.stack(
                                                                                                [pred_pts.min(dim=0).values, pred_pts.max(dim=0).values],
                                                                                                dim=0,
                                                                                        ).unsqueeze(0)
                                                                                        for gid in gt_ids:
                                                                                                gt_m = gt_masks_all[gid]
                                                                                                if gt_m.sum().item() == 0:
                                                                                                        continue
                                                                                                gt_pts = coords_full[gt_m]
                                                                                                if gt_pts.numel() == 0:
                                                                                                        continue
                                                                                                gt_box = torch.stack(
                                                                                                        [gt_pts.min(dim=0).values, gt_pts.max(dim=0).values],
                                                                                                        dim=0,
                                                                                                ).unsqueeze(0)
                                                                                                iou = float(get_batch_aabb_pair_ious(pred_box, gt_box)[0].item())
                                                                                                if iou > max_iou:
                                                                                                        max_iou = iou
                                                                        denom = float(len(gt_ids) + 1)
                                                                        m3d_bbox_f1_25_sum += (2.0 / denom) if max_iou >= 0.25 else 0.0
                                                                        m3d_bbox_f1_50_sum += (2.0 / denom) if max_iou >= 0.50 else 0.0

                                        if dbg_geom and int(getattr(self, "global_rank", 0)) == 0:
                                                try:
                                                        logger.warning(
                                                                f"[SSR3DLLM][geom_grd][debug] skip_all={dict(skip_all)} "
                                                                f"scan_total={int(scan_total)} m3d_total={int(m3d_total)} "
                                                                f"scan_expected={int(scan_total_expected)} m3d_expected={int(m3d_total_expected)}"
                                                        )
                                                except Exception:
                                                        pass

                                        # Ensure denominators are non-zero when the batch contains ScanRefer/M3DRef
                                        # items but we failed to compute IoU (e.g. no valid prediction/mask).
                                        try:
                                                scan_total = max(int(scan_total), int(scan_total_expected))
                                                m3d_total = max(int(m3d_total), int(m3d_total_expected))
                                        except Exception:
                                                pass

                                        losses["ssr3dllm_num_scanrefer_total"] = torch.tensor(
                                                float(scan_total), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_scanrefer_mask25_hit"] = torch.tensor(
                                                float(scan_mask25), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_scanrefer_mask50_hit"] = torch.tensor(
                                                float(scan_mask50), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_scanrefer_bbox25_hit"] = torch.tensor(
                                                float(scan_bbox25), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_scanrefer_bbox50_hit"] = torch.tensor(
                                                float(scan_bbox50), device=self.device, dtype=torch.float32
                                        )

                                        losses["ssr3dllm_num_m3dref_total"] = torch.tensor(
                                                float(m3d_total), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_m3dref_mask25_hit"] = torch.tensor(
                                                float(m3d_mask25), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_m3dref_mask50_hit"] = torch.tensor(
                                                float(m3d_mask50), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_m3dref_bbox25_hit"] = torch.tensor(
                                                float(m3d_bbox25), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_num_m3dref_bbox50_hit"] = torch.tensor(
                                                float(m3d_bbox50), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_sum_m3dref_bbox_f1_25"] = torch.tensor(
                                                float(m3d_bbox_f1_25_sum), device=self.device, dtype=torch.float32
                                        )
                                        losses["ssr3dllm_sum_m3dref_bbox_f1_50"] = torch.tensor(
                                                float(m3d_bbox_f1_50_sum), device=self.device, dtype=torch.float32
                                        )

                if self.llama_config.enable_llm and not getattr(self, "ssr3dllm_geom_only", False):
                        assert len(target) == 1
                        pred_inst_masks = (
                                        output["pred_masks"][0][target[0]["point2segment"].cpu()] > 0.).float().cpu().clone()
                        pred_inst_masks = self.get_full_res_mask(
                                pred_inst_masks, inverse_maps[0], target_full[0]['point2segment'])
                        pred_inst_masks = [pred_inst_masks]

                        # self.preds[file_names[bid]] = {
                        #     "pred_masks": (all_pred_masks[bid]).astype(bool), # pred_inst_masks
                        #     "pred_scores": all_pred_scores[bid], # similarity 100(queries) x 200 (classes)
                        #     'gt_ious': all_gt_ious[bid] if len(all_gt_ious) > 0 else (np.zeros((0,), dtype=float), np.zeros((0,), dtype=str), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)),
                        # }

                        try:
                                map_target_to_query, valid_target = batch_map_target_to_query[0]
                                # map_target_to_query = np.zeros((target_full[0]['labels'].shape[0]), dtype=int) - 1
                                # map_target_to_query[self.criterion.indices[0][1]] = self.criterion.indices[0][0]
                                inter = pred_inst_masks[0].to(
                                        bool)[:, map_target_to_query[valid_target]].T & target_full[0]['masks']
                                outer = pred_inst_masks[0].to(
                                        bool)[:, map_target_to_query[valid_target]].T | target_full[0]['masks']
                                instance_iou = inter.sum(1) / outer.sum(1)
                                out_json, score, bbox_score_25, bbox_score_50, mask_score_25, mask_score_50, m3dref_bbox_result = eval_llm_iou_score(out_json, {"pred_inst_masks"     : pred_inst_masks,
                                                                                                                                                                "target_full"         : target_full,
                                                                                                                                                                "batch_gt_inst_ids"   : batch_gt_inst_ids,
                                                                                                                                                                "original_coordinates": original_coordinates
                                                                                                                                                                })
                                with open(f"{self.llama_config.save_path}/{file_names[0]}.json", 'w') as json_file:
                                        json.dump({"prediction"   : out_json,
                                                   "score"        : score,
                                                   "bbox_score_25": bbox_score_25,
                                                   "bbox_score_50": bbox_score_50,
                                                   "mask_score_25": mask_score_25,
                                                   "mask_score_50": mask_score_50,
                                                   "seg_score"    : instance_iou.tolist()},
                                                  json_file, indent=4)
                                if m3dref_bbox_result:
                                        with open(f"{self.llama_config.save_path}/m3drefer/{file_names[0]}.pkl", 'wb') as f:
                                                pickle.dump(m3dref_bbox_result, f)
                        except Exception as e:
                                logger.error(f"Failed to save prediction for {file_names[0]}: {e}")
                                raise

                if self.config.data.test_mode != "test":
                        return {
                                f"val_{k}": v.detach().cpu().item() for k, v in losses.items()
                        }
                else:
                        return 0.0

        def test_step(self, batch, batch_idx):
                return self.eval_step(batch, batch_idx)

        def get_full_res_mask(
                        self, mask, inverse_map, point2segment_full, is_heatmap=False
        ):
                mask = mask.detach().cpu()[inverse_map]  # full res

                if self.eval_on_segments and is_heatmap == False:
                        mask = scatter_mean(
                                mask, point2segment_full, dim=0
                        )  # full res segments
                        mask = (mask > 0.5).float()
                        mask = mask.detach().cpu()[
                                point2segment_full.cpu()
                        ]  # full res points

                return mask

        def get_mask_and_scores(
                        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
        ):
                if device is None:
                        device = self.device
                labels = (
                        torch.arange(num_classes, device=device)
                        .unsqueeze(0)
                        .repeat(num_queries, 1)
                        .flatten(0, 1)
                )

                if self.config.general.topk_per_image != -1:
                        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                                self.config.general.topk_per_image, sorted=True
                        )
                else:
                        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                                num_queries, sorted=True
                        )

                labels_per_query = labels[topk_indices]
                topk_indices = torch.div(topk_indices, torch.tensor(
                        num_classes), rounding_mode='floor')  # class share the same mask
                mask_pred = mask_pred[:, topk_indices]

                result_pred_mask = (mask_pred > 0).float()
                heatmap = mask_pred.float().sigmoid()

                mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
                                result_pred_mask.sum(0) + 1e-6
                )
                score = scores_per_query * mask_scores_per_image
                # final query score = (scores of query) x sum(mask_pred.sigmoid() * (mask_pred > 0)) / sum(mask_pred > 0)
                # final query mask is shared across mask
                classes = labels_per_query

                return score, result_pred_mask, classes, heatmap

        def eval_instance_step(
                        self,
                        output,
                        target_low_res,
                        target_full_res,
                        inverse_maps,
                        file_names,
                        full_res_coords,
                        original_colors,
                        original_normals,
                        raw_coords,
                        idx,
                        extra_lang=None,
        ):
                # Some training/eval modes (e.g. step-token SFT / rel3d-only) intentionally
                # disable detection/segmentation outputs. Guard to keep validation sane.
                if output.get("pred_logits", None) is None or output.get("pred_masks", None) is None:
                        return
                label_offset = self.validation_dataset.label_offset
                if 'aux_outputs' in output:
                        prediction = output["aux_outputs"]
                else:
                        print('No aux outputs are found.')
                        prediction = []
                prediction.append(
                        {
                                "pred_logits": output["pred_logits"],
                                "pred_masks" : output["pred_masks"],
                        }
                )

                assert self.config.model.num_classes - \
                       1 == self.config.data.num_labels - label_offset
                pred_lang_logits = []
                if self.config.model.language_model and not self.config.model.softmax_mode:
                        pred_logits = []
                        for pred_logit in prediction[self.decoder_id]["pred_logits"]:
                                if not self.config.data.sample_class_labels or not self.training:
                                        pred_logits.append(
                                                pred_logit[:, :self.config.model.num_classes - 1].sigmoid())
                                        pred_lang_logits.append(
                                                pred_logit[:, self.config.model.num_classes - 1:].sigmoid())
                                else:
                                        pred_lang_logits.append(pred_logit.sigmoid())

                        if not self.config.data.sample_class_labels or not self.training:
                                prediction[self.decoder_id][
                                        "pred_logits"
                                ] = torch.stack(pred_logits, dim=0)
                        prediction[self.decoder_id][
                                "pred_lang_logits"
                        ] = pred_lang_logits
                elif not self.config.model.language_model and not self.config.model.softmax_mode:
                        if isinstance(prediction[self.decoder_id]["pred_logits"], list):
                                prediction[self.decoder_id]["pred_logits"] = torch.stack(
                                        prediction[self.decoder_id]["pred_logits"], 0)
                        prediction[self.decoder_id][
                                "pred_logits"
                        ] = prediction[self.decoder_id]["pred_logits"].sigmoid()
                else:
                        assert not self.config.data.sample_class_labels
                        if isinstance(prediction[self.decoder_id]["pred_logits"], list):
                                prediction[self.decoder_id]["pred_logits"] = torch.stack(
                                        prediction[self.decoder_id]["pred_logits"], 0)
                        prediction[self.decoder_id][
                                "pred_logits"
                        ] = torch.functional.F.softmax(
                                prediction[self.decoder_id]["pred_logits"], dim=-1
                        )[
                                ..., :-1
                        ]

                all_pred_classes = list()
                all_pred_masks = list()
                all_pred_scores = list()
                all_heatmaps = list()

                all_extra_query_texts = list()
                all_pred_extra_masks = list()
                all_pred_extra_masks_instance_coordscore = list()
                all_gt_extra_masks = list()
                all_gt_ious = []
                all_raw_pred_instance_masks = list()
                all_iou_25_f1_score = []
                all_iou_50_f1_score = []

                offset_coords_idx = 0
                for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                        if self.model.train_on_segments:
                                masks = (
                                        prediction[self.decoder_id]["pred_masks"][bid]
                                        .detach()
                                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                                )  # map back to raw points
                        else:
                                masks = (
                                        prediction[self.decoder_id]["pred_masks"][bid]
                                        .detach()
                                        .cpu()
                                )
                        if not self.config.data.sample_class_labels or not self.training:
                                if self.config.general.use_dbscan:
                                        new_preds = {
                                                "pred_masks"      : list(),
                                                "pred_logits"     : list(),
                                                "pred_lang_logits": list(),
                                        }

                                        curr_coords_idx = masks.shape[0]
                                        curr_coords = raw_coords[
                                                offset_coords_idx: curr_coords_idx + offset_coords_idx
                                        ]
                                        offset_coords_idx += curr_coords_idx

                                        # for each query in num_queries
                                        for curr_query in range(masks.shape[1]):
                                                # [num_points, query_i]
                                                curr_masks = masks[:, curr_query] > 0

                                                if curr_coords[curr_masks].shape[0] > 0:
                                                        clusters = (
                                                                DBSCAN(
                                                                        eps=self.config.general.dbscan_eps,
                                                                        min_samples=self.config.general.dbscan_min_points,
                                                                        n_jobs=-1,
                                                                )
                                                                .fit(curr_coords[curr_masks])
                                                                .labels_
                                                        )

                                                        new_mask = torch.zeros(curr_masks.shape, dtype=int)
                                                        new_mask[curr_masks] = (
                                                                        torch.from_numpy(clusters) + 1
                                                        )

                                                        for cluster_id in np.unique(clusters):
                                                                original_pred_masks = masks[:, curr_query]
                                                                if cluster_id != -1:
                                                                        new_preds["pred_masks"].append(  # current mask divided into cluster
                                                                                original_pred_masks
                                                                                * (new_mask == cluster_id + 1)
                                                                        )
                                                                        new_preds["pred_logits"].append(  # copy score
                                                                                prediction[self.decoder_id][
                                                                                        "pred_logits"
                                                                                ][bid, curr_query]
                                                                        )
                                                                        if len(pred_lang_logits) > 0:
                                                                                new_preds["pred_lang_logits"].append(  # copy score
                                                                                        prediction[self.decoder_id][
                                                                                                "pred_lang_logits"
                                                                                        ][bid][curr_query]
                                                                                )

                                        if len(pred_lang_logits) > 0:
                                                new_masks = torch.stack(new_preds["pred_masks"], dim=1)

                                                # for computing (num_points, num_query)
                                                raw_masks = (new_masks > 0.).float().cpu().clone()
                                                raw_heatmap = new_masks.float().cpu().clone()
                                                raw_masks = self.get_full_res_mask(
                                                        raw_masks, inverse_maps[bid], target_full_res[bid]['point2segment'])
                                                raw_heatmap = self.get_full_res_mask(
                                                        raw_heatmap, inverse_maps[bid], target_full_res[bid]['point2segment'], is_heatmap=True)
                                                if len(new_preds["pred_lang_logits"]) > 0:
                                                        pred_lang_logits = torch.stack(
                                                                new_preds["pred_lang_logits"])
                                                else:
                                                        pred_lang_logits = torch.zeros(
                                                                (0, prediction[self.decoder_id]["pred_lang_logits"][bid].shape[1]), dtype=torch.float32, device='cuda')
                                                prediction[self.decoder_id]['pred_lang_logits'][bid] = pred_lang_logits
                                        else:
                                                raw_masks = None

                                        scores, masks, classes, heatmap = self.get_mask_and_scores(
                                                torch.stack(new_preds["pred_logits"]).cpu(),
                                                torch.stack(new_preds["pred_masks"]).T,
                                                len(new_preds["pred_logits"]),
                                                self.model.num_classes - 1,
                                        )
                                else:
                                        # # for computing (num_points, num_query)
                                        raw_masks = (masks > 0.).float().cpu().clone()
                                        raw_heatmap = masks.float().cpu().clone()
                                        raw_masks = self.get_full_res_mask(
                                                raw_masks, inverse_maps[bid], target_full_res[bid]['point2segment'])
                                        raw_heatmap = self.get_full_res_mask(
                                                raw_heatmap, inverse_maps[bid], target_full_res[bid]['point2segment'], is_heatmap=True)

                                        scores, masks, classes, heatmap = self.get_mask_and_scores(
                                                prediction[self.decoder_id]["pred_logits"][bid]
                                                .detach()
                                                .cpu(),
                                                masks,
                                                prediction[self.decoder_id]["pred_logits"][bid].shape[
                                                        0
                                                ],
                                                self.model.num_classes - 1,
                                        )

                        all_raw_pred_instance_masks.append(raw_masks)

                        if not self.config.data.sample_class_labels or not self.training:
                                masks = self.get_full_res_mask(
                                        masks,
                                        inverse_maps[bid],
                                        target_full_res[bid]["point2segment"],
                                )

                                heatmap = self.get_full_res_mask(
                                        heatmap,
                                        inverse_maps[bid],
                                        target_full_res[bid]["point2segment"],
                                        is_heatmap=True,
                                )

                        if not self.config.data.sample_class_labels or not self.training:
                                masks = masks.numpy()
                                heatmap = heatmap.numpy()

                                sort_scores = scores.sort(descending=True)
                                sort_scores_index = sort_scores.indices.cpu().numpy()
                                sort_scores_values = sort_scores.values.cpu().numpy()
                                sort_classes = classes[sort_scores_index]

                                sorted_masks = masks[:, sort_scores_index]
                                sorted_heatmap = heatmap[:, sort_scores_index]

                        if not self.config.data.sample_class_labels or not self.training:
                                all_pred_classes.append(sort_classes)
                                all_pred_masks.append(sorted_masks)
                                all_pred_scores.append(sort_scores_values)
                                all_heatmaps.append(sorted_heatmap)

                        if len(extra_lang) > 0 and self.config.model.language_model:
                                gt_ious, gt_extra_masks, extra_query_texts, pred_extra_masks, pred_extra_masks_instance_coordscore = eval_seg_model(bid=bid,
                                                                                                                                                    config=self.config,
                                                                                                                                                    extra_lang=extra_lang,
                                                                                                                                                    full_res_coords=full_res_coords,
                                                                                                                                                    raw_masks=raw_masks,
                                                                                                                                                    raw_heatmap=raw_heatmap,
                                                                                                                                                    target_full_res=target_full_res,
                                                                                                                                                    pred_lang_logits=prediction[self.decoder_id][
                                                                                                                                                            'pred_lang_logits'][bid],
                                                                                                                                                    training=self.training,
                                                                                                                                                    )
                                all_gt_ious.append(gt_ious)
                                all_gt_extra_masks.append(gt_extra_masks)
                                all_extra_query_texts.append(extra_query_texts)
                                all_pred_extra_masks.append(pred_extra_masks)
                                all_pred_extra_masks_instance_coordscore.append(
                                        pred_extra_masks_instance_coordscore)

                if self.validation_dataset.dataset_name == "scannet200":
                        # remap gt labels
                        # this code originally is out of the bid loop, which seems to be a bug.
                        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                                if self.config.data.test_mode != "test":
                                        target_full_res[bid]["labels"][
                                                target_full_res[bid]["labels"] == 0
                                                ] = -1

                for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                        if (
                                        self.config.data.test_mode != "test"
                                        and len(target_full_res) != 0
                        ):
                                target_full_res[bid][
                                        "labels"
                                ] = self.validation_dataset._remap_model_output(
                                        target_full_res[bid]["labels"].cpu() + label_offset
                                )

                                # GT BOX
                                bbox_data = []
                                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                                        if target_full_res[bid]["labels"][obj_id].item() == 255:
                                                continue

                                        obj_coords = full_res_coords[bid][
                                                target_full_res[bid]["masks"][obj_id, :]
                                                .cpu()
                                                .detach()
                                                .numpy()
                                                .astype(bool),
                                                :,
                                        ]
                                        if obj_coords.shape[0] > 0:
                                                obj_center = obj_coords.mean(axis=0)
                                                obj_axis_length = obj_coords.max(
                                                        axis=0
                                                ) - obj_coords.min(axis=0)

                                                bbox = np.concatenate((obj_center, obj_axis_length))
                                                bbox_data.append(
                                                        (
                                                                target_full_res[bid]["labels"][obj_id].item(),
                                                                bbox,
                                                        )
                                                )

                                self.bbox_gt[file_names[bid]] = bbox_data

                if not self.config.data.sample_class_labels or not self.training:
                        if self.validation_dataset.dataset_name == "scannet200":
                                all_pred_classes[bid][all_pred_classes[bid] == 0] = -1

                        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                                all_pred_classes[
                                        bid
                                ] = self.validation_dataset._remap_model_output(
                                        all_pred_classes[bid].cpu() + label_offset
                                )

                                if (
                                                self.config.data.test_mode != "test"
                                                and len(target_full_res) != 0
                                ):
                                        bbox_data = []
                                        for query_id in range(
                                                        all_pred_masks[bid].shape[1]
                                        ):  # self.model.num_queries
                                                obj_coords = full_res_coords[bid][
                                                        all_pred_masks[bid][:, query_id].astype(bool), :
                                                ]
                                                if obj_coords.shape[0] > 0:
                                                        obj_center = obj_coords.mean(axis=0)
                                                        obj_axis_length = obj_coords.max(
                                                                axis=0
                                                        ) - obj_coords.min(axis=0)

                                                        bbox = np.concatenate(
                                                                (obj_center, obj_axis_length))

                                                        bbox_data.append(
                                                                (
                                                                        all_pred_classes[bid][query_id].item(),
                                                                        bbox,
                                                                        all_pred_scores[bid][query_id],
                                                                )
                                                        )
                                        self.bbox_preds[file_names[bid]] = bbox_data

                                self.preds[file_names[bid]] = {
                                        "pred_masks"  : (all_pred_masks[bid]).astype(bool),
                                        "pred_scores" : all_pred_scores[bid],
                                        "pred_classes": all_pred_classes[bid],
                                        'gt_ious'     : all_gt_ious[bid] if len(all_gt_ious) > 0 else (np.zeros((0,), dtype=float), np.zeros((0,), dtype=str), np.zeros((0,), dtype=float),
                                                                                                       np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)),
                                }
                                if self.config.general.export:
                                        self.export(
                                                self.preds[file_names[bid]]["pred_masks"],
                                                self.preds[file_names[bid]]["pred_scores"],
                                                self.preds[file_names[bid]]["pred_classes"],
                                                file_names[bid],
                                                self.decoder_id,
                                        )

                                if 'gt_ious' in self.preds[file_names[bid]]:
                                        gt_ious = self.preds[file_names[bid]]['gt_ious'][0]
                                        # if len(gt_ious) > 0:
                                        #     print(file_names[bid], f'iou_0.25: {(gt_ious > 0.25).sum() / (gt_ious.shape[0]+1e-8):.3f}', f'iou_0.5: {(gt_ious > 0.5).sum() / (gt_ious.shape[0]+1e-8):.3f}')
                                        multi_iou_25_f1_score = self.preds[file_names[bid]
                                        ]['gt_ious'][3]
                                        multi_iou_50_f1_score = self.preds[file_names[bid]
                                        ]['gt_ious'][4]
                                        assert len(multi_iou_25_f1_score) == len(
                                                multi_iou_50_f1_score) == len(gt_ious)
                                        # if len(multi_iou_25_f1_score) > 0:
                                        #     print(file_names[bid], f'multi_iou_0.25: {np.mean(multi_iou_25_f1_score):.3f}', f'multi_iou_0.5: {np.mean(multi_iou_50_f1_score):.3f}')

                                if self.config.general.gpus > 1:
                                        dump_bbox = [self.bbox_preds[file_names[bid]],
                                                     self.bbox_gt[file_names[bid]]]
                                        # type: ignore
                                        with open(osp.join(self.tmpdir, file_names[bid] + "_bbox.pkl"), 'wb') as f:
                                                pickle.dump(dump_bbox, f, protocol=2)

                                        np.savez_compressed(osp.join(self.tmpdir, file_names[bid] + '_preds.npz'),
                                                            pred_masks=self.preds[file_names[bid]]['pred_masks'].astype(
                                                                    bool),
                                                            pred_scores=self.preds[file_names[bid]
                                                            ]['pred_scores'],
                                                            pred_classes=self.preds[file_names[bid]
                                                            ]['pred_classes'],
                                                            gt_ious=self.preds[file_names[bid]
                                                            ]['gt_ious'] if 'gt_ious' in self.preds[file_names[bid]] else None,
                                                            )
                                else:
                                        # Optional: stream per-scene predictions to disk even on single-GPU to
                                        # avoid OOM when `self.preds` would otherwise hold all scans in RAM.
                                        try:
                                                _stream = str(os.environ.get("SSR3DLLM_STREAM_INSTANCE_EVAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
                                        except Exception:
                                                _stream = False
                                        if _stream:
                                                dump_bbox = [self.bbox_preds.get(file_names[bid], []),
                                                             self.bbox_gt.get(file_names[bid], [])]
                                                with open(osp.join(self.tmpdir, file_names[bid] + "_bbox.pkl"), 'wb') as f:
                                                        pickle.dump(dump_bbox, f, protocol=2)
                                                np.savez_compressed(
                                                        osp.join(self.tmpdir, file_names[bid] + '_preds.npz'),
                                                        pred_masks=self.preds[file_names[bid]]['pred_masks'].astype(bool),
                                                        pred_scores=self.preds[file_names[bid]]['pred_scores'],
                                                        pred_classes=self.preds[file_names[bid]]['pred_classes'],
                                                        gt_ious=self.preds[file_names[bid]].get('gt_ious', None),
                                                )

                for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                        if self.config.general.save_visualizations:
                                self.save_visualizations(
                                        target_full_res[bid],
                                        full_res_coords[bid],
                                        [self.preds[file_names[bid]]["pred_masks"]
                                         ] if not self.config.data.sample_class_labels or not self.training else None,
                                        [self.preds[file_names[bid]]["pred_classes"]
                                         ] if not self.config.data.sample_class_labels or not self.training else None,
                                        file_names[bid],
                                        original_colors[bid],
                                        original_normals[bid],
                                        [self.preds[file_names[bid]]["pred_scores"]
                                         ] if not self.config.data.sample_class_labels or not self.training else None,
                                        point_size=self.config.general.visualization_point_size,
                                        query_text=all_extra_query_texts[bid] if len(
                                                extra_lang) > 0 else None,
                                        query_mask=all_pred_extra_masks[bid] if len(
                                                extra_lang) > 0 else None,
                                        gt_query_mask=all_gt_extra_masks[bid] if len(
                                                extra_lang) > 0 else None,
                                        query_mask_instance_coordscore=all_pred_extra_masks_instance_coordscore[bid] if len(
                                                extra_lang) > 0 else None,
                                )

                # If we streamed predictions to disk (single-GPU), drop in-memory copies to keep RAM bounded.
                try:
                        _stream = str(os.environ.get("SSR3DLLM_STREAM_INSTANCE_EVAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                        _stream = False
                if _stream and (not self.config.general.save_visualizations):
                        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
                                try:
                                        self.preds.pop(file_names[bid], None)
                                        self.bbox_preds.pop(file_names[bid], None)
                                        self.bbox_gt.pop(file_names[bid], None)
                                except Exception:
                                        pass

                return all_raw_pred_instance_masks

        def eval_instance_epoch_end(self, all_preds, all_bbox_preds, all_bbox_gt, pred_npz_files=None):
                log_prefix = f"val"
                ap_results = {}

                # Fast sanity eval mode:
                # - Skip expensive instance AP (eval_det + official instance evaluation)
                # - Keep grounding metrics (ScanRefer/M3DRef) via collect_grounding_score
                #
                # This is intended for per-epoch "health checks" during finetune to quickly
                # catch collapses (e.g., 0.00X grounding IoU), not for paper numbers.
                #
                # Env:
                #   - SSR3DLLM_SKIP_INSTANCE_AP_EVAL=1
                #   - SSR3DLLM_FAST_EVAL=1 (alias)
                try:
                        _skip_ap = str(os.environ.get("SSR3DLLM_SKIP_INSTANCE_AP_EVAL", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                except Exception:
                        _skip_ap = False
                try:
                        _skip_ap = _skip_ap or (str(os.environ.get("SSR3DLLM_FAST_EVAL", "0")).strip().lower() in {"1", "true", "yes", "y", "on"})
                except Exception:
                        pass
                if _skip_ap:
                        ap_results = collect_grounding_score(all_preds, ap_results, log_prefix)
                        try:
                                with open(self.tmpdir + '/ap_results.pkl', 'wb') as f:
                                        pickle.dump(ap_results, f, protocol=2)
                        except Exception:
                                pass
                        return ap_results

                head_results, tail_results, common_results = [], [], []

                box_ap_50 = eval_det(
                        all_bbox_preds, all_bbox_gt, ovthresh=0.5, use_07_metric=False
                )
                box_ap_25 = eval_det(
                        all_bbox_preds, all_bbox_gt, ovthresh=0.25, use_07_metric=False
                )
                mean_box_ap_25 = sum([v for k, v in box_ap_25[-1].items()]) / len(
                        box_ap_25[-1].keys()
                )
                mean_box_ap_50 = sum([v for k, v in box_ap_50[-1].items()]) / len(
                        box_ap_50[-1].keys()
                )

                ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
                ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50

                for class_id in box_ap_50[-1].keys():
                        try:
                                class_name = self.labels_info[class_id]["name"]
                                ap_results[f"{log_prefix}_{class_name}_val_box_ap_50"] = box_ap_50[
                                        -1
                                ][class_id]
                        except Exception as e:
                                print(e)
                                class_name = 'invalid'
                                continue

                for class_id in box_ap_25[-1].keys():
                        try:
                                class_name = self.labels_info[class_id]["name"]
                                ap_results[f"{log_prefix}_{class_name}_val_box_ap_25"] = box_ap_25[
                                        -1
                                ][class_id]
                        except Exception as e:
                                print(e)
                                class_name = 'invalid'
                                continue

                base_path = os.path.join(
                        self.config.general.save_dir,
                        "eval_output",
                        f"instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}",
                )

                if self.validation_dataset.dataset_name in [
                        "scannet",
                        "scannet200",
                ]:
                        gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/{self.validation_dataset.mode}"
                else:
                        gt_data_path = f"{self.validation_dataset.data_dir[0]}/instance_gt/Area_{self.config.general.area}"

                pred_path = f"{base_path}/tmp_output.txt"

                log_prefix = f"val"

                if not os.path.exists(base_path):
                        os.makedirs(base_path)

                try:
                        if pred_npz_files is not None:
                                try:
                                        from benchmark.evaluate_semantic_instance import (  # type: ignore
                                                evaluate_from_npz_files,
                                        )
                                except Exception as exc:
                                        print(
                                                "[baseline][warn] missing `benchmark` helpers; "
                                                "skip semantic instance evaluation (npz mode). "
                                                f"{type(exc).__name__}: {exc}",
                                                flush=True,
                                        )
                                        ap_results[f"{log_prefix}_mean_ap"] = 0.0
                                        ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
                                        ap_results[f"{log_prefix}_mean_ap_25"] = 0.0
                                        return ap_results
                                evaluate_from_npz_files(
                                        pred_npz_files,
                                        gt_data_path,
                                        pred_path,
                                        dataset=self.validation_dataset.dataset_name,
                                )
                        else:
                                if evaluate is None:
                                        print(
                                                "[baseline][warn] missing `benchmark` helpers; "
                                                "skip semantic instance evaluation.",
                                                flush=True,
                                        )
                                        ap_results[f"{log_prefix}_mean_ap"] = 0.0
                                        ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
                                        ap_results[f"{log_prefix}_mean_ap_25"] = 0.0
                                        return ap_results
                                evaluate(
                                        all_preds,
                                        gt_data_path,
                                        pred_path,
                                        dataset=self.validation_dataset.dataset_name,
                                )

                        with open(pred_path, "r") as fin:
                                for line_id, line in enumerate(fin):
                                        if line_id == 0:
                                                # ignore header
                                                continue
                                        class_name, _, ap, ap_50, ap_25 = line.strip().split(",")

                                        if self.validation_dataset.dataset_name == "scannet200":
                                                if class_name in VALID_CLASS_IDS_200_VALIDATION:
                                                        ap_results[
                                                                f"{log_prefix}_{class_name}_val_ap"
                                                        ] = float(ap)
                                                        ap_results[
                                                                f"{log_prefix}_{class_name}_val_ap_50"
                                                        ] = float(ap_50)
                                                        ap_results[
                                                                f"{log_prefix}_{class_name}_val_ap_25"
                                                        ] = float(ap_25)

                                                        if class_name in HEAD_CATS_SCANNET_200:
                                                                head_results.append(
                                                                        np.array(
                                                                                (float(ap), float(ap_50), float(ap_25))
                                                                        )
                                                                )
                                                        elif class_name in COMMON_CATS_SCANNET_200:
                                                                common_results.append(
                                                                        np.array(
                                                                                (float(ap), float(ap_50), float(ap_25))
                                                                        )
                                                                )
                                                        elif class_name in TAIL_CATS_SCANNET_200:
                                                                tail_results.append(
                                                                        np.array(
                                                                                (float(ap), float(ap_50), float(ap_25))
                                                                        )
                                                                )
                                                        else:
                                                                raise ValueError("class not known!")
                                        else:
                                                ap_results[
                                                        f"{log_prefix}_{class_name}_val_ap"
                                                ] = float(ap)
                                                ap_results[
                                                        f"{log_prefix}_{class_name}_val_ap_50"
                                                ] = float(ap_50)
                                                ap_results[
                                                        f"{log_prefix}_{class_name}_val_ap_25"
                                                ] = float(ap_25)

                        if self.validation_dataset.dataset_name == "scannet200":
                                head_results = np.stack(head_results)
                                common_results = np.stack(common_results)
                                tail_results = np.stack(tail_results)

                                mean_tail_results = np.nanmean(tail_results, axis=0)
                                mean_common_results = np.nanmean(common_results, axis=0)
                                mean_head_results = np.nanmean(head_results, axis=0)

                                ap_results[
                                        f"{log_prefix}_mean_tail_ap_25"
                                ] = mean_tail_results[0]
                                ap_results[
                                        f"{log_prefix}_mean_common_ap_25"
                                ] = mean_common_results[0]
                                ap_results[
                                        f"{log_prefix}_mean_head_ap_25"
                                ] = mean_head_results[0]

                                ap_results[
                                        f"{log_prefix}_mean_tail_ap_50"
                                ] = mean_tail_results[1]
                                ap_results[
                                        f"{log_prefix}_mean_common_ap_50"
                                ] = mean_common_results[1]
                                ap_results[
                                        f"{log_prefix}_mean_head_ap_50"
                                ] = mean_head_results[1]

                                ap_results[
                                        f"{log_prefix}_mean_tail_ap_25"
                                ] = mean_tail_results[2]
                                ap_results[
                                        f"{log_prefix}_mean_common_ap_25"
                                ] = mean_common_results[2]
                                ap_results[
                                        f"{log_prefix}_mean_head_ap_25"
                                ] = mean_head_results[2]

                                overall_ap_results = np.nanmean(
                                        np.vstack((head_results, common_results, tail_results)),
                                        axis=0,
                                )

                                ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
                                ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
                                ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]

                                ap_results = {
                                        key: 0.0 if math.isnan(score) else score
                                        for key, score in ap_results.items()
                                }
                        else:
                                mean_ap = statistics.mean(
                                        [
                                                item
                                                for key, item in ap_results.items()
                                                if key.endswith("val_ap")
                                        ]
                                )
                                mean_ap_50 = statistics.mean(
                                        [
                                                item
                                                for key, item in ap_results.items()
                                                if key.endswith("val_ap_50")
                                        ]
                                )
                                mean_ap_25 = statistics.mean(
                                        [
                                                item
                                                for key, item in ap_results.items()
                                                if key.endswith("val_ap_25")
                                        ]
                                )

                                ap_results[f"{log_prefix}_mean_ap"] = mean_ap
                                ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
                                ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

                                ap_results = {
                                        key: 0.0 if math.isnan(score) else score
                                        for key, score in ap_results.items()
                                }
                except (IndexError, OSError) as e:
                        print("NO SCORES!!!")
                        ap_results[f"{log_prefix}_mean_ap"] = 0.0
                        ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
                        ap_results[f"{log_prefix}_mean_ap_25"] = 0.0

                ap_results = collect_grounding_score(all_preds, ap_results, log_prefix)

                with open(self.tmpdir + '/ap_results.pkl', 'wb') as f:
                        pickle.dump(ap_results, f, protocol=2)

                try:
                        if not self.config.general.export:
                                shutil.rmtree(base_path)
                except FileNotFoundError as e:
                        pass

                return ap_results

        def test_epoch_end(self, outputs):
                if self.config.general.export:
                        return
                step_sft = self._is_step_token_sft()
                if _ssr3dllm_env_flag("SSR3DLLM_DEBUG_RESOURCE", "0") and int(getattr(self, "global_rank", 0)) == 0:
                        _ssr3dllm_log_resource(
                                "test_epoch_end enter",
                                save_dir=str(self.config.general.save_dir),
                                tmp_dir=str(getattr(self, "tmpdir", "")),
                                device=getattr(self, "device", None),
                        )
                try:
                        _stream = str(os.environ.get("SSR3DLLM_STREAM_INSTANCE_EVAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
                except Exception:
                        _stream = False
                pred_npz_files = glob.glob(self.tmpdir + '/*_preds.npz') if _stream or self.config.general.gpus > 1 else []

                if not step_sft and (len(self.preds) == 0) and (len(pred_npz_files) == 0):
                        print('===================== found zero prediction ===================')
                        return

                # multi-gpu temporarilly saved the file into .dist_test for evaluation
                if (not step_sft) and (not self.config.data.sample_class_labels or not self.training):
                        if self.config.general.gpus > 1 or _stream:
                                # clean
                                del self.preds
                                del self.bbox_preds
                                del self.bbox_gt

                                gc.collect()

                                # Synchronize ranks before reading prediction files (DDP).
                                try:
                                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                                torch.distributed.barrier()
                                except Exception:
                                        pass

                                # Stream predictions from disk to avoid holding all pred_masks in memory.
                                all_preds = {}
                                all_bbox_preds = {}
                                all_bbox_gt = {}
                                pred_npz_files = glob.glob(self.tmpdir + '/*_preds.npz')
                                for i in pred_npz_files:
                                        scene_name = i.split('/')[-1].split('_preds.npz')[0]
                                        try:
                                                with np.load(i, allow_pickle=True) as data:
                                                        all_preds[scene_name] = dict(gt_ious=data["gt_ious"])
                                        except Exception:
                                                all_preds[scene_name] = dict(gt_ious=(np.zeros((0,), dtype=float), np.zeros((0,), dtype=str), np.zeros((0,), dtype=float),
                                                                                       np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)))
                                        bbox_path = i.replace('_preds.npz', '_bbox.pkl')
                                        if osp.exists(bbox_path):
                                                with open(bbox_path, 'rb') as f:
                                                        all_bbox_preds[scene_name], all_bbox_gt[scene_name] = pickle.load(f)
                                        else:
                                                all_bbox_preds[scene_name], all_bbox_gt[scene_name] = [], []

                                # DDP uses a filesystem rendezvous for metrics; single-process streaming
                                # should just return the computed dict directly (no ap_results.pkl needed).
                                is_ddp = False
                                try:
                                        is_ddp = bool(
                                                self.config.general.gpus > 1
                                                and torch.distributed.is_available()
                                                and torch.distributed.is_initialized()
                                        )
                                except Exception:
                                        is_ddp = False

                                ap_results = {}
                                if self.global_rank == 0:
                                        if _ssr3dllm_env_flag("SSR3DLLM_DEBUG_RESOURCE", "0"):
                                                _ssr3dllm_log_resource(
                                                        f"test_epoch_end before eval_instance_epoch_end pred_npz_files={len(pred_npz_files)}",
                                                        save_dir=str(self.config.general.save_dir),
                                                        tmp_dir=str(getattr(self, "tmpdir", "")),
                                                        device=getattr(self, "device", None),
                                                )
                                        ap_results = self.eval_instance_epoch_end(
                                                all_preds, all_bbox_preds, all_bbox_gt, pred_npz_files=pred_npz_files
                                        ) or {}
                                        if is_ddp:
                                                try:
                                                        with open(self.tmpdir + '/ap_results.pkl', 'wb') as f:
                                                                pickle.dump(ap_results, f, protocol=2)
                                                except Exception:
                                                        pass

                                if is_ddp:
                                        try:
                                                torch.distributed.barrier()
                                        except Exception:
                                                pass
                                        try:
                                                with open(self.tmpdir + '/ap_results.pkl', 'rb') as f:
                                                        ap_results = pickle.load(f)
                                        except FileNotFoundError:
                                                ap_results = {}

                                if self.global_rank == 0:
                                        for i in glob.glob(self.tmpdir + '/*_preds.npz'):
                                                os.remove(i)
                                                os.remove(i.replace('_preds.npz', '_bbox.pkl'))
                        else:
                                all_preds = self.preds
                                all_bbox_preds = self.bbox_preds
                                all_bbox_gt = self.bbox_gt
                                ap_results = self.eval_instance_epoch_end(
                                        all_preds, all_bbox_preds, all_bbox_gt)

                                # clean
                                del self.preds
                                del self.bbox_preds
                                del self.bbox_gt

                                gc.collect()

                        # Log mean metrics, but also persist a full metrics snapshot (including ssr3dllm counters)
                        # to make offline analysis easier.
                        ap_results_mean = {k: v for k, v in ap_results.items() if k.startswith('val_mean')}
                        self.log_dict(ap_results_mean)
                        print({k: f'{v:.4f}' for k, v in ap_results_mean.items()})

                        # Persist metrics for later comparison/debugging
                        try:
                                from pathlib import Path
                                import json

                                metrics_dir = Path(self.config.general.save_dir)
                                metrics_dir.mkdir(parents=True, exist_ok=True)
                                # Keep legacy mean-only file.
                                metrics_path = metrics_dir / "metrics.json"
                                with metrics_path.open("w", encoding="utf-8") as fp:
                                        json.dump(ap_results_mean, fp, indent=2)
                                # Save full snapshot (includes val_ssr3dllm_* if present).
                                full_path = metrics_dir / "metrics_full.json"
                                with full_path.open("w", encoding="utf-8") as fp:
                                        json.dump(ap_results, fp, indent=2)

                                # Optional: fast "capability preservation" language metrics on the saved per-scene
                                # prediction JSONs (used for quick health checks during training).
                                #
                                # This does NOT replace the paper protocol; use
                                # `final_scripts/eval_step4_capability_preservation_ckpt.sh` for full evaluation.
                                do_fast_cap = _ssr3dllm_env_flag("SSR3DLLM_FAST_CAP_PRESERVE", "0")
                                if do_fast_cap:
                                        # Ensure all ranks finished writing per-scene JSONs before summarizing.
                                        try:
                                                if torch.distributed.is_available() and torch.distributed.is_initialized():
                                                        torch.distributed.barrier()
                                        except Exception:
                                                pass
                                        if int(getattr(self, "global_rank", 0)) == 0:
                                                try:
                                                        from pathlib import Path as _Path
                                                        from tools.summarize_capability_preservation import (  # type: ignore
                                                                _summarize_language_metrics,
                                                                _summarize_grounding_metrics,
                                                        )

                                                        pred_dir = _Path(str(getattr(self.llama_config, "save_path", "") or "")).expanduser()
                                                        if pred_dir.exists():
                                                                lang = _summarize_language_metrics(pred_dir) or {}
                                                                grd = _summarize_grounding_metrics(metrics_path) if metrics_path.exists() else {}

                                                                def _fmt(x):
                                                                        try:
                                                                                return "NA" if x is None else f"{float(x):.4f}"
                                                                        except Exception:
                                                                                return str(x)

                                                                print(
                                                                        "[SSR3DLLM][cap_preserve_fast] "
                                                                        f"Dialog_CIDEr={_fmt(lang.get('dialog_cider'))} "
                                                                        f"ScanQA_EM={_fmt(lang.get('scanqa_em'))} "
                                                                        f"Scan2Cap_CIDEr@0.5={_fmt(lang.get('scan2cap_cider_50'))} "
                                                                        f"ObjDesc_CIDEr={_fmt(lang.get('objdesc_cider'))} "
                                                                        f"(n_dialog={lang.get('dialog_n','NA')} n_scanqa={lang.get('scanqa_n','NA')} "
                                                                        f"n_scan2cap={lang.get('scan2cap_n','NA')} n_objdesc={lang.get('objdesc_n','NA')})",
                                                                        flush=True,
                                                                )
                                                                if grd:
                                                                        print(
                                                                                "[SSR3DLLM][cap_preserve_fast] "
                                                                                f"ScanRefer_Acc@0.25(bbox)={_fmt(grd.get('scanrefer_bbox_acc_25'))} "
                                                                                f"M3DRef_F1@0.25(bbox)={_fmt(grd.get('m3dref_bbox_f1_25'))}",
                                                                                flush=True,
                                                                        )
                                                        else:
                                                                print(
                                                                        f"[SSR3DLLM][cap_preserve_fast][warn] pred_dir not found: {pred_dir}",
                                                                        flush=True,
                                                                )
                                                except Exception as _exc:
                                                        print(
                                                                f"[SSR3DLLM][cap_preserve_fast][warn] failed to summarize language metrics: {_exc}",
                                                                flush=True,
                                                        )
                        except Exception as exc:  # pragma: no cover - best effort logging
                                print(f"[Warning] Failed to save metrics json: {exc}")

                self.preds = dict()
                self.bbox_preds = dict()
                self.bbox_gt = dict()

                def gather_cpu(obj, tmpdir, rank, total_rank):
                        with open(tmpdir + f'{rank}.pkl', 'wb') as f:
                                pickle.dump(obj, f, protocol=2)
                        torch.distributed.barrier()
                        objs = []
                        for i in range(total_rank):
                                with open(tmpdir + f'{rank}.pkl', 'rb') as f:
                                        objs.append(pickle.load(f))
                        return objs

                dd = defaultdict(list)
                for output in outputs:
                        if not isinstance(output, dict):
                                continue
                        for key, val in output.items():
                                dd[key].append(val)

                # ---------------- SSR3DLLM: aggregate rel3dref geometry metrics ----------------
                # These counters are emitted by `eval_step()` as per-scene counts. They must be
                # SUM-reduced (not mean-reduced) across batches/ranks, then converted to rates.
                ssr3_num_rel_local = float(sum(dd.get("val_ssr3dllm_num_rel", [])))
                ssr3_num_target_hit_local = float(sum(dd.get("val_ssr3dllm_num_target_hit", [])))
                ssr3_num_chain_hit_local = float(sum(dd.get("val_ssr3dllm_num_chain_hit", [])))
                ssr3_iou_total_local = float(sum(dd.get("val_ssr3dllm_num_iou_total", [])))
                ssr3_iou25_hit_local = float(sum(dd.get("val_ssr3dllm_num_iou25_hit", [])))
                ssr3_iou50_hit_local = float(sum(dd.get("val_ssr3dllm_num_iou50_hit", [])))

                # ---------------- SSR3DLLM: aggregate geom-chain grounding metrics (ScanRefer/M3DRef) ----------------
                scan_total_local = float(sum(dd.get("val_ssr3dllm_num_scanrefer_total", [])))
                scan_mask25_local = float(sum(dd.get("val_ssr3dllm_num_scanrefer_mask25_hit", [])))
                scan_mask50_local = float(sum(dd.get("val_ssr3dllm_num_scanrefer_mask50_hit", [])))
                scan_bbox25_local = float(sum(dd.get("val_ssr3dllm_num_scanrefer_bbox25_hit", [])))
                scan_bbox50_local = float(sum(dd.get("val_ssr3dllm_num_scanrefer_bbox50_hit", [])))

                m3d_total_local = float(sum(dd.get("val_ssr3dllm_num_m3dref_total", [])))
                m3d_mask25_local = float(sum(dd.get("val_ssr3dllm_num_m3dref_mask25_hit", [])))
                m3d_mask50_local = float(sum(dd.get("val_ssr3dllm_num_m3dref_mask50_hit", [])))
                m3d_bbox25_local = float(sum(dd.get("val_ssr3dllm_num_m3dref_bbox25_hit", [])))
                m3d_bbox50_local = float(sum(dd.get("val_ssr3dllm_num_m3dref_bbox50_hit", [])))
                m3d_bbox_f1_25_sum_local = float(sum(dd.get("val_ssr3dllm_sum_m3dref_bbox_f1_25", [])))
                m3d_bbox_f1_50_sum_local = float(sum(dd.get("val_ssr3dllm_sum_m3dref_bbox_f1_50", [])))

                ssr3_agg = torch.tensor(
                        [
                                ssr3_num_rel_local,
                                ssr3_num_target_hit_local,
                                ssr3_num_chain_hit_local,
                                ssr3_iou_total_local,
                                ssr3_iou25_hit_local,
                                ssr3_iou50_hit_local,
                        ],
                        device=self.device,
                        dtype=torch.float32,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(ssr3_agg, op=torch.distributed.ReduceOp.SUM)

                ssr3_num_rel_total = float(ssr3_agg[0].item())
                ssr3_num_target_hit_total = float(ssr3_agg[1].item())
                ssr3_num_chain_hit_total = float(ssr3_agg[2].item())
                ssr3_iou_total = float(ssr3_agg[3].item())
                ssr3_iou25_hit = float(ssr3_agg[4].item())
                ssr3_iou50_hit = float(ssr3_agg[5].item())

                ssr3_geom_agg = torch.tensor(
                        [
                                scan_total_local,
                                scan_mask25_local,
                                scan_mask50_local,
                                scan_bbox25_local,
                                scan_bbox50_local,
                                m3d_total_local,
                                m3d_mask25_local,
                                m3d_mask50_local,
                                m3d_bbox25_local,
                                m3d_bbox50_local,
                                m3d_bbox_f1_25_sum_local,
                                m3d_bbox_f1_50_sum_local,
                        ],
                        device=self.device,
                        dtype=torch.float32,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(ssr3_geom_agg, op=torch.distributed.ReduceOp.SUM)

                scan_total = float(ssr3_geom_agg[0].item())
                scan_mask25 = float(ssr3_geom_agg[1].item())
                scan_mask50 = float(ssr3_geom_agg[2].item())
                scan_bbox25 = float(ssr3_geom_agg[3].item())
                scan_bbox50 = float(ssr3_geom_agg[4].item())
                m3d_total = float(ssr3_geom_agg[5].item())
                m3d_mask25 = float(ssr3_geom_agg[6].item())
                m3d_mask50 = float(ssr3_geom_agg[7].item())
                m3d_bbox25 = float(ssr3_geom_agg[8].item())
                m3d_bbox50 = float(ssr3_geom_agg[9].item())
                m3d_bbox_f1_25_sum = float(ssr3_geom_agg[10].item())
                m3d_bbox_f1_50_sum = float(ssr3_geom_agg[11].item())

                if self.config.general.gpus > 1:
                        # sync multi gpu
                        dd_mgpu = gather_cpu(dd, self.tmpdir + '/losses',
                                             self.global_rank, self.config.general.gpus)
                        dd = {k: [] for k in dd_mgpu[0]}
                        for bd in dd_mgpu:
                                for k in dd:
                                        dd[k].extend(bd[k])

                def _to_float(x):
                        try:
                                import numpy as _np  # local import to keep global deps minimal
                        except Exception:  # pragma: no cover
                                _np = None
                        try:
                                import torch as _torch  # local import
                        except Exception:  # pragma: no cover
                                _torch = None

                        if x is None:
                                return None
                        if isinstance(x, (int, float)):
                                return float(x)
                        if _np is not None:
                                try:
                                        if isinstance(x, (_np.floating, _np.integer)):
                                                return float(x)
                                        if isinstance(x, _np.ndarray):
                                                if x.size == 0:
                                                        return None
                                                return float(_np.mean(x))
                                except Exception:
                                        pass
                        if _torch is not None and _torch.is_tensor(x):
                                try:
                                        if x.numel() == 0:
                                                return None
                                        # Accept any tensor shape; reduce to scalar mean.
                                        return float(x.detach().float().mean().item())
                                except Exception:
                                        return None
                        # Skip non-numeric types (e.g., strings / dicts).
                        return None

                def _mean_safe(vals):
                        nums = []
                        for it in vals:
                                v = _to_float(it)
                                if v is not None:
                                        nums.append(v)
                        return statistics.mean(nums) if nums else float("nan")

                dd = {k: _mean_safe(v) for k, v in dd.items()}

                # In some experiment modes (e.g. step-token SFT / rel3d-only),
                # segmentation/detection losses can be intentionally disabled,
                # so `dd` may contain no `loss_ce|loss_mask|loss_dice` keys.
                # Guard against empty statistics.mean().
                loss_ce_vals = [v for k, v in dd.items() if "loss_ce" in k]
                loss_mask_vals = [v for k, v in dd.items() if "loss_mask" in k]
                loss_dice_vals = [v for k, v in dd.items() if "loss_dice" in k]

                dd["val_mean_loss_ce"] = statistics.mean(loss_ce_vals) if loss_ce_vals else float("nan")
                dd["val_mean_loss_mask"] = statistics.mean(loss_mask_vals) if loss_mask_vals else float("nan")
                dd["val_mean_loss_dice"] = statistics.mean(loss_dice_vals) if loss_dice_vals else float("nan")

                # Add SSR3DLLM aggregated metrics to logs (global sums/rates across ranks).
                dd["val_ssr3dllm_num_rel_total"] = ssr3_num_rel_total
                dd["val_ssr3dllm_iou_total"] = ssr3_iou_total
                dd["val_ssr3dllm_iou25_hit"] = ssr3_iou25_hit
                dd["val_ssr3dllm_iou50_hit"] = ssr3_iou50_hit
                dd["val_ssr3dllm_target_acc"] = (
                        ssr3_num_target_hit_total / ssr3_num_rel_total
                        if ssr3_num_rel_total > 0
                        else float("nan")
                )
                dd["val_ssr3dllm_chain_acc"] = (
                        ssr3_num_chain_hit_total / ssr3_num_rel_total
                        if ssr3_num_rel_total > 0
                        else float("nan")
                )
                dd["val_ssr3dllm_iou25"] = (
                        ssr3_iou25_hit / ssr3_iou_total if ssr3_iou_total > 0 else 0.0
                )
                dd["val_ssr3dllm_iou50"] = (
                        ssr3_iou50_hit / ssr3_iou_total if ssr3_iou_total > 0 else 0.0
                )

                # Add SSR3DLLM geom-chain grounding metrics (ScanRefer/M3DRef) to logs.
                dd["val_ssr3dllm_scanrefer_geom_mask_iou25"] = (
                        scan_mask25 / scan_total if scan_total > 0 else 0.0
                )
                dd["val_ssr3dllm_scanrefer_geom_mask_iou50"] = (
                        scan_mask50 / scan_total if scan_total > 0 else 0.0
                )
                dd["val_ssr3dllm_scanrefer_geom_bbox_iou25"] = (
                        scan_bbox25 / scan_total if scan_total > 0 else 0.0
                )
                dd["val_ssr3dllm_scanrefer_geom_bbox_iou50"] = (
                        scan_bbox50 / scan_total if scan_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_mask_iou25"] = (
                        m3d_mask25 / m3d_total if m3d_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_mask_iou50"] = (
                        m3d_mask50 / m3d_total if m3d_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_bbox_iou25"] = (
                        m3d_bbox25 / m3d_total if m3d_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_bbox_iou50"] = (
                        m3d_bbox50 / m3d_total if m3d_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_bbox_f1_25"] = (
                        m3d_bbox_f1_25_sum / m3d_total if m3d_total > 0 else 0.0
                )
                dd["val_ssr3dllm_m3dref_geom_bbox_f1_50"] = (
                        m3d_bbox_f1_50_sum / m3d_total if m3d_total > 0 else 0.0
                )

                # Make SSR3DLLM rel3dref metrics visible in logs (not only in tensorboard).
                # Print once on rank0 to avoid multi-GPU spam.
                if self.global_rank == 0:
                        try:
                                print(
                                        "[SSR3DLLM][eval] rel3dref_total="
                                        f"{int(ssr3_num_rel_total)} "
                                        "target_acc="
                                        f"{dd['val_ssr3dllm_target_acc']:.4f} "
                                        "chain_acc="
                                        f"{dd['val_ssr3dllm_chain_acc']:.4f} "
                                        "iou25="
                                        f"{dd['val_ssr3dllm_iou25']:.4f} "
                                        "iou50="
                                        f"{dd['val_ssr3dllm_iou50']:.4f}"
                                , flush=True)
                        except Exception:
                                # Best effort; never break training due to logging.
                                pass

                        if not step_sft:
                                try:
                                        print(
                                                "[SSR3DLLM][eval_geom] "
                                                f"scanrefer_total={int(scan_total)} "
                                                f"mask25={dd['val_ssr3dllm_scanrefer_geom_mask_iou25']:.4f} "
                                                f"mask50={dd['val_ssr3dllm_scanrefer_geom_mask_iou50']:.4f} "
                                                f"bbox25={dd['val_ssr3dllm_scanrefer_geom_bbox_iou25']:.4f} "
                                                f"bbox50={dd['val_ssr3dllm_scanrefer_geom_bbox_iou50']:.4f} "
                                                f"| m3dref_total={int(m3d_total)} "
                                                f"mask25={dd['val_ssr3dllm_m3dref_geom_mask_iou25']:.4f} "
                                                f"mask50={dd['val_ssr3dllm_m3dref_geom_mask_iou50']:.4f} "
                                                f"bbox25={dd['val_ssr3dllm_m3dref_geom_bbox_iou25']:.4f} "
                                                f"bbox50={dd['val_ssr3dllm_m3dref_geom_bbox_iou50']:.4f}"
                                        , flush=True)
                                except Exception:
                                        pass

                # Optional: quick ReferIt3D (Vigor) evaluation on small CSV/JSON splits.
                def _env_flag(name: str, default: str = "0") -> bool:
                        v = os.environ.get(name, default)
                        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

                quick_enabled = _env_flag("SSR3DLLM_VIGOR_QUICK_EVAL", "0")
                quick_metrics: Dict[str, float] = {}
                if quick_enabled:
                        try:
                                quick_metrics = self._run_vigor_quick_eval()
                        except Exception as exc:
                                if self.global_rank == 0:
                                        print(f"[SSR3DLLM][vigor_quick][warn] {type(exc).__name__}: {exc}", flush=True)
                                quick_metrics = {}
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                obj_list = [quick_metrics]
                                torch.distributed.broadcast_object_list(obj_list, src=0)
                                quick_metrics = obj_list[0] if obj_list else {}
                        if isinstance(quick_metrics, dict) and quick_metrics:
                                dd.update(quick_metrics)

                # Optional: ReferIt3D quick eval aligned with official benchmark API.
                referit_enabled = _env_flag("SSR3DLLM_REFERIT_QUICK_EVAL", "0")
                referit_metrics: Dict[str, float] = {}
                if referit_enabled:
                        try:
                                referit_metrics = self._run_referit_quick_eval()
                        except Exception as exc:
                                if self.global_rank == 0:
                                        print(f"[SSR3DLLM][referit_quick][warn] {type(exc).__name__}: {exc}", flush=True)
                                referit_metrics = {}
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                obj_list = [referit_metrics]
                                torch.distributed.broadcast_object_list(obj_list, src=0)
                                referit_metrics = obj_list[0] if obj_list else {}
                        if isinstance(referit_metrics, dict) and referit_metrics:
                                dd.update(referit_metrics)

                # Optional: run the *same* Vigor step-slot evaluation (hard/easy/v-dep/v-indep/among-true)
                # that we use for the BERT Mask3D-Vigor baselines, but with the current SSR3DLLM
                # (LLM + geometry-chain head) model providing the per-object logits.
                vigor_stepslot_enabled = _env_flag("SSR3DLLM_VIGOR_STEPSLOT_EVAL", "0")
                vigor_stepslot_metrics: Dict[str, float] = {}
                if vigor_stepslot_enabled:
                        try:
                                vigor_stepslot_metrics = self._run_vigor_stepslot_eval()
                        except Exception as exc:
                                if self.global_rank == 0:
                                        print(f"[SSR3DLLM][vigor_stepslot][warn] {type(exc).__name__}: {exc}", flush=True)
                                        if _env_flag("SSR3DLLM_VIGOR_STEPSLOT_TRACEBACK", "0"):
                                                import traceback
                                                print("[SSR3DLLM][vigor_stepslot][traceback]\n" + traceback.format_exc(), flush=True)
                                vigor_stepslot_metrics = {}
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                                obj_list = [vigor_stepslot_metrics]
                                torch.distributed.broadcast_object_list(obj_list, src=0)
                                vigor_stepslot_metrics = obj_list[0] if obj_list else {}
                        if isinstance(vigor_stepslot_metrics, dict) and vigor_stepslot_metrics:
                                dd.update(vigor_stepslot_metrics)

                self.log_dict(dd)

                print(self.config.general.experiment_name)

        def _run_vigor_quick_eval(self) -> Dict[str, float]:
                def _env_flag(name: str, default: str = "0") -> bool:
                        v = os.environ.get(name, default)
                        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

                def _env_int(name: str, default: int = 0) -> int:
                        try:
                                return int(str(os.environ.get(name, str(default))).strip())
                        except Exception:
                                return int(default)

                def _split_paths(raw: str) -> list[str]:
                        return [p.strip() for p in raw.split(",") if p.strip()]

                if self.global_rank != 0:
                        return {}

                rel3d_raw = os.environ.get("SSR3DLLM_VIGOR_QUICK_REL3D_JSON", "").strip()
                if not rel3d_raw:
                        return {}

                if not getattr(self, "enable_ssr3dllm_geom", False) or not getattr(self, "ssr3dllm_geom_head", None):
                        return {}

                mask3d_root = os.environ.get("SSR3DLLM_VIGOR_QUICK_MASK3D_FEATS", "").strip()
                if not mask3d_root:
                        try:
                                mask3d_root = self.ssr3dllm_geom_head._get_mask3d_feat_root()
                        except Exception:
                                mask3d_root = ""
                if not mask3d_root:
                        return {}

                mask3d_root_path = Path(mask3d_root)
                if not mask3d_root_path.exists():
                        print(f"[SSR3DLLM][vigor_quick][warn] mask3d feats not found: {mask3d_root_path}", flush=True)
                        return {}

                use_scannet_boxes = _env_flag(
                        "SSR3DLLM_VIGOR_QUICK_USE_SCANNET_BOXES",
                        os.environ.get("SSR3DLLM_VIGOR_USE_SCANNET_BOXES", "0"),
                )
                use_pred_boxes = _env_flag(
                        "SSR3DLLM_VIGOR_QUICK_USE_PRED_BOX_INFO",
                        os.environ.get("SSR3DLLM_VIGOR_USE_PRED_BOX_INFO", "0"),
                )
                scannet_pkl = os.environ.get("SSR3DLLM_VIGOR_QUICK_SCANNET_PKL", "").strip()
                if not scannet_pkl:
                        scannet_pkl = os.environ.get("SSR3DLLM_VIGOR_SCANNET_PKL", "").strip()
                if not scannet_pkl:
                        scannet_pkl = os.environ.get("SCANNET_PKL", "").strip()

                filter_source = os.environ.get("SSR3DLLM_VIGOR_QUICK_FILTER_SOURCE", "vigor").strip()
                filter_dataset = os.environ.get("SSR3DLLM_VIGOR_QUICK_FILTER_DATASET", "").strip()
                max_samples = _env_int("SSR3DLLM_VIGOR_QUICK_MAX_SAMPLES", 0)
                debug_first = _env_int("SSR3DLLM_VIGOR_QUICK_DEBUG_FIRST", 0)
                print_every = _env_int("SSR3DLLM_VIGOR_QUICK_PRINT_EVERY", 0)
                cascading = _env_flag("SSR3DLLM_VIGOR_CASCADING", "1")

                vigor = None
                try:
                        vigor = self.ssr3dllm_geom_head._get_vigor_runtime(self.device)
                except Exception:
                        vigor = None

                scannet_scans = None
                if use_scannet_boxes and not use_pred_boxes:
                        try:
                                scannet_scans = self.ssr3dllm_geom_head._get_vigor_scans()
                        except Exception:
                                scannet_scans = None

                vigor_ckpt = os.environ.get("SSR3DLLM_VIGOR_LISTENER_CKPT", "").strip()
                if not vigor_ckpt:
                        vigor_ckpt = os.environ.get("VIGOR_CKPT", "").strip()

                def _tag_for_path(p: Path) -> str:
                        name = p.name.lower()
                        tag = "vigor"
                        if "sr3d" in name:
                                tag = "sr3d"
                        elif "nr3d" in name:
                                tag = "nr3d"
                        if "train" in name:
                                tag += "_train"
                        elif "test" in name:
                                tag += "_test"
                        if "0.025" in name or "_0.025" in name:
                                tag += "_0p025"
                        elif "0.01" in name or "_0.01" in name:
                                tag += "_0p01"
                        elif "0.05" in name or "_0.05" in name:
                                tag += "_0p05"
                        elif "0.1" in name or "_0.1" in name:
                                tag += "_0p1"
                        return tag

                metrics: Dict[str, float] = {}
                seen_tags: set[str] = set()
                # Lazy import: keep this branch optional in eval-only public releases.
                try:
                        from train.eval_referit3d_listener_rel3d_json import (  # type: ignore
                                run_listener_rel3d_eval,
                        )
                except Exception as exc:
                        print(
                                "[SSR3DLLM][vigor_quick][warn] optional listener eval module is unavailable; "
                                f"skip vigor_quick eval. detail={exc}",
                                flush=True,
                        )
                        return metrics
                for raw in _split_paths(rel3d_raw):
                        p = Path(raw)
                        if not p.exists():
                                print(f"[SSR3DLLM][vigor_quick][warn] missing rel3d file: {p}", flush=True)
                                continue
                        tag = _tag_for_path(p)
                        if tag in seen_tags:
                                suffix = len(seen_tags)
                                tag = f"{tag}_{suffix}"
                        seen_tags.add(tag)
                        res = run_listener_rel3d_eval(
                                rel3d_json=p,
                                mask3d_feats=mask3d_root_path,
                                listener_ckpt=vigor_ckpt,
                                device=self.device,
                                scannet_pkl=scannet_pkl,
                                use_scannet_boxes=use_scannet_boxes,
                                use_pred_boxes=use_pred_boxes,
                                max_samples=max_samples,
                                filter_source=filter_source,
                                filter_dataset=filter_dataset,
                                debug_first=debug_first,
                                print_every=print_every,
                                cascading=cascading,
                                vigor=vigor,
                                scannet_scans=scannet_scans,
                                verbose=False,
                        )
                        if not isinstance(res, dict):
                                continue
                        acc = float(res.get("acc", float("nan")))
                        used = float(res.get("used", 0))
                        metrics[f"val_vigor_quick_{tag}_acc"] = acc
                        metrics[f"val_vigor_quick_{tag}_used"] = used
                        if res.get("skipped") is not None:
                                print(
                                        f"[SSR3DLLM][vigor_quick] {tag} used={int(used)} acc={acc:.4f} "
                                        f"skipped={res.get('skipped')}",
                                        flush=True,
                                )
                        else:
                                print(
                                        f"[SSR3DLLM][vigor_quick] {tag} used={int(used)} acc={acc:.4f}",
                                        flush=True,
                                )
                return metrics

        def _run_vigor_stepslot_eval(self) -> Dict[str, float]:
                """
                Evaluate the current SSR3DLLM model on Vigor's ReferIt3D step-slot setting:
                - Uses Vigor's dataset construction/sampling (max_test_objects, view-dep splits, hardness).
                - Uses the same metric breakdown: hard/easy/v-dep/v-indep/all/among-true.

                This is intentionally *optional* (env-gated) because it runs a full pass over the
                Vigor test loader and includes an LLM teacher-forcing forward to obtain
                `lang_info.llm_text_init/llm_text_tokens` from the `<stepK>` supervision.
                """
                if self.global_rank != 0:
                        return {}
                if not (getattr(self, "llama_config", None) and getattr(self.llama_config, "enable_llm", False)):
                        return {}
                if not getattr(self, "enable_ssr3dllm_geom", False) or not getattr(self, "ssr3dllm_geom_head", None):
                        return {}

                import sys
                import importlib
                import traceback
                from types import SimpleNamespace
                from pathlib import Path
                import numpy as np

                # Resolve Vigor package path (vendored under `third_party/Vigor`).
                repo_root = Path(__file__).resolve().parents[3]
                vigor_root = repo_root / "third_party" / "Vigor"
                if vigor_root.exists():
                        sys.path.insert(0, str(vigor_root))

                # Avoid module-cache conflicts between different `referit3d` copies
                # (e.g., `benchmark/referit3d` vs Vigor's bundled `referit3d`).
                # In long-running training, `referit3d` might have been imported already;
                # force re-import from Vigor by purging the cache.
                for k in list(sys.modules.keys()):
                        if k == "referit3d" or k.startswith("referit3d."):
                                try:
                                        del sys.modules[k]
                                except Exception:
                                        pass

                from referit3d.in_out.neural_net_oriented import (  # type: ignore
                        load_scan_related_data,
                        load_referential_data,
                        compute_auxiliary_data,
                )
                from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders  # type: ignore
                from referit3d.analysis.utterances import is_explicitly_view_dependent  # type: ignore
                from referit3d.data_generation.nr3d import decode_stimulus_string  # type: ignore

                def _env_int(name: str, default: int) -> int:
                        try:
                                return int(str(os.environ.get(name, str(default))).strip())
                        except Exception:
                                return int(default)

                def _env_str(name: str, default: str = "") -> str:
                        v = os.environ.get(name, default)
                        return str(v).strip()

                def _pad_truncate_steps(steps: list[str], order_len: int) -> list[str]:
                        steps = list(steps)
                        while len(steps) > int(order_len):
                                steps.pop(0)
                        if not steps:
                                return ["unknown"] * int(order_len)
                        if len(steps) < int(order_len):
                                steps = steps + [steps[-1]] * (int(order_len) - len(steps))
                        return steps

                def _format_step_output(order: list[str], order_len: int) -> str:
                        order = [str(x).strip().strip("*").strip() for x in order if str(x).strip()]
                        order = _pad_truncate_steps(order, order_len=order_len)
                        chunks = []
                        for i in range(int(order_len)):
                                tok = f"<step{i+1}>"
                                # IMPORTANT (causal LLM):
                                # We extract the hidden state at the <stepK> position to drive the geometry head.
                                # For a causal decoder, <stepK> cannot attend to tokens that appear AFTER it.
                                # Therefore, the step text must appear BEFORE <stepK> (e.g., "door <step1>"),
                                # so the <stepK> hidden state is conditioned on the step phrase.
                                chunks.append(f"{order[i]} {tok}".strip())
                        return " ".join(chunks).strip()

                # Required paths.
                scannet_file = _env_str("SSR3DLLM_VIGOR_SCANNET_PKL") or _env_str("SCANNET_PKL")
                if not scannet_file:
                        return {}
                mask3d_root = (
                        _env_str("SSR3DLLM_VIGOR_STEPSLOT_MASK3D_FEATS_TEST")
                        or _env_str("MASK3D_FEATS_TEST")
                        or _env_str("VIGOR_MASK3D_FEATS_ROOT")
                )
                if not mask3d_root:
                        return {}

                # CSVs: allow overriding; default to Vigor-provided step4 files.
                sr3d_train_csv = _env_str("SSR3DLLM_VIGOR_STEPSLOT_SR3D_TRAIN_CSV") or str(
                        vigor_root / "referit3d" / "data" / "csv_data" / "sr3d_train_LLM_step4_485.csv"
                )
                nr3d_train_csv = _env_str("SSR3DLLM_VIGOR_STEPSLOT_NR3D_TRAIN_CSV") or str(
                        vigor_root / "referit3d" / "data" / "csv_data" / "nr3d_train_LLM_step4_485.csv"
                )

                # Eval hyperparams (match common Mask3D-Vigor settings by default).
                order_len = _env_int("SSR3DLLM_STEP_ORDER_LEN", 4)
                max_test_objects = _env_int("SSR3DLLM_VIGOR_STEPSLOT_MAX_TEST_OBJECTS", 88)
                max_distractors = _env_int("SSR3DLLM_VIGOR_STEPSLOT_MAX_DISTRACTORS", 51)
                batch_size = _env_int("SSR3DLLM_VIGOR_STEPSLOT_BATCH_SIZE", 16)
                max_seq_len = _env_int("SSR3DLLM_VIGOR_STEPSLOT_MAX_SEQ_LEN", 24)
                points_per_object = _env_int("SSR3DLLM_VIGOR_STEPSLOT_POINTS_PER_OBJECT", 1024)
                n_workers = _env_int("SSR3DLLM_VIGOR_STEPSLOT_N_WORKERS", 2)
                random_seed = _env_int("SSR3DLLM_VIGOR_STEPSLOT_SEED", int(getattr(self.config.general, "seed", 2020)))
                max_examples = _env_int("SSR3DLLM_VIGOR_STEPSLOT_MAX_EXAMPLES", 0)

                # Build minimal args expected by Vigor loader utils.
                args = SimpleNamespace(
                        scannet_file=scannet_file,
                        referit3D_file=None,
                        augment_with_sr3d=None,
                        s_vs_n_weight=None,
                        mentions_target_class_only=True,
                        max_seq_len=max_seq_len,
                        min_word_freq=3,
                        vocab_file=None,
                        unit_sphere_norm=True,
                        points_per_object=points_per_object,
                        max_distractors=max_distractors,
                        max_test_objects=max_test_objects,
                        batch_size=batch_size,
                        n_workers=n_workers,
                        mode="evaluate",
                        lang_multilabel=True,
                        multilabel_pretraining=True,
                        cascading=True,
                        order_len=order_len,
                        random_seed=random_seed,
                        mask3d_feature_root=mask3d_root,
                        mask3d_feature_root_test=mask3d_root,
                )

                # Load ScanNet scans + referential data (Vigor does train+test concat internally).
                all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

                def _eval_one(name: str, train_csv: str) -> Dict[str, float]:
                        args.referit3D_file = train_csv
                        referit_data = load_referential_data(args, train_csv, scans_split)
                        mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
                        data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
                        loader = data_loaders["test"]
                        dataset = loader.dataset
                        refs = dataset.references

                        got = []
                        got_among_true = []

                        self.llama_model.eval()
                        self.ssr3dllm_geom_head.eval()
                        device = self.device
                        seen = 0
                        feat_cache: Dict[str, dict] = {}
                        for batch in loader:
                                if max_examples > 0 and seen >= max_examples:
                                        break
                                # Move tensors.
                                objects = batch["objects"].to(device=device)
                                # In Vigor's Mask3D mode, `batch["objects"]` are per-object point samples
                                # with shape [B,N,P,C]. We must map them to Mask3D query embeddings using
                                # `mask3d_feature_path` + `instance_ids` (aligned with Vigor training).
                                if objects.dim() == 4:
                                        feat_paths = batch.get("mask3d_feature_path", None)
                                        inst_ids = batch.get("instance_ids", None)
                                        if feat_paths is None or inst_ids is None:
                                                raise RuntimeError(
                                                        "Vigor step-slot eval expects `mask3d_feature_path` and `instance_ids` "
                                                        "when `objects` is 4D (point samples)."
                                                )
                                        # `feat_paths` is typically a list[str] after collation.
                                        if isinstance(feat_paths, str):
                                                feat_paths = [feat_paths] * int(objects.size(0))
                                        if torch.is_tensor(feat_paths):
                                                feat_paths = feat_paths.detach().cpu().tolist()
                                        if not isinstance(feat_paths, (list, tuple)):
                                                feat_paths = [str(feat_paths)] * int(objects.size(0))

                                        inst_ids_t = inst_ids.to(device="cpu")
                                        B = int(objects.size(0))
                                        N = int(objects.size(1))
                                        # Load one scene feature file to infer embedding dim.
                                        sample_path = str(feat_paths[0])
                                        if sample_path not in feat_cache:
                                                feat_cache[sample_path] = torch.load(sample_path, map_location="cpu")
                                        sample_feat = feat_cache[sample_path]
                                        oq0 = sample_feat.get("object_queries", None)
                                        if oq0 is None:
                                                raise RuntimeError(f"Missing `object_queries` in: {sample_path}")
                                        D0 = int(oq0.shape[-1])

                                        obj_emb = torch.zeros((B, N, D0), dtype=torch.float32)
                                        debug_map = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_MAP", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                        map_hits = 0
                                        map_total = 0
                                        for bi in range(B):
                                                path = str(feat_paths[bi])
                                                if path not in feat_cache:
                                                        feat_cache[path] = torch.load(path, map_location="cpu")
                                                feat = feat_cache[path]
                                                oq = feat.get("object_queries", None)
                                                gt_map = feat.get("gt_to_query_map", {}) or {}
                                                if oq is None:
                                                        continue
                                                oq = torch.as_tensor(oq, dtype=torch.float32)
                                                # Fill each context slot from the mapped Mask3D query.
                                                for j in range(N):
                                                        inst_id = int(inst_ids_t[bi, j].item())
                                                        if inst_id < 0:
                                                                continue
                                                        map_total += 1
                                                        qidx = gt_map.get(inst_id, None)
                                                        if qidx is None:
                                                                # Common mismatch: referit3d uses 0-based object_id while some
                                                                # mappings are stored 1-based (instance_id starts at 1).
                                                                if inst_id >= 0 and (inst_id + 1) in gt_map and 0 not in gt_map and 1 in gt_map:
                                                                        qidx = gt_map.get(inst_id + 1, None)
                                                        try:
                                                                qidx = int(qidx) if qidx is not None else None
                                                        except Exception:
                                                                qidx = None
                                                        if qidx is None or not (0 <= qidx < int(oq.shape[0])):
                                                                continue
                                                        map_hits += 1
                                                        obj_emb[bi, j, :] = oq[qidx]

                                        objects = obj_emb.to(device=device, dtype=torch.float32)
                                        if debug_map and map_total > 0 and not getattr(self, "_vigor_stepslot_debug_map_printed", False):
                                                self._vigor_stepslot_debug_map_printed = True
                                                print(
                                                        f"[SSR3DLLM][vigor_stepslot][map] "
                                                        f"hit_rate={float(map_hits) / float(map_total):.4f} "
                                                        f"hits={map_hits} total={map_total} "
                                                        f"D={D0}",
                                                        flush=True,
                                                )
                                context_size = batch["context_size"].to(device=device)
                                target_pos = batch["target_pos"].to(device=device)
                                # Ensure boolean mask semantics (Vigor dataset stores it as bool numpy, but collate
                                # can sometimes cast to uint8/int64 depending on the pipeline).
                                target_class_mask = batch["target_class_mask"].to(device=device)
                                if target_class_mask.dtype != torch.bool:
                                        target_class_mask = target_class_mask > 0
                                # Box features: [cx,cy,cz,vol]
                                # IMPORTANT: use Vigor dataset's `box_info` directly so we stay aligned with
                                # Mask3D-Vigor training (predbox mode uses pred_box_info[qidx]).
                                box_info = batch.get("box_info", None)
                                if box_info is None:
                                        raise KeyError(
                                                "[SSR3DLLM][vigor_stepslot] missing batch['box_info'] (expected shape [B,N,4] "
                                                "with fields [cx,cy,cz,vol]); this eval intentionally fails-fast to avoid "
                                                "silent misalignment with the Mask3D-Vigor predbox setup."
                                        )
                                if torch.is_tensor(box_info):
                                        box_info = box_info.to(device=device, dtype=torch.float32)
                                else:
                                        box_info = torch.as_tensor(box_info, device=device, dtype=torch.float32)
                                if box_info.dim() == 2:
                                        box_info = box_info.unsqueeze(0)
                                if box_info.dim() != 3 or int(box_info.size(-1)) != 4:
                                        raise ValueError(
                                                f"[SSR3DLLM][vigor_stepslot] invalid box_info.shape={tuple(box_info.shape)}; "
                                                "expected [B,N,4] with fields [cx,cy,cz,vol]."
                                        )

                                B, N, D = objects.shape
                                # Build per-sample step-token supervision from referential_order.
                                ref_order = batch.get("referential_order", None)
                                orders: list[list[str]] = []
                                if isinstance(ref_order, list):
                                        # Expected shape: [B][order_len] (strings)
                                        for i in range(B):
                                                row = ref_order[i] if i < len(ref_order) else []
                                                orders.append(list(row) if isinstance(row, (list, tuple)) else [])
                                else:
                                        orders = [[] for _ in range(B)]

                                utterances = batch.get("utterance", None)
                                if utterances is None:
                                        utterances = [" ".join(x) for x in batch.get("tokens", [[]])]

                                input_texts = [f"<geom> {str(u).strip()}" for u in utterances]
                                output_texts = [_format_step_output(orders[i], order_len=order_len) for i in range(B)]

                                # Construct minimal lang_info_data for LLM to attach llm_text_init/tokens.
                                from baseline.dataset.dataset_code.language_info import lang_info_data  # local import

                                eval_type = "rel3dref:vigor_steps:text_only"
                                lang_infos = [
                                        lang_info_data(
                                                question=input_texts[i],
                                                answer=output_texts[i],
                                                lang_type=eval_type,
                                                positives_question=[],
                                                inst_ids_question=[],
                                                query_ids_question=[],
                                                positives_answer=[],
                                                inst_ids_answer=[],
                                                query_ids_answer=[],
                                        )
                                        for i in range(B)
                                ]
                                gt_inst_ids = [(int(i), lang_infos[i]) for i in range(B)]

                                # Normalize instance embeddings for LLM input.
                                q_norm = objects / (objects.norm(dim=-1, keepdim=True) + 1e-8)
                                batch_instance_queries_hidden_state = [objects[i] for i in range(B)]
                                batch_instance_queries_normalized_embed = [q_norm[i] for i in range(B)]
                                batch_eval_types = [eval_type for _ in range(B)]

                                with torch.no_grad():
                                        _ = self.llama_model(
                                                batch_input_text_list=input_texts,
                                                batch_output_text_list=output_texts,
                                                batch_instance_queries_hidden_state=batch_instance_queries_hidden_state,
                                                batch_instance_queries_normalized_embed=batch_instance_queries_normalized_embed,
                                                batch_eval_types=batch_eval_types,
                                                batch_gt_inst_ids=gt_inst_ids,
                                        )

                                # Backend switch:
                                # - decoder: SSR3DLLM relation_field + pointer decoder (legacy)
                                # - vigor:   Mask3D-Vigor listener (no relation_field / pointer decoder)
                                backend = "decoder"
                                try:
                                        backend = str(self.ssr3dllm_geom_head._get_geom_backend()).strip().lower()
                                except Exception:
                                        backend = "decoder"

                                if backend == "vigor":
                                        runtime = self.ssr3dllm_geom_head._get_vigor_runtime(device)
                                        if runtime is None:
                                                return {}

                                        if not getattr(self, "_vigor_stepslot_using_vigor_backend_printed", False):
                                                self._vigor_stepslot_using_vigor_backend_printed = True
                                                try:
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot] backend=vigor "
                                                                f"order_len={int(getattr(runtime, 'order_len', order_len))} "
                                                                f"mask3d_dim={int(objects.size(-1))}",
                                                                flush=True,
                                                        )
                                                except Exception:
                                                        pass

                                        inner_dim = int(getattr(runtime.model, "inner_dim", 768))
                                        proj = self.ssr3dllm_geom_head._get_vigor_step_proj(inner_dim=inner_dim, device=device)
                                        O = int(getattr(runtime, "order_len", order_len))

                                        # Order-embedding source for step-slot evaluation.
                                        # - "llm" (default): use LLM <stepK> hidden states (projected to Vigor inner_dim).
                                        # - "bert": use Vigor listener's own BERT encoding on "<stepK> step_text"
                                        #   (matches the original BERT step-slot pipeline more closely).
                                        order_src = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_ORDER_SOURCE", "llm")).strip().lower()
                                        if order_src not in {"llm", "bert"}:
                                                order_src = "llm"

                                        # Build per-step order embeddings from LLM <stepK> token hidden states.
                                        order_embeds = None
                                        if order_src == "llm":
                                                orders_emb = []
                                                for i in range(B):
                                                        li = lang_infos[i]
                                                        step_emb = getattr(li, "llm_step_embeds", None)
                                                        step_emb_dim = None
                                                        if torch.is_tensor(step_emb) and step_emb.dim() == 2:
                                                                step_emb_dim = int(step_emb.size(-1))
                                                        # Accept:
                                                        # - legacy: [O, mask3d_dim] -> project to inner_dim
                                                        # - llama-stepslot: [O, inner_dim] -> use directly
                                                        if step_emb_dim == int(inner_dim):
                                                                step_emb = step_emb.to(device=device, dtype=torch.float32)
                                                        elif step_emb_dim == int(objects.size(-1)):
                                                                step_emb = step_emb.to(device=device, dtype=torch.float32)
                                                        else:
                                                                step_emb = torch.zeros((int(O), int(inner_dim)), device=device, dtype=torch.float32)
                                                        if int(step_emb.size(0)) < int(O):
                                                                pad = (
                                                                        step_emb[-1:].repeat(int(O - int(step_emb.size(0))), 1)
                                                                        if int(step_emb.size(0)) > 0
                                                                        else torch.zeros(
                                                                                (int(O), int(step_emb.size(-1))),
                                                                                device=device,
                                                                                dtype=torch.float32,
                                                                        )
                                                                )
                                                                step_emb = torch.cat([step_emb, pad], dim=0)
                                                        elif int(step_emb.size(0)) > int(O):
                                                                step_emb = step_emb[: int(O)]
                                                        if int(step_emb.size(-1)) == int(inner_dim):
                                                                orders_emb.append(step_emb.unsqueeze(1))  # [O,1,D]
                                                        else:
                                                                orders_emb.append(proj(step_emb).unsqueeze(1))  # [O,1,D]
                                                order_embeds = torch.stack(orders_emb, dim=0)  # [B,O,1,D]

                                        # pred_class_mask (per-step gating over context slots)
                                        use_predmask = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_USE_PREDMASK", "1")).strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "y",
                                                "on",
                                        }
                                        pcm = batch.get("pred_class_mask", None)
                                        if use_predmask and torch.is_tensor(pcm) and pcm.dim() == 3:
                                                pred_class_mask = pcm.to(device=device, dtype=torch.float32)
                                        else:
                                                pred_class_mask = torch.ones((int(B), int(O), int(N)), device=device, dtype=torch.float32)

                                        # -------- Diagnostics: verify target_pos aligns with slot order --------
                                        dbg_align = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DIAG_ALIGN", "0")).strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "y",
                                                "on",
                                        }
                                        if dbg_align and not getattr(self, "_vigor_stepslot_diag_align_printed", False):
                                                self._vigor_stepslot_diag_align_printed = True
                                                try:
                                                        cls_labels = batch.get("class_labels", None)
                                                        tgt_cls = batch.get("target_class", None)
                                                        inst_ids_dbg = batch.get("instance_ids", None)
                                                        if torch.is_tensor(cls_labels):
                                                                cls_labels = cls_labels.to(device=device)
                                                        if torch.is_tensor(tgt_cls):
                                                                tgt_cls = tgt_cls.to(device=device)
                                                        if torch.is_tensor(inst_ids_dbg):
                                                                inst_ids_dbg = inst_ids_dbg.to(device=device)

                                                        ok = True
                                                        if not (torch.is_tensor(cls_labels) and torch.is_tensor(tgt_cls)):
                                                                ok = False

                                                        if ok:
                                                                idx = target_pos.clamp(min=0, max=int(N) - 1).view(-1, 1)
                                                                cls_at_tp = cls_labels.gather(1, idx).view(-1)
                                                                cls_mismatch = (cls_at_tp != tgt_cls.view(-1)).detach()
                                                                tcm_at_tp = target_class_mask.gather(1, idx).view(-1).detach()
                                                                mismatch_rate = float(cls_mismatch.float().mean().cpu())
                                                                tcm_true_rate = float(tcm_at_tp.float().mean().cpu())
                                                                print(
                                                                        f"[SSR3DLLM][vigor_stepslot][align] "
                                                                        f"class_label_at_target_matches={1.0 - mismatch_rate:.4f} "
                                                                        f"target_in_target_class_mask_rate={tcm_true_rate:.4f} "
                                                                        f"use_predmask={int(use_predmask)}",
                                                                        flush=True,
                                                                )
                                                                # Print a few concrete samples for manual inspection.
                                                                for ii in range(int(min(4, B))):
                                                                        tp = int(target_pos[ii].item())
                                                                        cs = int(context_size[ii].item())
                                                                        ca = int(cls_at_tp[ii].item())
                                                                        tc = int(tgt_cls[ii].item())
                                                                        inst_at = None
                                                                        if torch.is_tensor(inst_ids_dbg):
                                                                                inst_at = int(inst_ids_dbg[ii, tp].item())
                                                                        bi = box_info[ii, tp].detach().cpu().tolist()
                                                                        print(
                                                                                f"[SSR3DLLM][vigor_stepslot][align] "
                                                                                f"i={ii} ctx={cs} tp={tp} "
                                                                                f"class_at_tp={ca} target_class={tc} "
                                                                                f"inst_id_at_tp={inst_at} "
                                                                                f"box_at_tp={[round(float(x),4) for x in bi]}",
                                                                                flush=True,
                                                                        )
                                                except Exception as e:
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][align][warn] {type(e).__name__}: {e}",
                                                                flush=True,
                                                        )

                                        # -------- Diagnostics: verify LLM step positions/embeddings --------
                                        dbg_steps = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DIAG_STEPS", "0")).strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "y",
                                                "on",
                                        }
                                        if dbg_steps and not getattr(self, "_vigor_stepslot_diag_steps_printed", False):
                                                self._vigor_stepslot_diag_steps_printed = True
                                                try:
                                                        step_pos_all = []
                                                        step_vecs = []
                                                        for i in range(B):
                                                                li = lang_infos[i]
                                                                sp = getattr(li, "llm_step_pos", None)
                                                                se = getattr(li, "llm_step_embeds", None)
                                                                if isinstance(sp, list):
                                                                        step_pos_all.extend([int(x) for x in sp[: int(O)]])
                                                                if torch.is_tensor(se) and se.dim() == 2:
                                                                        step_vecs.append(se[: int(O)].to(device=device, dtype=torch.float32))
                                                        miss = sum(1 for x in step_pos_all if int(x) < 0)
                                                        tot = max(1, len(step_pos_all))
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][diag] "
                                                                f"order_src={order_src} step_pos_missing={miss}/{tot} "
                                                                f"(frac={float(miss) / float(tot):.4f})",
                                                                flush=True,
                                                        )
                                                        if step_vecs:
                                                                sv = torch.stack(step_vecs, dim=0)  # [B,O,D]
                                                                norms = torch.linalg.vector_norm(sv, dim=-1)  # [B,O]
                                                                n_mean = float(norms.mean().detach().cpu())
                                                                n_std = float(norms.std(unbiased=False).detach().cpu())
                                                                n_min = float(norms.min().detach().cpu())
                                                                n_max = float(norms.max().detach().cpu())
                                                                cos_means = []
                                                                for kk in range(int(min(O, norms.size(1)))):
                                                                        v = sv[:, kk, :]
                                                                        v = v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-8)
                                                                        g = v @ v.t()
                                                                        if int(g.size(0)) > 1:
                                                                                off = (g.sum() - torch.diagonal(g).sum()) / float(
                                                                                        int(g.size(0)) * (int(g.size(0)) - 1)
                                                                                )
                                                                                cos_means.append(float(off.detach().cpu()))
                                                                        else:
                                                                                cos_means.append(1.0)
                                                                print(
                                                                        f"[SSR3DLLM][vigor_stepslot][diag] "
                                                                        f"step_embed_norm(mean,std,min,max)=({n_mean:.3g},{n_std:.3g},{n_min:.3g},{n_max:.3g}) "
                                                                        f"cos_offdiag_by_step={[(round(x,4)) for x in cos_means]}",
                                                                        flush=True,
                                                                )
                                                except Exception as e:
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][diag][warn] {type(e).__name__}: {e}",
                                                                flush=True,
                                                        )

                                        # -------- Main logits --------
                                        lang_tokens = runtime.tokenizer(
                                                utterances if utterances is not None else [""] * int(B),
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                        )
                                        if order_src == "bert":
                                                # BERT-step style: let the Vigor listener build order embeddings from order_tokens.
                                                order_texts_flat = []
                                                for i in range(B):
                                                        st = _pad_truncate_steps(orders[i], order_len=int(O))
                                                        for kk in range(int(O)):
                                                                order_texts_flat.append(f"<step{kk+1}> {st[kk]}".strip())
                                                order_tokens = runtime.tokenizer(
                                                        order_texts_flat,
                                                        return_tensors="pt",
                                                        padding=True,
                                                        truncation=True,
                                                )
                                                for k in list(order_tokens.keys()):
                                                        v = order_tokens[k]
                                                        if torch.is_tensor(v) and v.dim() == 2:
                                                                order_tokens[k] = v.reshape(int(B), int(O), v.size(1))
                                                lang_tokens = {k: v.to(runtime.device) for k, v in lang_tokens.items()}
                                                order_tokens = {k: v.to(runtime.device) for k, v in order_tokens.items()}
                                                batch_v = {
                                                        "inference": True,
                                                        "mask3d_object_queries": objects.to(runtime.device, dtype=torch.float32),
                                                        "box_info": box_info.to(runtime.device, dtype=torch.float32),
                                                        "lang_tokens": lang_tokens,
                                                        "order_tokens": order_tokens,
                                                        "pred_class_mask": pred_class_mask.to(runtime.device, dtype=torch.float32),
                                                        "class_labels": torch.zeros((int(B), int(N)), device=runtime.device, dtype=torch.long),
                                                        "target_pos": torch.zeros((int(B),), device=runtime.device, dtype=torch.long),
                                                        "target_class": torch.zeros((int(B),), device=runtime.device, dtype=torch.long),
                                                }
                                                _, _, _, logits, _, _ = runtime.model(batch_v)
                                        else:
                                                logits = runtime.forward_logits_with_order_embeds(
                                                        lang_tokens=lang_tokens,
                                                        order_embeds=order_embeds,
                                                        mask3d_queries=objects,
                                                        box_info=box_info,
                                                        pred_class_mask=pred_class_mask,
                                                )  # [B,N]
                                        logits = logits.to(device=device)

                                        # Optional diagnostic: compare a BERT-step forward on the first batch.
                                        dbg_cmp = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DIAG_COMPARE_BERT", "0")).strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "y",
                                                "on",
                                        }
                                        if dbg_cmp and not getattr(self, "_vigor_stepslot_diag_compare_printed", False) and order_src != "bert":
                                                self._vigor_stepslot_diag_compare_printed = True
                                                try:
                                                        order_texts_flat = []
                                                        for i in range(B):
                                                                st = _pad_truncate_steps(orders[i], order_len=int(O))
                                                                for kk in range(int(O)):
                                                                        order_texts_flat.append(f"<step{kk+1}> {st[kk]}".strip())
                                                        order_tokens = runtime.tokenizer(
                                                                order_texts_flat,
                                                                return_tensors="pt",
                                                                padding=True,
                                                                truncation=True,
                                                        )
                                                        for k in list(order_tokens.keys()):
                                                                v = order_tokens[k]
                                                                if torch.is_tensor(v) and v.dim() == 2:
                                                                        order_tokens[k] = v.reshape(int(B), int(O), v.size(1))
                                                        lang_tokens2 = {k: v.to(runtime.device) for k, v in lang_tokens.items()}
                                                        order_tokens2 = {k: v.to(runtime.device) for k, v in order_tokens.items()}
                                                        batch_v = {
                                                                "inference": True,
                                                                "mask3d_object_queries": objects.to(runtime.device, dtype=torch.float32),
                                                                "box_info": box_info.to(runtime.device, dtype=torch.float32),
                                                                "lang_tokens": lang_tokens2,
                                                                "order_tokens": order_tokens2,
                                                                "pred_class_mask": pred_class_mask.to(runtime.device, dtype=torch.float32),
                                                                "class_labels": torch.zeros((int(B), int(N)), device=runtime.device, dtype=torch.long),
                                                                "target_pos": torch.zeros((int(B),), device=runtime.device, dtype=torch.long),
                                                                "target_class": torch.zeros((int(B),), device=runtime.device, dtype=torch.long),
                                                        }
                                                        _, _, _, logits_b, _, _ = runtime.model(batch_v)
                                                        logits_b = logits_b.to(device=device)
                                                        p_llm = logits.argmax(dim=-1)
                                                        p_bert = logits_b.argmax(dim=-1)
                                                        acc_llm = float((p_llm == target_pos).float().mean().detach().cpu() * 100.0)
                                                        acc_bert = float((p_bert == target_pos).float().mean().detach().cpu() * 100.0)
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][diag] compare_first_batch "
                                                                f"acc_llm={acc_llm:.2f} acc_bert={acc_bert:.2f} "
                                                                f"use_predmask={int(use_predmask)}",
                                                                flush=True,
                                                        )
                                                except Exception as e:
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][diag][warn] compare_bert failed: {type(e).__name__}: {e}",
                                                                flush=True,
                                                        )

                                        for i in range(B):
                                                c = int(context_size[i].item())
                                                if c < int(N):
                                                        logits[i, c:] = -1e9

                                        pred = logits.argmax(dim=-1)
                                        got.append((pred == target_pos).detach().cpu().numpy())

                                        target_class_mask_b = target_class_mask[:, : int(N)].clone()
                                        for i in range(B):
                                                c = int(context_size[i].item())
                                                if c < int(N):
                                                        target_class_mask_b[i, c:] = False
                                        masked_logits = logits.masked_fill(~target_class_mask_b, -1e9)
                                        pred2 = masked_logits.argmax(dim=-1)
                                        got_among_true.append((pred2 == target_pos).detach().cpu().numpy())

                                        dbg_among_true = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_AMONG_TRUE", "0")).strip().lower() in {
                                                "1",
                                                "true",
                                                "yes",
                                                "y",
                                                "on",
                                        }
                                        if dbg_among_true and not getattr(self, "_vigor_stepslot_debug_among_true_printed", False):
                                                self._vigor_stepslot_debug_among_true_printed = True
                                                try:
                                                        mask_sum = target_class_mask_b.sum(dim=-1).detach().cpu().tolist()
                                                        tgt_in = target_class_mask_b[
                                                                torch.arange(B, device=device),
                                                                target_pos.clamp(min=0, max=int(N) - 1),
                                                        ].detach().cpu().tolist()
                                                        log_min = float(logits.detach().min().cpu())
                                                        log_max = float(logits.detach().max().cpu())
                                                        log_mean = float(logits.detach().mean().cpu())
                                                        print(
                                                                f"[SSR3DLLM][vigor_stepslot][among_true_dbg] "
                                                                f"N={int(N)} B={int(B)} "
                                                                f"logits(min,max,mean)=({log_min:.3g},{log_max:.3g},{log_mean:.3g}) "
                                                                f"mask_sum={mask_sum[:8]} target_in_mask={tgt_in[:8]}",
                                                                flush=True,
                                                        )
                                                except Exception:
                                                        pass

                                        seen += int(B)

                                        # proceed to next batch
                                        continue

                                # Geometry-chain head: build student-space object tokens.
                                q_hidden_q = objects.detach().to(dtype=torch.float32)
                                obj_tokens = self.ssr3dllm_geom_head.query_up(q_hidden_q)
                                field_s, _ = self.ssr3dllm_geom_head.relation_field(center_coors.to(dtype=torch.float32))
                                obj_tokens = obj_tokens + field_s.to(dtype=obj_tokens.dtype)

                                # text_init + token-level features from LLM.
                                text_init = torch.zeros((B, obj_tokens.size(-1)), device=device, dtype=obj_tokens.dtype)
                                text_tokens_list = []
                                max_L = 0
                                dbg_llm = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_LLM", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                llm_init_ok = 0
                                llm_tok_ok = 0
                                for i in range(B):
                                        li = lang_infos[i]
                                        t0 = getattr(li, "llm_text_init", None)
                                        if torch.is_tensor(t0) and t0.numel() == D:
                                                text_init[i] = self.ssr3dllm_geom_head.query_up(t0.view(1, -1).to(dtype=torch.float32)).to(dtype=obj_tokens.dtype)[0]
                                                llm_init_ok += 1
                                        tt = getattr(li, "llm_text_tokens", None)
                                        if torch.is_tensor(tt) and tt.dim() == 2 and int(tt.size(-1)) == int(D):
                                                tts = self.ssr3dllm_geom_head.query_up(tt.to(dtype=torch.float32)).to(dtype=obj_tokens.dtype)
                                                llm_tok_ok += 1
                                        else:
                                                tts = torch.zeros((1, obj_tokens.size(-1)), device=device, dtype=obj_tokens.dtype)
                                        text_tokens_list.append(tts)
                                        max_L = max(max_L, int(tts.size(0)))
                                if dbg_llm and not getattr(self, "_vigor_stepslot_debug_llm_printed", False):
                                        self._vigor_stepslot_debug_llm_printed = True
                                        print(
                                                f"[SSR3DLLM][vigor_stepslot][llm] "
                                                f"llm_text_init_ok={llm_init_ok}/{B} "
                                                f"llm_text_tokens_ok={llm_tok_ok}/{B} "
                                                f"D={int(D)}",
                                                flush=True,
                                        )

                                text_tokens = None
                                if max_L > 0:
                                        padded = []
                                        for tts in text_tokens_list:
                                                if int(tts.size(0)) < max_L:
                                                        pad = torch.zeros((max_L - int(tts.size(0)), tts.size(1)), device=device, dtype=tts.dtype)
                                                        padded.append(torch.cat([tts, pad], dim=0))
                                                else:
                                                        padded.append(tts)
                                        text_tokens = torch.stack(padded, dim=0)  # [B,L,D]

                                # Greedy autoregressive decode to get a sequence of selected indices.
                                # For Vigor-style step-slot evaluation, the final prediction should come from
                                # the *last step* (order_len-1). We do not use an adaptive STOP criterion here.
                                T = int(order_len)
                                stop_idx = int(N)  # reserved STOP slot in the decoder output space
                                order_labels = torch.full((B, T), fill_value=stop_idx, device=device, dtype=torch.long)
                                pointer_logits = None
                                pred_class_mask = batch.get("pred_class_mask", None)
                                use_predmask = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_USE_PREDMASK", "1")).strip().lower() in {
                                        "1",
                                        "true",
                                        "yes",
                                        "y",
                                        "on",
                                }
                                if use_predmask and torch.is_tensor(pred_class_mask) and pred_class_mask.dim() == 3:
                                        pred_class_mask = pred_class_mask.to(device=device)
                                        if pred_class_mask.dtype != torch.bool:
                                                pred_class_mask = pred_class_mask > 0
                                else:
                                        pred_class_mask = None
                                dbg_predmask = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_PREDMASK", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                if dbg_predmask and pred_class_mask is not None and not getattr(self, "_vigor_stepslot_debug_predmask_printed", False):
                                        self._vigor_stepslot_debug_predmask_printed = True
                                        try:
                                                tt0 = 0
                                                ttl = int(pred_class_mask.size(1)) - 1
                                                cnt0 = pred_class_mask[:, tt0, : int(N)].sum(dim=-1).detach().cpu().tolist()
                                                cntl = pred_class_mask[:, ttl, : int(N)].sum(dim=-1).detach().cpu().tolist()
                                                tpos = target_pos.clamp(min=0, max=int(N) - 1)
                                                tin0 = pred_class_mask[torch.arange(B, device=device), tt0, tpos].detach().cpu().tolist()
                                                tinl = pred_class_mask[torch.arange(B, device=device), ttl, tpos].detach().cpu().tolist()
                                                print(
                                                        f"[SSR3DLLM][vigor_stepslot][predmask_dbg] "
                                                        f"tt0={tt0} ttl={ttl} "
                                                        f"cnt0={cnt0[:8]} cntl={cntl[:8]} "
                                                        f"target_in0={tin0[:8]} target_inl={tinl[:8]}",
                                                        flush=True,
                                                )
                                        except Exception:
                                                pass
                                for t in range(T):
                                        obj_in = obj_tokens
                                        step_mask = None
                                        if pred_class_mask is not None:
                                                tt = min(int(t), int(pred_class_mask.size(1)) - 1)
                                                step_mask = pred_class_mask[:, tt, : int(N)]
                                                step_mask_f = step_mask.to(dtype=obj_tokens.dtype).unsqueeze(-1)  # [B,N,1]
                                                obj_in = obj_tokens * step_mask_f
                                        pointer_logits = self.ssr3dllm_geom_head.decoder(
                                                obj_tokens=obj_in,
                                                text_tokens=text_tokens,
                                                order_labels=order_labels,
                                                text_init=text_init,
                                                obj_padding_mask=None,
                                        )  # [B,T,N+1]
                                        # Note: In Vigor, `pred_class_mask` gates *features* during the iterative
                                        # update, but does not hard-remove candidates from the output space.
                                        step_logits = pointer_logits[:, t, : int(N)]
                                        pred_t = step_logits.argmax(dim=-1)
                                        order_labels[:, t] = pred_t
                                if pointer_logits is None:
                                        logits = torch.full((B, N), -1e9, device=device)
                                else:
                                        # Use the final step logits as the ReferIt3D prediction (Vigor-compatible).
                                        logits = pointer_logits[:, T - 1, : int(N)].clone()

                                # Mask padding slots beyond context_size.
                                for i in range(B):
                                        c = int(context_size[i].item())
                                        if c < int(N):
                                                logits[i, c:] = -1e9

                                pred = logits.argmax(dim=-1)
                                got.append((pred == target_pos).detach().cpu().numpy())

                                # "Among-true" accuracy: constrain prediction to target's instance-class only.
                                # For correctness, the constraint must never exclude the target itself; we also
                                # exclude padding beyond context_size.
                                target_class_mask = target_class_mask[:, : int(N)].clone()
                                for i in range(B):
                                        c = int(context_size[i].item())
                                        if c < int(N):
                                                target_class_mask[i, c:] = False

                                dbg_among_true = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_AMONG_TRUE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                if dbg_among_true and not getattr(self, "_vigor_stepslot_debug_among_true_printed", False):
                                        self._vigor_stepslot_debug_among_true_printed = True
                                        try:
                                                mask_sum = target_class_mask.sum(dim=-1).detach().cpu().tolist()
                                                tgt_in = target_class_mask[torch.arange(B, device=device), target_pos.clamp(min=0, max=int(N) - 1)].detach().cpu().tolist()
                                                log_min = float(logits.detach().min().cpu())
                                                log_max = float(logits.detach().max().cpu())
                                                log_mean = float(logits.detach().mean().cpu())
                                                print(
                                                        f"[SSR3DLLM][vigor_stepslot][among_true_dbg] "
                                                        f"N={int(N)} B={int(B)} "
                                                        f"logits(min,max,mean)=({log_min:.3g},{log_max:.3g},{log_mean:.3g}) "
                                                        f"mask_sum={mask_sum[:8]} target_in_mask={tgt_in[:8]}",
                                                        flush=True,
                                                )
                                        except Exception:
                                                pass

                                masked_logits = logits.masked_fill(~target_class_mask, -1e9)
                                pred2 = masked_logits.argmax(dim=-1)
                                got_among_true.append((pred2 == target_pos).detach().cpu().numpy())

                                dbg_logits = str(os.environ.get("SSR3DLLM_VIGOR_STEPSLOT_DEBUG_LOGITS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                if dbg_logits and not getattr(self, "_vigor_stepslot_debug_logits_printed", False):
                                        self._vigor_stepslot_debug_logits_printed = True
                                        try:
                                                i0 = 0
                                                c0 = int(context_size[i0].item())
                                                tgt0 = int(target_pos[i0].item())
                                                row = logits[i0, :c0].detach().cpu()
                                                row_m = masked_logits[i0, :c0].detach().cpu()
                                                topk = min(8, int(row.numel()))
                                                v, idx = torch.topk(row, k=topk)
                                                v2, idx2 = torch.topk(row_m, k=topk)
                                                print(
                                                        f"[SSR3DLLM][vigor_stepslot][logits_dbg] "
                                                        f"use_predmask={int(use_predmask)} c={c0} tgt={tgt0} "
                                                        f"pred={int(pred[i0].item())} pred2={int(pred2[i0].item())} "
                                                        f"tgt_step={int(T - 1)} "
                                                        f"top_idx={idx.tolist()} top_val={[float(x) for x in v.tolist()]} "
                                                        f"top_idx_among={idx2.tolist()} top_val_among={[float(x) for x in v2.tolist()]}",
                                                        flush=True,
                                                )
                                        except Exception:
                                                pass

                                seen += int(B)

                        got_it_right = np.hstack(got) if got else np.zeros((0,), dtype=bool)
                        got_it_right_true = np.hstack(got_among_true) if got_among_true else np.zeros((0,), dtype=bool)

                        if len(got_it_right) != len(refs):
                                # Best-effort: align with evaluated subset.
                                refs = refs.iloc[: len(got_it_right)]

                        hardness = refs.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
                        view_dep_mask = is_explicitly_view_dependent(refs)
                        easy_context_mask = hardness <= 2

                        def _mean(mask) -> float:
                                if len(got_it_right) == 0:
                                        return 0.0
                                arr = got_it_right[mask] if mask is not None else got_it_right
                                return float(arr.mean() * 100.0) if arr.size > 0 else 0.0

                        def _mean_true(mask) -> float:
                                if len(got_it_right_true) == 0:
                                        return 0.0
                                arr = got_it_right_true[mask] if mask is not None else got_it_right_true
                                return float(arr.mean() * 100.0) if arr.size > 0 else 0.0

                        out = {
                                f"val_vigor_stepslot_{name}_hard": _mean(~easy_context_mask.values),
                                f"val_vigor_stepslot_{name}_easy": _mean(easy_context_mask.values),
                                f"val_vigor_stepslot_{name}_v_dep": _mean(view_dep_mask.values),
                                f"val_vigor_stepslot_{name}_v_indep": _mean((~view_dep_mask).values),
                                f"val_vigor_stepslot_{name}_all": _mean(None),
                                f"val_vigor_stepslot_{name}_among_true": _mean_true(None),
                                f"val_vigor_stepslot_{name}_n": float(len(got_it_right)),
                        }
                        print(
                                f"[SSR3DLLM][vigor_stepslot] {name} "
                                f"hard={out[f'val_vigor_stepslot_{name}_hard']:.1f} "
                                f"easy={out[f'val_vigor_stepslot_{name}_easy']:.1f} "
                                f"v-dep={out[f'val_vigor_stepslot_{name}_v_dep']:.1f} "
                                f"v-indep={out[f'val_vigor_stepslot_{name}_v_indep']:.1f} "
                                f"all={out[f'val_vigor_stepslot_{name}_all']:.1f} "
                                f"among-true={out[f'val_vigor_stepslot_{name}_among_true']:.1f} "
                                f"n={int(out[f'val_vigor_stepslot_{name}_n'])}",
                                flush=True,
                        )
                        return out

                metrics: Dict[str, float] = {}
                if Path(sr3d_train_csv).exists():
                        metrics.update(_eval_one("sr3d", sr3d_train_csv))
                if Path(nr3d_train_csv).exists():
                        metrics.update(_eval_one("nr3d", nr3d_train_csv))
                return metrics

        def _run_referit_quick_eval(self) -> Dict[str, float]:
                def _env_flag(name: str, default: str = "0") -> bool:
                        v = os.environ.get(name, default)
                        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

                def _env_int(name: str, default: int = 0) -> int:
                        try:
                                return int(str(os.environ.get(name, str(default))).strip())
                        except Exception:
                                return int(default)

                if self.global_rank != 0:
                        return {}

                scannet_file = os.environ.get("SSR3DLLM_REFERIT_SCANNET_FILE", "").strip()
                if not scannet_file:
                        scannet_file = os.environ.get("REFERIT_SCANNET_FILE", "").strip()
                nr3d_file = os.environ.get("SSR3DLLM_REFERIT_NR3D_FILE", "").strip()
                sr3d_file = os.environ.get("SSR3DLLM_REFERIT_SR3D_FILE", "").strip()
                instance_map_path = os.environ.get("SSR3DLLM_REFERIT_INSTANCE_MAP", "").strip()
                if not instance_map_path:
                        instance_map_path = os.environ.get("REFERIT_INSTANCE_MAP", "").strip()

                if not scannet_file or not instance_map_path:
                        return {}
                if not nr3d_file and not sr3d_file:
                        return {}

                if not getattr(self, "enable_ssr3dllm_geom", False) or not getattr(self, "ssr3dllm_geom_head", None):
                        return {}

                mask3d_root = os.environ.get("SSR3DLLM_REFERIT_QUICK_MASK3D_FEATS", "").strip()
                if not mask3d_root:
                        try:
                                mask3d_root = self.ssr3dllm_geom_head._get_mask3d_feat_root()
                        except Exception:
                                mask3d_root = ""
                if not mask3d_root:
                        return {}
                mask3d_root_path = Path(mask3d_root)
                if not mask3d_root_path.exists():
                        print(f"[SSR3DLLM][referit_quick][warn] mask3d feats not found: {mask3d_root_path}", flush=True)
                        return {}

                use_scannet_boxes = _env_flag(
                        "SSR3DLLM_REFERIT_QUICK_USE_SCANNET_BOXES",
                        os.environ.get("SSR3DLLM_VIGOR_USE_SCANNET_BOXES", "0"),
                )
                use_pred_boxes = _env_flag(
                        "SSR3DLLM_REFERIT_QUICK_USE_PRED_BOX_INFO",
                        os.environ.get("SSR3DLLM_VIGOR_USE_PRED_BOX_INFO", "0"),
                )

                # Load instance->query mapping (required for official ReferIt3D evaluation).
                try:
                        with open(instance_map_path, "rb") as f:
                                inst_map = pickle.load(f)
                except Exception as exc:
                        print(f"[SSR3DLLM][referit_quick][warn] failed to load instance map: {exc}", flush=True)
                        return {}
                if not isinstance(inst_map, dict) or not inst_map:
                        return {}

                vigor = None
                try:
                        vigor = self.ssr3dllm_geom_head._get_vigor_runtime(self.device)
                except Exception:
                        vigor = None
                if vigor is None:
                        return {}

                order_len = int(getattr(vigor, "order_len", 4))

                # Cache scans for GT box_info.
                scannet_scans = None
                if use_scannet_boxes and not use_pred_boxes:
                        try:
                                if not hasattr(self, "_referit_quick_scans"):
                                        from benchmark.referit3d.referit3d.in_out.neural_net_oriented import (
                                                load_scan_related_data,
                                        )
                                        scans, _, _ = load_scan_related_data(scannet_file, verbose=False, add_pad=False)
                                        self._referit_quick_scans = scans
                                scannet_scans = getattr(self, "_referit_quick_scans", None)
                        except Exception:
                                scannet_scans = None

                # Cache ReferIt benchmark adapters to avoid reloading each epoch.
                if not hasattr(self, "_referit_quick_adapters"):
                        self._referit_quick_adapters = {}

                def _get_adapter(referit_file: str):
                        key = (scannet_file, referit_file)
                        if key in self._referit_quick_adapters:
                                return self._referit_quick_adapters[key]
                        from benchmark.unified.referit_adapter import ReferItAdapter, ReferItConfig

                        cfg = ReferItConfig(
                                scannet_file=scannet_file,
                                referit3d_file=referit_file,
                        )
                        adapter = ReferItAdapter(cfg, device=str(self.device), channel_last=True)
                        self._referit_quick_adapters[key] = adapter
                        return adapter

                # Prepare a small inference helper based on Vigor listener.
                feat_cache: Dict[str, dict] = {}

                def _load_feat(scene_id: str) -> Optional[dict]:
                        if scene_id in feat_cache:
                                return feat_cache[scene_id]
                        p = mask3d_root_path / f"{scene_id}.pt"
                        if not p.exists():
                                return None
                        try:
                                feat = torch.load(str(p), map_location="cpu")
                        except Exception:
                                return None
                        if not isinstance(feat, dict):
                                return None
                        feat_cache[scene_id] = feat
                        return feat

                def _build_box_info(scene_id: str, feat: dict, q_expected: int) -> torch.Tensor:
                        # pred boxes first (optional)
                        if use_pred_boxes:
                                pred_box = feat.get("pred_box_info", None)
                                if pred_box is not None:
                                        try:
                                                pb = torch.as_tensor(pred_box, dtype=torch.float32)
                                                if pb.ndim == 2 and int(pb.size(0)) == int(q_expected) and int(pb.size(1)) == 4:
                                                        return pb
                                        except Exception:
                                                pass
                                pred_aabb = feat.get("pred_aabb", None)
                                if pred_aabb is not None:
                                        try:
                                                aabb = torch.as_tensor(pred_aabb, dtype=torch.float32)
                                                if aabb.ndim == 2 and int(aabb.size(0)) == int(q_expected) and int(aabb.size(1)) == 6:
                                                        mn = aabb[:, 0:3]
                                                        mx = aabb[:, 3:6]
                                                        center = (mn + mx) * 0.5
                                                        size = (mx - mn).clamp(min=0.0)
                                                        vol = size[:, 0] * size[:, 1] * size[:, 2]
                                                        box = torch.zeros((int(q_expected), 4), dtype=torch.float32)
                                                        box[:, 0:3] = center
                                                        box[:, 3] = vol
                                                        return box
                                        except Exception:
                                                pass
                        # GT box_info from ScanNet (if available)
                        if use_scannet_boxes and isinstance(scannet_scans, dict):
                                if scene_id and scene_id in scannet_scans:
                                        gt_map = feat.get("gt_to_query_map", None)
                                        if isinstance(gt_map, dict):
                                                scan = scannet_scans[scene_id]
                                                box = torch.zeros((int(q_expected), 4), dtype=torch.float32)
                                                for inst_id, qidx in gt_map.items():
                                                        try:
                                                                qi = int(qidx)
                                                                ii = int(inst_id)
                                                        except Exception:
                                                                continue
                                                        if not (0 <= qi < int(q_expected)):
                                                                continue
                                                        try:
                                                                obj = scan.three_d_objects[int(ii)]
                                                                bb = obj.get_bbox()
                                                                box[qi, 0] = float(bb.cx)
                                                                box[qi, 1] = float(bb.cy)
                                                                box[qi, 2] = float(bb.cz)
                                                                box[qi, 3] = float(bb.volume())
                                                        except Exception:
                                                                continue
                                                return box
                        # fallback: sampled coords -> center + volume=1
                        coords = feat.get("sampled_coords", None)
                        try:
                                coords_t = torch.as_tensor(coords, dtype=torch.float32) if coords is not None else None
                        except Exception:
                                coords_t = None
                        box = torch.zeros((int(q_expected), 4), dtype=torch.float32)
                        if torch.is_tensor(coords_t) and coords_t.ndim == 2 and int(coords_t.size(0)) == int(q_expected) and int(coords_t.size(1)) == 3:
                                box[:, :3] = coords_t
                        box[:, 3] = 1.0
                        return box

                def _infer(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
                        from benchmark.referit3d.referit3d.data_generation.nr3d.stimuli_generation import (
                                SameInstanceStimulus,
                        )

                        context_size = batch["context_size"]
                        B = int(context_size.shape[0])
                        max_ctx = int(context_size.max().item())
                        logits = torch.full((B, max_ctx), -1e9, device=batch["context_size"].device)

                        if "stimulus_id" not in batch or "utterance" not in batch or "object_ids" not in batch:
                                return logits

                        stimulus_ids = batch["stimulus_id"]
                        utterances = batch["utterance"]
                        object_ids = batch["object_ids"]
                        if torch.is_tensor(object_ids):
                                object_ids = object_ids.cpu().numpy()

                        for i in range(B):
                                try:
                                        scene_id, _, _, _, _ = SameInstanceStimulus.decode_stimulus_string(stimulus_ids[i])
                                except Exception:
                                        continue
                                scene_map = inst_map.get(scene_id, None)
                                if not scene_map:
                                        continue
                                feat = _load_feat(scene_id)
                                if not isinstance(feat, dict):
                                        continue
                                oq = feat.get("object_queries", None)
                                if not torch.is_tensor(oq) or oq.ndim != 2:
                                        continue
                                Q = int(oq.shape[0])
                                text = utterances[i]
                                text = str(text).replace("<geom>", "").strip()
                                if not text:
                                        continue
                                box_info = _build_box_info(scene_id, feat, Q).to(device=self.device, dtype=torch.float32)
                                pred_mask = torch.ones((int(order_len), int(Q)), device=self.device, dtype=torch.float32)
                                order_texts = [text] * int(order_len)
                                try:
                                        q_logits = vigor.predict_logits(
                                                text=text,
                                                mask3d_queries=oq.to(device=self.device, dtype=torch.float32),
                                                box_info=box_info,
                                                order_texts=order_texts,
                                                pred_class_mask=pred_mask,
                                        )
                                except Exception:
                                        continue
                                if not torch.is_tensor(q_logits) or q_logits.numel() != int(Q):
                                        continue
                                n_ctx = int(context_size[i].item())
                                for j in range(n_ctx):
                                        try:
                                                oid = int(object_ids[i][j])
                                        except Exception:
                                                continue
                                        entry = scene_map.get(oid, None)
                                        if not entry:
                                                continue
                                        qidx = entry.get("best_query", None)
                                        try:
                                                qi = int(qidx)
                                        except Exception:
                                                continue
                                        if not (0 <= qi < int(Q)):
                                                continue
                                        logits[i, j] = q_logits[qi]
                        return logits

                max_examples = _env_int("SSR3DLLM_REFERIT_QUICK_MAX_EXAMPLES", 0)
                old_max = os.environ.get("REFERIT_MAX_EXAMPLES", None)
                if max_examples > 0:
                        os.environ["REFERIT_MAX_EXAMPLES"] = str(max_examples)

                metrics: Dict[str, float] = {}
                for name, path in (("nr3d", nr3d_file), ("sr3d", sr3d_file)):
                        if not path:
                                continue
                        adapter = _get_adapter(path)
                        summary = adapter.evaluate(_infer)
                        if not isinstance(summary, dict):
                                continue
                        metrics[f"val_referit_quick_{name}_acc"] = float(summary.get("overall_acc", 0.0))
                        metrics[f"val_referit_quick_{name}_nr3d_acc"] = float(summary.get("nr3d_acc", 0.0))
                        metrics[f"val_referit_quick_{name}_sr3d_acc"] = float(summary.get("sr3d_acc", 0.0))
                        metrics[f"val_referit_quick_{name}_n_examples"] = float(summary.get("n_examples", 0.0))
                        print(
                                f"[SSR3DLLM][referit_quick] {name} overall_acc={metrics[f'val_referit_quick_{name}_acc']:.4f} "
                                f"nr3d_acc={metrics[f'val_referit_quick_{name}_nr3d_acc']:.4f} "
                                f"sr3d_acc={metrics[f'val_referit_quick_{name}_sr3d_acc']:.4f} "
                                f"n={int(metrics[f'val_referit_quick_{name}_n_examples'])}",
                                flush=True,
                        )

                if old_max is not None:
                        os.environ["REFERIT_MAX_EXAMPLES"] = old_max
                elif "REFERIT_MAX_EXAMPLES" in os.environ:
                        os.environ.pop("REFERIT_MAX_EXAMPLES", None)

                return metrics

        def configure_optimizers(self):
                # SSR3DLLM + Vigor backend:
                # The Vigor listener is created lazily (to avoid importing heavy deps in baseline runs).
                # However, if we want to fine-tune it (SSR3DLLM_VIGOR_FINETUNE=1), it MUST exist
                # before optimizer construction, otherwise its parameters will not be included in
                # the optimizer param groups and only BN running stats will change.
                try:
                        if getattr(self, "enable_ssr3dllm_geom", False) and getattr(self, "ssr3dllm_geom_head", None):
                                backend = str(self.ssr3dllm_geom_head._get_geom_backend()).strip().lower()
                                finetune = str(os.environ.get("SSR3DLLM_VIGOR_FINETUNE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                train_step_proj = str(os.environ.get("SSR3DLLM_VIGOR_TRAIN_STEP_PROJ", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
                                if backend == "vigor" and (finetune or train_step_proj):
                                        # IMPORTANT for DDP:
                                        # DDP syncs buffers at the start of the first forward and requires them to be
                                        # CUDA dense tensors. If we instantiate the Vigor runtime on CPU here, its
                                        # buffers will be CPU tensors and DDP will crash with:
                                        #   RuntimeError: Tensors must be CUDA and dense
                                        # Therefore, instantiate/move the runtime onto the module's current device.
                                        try:
                                                dev = next(self.parameters()).device
                                        except Exception:
                                                dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                                        runtime = self.ssr3dllm_geom_head._get_vigor_runtime(dev)
                                        # If we train the SSR3DLLM-side step projection (without finetuning the runtime),
                                        # ensure it is materialized BEFORE optimizer construction, otherwise it won't be
                                        # included in param groups.
                                        if train_step_proj:
                                                try:
                                                        inner_dim = int(getattr(getattr(runtime, "model", None), "inner_dim", 768))
                                                except Exception:
                                                        inner_dim = 768
                                                _ = self.ssr3dllm_geom_head._get_vigor_step_proj(inner_dim, dev)
                except Exception as e:
                        # Fail fast: if the user explicitly asked to finetune Vigor but we cannot
                        # instantiate it at optimizer-build time, training will silently not update it.
                        raise RuntimeError(f"[SSR3DLLM][vigor_train] failed to initialize Vigor runtime before optimizer: {e}") from e

                # Optional: use a dedicated learning rate for the Vigor listener.
                # Enable with:
                #   export SSR3DLLM_VIGOR_LISTENER_LR=1e-5
                vigor_listener_lr_raw = str(os.environ.get("SSR3DLLM_VIGOR_LISTENER_LR", "")).strip()
                vigor_listener_lr = None
                if vigor_listener_lr_raw:
                        try:
                                vigor_listener_lr = float(vigor_listener_lr_raw)
                        except Exception:
                                vigor_listener_lr = None

                # Optional: use a dedicated learning rate for the SSR3DLLM-side step projection.
                # Enable with:
                #   export SSR3DLLM_VIGOR_STEP_PROJ_LR=1e-5
                step_proj_lr_raw = str(os.environ.get("SSR3DLLM_VIGOR_STEP_PROJ_LR", "")).strip()
                step_proj_lr = None
                if step_proj_lr_raw:
                        try:
                                step_proj_lr = float(step_proj_lr_raw)
                        except Exception:
                                step_proj_lr = None

                vigor_trainable = []
                vigor_param_ids = set()
                if vigor_listener_lr is not None and vigor_listener_lr > 0:
                        try:
                                vigor_trainable = [
                                        (n, p) for n, p in self.named_parameters()
                                        if "ssr3dllm_geom_head._vigor_runtime." in n and getattr(p, "requires_grad", False)
                                ]
                                vigor_param_ids = {id(p) for _, p in vigor_trainable}
                        except Exception:
                                vigor_trainable = []
                                vigor_param_ids = set()

                step_proj_trainable = []
                step_proj_param_ids = set()
                if step_proj_lr is not None and step_proj_lr > 0:
                        try:
                                step_proj_trainable = [
                                        (n, p) for n, p in self.named_parameters()
                                        if "ssr3dllm_geom_head._vigor_step_proj." in n and getattr(p, "requires_grad", False)
                                ]
                                step_proj_param_ids = {id(p) for _, p in step_proj_trainable}
                        except Exception:
                                step_proj_trainable = []
                                step_proj_param_ids = set()

                excluded_param_ids = set(vigor_param_ids) | set(step_proj_param_ids)
                other_params = [p for n, p in self.named_parameters()
                                if 'language_model' not in n and id(p) not in excluded_param_ids]
                lang_params = [p for n, p in self.named_parameters()
                               if 'language_model' in n and id(p) not in excluded_param_ids]
                weight_decay = 1e-4

                params = [
                        {'params'      : other_params, "lr": self.config.optimizer.lr,
                         "weight_decay": weight_decay},
                        {'params': lang_params, "lr": self.config.optimizer.lr * 0.1,
                         "weight_decay": weight_decay},
                ]
                if vigor_trainable and vigor_listener_lr is not None and vigor_listener_lr > 0:
                        params.insert(
                                0,
                                {
                                        "params": [p for _, p in vigor_trainable],
                                        "lr": vigor_listener_lr,
                                        "weight_decay": weight_decay,
                                },
                        )
                if step_proj_trainable and step_proj_lr is not None and step_proj_lr > 0:
                        params.insert(
                                0,
                                {
                                        "params": [p for _, p in step_proj_trainable],
                                        "lr": step_proj_lr,
                                        "weight_decay": weight_decay,
                                },
                        )

                optimizer = instantiate(
                        self.config.optimizer, params=params
                )

                # Optional sanity print: confirm Vigor listener params are included in optimizer.
                # Enable with: export SSR3DLLM_DEBUG_OPTIM_VIGOR=1
                if str(os.environ.get("SSR3DLLM_DEBUG_OPTIM_VIGOR", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}:
                        try:
                                vigor_named = [
                                        (n, p) for n, p in self.named_parameters()
                                        if "ssr3dllm_geom_head._vigor_runtime." in n
                                ]
                                vigor_trainable_dbg = [(n, p) for n, p in vigor_named if getattr(p, "requires_grad", False)]
                                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g.get("params", [])}
                                included = sum(1 for _, p in vigor_trainable_dbg if id(p) in opt_param_ids)
                                lr_dbg = f"{vigor_listener_lr:.3g}" if (vigor_listener_lr is not None) else "<default>"
                                print(
                                        "[SSR3DLLM][vigor_train][optim_dbg] "
                                        f"vigor_params_total={len(vigor_named)} "
                                        f"vigor_params_trainable={len(vigor_trainable_dbg)} "
                                        f"vigor_params_in_optimizer={included} "
                                        f"vigor_listener_lr={lr_dbg}",
                                        flush=True,
                                )
                        except Exception as e:
                                print(f"[SSR3DLLM][vigor_train][optim_dbg][warn] {type(e).__name__}: {e}", flush=True)

                if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
                        self.config.scheduler.scheduler.steps_per_epoch = len(
                                self.train_dataloader()
                        )
                lr_scheduler = instantiate(
                        self.config.scheduler.scheduler, optimizer=optimizer
                )
                scheduler_config = {"scheduler": lr_scheduler}
                scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
                return [optimizer], [scheduler_config]

        def train_dataloader(self):
                c_fn = instantiate(self.config.data.train_collation)
                if self.config.general.gpus > 1:
                        sampler = torch.utils.data.distributed.DistributedSampler(
                                self.train_dataset, shuffle=True)
                        self.config.data.train_dataloader.shuffle = False
                return instantiate(
                        self.config.data.train_dataloader,
                        self.train_dataset,
                        collate_fn=c_fn,
                        sampler=sampler if self.config.general.gpus > 1 else None
                )

        def val_dataloader(self):
                c_fn = instantiate(self.config.data.validation_collation)
                if self.config.general.gpus > 1:
                        sampler = torch.utils.data.distributed.DistributedSampler(
                                self.validation_dataset, shuffle=False)
                return instantiate(
                        self.config.data.validation_dataloader,
                        self.validation_dataset,
                        collate_fn=c_fn,
                        sampler=sampler if self.config.general.gpus > 1 else None
                )

        def test_dataloader(self):
                c_fn = instantiate(self.config.data.test_collation)
                if self.config.general.gpus > 1:
                        sampler = torch.utils.data.distributed.DistributedSampler(
                                self.test_dataset, shuffle=False)
                return instantiate(
                        self.config.data.test_dataloader,
                        self.test_dataset,
                        collate_fn=c_fn,
                        sampler=sampler if self.config.general.gpus > 1 else None
                )
