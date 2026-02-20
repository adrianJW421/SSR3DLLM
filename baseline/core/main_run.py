import argparse
import sys
import traceback
import logging
import os
import signal
import time
from hashlib import md5
from uuid import uuid4
import yaml

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import MinkowskiEngine as ME

from config import (
    task_config,
    config_to_dict,
    instantiate,
    apply_overrides,
    refresh_links,
    clone_config,
    load_yaml_config,
)
from baseline.core.trainer.trainer import ModelingGrounded3DLLM
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
)

def _maybe_patch_lightning_cuda_device_count() -> None:
        """
        Some PyTorch Lightning versions call `device_parser.num_cuda_devices()` which
        internally spawns a `multiprocessing.Pool(..., context='fork')` to query CUDA.

        In some cluster environments, that fork-based probe can hang or get killed
        (e.g., by a watchdog), causing `trainer.test()` to terminate before any
        evaluation runs.

        Set `SSR3DLLM_PL_DISABLE_FORK_DEVICE_COUNT=1` to replace the probe with a
        direct `torch.cuda.device_count()` call.
        """
        enabled = os.environ.get("SSR3DLLM_PL_DISABLE_FORK_DEVICE_COUNT", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
        }
        if not enabled:
                return

        try:
                import torch  # type: ignore
                from pytorch_lightning.utilities import device_parser  # type: ignore

                def _num_cuda_devices_direct() -> int:
                        try:
                                return int(torch.cuda.device_count())
                        except Exception:
                                return 0

                device_parser.num_cuda_devices = _num_cuda_devices_direct  # type: ignore[attr-defined]
                print("[SSR3DLLM] Patched PL device_parser.num_cuda_devices (no fork).")
        except Exception as exc:
                print(f"[SSR3DLLM] WARN: failed to patch PL num_cuda_devices: {exc}")


def _destroy_process_group_best_effort() -> None:
        try:
                import torch.distributed as dist  # type: ignore
                if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
        except Exception:
                pass


def _kill_lightning_ddp_children_best_effort(trainer: "Trainer") -> None:
        """
        PyTorch Lightning's DDP ("strategy=ddp") launches N-1 child processes from rank0.
        If rank0 is killed (or Ctrl+C is swallowed by the shell pipeline), those children
        can keep occupying GPUs and appear as "zombie" training jobs.

        This function attempts to terminate all spawned subprocesses in a version-tolerant
        way, without depending on a specific Lightning internal API.
        """
        try:
                strategy = getattr(trainer, "strategy", None)
                launcher = getattr(strategy, "launcher", None) if strategy is not None else None
                if launcher is None and strategy is not None:
                        launcher = getattr(strategy, "_launcher", None)

                # Preferred: PL launcher exposes a kill() helper.
                kill_fn = getattr(launcher, "kill", None) if launcher is not None else None
                if callable(kill_fn):
                        try:
                                kill_fn()
                                return
                        except TypeError:
                                # Some versions have different signatures; fall back to manual.
                                pass

                # Fallback: try to access stored subprocess handles.
                procs = []
                if launcher is not None:
                        for attr in ("processes", "_processes", "children", "_children", "subprocesses", "_subprocesses"):
                                v = getattr(launcher, attr, None)
                                if isinstance(v, (list, tuple)):
                                        procs.extend([p for p in v if hasattr(p, "pid")])

                # Graceful terminate then hard kill.
                for p in procs:
                        try:
                                p.terminate()
                        except Exception:
                                pass
                if procs:
                        time.sleep(2.0)
                for p in procs:
                        try:
                                p.kill()
                        except Exception:
                                pass
        except Exception:
                # Best-effort only; never let cleanup raise.
                pass


def _install_fast_signal_exit(trainer: "Trainer") -> None:
        """
        Make Ctrl+C / SIGTERM reliably stop multi-GPU Lightning jobs.

        - On SIGINT/SIGTERM: try to kill DDP children + destroy process group, then
          hard-exit to avoid hanging inside NCCL barriers or Lightning teardown.
        """
        enabled = os.environ.get("SSR3DLLM_FAST_SIGNAL_EXIT", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
        }
        if not enabled:
                return

        hard_exit = os.environ.get("SSR3DLLM_FAST_SIGNAL_HARD_EXIT", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
        }

        # Only hard-exit for multi-process runs; for single-process, prefer a graceful exit
        # so logs flush and errors are visible.
        #
        # NOTE: Lightning's public attributes changed across versions (e.g. `trainer.gpus`
        # may not exist). Prefer env WORLD_SIZE and fall back to several trainer fields.
        world_size = 1
        try:
                world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        except Exception:
                world_size = 1
        if world_size <= 0:
                world_size = 1

        num_devices = None
        for attr in ("num_devices", "devices", "num_gpus", "gpus"):
                try:
                        v = getattr(trainer, attr, None)
                        if v is None:
                                continue
                        if isinstance(v, (list, tuple)):
                                num_devices = len(v)
                                break
                        num_devices = int(v)
                        break
                except Exception:
                        continue
        if num_devices is None:
                num_devices = 1
        if num_devices <= 0:
                num_devices = 1

        is_multiprocess = bool(world_size > 1 or num_devices > 1)
        hard_exit = bool(hard_exit and is_multiprocess)

        state = {"handled": False}

        def _handler(signum, frame):  # pragma: no cover - runtime behavior
                if state["handled"]:
                        # Second signal: exit immediately.
                        os._exit(1)
                state["handled"] = True
                print(f"[Signal] Received {signal.Signals(signum).name}; terminating all DDP workers...")
                try:
                        traceback.print_stack(frame)
                        sys.stdout.flush()
                        sys.stderr.flush()
                except Exception:
                        pass
                _kill_lightning_ddp_children_best_effort(trainer)
                _destroy_process_group_best_effort()
                if hard_exit:
                        # Hard-exit: avoid deadlocks in teardown/barrier for multi-process.
                        os._exit(1)
                raise SystemExit(1)

        try:
                signal.signal(signal.SIGINT, _handler)
                signal.signal(signal.SIGTERM, _handler)
        except Exception:
                # Some environments disallow overriding handlers; ignore.
                pass


class RegularCheckpointing(pl.Callback):
        def on_train_epoch_end(
                        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ):
                general = pl_module.config.general
                # Only let global-rank 0 write checkpoints; other ranks would race/corrupt the file.
                if getattr(trainer, "global_rank", 0) != 0:
                        return

                # Atomic save: write to a temp file then replace. This prevents partially-written
                # `last-epoch.ckpt` when the job is interrupted (SIGINT/SIGTERM) during save.
                ckpt_path = f"{general.save_dir}/last-epoch.ckpt"
                tmp_path = ckpt_path + ".tmp"
                trainer.save_checkpoint(tmp_path)
                os.replace(tmp_path, ckpt_path)
                print("Checkpoint created")


def get_parameters(cfg):
        logger = logging.getLogger(__name__)
        load_dotenv(".env")

        seed_everything(cfg.general.seed)

        if cfg.general.get("gpus", None) is None:
                env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if env_gpus is not None:
                        if "," in env_gpus:
                                cfg.general.gpus = len([gpu for gpu in env_gpus.split(",") if gpu.strip()])
                        else:
                                try:
                                        cfg.general.gpus = int(env_gpus)
                                except ValueError:
                                        cfg.general.gpus = 1
                else:
                        cfg.general.gpus = 1

        if isinstance(cfg.general.gpus, str):
                if "," in cfg.general.gpus:
                        cfg.general.gpus = len([gpu for gpu in cfg.general.gpus.split(",") if gpu.strip()])
                else:
                        try:
                                cfg.general.gpus = int(cfg.general.gpus)
                        except ValueError:
                                cfg.general.gpus = 1

        loggers = []

        cfg.general.experiment_id = "0"  # str(Repo(\"./\").commit())[:8]
        params = flatten_dict(config_to_dict(cfg))

        unique_id = "_" + str(uuid4())[:4]
        cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

        specified_ckpt = cfg.general.checkpoint
        if specified_ckpt and os.path.isfile(specified_ckpt):
                # NOTE:
                # When user explicitly specifies `general.checkpoint` (e.g. using
                # a Step-1 detector checkpoint to initialize Step-2/3), we do
                # *not* set `trainer.resume_from_checkpoint` here. Instead, we
                # rely on `load_checkpoint_with_missing_or_exsessive_keys`
                # below to load weights in a tolerant way. Letting Lightning
                # also try to "resume" from this checkpoint would trigger a
                # strict state_dict load and crash whenever the architectures
                # do not exactly match (typical for cross-stage initialization).
                last_ckpt = specified_ckpt
        else:
                if not os.path.exists(cfg.general.save_dir):
                        os.makedirs(cfg.general.save_dir)
                last_ckpt = f"{cfg.general.save_dir}/last-epoch.ckpt"
                if os.path.isfile(last_ckpt):
                        # This is the standard "resume training from last epoch
                        # of the same experiment" case; here the Lightning
                        # checkpoint and current architecture should match.
                        cfg["trainer"]["resume_from_checkpoint"] = last_ckpt
                        print(f"Load weights from: {last_ckpt}")
                        cfg.general.checkpoint = last_ckpt
                elif specified_ckpt:
                        print(f"Checkpoint not found at {specified_ckpt}")
                        cfg.general.checkpoint = None
                else:
                        print("Note that *No* checkpoint is found.")

        flat_cfg = flatten_dict(config_to_dict(cfg))

        # Explicit log for Rel3D geometry-aware query enhancement so that
        # it is easy to verify from stdout whether the relation feature
        # branch is enabled in the current run.
        rel3d_flag = getattr(cfg.model, "use_rel3d_geom", None)
        if rel3d_flag is not None:
                rel3d_num_dirs = getattr(cfg.model, "rel3d_num_dirs", "n/a")
                rel3d_sigma = getattr(cfg.model, "rel3d_sigma", "n/a")
                rel3d_mu_scale = getattr(cfg.model, "rel3d_mu_scale", "n/a")
                print(
                        f"[Rel3D] geometry enhancement: use_rel3d_geom={rel3d_flag}, "
                        f"num_dirs={rel3d_num_dirs}, sigma={rel3d_sigma}, mu_scale={rel3d_mu_scale}"
                )

        # Log and optionally enable GT proposal / oracle proposal experiment.
        gt_prop_flag = getattr(cfg.general, "use_gt_proposals_for_llm", False)
        print(f"[OracleProposal] use_gt_proposals_for_llm={gt_prop_flag}")

        # Optionally dump the full runtime config to YAML for reproducibility.
        if getattr(cfg.general, "save_runtime_config", True):
                # Ensure save_dir exists even when we are initializing from a
                # cross-stage checkpoint (specified_ckpt branch above).
                if not os.path.exists(cfg.general.save_dir):
                        os.makedirs(cfg.general.save_dir, exist_ok=True)
                runtime_cfg_path = os.path.join(cfg.general.save_dir, "runtime_cfg.yaml")
                try:
                        with open(runtime_cfg_path, "w", encoding="utf-8") as f:
                                yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)
                        print(f"[RuntimeConfig] Saved runtime config to: {runtime_cfg_path}")
                except Exception as exc:  # pragma: no cover - best-effort logging
                        print(f"[RuntimeConfig] Failed to save runtime config: {exc}")

        for log in cfg.logging:
                logger_instance = instantiate(log)
                # PL TensorBoardLogger has version-dependent behavior: when `metrics` is omitted,
                # some releases attempt to log a non-dict sentinel (e.g. -1), which raises.
                # Always pass a dict to make this robust across environments.
                try:
                        logger_instance.log_hyperparams(flat_cfg, metrics={"hp_metric": 0.0})
                except TypeError:
                        # Backward compatibility for loggers that don't accept `metrics=`.
                        logger_instance.log_hyperparams(flat_cfg)
                loggers.append(logger_instance)


        model = ModelingGrounded3DLLM(cfg)
        if cfg.general.gpus and cfg.general.gpus > 1:
                model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.general.checkpoint is not None:
                print(f"Loading checkpoint from: {cfg.general.checkpoint}")
                cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

                logger.info(flat_cfg)
                return cfg, model, loggers

        # Return config and model even when no checkpoint is specified.
        logger.info(flat_cfg)
        return cfg, model, loggers


def run_train(cfg):
        cfg, model, loggers = get_parameters(cfg)
        callbacks = [instantiate(cb) for cb in cfg.callbacks]
        callbacks.append(RegularCheckpointing())

        # Patch Lightning CUDA device count probe *before* Trainer/strategy setup.
        # Some PL versions call `device_parser.num_cuda_devices()` during strategy init,
        # which may spawn a fork-based multiprocessing probe and get killed on clusters.
        _maybe_patch_lightning_cuda_device_count()

        # DDP robustness:
        # Some training regimes (e.g. step-token / rel3dref SFT) may drop all candidate samples
        # on a given rank for a step (e.g. filtering/truncation), causing "unused parameters"
        # and potential DDP deadlocks when `find_unused_parameters=False`.
        # Allow enabling the safer DDP variant via env flag to avoid hanging multi-GPU jobs.
        strategy = None
        if cfg.general.gpus and cfg.general.gpus > 1:
                ddp_find_unused = os.environ.get("SSR3DLLM_DDP_FIND_UNUSED", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                }
                if ddp_find_unused:
                        # NOTE: Different PyTorch Lightning versions expose different strategy
                        # registry names. Some (e.g. PL 1.7) do not register
                        # "ddp_find_unused_parameters_true", only the *_false variants.
                        # Using an explicit DDPStrategy instance is the most compatible way.
                        try:
                                from pytorch_lightning.strategies import DDPStrategy  # type: ignore

                                strategy = DDPStrategy(find_unused_parameters=True)
                        except Exception:
                                # Fallback to default ddp (may still work depending on PL version).
                                strategy = "ddp"
                else:
                        strategy = "ddp"

        trainer = Trainer(
                logger=loggers,
                gpus=cfg.general.gpus,
                accelerator="gpu" if cfg.general.gpus and cfg.general.gpus > 1 else None,
                strategy=strategy,
                callbacks=callbacks,
                weights_save_path=str(cfg.general.save_dir),
                **config_to_dict(cfg.trainer),
        )
        _install_fast_signal_exit(trainer)
        try:
                trainer.fit(model)
        finally:
                # Ensure we never leave distributed state around if the process exits early.
                _kill_lightning_ddp_children_best_effort(trainer)
                _destroy_process_group_best_effort()


def run_test(cfg):
        cfg, model, loggers = get_parameters(cfg)
        strategy = None
        if cfg.general.gpus and cfg.general.gpus > 1:
                ddp_find_unused = os.environ.get("SSR3DLLM_DDP_FIND_UNUSED", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                }
                if ddp_find_unused:
                        try:
                                from pytorch_lightning.strategies import DDPStrategy  # type: ignore

                                strategy = DDPStrategy(find_unused_parameters=True)
                        except Exception:
                                strategy = "ddp"
                else:
                        strategy = "ddp"
        # Patch Lightning CUDA device count probe *before* Trainer/strategy setup.
        _maybe_patch_lightning_cuda_device_count()
        trainer = Trainer(
                gpus=cfg.general.gpus,
                logger=loggers,
                accelerator="gpu" if cfg.general.gpus and cfg.general.gpus > 1 else None,
                strategy=strategy,
                weights_save_path=str(cfg.general.save_dir),
                **config_to_dict(cfg.trainer),
        )
        _install_fast_signal_exit(trainer)
        try:
                trainer.test(model)
        finally:
                _kill_lightning_ddp_children_best_effort(trainer)
                _destroy_process_group_best_effort()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Grounded 3D LLM training or evaluation without Hydra."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        help="Override `general.train_mode` from config.py.",
    )
    parser.add_argument("--gpus", type=int, help="Override number of GPUs.")
    parser.add_argument("--checkpoint", help="Path to model checkpoint.")
    parser.add_argument("--data-config", help="YAML config to override data settings.")
    parser.add_argument("--model-config", help="YAML config to override model settings.")
    parser.add_argument("--trainer-config", help="YAML config to override trainer settings.")
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Override `general.llm_config` path. Defaults to the value defined in config.py.",
    )
    parser.add_argument("--llm-data-config", help="Override `general.llm_data_config` path.")
    args, unknown = parser.parse_known_args()

    cfg = clone_config()

    # Parse CLI overrides first, but apply them only after YAML configs
    # are loaded so that command-line flags take precedence.
    cli_overrides = _parse_unknown_overrides(unknown)
    if args.mode is not None:
        cli_overrides["general.train_mode"] = args.mode == "train"
    if args.gpus is not None:
        cli_overrides["general.gpus"] = args.gpus
    if args.checkpoint:
        cli_overrides["general.checkpoint"] = args.checkpoint
    if args.llm_config:
        cli_overrides["general.llm_config"] = args.llm_config
    if args.llm_data_config:
        cli_overrides["general.llm_data_config"] = args.llm_data_config
    if args.data_config:
        apply_overrides(cfg.data, load_yaml_config(args.data_config, cfg))
    if args.model_config:
        apply_overrides(cfg.model, load_yaml_config(args.model_config, cfg))
    if args.trainer_config:
        apply_overrides(cfg.trainer, load_yaml_config(args.trainer_config, cfg))

    if cli_overrides:
        apply_overrides(cfg, cli_overrides)

    refresh_links(cfg)

    if cfg.general.train_mode:
        run_train(cfg)
    else:
        run_test(cfg)


def _parse_unknown_overrides(unknown_args):
        """
        Translate CLI overrides of the form
        `--general.experiment_name sanity_eval --optimizer.lr 0.0001`
        (or `--general.experiment_name=sanity_eval`) into a dictionary
        that `apply_overrides` understands.
        """
        overrides = {}
        it = iter(unknown_args)
        for token in it:
                if not token.startswith("--"):
                        raise ValueError(f"Unexpected argument '{token}'. Use --key value pairs.")

                key_token = token[2:]
                if "=" in key_token:
                        key, value = key_token.split("=", 1)
                else:
                        try:
                                value = next(it)
                        except StopIteration as exc:
                                raise ValueError(f"Missing value for override '{token}'.") from exc
                        if value.startswith("--"):
                                raise ValueError(f"Missing value for override '{token}'.")
                        key = key_token

                overrides[key] = _convert_literal(value)
        return overrides


def _convert_literal(value: str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
                return lowered == "true"
        if lowered in {"none", "null"}:
                return None
        try:
                if "." in value:
                        return float(value)
                return int(value)
        except ValueError:
                return value


if __name__ == "__main__":
        main()
