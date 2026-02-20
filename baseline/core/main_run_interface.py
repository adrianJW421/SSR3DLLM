import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from baseline.api.baseline_interface import Grounded3DLLMBaselineInterface


def _resolve_scene_ids(
                interface: Grounded3DLLMBaselineInterface,
                requested_scene: Optional[str],
                limit: int,
) -> List[str]:
        """
        Determine which scenes to evaluate.

        If ``requested_scene`` is provided we only run that one. Otherwise we take
        the first ``limit`` unique scene ids following the dataset ordering so that
        the batch matches the Lightning dataloader sequence.
        """
        if requested_scene:
                return [requested_scene]

        seen = set()
        ordered_scenes: List[str] = []
        for entry in interface.dataset.data:
                scene_id = Path(entry["instance_gt_filepath"]).stem
                if scene_id not in seen:
                        seen.add(scene_id)
                        ordered_scenes.append(scene_id)
                if len(ordered_scenes) >= limit:
                        break

        if not ordered_scenes:
                raise RuntimeError("Unable to resolve any scenes from the dataset metadata.")

        return ordered_scenes


def _save_prediction(prediction: Dict[str, object], destination: Path) -> None:
        """
        Persist the baseline interface prediction using the same tensor layout as
        the official evaluation loop so downstream comparisons can simply load both
        npz bundles and diff them element-wise.
        """
        payload: Dict[str, np.ndarray] = {}

        payload["pred_masks"] = np.asarray(prediction["pred_masks"], dtype=bool)
        payload["pred_scores"] = np.asarray(prediction["pred_scores"], dtype=np.float32)
        payload["pred_classes"] = np.asarray(prediction["pred_classes"], dtype=np.int64)

        if "gt_ious" in prediction and prediction["gt_ious"] is not None:
                gt_ious = prediction["gt_ious"]
                if isinstance(gt_ious, (list, tuple)) and len(gt_ious) >= 5:
                        payload["gt_ious_mask"] = np.asarray(gt_ious[0], dtype=np.float32)
                        payload["gt_eval_types"] = np.asarray(gt_ious[1], dtype=object)
                        payload["gt_bbox_iou"] = np.asarray(gt_ious[2], dtype=np.float32)
                        payload["gt_multi_iou25_f1"] = np.asarray(gt_ious[3], dtype=np.float32)
                        payload["gt_multi_iou50_f1"] = np.asarray(gt_ious[4], dtype=np.float32)
                else:
                        payload["gt_ious_raw"] = np.asarray(gt_ious, dtype=object)

        if "bbox_preds" in prediction:
                payload["bbox_preds"] = np.asarray(prediction["bbox_preds"], dtype=object)

        np.savez_compressed(destination, **payload)


def main(args: Optional[Iterable[str]] = None) -> None:
        parser = argparse.ArgumentParser(
                description=(
                        "Mirror the official Lightning evaluation pipeline via the "
                        "Grounded3DLLM baseline interface. The script reproduces the "
                        "command-line run used for `main_run.py` but limits execution to "
                        "a single batch and stores the predictions for side-by-side checks."
                )
        )
        parser.add_argument("--checkpoint", default="saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt")
        parser.add_argument("--data-config", default="baseline/core/conf/data/indoor_dialog.yaml")
        parser.add_argument("--model-config", default="baseline/core/conf/model/mask3d_lang.yaml")
        parser.add_argument("--trainer-config", default="baseline/core/conf/trainer/trainer50.yaml")
        parser.add_argument("--llm-config", default="baseline/core/conf/llm/tiny_vicuna_len512_bs4.json")
        parser.add_argument("--llm-data-config", default="baseline/core/conf/llm/det10.json")
        parser.add_argument("--experiment-name", default="sanity_eval_interface")
        parser.add_argument("--project-name", default="scannet")
        parser.add_argument("--data-split", default="validation", choices=["train", "validation", "test"])
        parser.add_argument("--device", default=None, help="Torch device string; defaults to CUDA when available.")
        parser.add_argument("--topk-per-image", type=int, default=100)
        parser.add_argument(
                "--limit-scenes",
                type=int,
                default=1,
                help="Maximum number of scenes to evaluate (mirrors limit_test_batches for batch size 1).",
        )
        parser.add_argument(
                "--scene-id",
                default=None,
                help="Optional explicit scene id. When omitted we follow dataset ordering.",
        )
        parser.add_argument(
                "--output-dir",
                default=None,
                help="Directory where predictions will be stored. "
                     "Defaults to saved/<experiment-name>/baseline_interface_outputs.",
        )

        cli_args = parser.parse_args(args=args)

        interface = Grounded3DLLMBaselineInterface(
                checkpoint=cli_args.checkpoint,
                data_split=cli_args.data_split,
                data_config=cli_args.data_config,
                model_config=cli_args.model_config,
                trainer_config=cli_args.trainer_config,
                experiment_name=cli_args.experiment_name,
                project_name=cli_args.project_name,
                device=cli_args.device,
                llm_config=cli_args.llm_config,
                llm_data_config=cli_args.llm_data_config,
                topk_per_image=cli_args.topk_per_image,
        )

        output_dir = (
                Path(cli_args.output_dir)
                if cli_args.output_dir is not None
                else Path("saved") / cli_args.experiment_name / "baseline_interface_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        scenes = _resolve_scene_ids(interface, cli_args.scene_id, cli_args.limit_scenes)
        manifest: Dict[str, object] = {
                "experiment_name": cli_args.experiment_name,
                "checkpoint"     : cli_args.checkpoint,
                "data_split"     : cli_args.data_split,
                "scenes"         : [],
                "metrics"        : "metrics.json",
        }

        for scene_id in scenes:
                prediction = interface.predict_scene(scene_id)
                scene_file = output_dir / f"{scene_id}.npz"
                _save_prediction(prediction, scene_file)
                manifest["scenes"].append(
                        {
                                "scene_id" : scene_id,
                                "file_name": prediction.get("file_name"),
                                "npz"      : scene_file.name,
                        }
                )

        metrics = interface.collect_metrics()
        if metrics:
                metrics_path = output_dir / "metrics.json"
                with metrics_path.open("w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2, ensure_ascii=False)
        else:
                manifest["metrics"] = None

        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2, ensure_ascii=False)

        print(f"[baseline-interface] Saved {len(scenes)} scene(s) to {output_dir}")


if __name__ == "__main__":
        main()
