# Data Setup (Evaluation Release)

This release is evaluation-focused.
`data/` contains placeholders only. Do not commit real datasets or checkpoints.

## Current Release Scope

- Released: evaluation/inference code
- Released: evaluation checkpoints and evaluation datasets (download externally)
- Not released yet: full training code/data pipeline

## Required for Evaluation

Fill these paths (via `configs/paths.sh`) to run the public eval/demo scripts:

| Path | What it should contain | Download link |
|---|---|---|
| `data/SCANNET200_ROOT/` | Processed ScanNet200 root (`train/`, `validation/`, `instance_gt/`, `*_database.yaml`) |  |
| `data/rawscannet/scans/` | Raw ScanNet `scans/` (axis-alignment txt files) |  |
| `data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl` | ReferIt3D ScanNet metadata pkl |  |
| `data/grounded3dllm_ckpts/step3/last-epoch.ckpt` | Baseline Step-3 checkpoint for comparison |  |
| `data/SSR3DLLM_CKPT/SSR3DLLM.ckpt` | Packed SSR3DLLM checkpoint |  |
| `data/BERT_PATH/bert-base-uncased/` | BERT weights (HF format) |  |
| `data/LLM_PATH/Tiny-Vicuna-1B/` | LLM weights (HF format) |  |
| `data/LISTENER_INIT_CKPT_BERT/best_model.pth` | Listener init checkpoint |  |
| `data/NR3D_TRAIN_CSV/` | ReferIt3D CSVs used by eval scripts |  |
| `data/SR3D_TRAIN_CSV/` | ReferIt3D CSVs used by eval scripts |  |
| `data/MASK3D_FEATS_TRAIN/` | Pre-exported train feature shards (`*.pt`) |  |
| `data/MASK3D_FEATS_TEST/` | Pre-exported val/test feature shards (`*.pt`) |  |

## Optional (Not Needed for Main Eval)

| Path | Use case |
|---|---|
| `data/grounded3dllm_ckpts/step2/last-epoch.ckpt` | Only if you want to re-export Mask3D features locally |
| `data/LLAMA_STEPSLOT_EVAL_CKPT/` | Optional extra eval profile |
| `data/LLAMA_STEPSLOT_EVAL_CKPT_UB/` | Optional UB profile |
| `data/MASK3D_FEATS_TRAIN_predbox_qnorm/`, `data/MASK3D_FEATS_TEST_predbox_qnorm/` | Optional qnorm ablation assets |
| `data/langdata/` | Optional local language payloads for custom experiments |
| `data/processed/` | Legacy compatibility symlink path only |

## Clarification on Potentially Confusing Paths

- The legacy `data/STEP2_CKPT/` path is intentionally not used in this release.
  Use `data/grounded3dllm_ckpts/step2/` if you need a Step-2 ckpt for local feature export.
- `data/processed/` is a legacy compatibility location (often a symlink target).
  It is not required for the main eval path.
- `LLAMA_STEPSLOT_EVAL_CKPT_519` remains as a compatibility env var, but in this release
  it points to the same UB checkpoint path by default.
- `data/rawscannet/` is **not** duplicated with `data/SCANNET200_ROOT/`.
  `rawscannet` provides raw scan metadata (axis alignment), while `SCANNET200_ROOT` is processed data for model/eval.

## Check

```bash
bash scripts/check_paths.sh
```
