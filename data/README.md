# Data Setup (Evaluation Release)

`data/` is placeholder-only in this repository.
Do not commit real datasets or checkpoints.

## Release Policy (Anonymous Phase)

To fit anonymous storage limits, assets are split into:

1. Hosted by us (Figshare): project-owned checkpoints and small protocol files.
2. Downloaded from official sources: third-party datasets and base language models.
3. Optional fallback: rebuild visual frontend cache (`MASK3D_FEATS_*`) from Step-2 ckpt.

## A. Hosted by Us (Figshare)

Single anonymous-review bundle link (contains all project-owned assets in this section):

`https://figshare.com/s/b4f92c34ceda0b17626d`

Note: this is a private review link and may have an expiry date.

| Path | Purpose | Size (approx) |
|---|---|---:|
| `data/SSR3DLLM_CKPT/SSR3DLLM.ckpt` | Packed one-file SSR3DLLM checkpoint | 8.4 GB |
| `data/grounded3dllm_ckpts/step3/last-epoch.ckpt` | Baseline Step-3 checkpoint (for baseline-vs-ours eval) | 3.4 GB |
| `data/grounded3dllm_ckpts/step2/last-epoch.ckpt` | Step-2 checkpoint (for local feature export) | 2.2 GB |
| `data/LISTENER_INIT_CKPT_BERT/best_model.pth` | Listener init ckpt for ReferIt3D suite | 1.1 GB |
| `data/NR3D_TRAIN_CSV/` | NR3D train/test CSV pair used by release scripts | 0.1 GB |
| `data/SR3D_TRAIN_CSV/` | SR3D train/test CSV pair used by release scripts | 0.1 GB |
| `data/SCANREFER_TRAIN_CSV/` | ScanRefer train/test CSV pair in Vigor-style chain format | varies |
| `data/MULTI3DREF_TRAIN_CSV/` | Multi3DRef train/test CSV pair in Vigor-style chain format | varies |
| `data/MASK3D_FEATS_TRAIN` (zip) | Pre-exported train feature shards | 0.07 GB |
| `data/MASK3D_FEATS_TEST` (zip) | Pre-exported validation/test feature shards | 0.02 GB |
| `data/SCANREFER_MASK3D_FEATS_TRAIN` (zip) | ScanRefer train proposal-level feature shards | varies |
| `data/SCANREFER_MASK3D_FEATS_TEST` (zip) | ScanRefer validation/test proposal-level feature shards | varies |
| `data/MULTI3DREF_MASK3D_FEATS_TRAIN` (zip) | Multi3DRef train proposal-level feature shards | varies |
| `data/MULTI3DREF_MASK3D_FEATS_TEST` (zip) | Multi3DRef validation/test proposal-level feature shards | varies |
| `data/MASK3D_FEATS_TRAIN_predbox_qnorm` (zip) | Optional qnorm train feature variant | 0.13 GB |
| `data/MASK3D_FEATS_TEST_predbox_qnorm` (zip) | Optional qnorm validation/test feature variant | 0.03 GB |
| `data/SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT/best_model.pth` | SSR3D-LLM ScanRefer evaluation checkpoint | varies |
| `data/MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT/best_model.pth` | SSR3D-LLM Multi3DRef evaluation checkpoint | varies |

## B. Download from Official Sources

| Path | What it contains | Official source link |
|---|---|---|
| `data/SCANNET200_ROOT/` | Processed ScanNet200 root (`train/`, `validation/`, `instance_gt/`, `*_database.yaml`) |  |
| `data/rawscannet/scans/` | Raw ScanNet `scans/` (axis-alignment txt files) |  |
| `data/BERT_PATH/bert-base-uncased/` | BERT weights (HF format) |  |
| `data/LLM_PATH/Tiny-Vicuna-1B/` | LLM weights (HF format) |  |
| `data/REFERIT_SCANNET_FILE/keep_all_points_with_global_scan_alignment.pkl` | ReferIt3D ScanNet metadata pkl (needed by ReferIt3D suite) |  |

## C. Rebuild Locally (Optional Fallback)

The Figshare bundle already contains zipped `MASK3D_FEATS_*`.
If download is unavailable in your environment, generate them locally:

```bash
# validation side
bash scripts/run_export_mask3d_features.sh validation

# train side
bash scripts/run_export_mask3d_features.sh train
```

Then set in `configs/paths.sh`:

- `MASK3D_FEATS_TEST=<your validation export dir>`
- `MASK3D_FEATS_TRAIN=<your train export dir>`

## D. Optional DINO Appearance Sidecar Cache

The DINO sidecar cache is not hosted in the anonymous Figshare bundle because the bundle has a 20GB storage limit. It is optional: if `SCANREFER_DINO_SAMPLE_CACHE_ROOT` and `MULTI3DREF_DINO_SAMPLE_CACHE_ROOT` are unset, the evaluation scripts disable DINO fusion.

To rebuild it locally, use the official ScanNet RGB-D frames and camera poses, the same Mask3D/CLASP proposal cache referenced by the evaluation CSV, and a DINOv2 multi-view object-feature extractor. Save one `.pt` file per query sample with the same relative path stored in `mask3d_sample_cache_relpath` or `mask3d_sample_cache_path`.

Expected `.pt` schema:

| Key | Shape / type |
|---|---|
| `proposal_dino_features` | `FloatTensor [num_proposals, 1024]` |
| `proposal_dino_valid_mask` | `BoolTensor [num_proposals]` |
| `gt_to_query_map` | dict mapping ScanNet object id to proposal row |

Then set in `configs/paths.sh`:

- `SCANREFER_DINO_SAMPLE_CACHE_ROOT=<your ScanRefer DINO sidecar root>`
- `MULTI3DREF_DINO_SAMPLE_CACHE_ROOT=<your Multi3DRef DINO sidecar root>`
- `DINO_FEATURE_DIM=1024`
- `DINO_ALPHA=2.0`

## E. If You Download Feature ZIPs: Expected Extraction Paths

If you use the four feature zip files from the Figshare bundle, extract them to:

- `data/MASK3D_FEATS_TRAIN/`
- `data/MASK3D_FEATS_TEST/`
- `data/SCANREFER_MASK3D_FEATS_TRAIN/`
- `data/SCANREFER_MASK3D_FEATS_TEST/`
- `data/MULTI3DREF_MASK3D_FEATS_TRAIN/`
- `data/MULTI3DREF_MASK3D_FEATS_TEST/`
- `data/MASK3D_FEATS_TRAIN_predbox_qnorm/` (optional)
- `data/MASK3D_FEATS_TEST_predbox_qnorm/` (optional)

Example:

```bash
cd <repo-root>/data
unzip MASK3D_FEATS_TRAIN.zip -d .
unzip MASK3D_FEATS_TEST.zip -d .
# optional qnorm
unzip MASK3D_FEATS_TRAIN_predbox_qnorm.zip -d .
unzip MASK3D_FEATS_TEST_predbox_qnorm.zip -d .
```

After extraction, make sure `configs/paths.sh` points to the extracted directories.

## Optional Paths (Not Needed for Main Demo)

| Path | Use case |
|---|---|
| `data/LLAMA_STEPSLOT_EVAL_CKPT/` | Optional extra eval profile |
| `data/LLAMA_STEPSLOT_EVAL_CKPT_UB/` | Optional UB profile |
| `data/SCANREFER_LLAMA_STEPSLOT_EVAL_CKPT/` | ScanRefer SSR3D-LLM evaluation profile |
| `data/MULTI3DREF_LLAMA_STEPSLOT_EVAL_CKPT/` | Multi3DRef SSR3D-LLM evaluation profile |
| `data/MASK3D_FEATS_TRAIN_predbox_qnorm/`, `data/MASK3D_FEATS_TEST_predbox_qnorm/` | Optional qnorm ablation assets (also included as zip in the bundle) |
| `data/langdata/` | Optional local language payloads for custom experiments |
| `data/processed/` | Legacy compatibility path only |

## Quick Path Check

```bash
bash scripts/check_paths.sh
```
