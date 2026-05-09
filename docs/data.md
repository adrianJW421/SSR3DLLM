# Data & Checkpoints

This release does not include datasets or model weights.
Use placeholders under `data/`, then fetch assets from your external release channels.

## 1) Where to fill download links

Use this registry and fill links before public release:
- `data/README.md`

Current anonymous-review bundle link (project-owned assets):
- `https://figshare.com/s/b4f92c34ceda0b17626d`

The table there is organized by folder and includes:
- required/optional status
- expected payload format
- source (fill)
- download link (for project-owned assets, one shared bundle link is used)
- SHA256 (fill)

## 2) Path configuration

Set all local/external paths in:
- `configs/paths.sh`

You can choose either mode:
- Put assets into repo-local `data/...` placeholders.
- Keep assets outside the repo and only map paths in `configs/paths.sh`.

## 3) Optional preprocessing from raw

The anonymous Figshare bundle already includes zipped `MASK3D_FEATS_*`.
If those zip assets are not available in your environment, generate features locally:

```bash
# validation side
bash scripts/run_export_mask3d_features.sh validation

# train side
bash scripts/run_export_mask3d_features.sh train
```

Then point `MASK3D_FEATS_TEST` and `MASK3D_FEATS_TRAIN` in `configs/paths.sh`
to your exported directories.

## 4) Optional DINO sidecar cache

DINO appearance sidecar caches are not hosted in the anonymous Figshare bundle because of storage limits. They are optional. If `SCANREFER_DINO_SAMPLE_CACHE_ROOT` and `MULTI3DREF_DINO_SAMPLE_CACHE_ROOT` are unset, scripts run without DINO fusion.

To regenerate them, prepare DINOv2 multi-view object features from official ScanNet RGB-D frames and camera poses, and use the same Mask3D/CLASP proposal cache referenced by the evaluation CSV. Then run:

```bash
export DATASET=scanrefer
export SCANREFER_QCOND_CACHE_ROOT=/path/to/scanrefer_query_conditioned_mask3d_cache
export SCANREFER_DINO_SAMPLE_CACHE_ROOT=/path/to/scanrefer_dino_sidecars
export DINO_SOURCE_FEATURES=/path/to/scannet_dinov2_multiview_object_features.pt
export DINO_SOURCE_ATTRIBUTES=/path/to/scannet_mask3d_object_attributes.pt
bash scripts/run_build_dino_sidecar_cache.sh
```

For Multi3DRef, set `DATASET=multi3dref`, `MULTI3DREF_QCOND_CACHE_ROOT`, and `MULTI3DREF_DINO_SAMPLE_CACHE_ROOT`. The sidecar root mirrors the CSV `mask3d_sample_cache_relpath` or `mask3d_sample_cache_path` values.

`DINO_SOURCE_FEATURES` should be a PyTorch dictionary of multi-view DINOv2 object features keyed as `scene_id_objectid`, for example `scene0000_00_03`. `DINO_SOURCE_ATTRIBUTES` should be a PyTorch dictionary keyed by `scene_id`; each value should contain `locs` as `[num_objects, 6]` center-size boxes and optionally `obj_ids`.

Each sidecar `.pt` should contain:

- `proposal_dino_features`: `FloatTensor [num_proposals, 1024]`
- `proposal_dino_valid_mask`: `BoolTensor [num_proposals]`
- `gt_to_query_map`: dict mapping ScanNet object id to proposal row

Set these paths in `configs/paths.sh`:

- `SCANREFER_DINO_SAMPLE_CACHE_ROOT`
- `MULTI3DREF_DINO_SAMPLE_CACHE_ROOT`
- `DINO_FEATURE_DIM=1024`
- `DINO_ALPHA=2.0`

If you use the zipped feature assets from Figshare instead of local export, extract
them under `<repo-root>/data/` and keep directory names unchanged:

- `MASK3D_FEATS_TRAIN/`
- `MASK3D_FEATS_TEST/`
- `SCANREFER_MASK3D_FEATS_TRAIN/`
- `SCANREFER_MASK3D_FEATS_TEST/`
- `MULTI3DREF_MASK3D_FEATS_TRAIN/`
- `MULTI3DREF_MASK3D_FEATS_TEST/`
- `MASK3D_FEATS_TRAIN_predbox_qnorm/` (optional)
- `MASK3D_FEATS_TEST_predbox_qnorm/` (optional)

## 5) Release recommendation

Before tagging a public release:
- keep `data/` as placeholders only
- publish one external asset index with source + download links + SHA256
- keep anonymous storage focused on project-owned checkpoints; point third-party assets to official sources
- run `bash scripts/check_paths.sh`
