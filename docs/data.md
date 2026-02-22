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

If you use the zipped feature assets from Figshare instead of local export, extract
them under `<repo-root>/data/` and keep directory names unchanged:

- `MASK3D_FEATS_TRAIN/`
- `MASK3D_FEATS_TEST/`
- `MASK3D_FEATS_TRAIN_predbox_qnorm/` (optional)
- `MASK3D_FEATS_TEST_predbox_qnorm/` (optional)

## 4) Release recommendation

Before tagging a public release:
- keep `data/` as placeholders only
- publish one external asset index with source + download links + SHA256
- keep anonymous storage focused on project-owned checkpoints; point third-party assets to official sources
- run `bash scripts/check_paths.sh`
