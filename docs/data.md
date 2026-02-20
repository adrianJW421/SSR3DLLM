# Data & Checkpoints

This release does not include datasets or model weights.
Use placeholders under `data/`, then fetch assets from your external release channels.

## 1) Where to fill download links

Use this registry and fill links before public release:
- `data/README.md`

The table there is organized by folder and includes:
- required/optional status
- expected payload format
- source (fill)
- download link (fill)
- SHA256 (fill)

## 2) Path configuration

Set all local/external paths in:
- `configs/paths.sh`

You can choose either mode:
- Put assets into repo-local `data/...` placeholders.
- Keep assets outside the repo and only map paths in `configs/paths.sh`.

## 3) Optional preprocessing from raw

This release is evaluation-focused and does not ship preprocessing utilities.
If you only have raw ScanNet / metadata, use your own preprocessing pipeline
and make sure outputs match the folder contracts in `data/README.md`.

## 4) Release recommendation

Before tagging a public release:
- keep `data/` as placeholders only
- publish one external asset index with source + download links + SHA256
- run `bash scripts/check_paths.sh`
