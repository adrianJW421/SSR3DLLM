# Third-party notices (vendored & customized)

This repository vendors several third-party codebases under `third_party/` for reproducibility.
**These copies are customized** (i.e., they may differ from upstream).

Where possible, we keep upstream LICENSE files inside each vendored folder.
If a vendored folder does not include an explicit LICENSE file, please refer to license headers in source files and the original upstream repository.

## Vendored under `third_party/`

- `third_party/Vigor/`
  - Customized copy used by the ReferIt3D training/evaluation pipeline.
  - License file: `third_party/Vigor/LICENSE`
- `third_party/pointnet2/`
  - Vendored PointNet++ CUDA ops wrapper used by Mask3D-related components (baseline code under `baseline/core/models/`).

## Archived (not required for mainline reproduction)

The following vendored trees are kept under `unrelated_to_release/third_party/` for internal reference,
but are not part of the SSR3DLLM + baseline training/evaluation mainline:

- `unrelated_to_release/third_party/STAMP/` (customized copy; license file inside folder)
- `unrelated_to_release/third_party/univlg/` (customized copy; license file inside folder)
- `unrelated_to_release/third_party/vil3dref/` (customized copy; license file inside folder)
- `unrelated_to_release/third_party/MiKASA-3DVG/` (archived MiKASA baseline; license file inside folder)

If you plan to publish this repository, please ensure all third-party licenses are correctly preserved and attributed.
