# Release notes (snapshot)

This directory is intended to be published as the **public GitHub repository** root.

## Included

- Baseline Grounded 3D-LLM code (hydra-free pipeline):
  - Core entrypoints (repo root): `config.py`, `main_run.py` (`--entry standard|interface`)
  - Baseline configs: `baseline/core/conf/`
  - Core implementation (moved under baseline): `baseline/core/models/`, `baseline/core/trainer/`, `baseline/core/utils/`
  - Shared packages: `models/`, `utils/` (SSR3DLLM modules live at top-level; baseline modules are reachable via `baseline/core/...` search-path shims)
  - Baseline dataset implementation: `baseline/dataset/datasets/`, `baseline/dataset/dataset_code/`
- SSR3D-LLM (SSR3DLLM) code is the **primary** top-level implementation:
  - Eval entrypoint shim: `train/step3_train_ssr3dllm_geom_entry.py` (used by appendix-style eval)
  - Core modules: `models/`, `utils/`
  - Customized third-party dependencies: `third_party/`
  - ReferIt3D listener runtime adapter (customized from ViGOR): `models/referit3d_listener_runtime.py`
- Public-facing scripts under `scripts/` (evaluation-focused):
  - `run_eval_appendix_examples.sh`, `run_eval_dialog_demo.sh`, `run_eval_unified.sh`
  - `run_eval_stepslot_varlen.sh`, `run_eval_referit3d_suite.sh`
  - `eval_llm.sh` (metric helper used by appendix protocol)

## Not included

- Archived non-mainline material is kept under `unrelated_to_release/` (not needed for mainline training/evaluation).
- Datasets and large evaluation artifacts:
  - ScanNet / ScanNet200 raw and processed data
  - ReferIt3D / ViGOR-format CSVs and other derived files
- Pretrained model weights and training outputs:
  - `pretrained/`, `saved/`, `outputs/`
- Internal experimental scripts and legacy branches not needed for the main training/evaluation path
- Unrelated research modules that are not part of the baseline or SSR3DLLM mainline (e.g., the previously vendored GVerifier code path)

## Path configuration

- Copy `configs/paths.example.sh` → `configs/paths.sh` and edit it for your machine.
- `configs/paths.sh` is ignored by git (see `.gitignore`).
