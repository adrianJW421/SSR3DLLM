# Release Checklist (Maintainers)

Use this list before pushing a release tag.
This checklist is maintainer-only and is not required for regular users running eval/demo scripts.

## 0) Environment + paths

- [ ] `configs/paths.sh` is set for the target machine.
- [ ] `bash scripts/check_paths.sh` passes.
- [ ] `SCANNET200_ROOT`, `REFERIT_SCANNET_FILE`, `BERT_PATH`, `LLM_PATH` exist.
- [ ] `SSR3DLLM_RAWSCANNET_SCANS_DIR` points to raw ScanNet `scans/` (axisAlignment txt files).

## 1) Artifacts

- [ ] packed checkpoint exists: `data/SSR3DLLM_CKPT/SSR3DLLM.ckpt`
- [ ] baseline checkpoint exists: `data/grounded3dllm_ckpts/step3/last-epoch.ckpt`
- [ ] publish checksum for packed checkpoint:

```bash
sha256sum data/SSR3DLLM_CKPT/SSR3DLLM.ckpt
```

## 2) Script health

- [ ] appendix eval entry works:

```bash
bash scripts/run_eval_appendix_examples.sh
```

- [ ] dialog demo entry works:

```bash
bash scripts/run_eval_dialog_demo.sh
```

## 3) Repro entrypoints

- [ ] paper eval path works:

```bash
bash scripts/run_eval_appendix_examples.sh
```

- [ ] unified single-dialog demo works:

```bash
bash scripts/run_eval_dialog_demo.sh
```

## 4) Docs and messaging

- [ ] `README.md` includes official entrypoints and expected outputs.
- [ ] known warnings/limitations are documented (SPICE Java warning, planning length zero if no data, demo vs full protocol).
- [ ] release notes mention what is included/excluded.

## 5) Final gate

- [ ] clean run logs saved under `outputs/ssr3dllm/`
- [ ] no blocker from smoke checks
- [ ] tag + upload
