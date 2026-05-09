# Script Guide (Human-Friendly Entrypoints)

To make release usage clearer, this repo provides `run_*.sh` aliases.
These aliases call the original scripts internally, so old commands still work.

## Recommended Entrypoints

| Task | Use this script | Delegates to |
|---|---|---|
| Appendix capability eval | `scripts/run_eval_appendix_examples.sh` | `scripts/eval_ssr3dllm_appendix_cap_examples_from_ckpt.sh` |
| Dialog demo | `scripts/run_eval_dialog_demo.sh` | `scripts/eval_ssr3dllm_dialog_demo_from_ckpt.sh` |
| Unified eval/ask entry | `scripts/run_eval_unified.sh` | `scripts/eval_ssr3dllm_unified.sh` |
| Eval step-slot varlen chain | `scripts/run_eval_stepslot_varlen.sh` | `scripts/eval_llama_stepslot_varlen_chain_onepass_pred.sh` |
| Eval ReferIt3D suite | `scripts/run_eval_referit3d_suite.sh` | `scripts/eval_llama_stepslot_referit3d_suite.sh` |
| Eval ScanRefer/Multi3DRef | `scripts/run_eval_scanrefer_multi3dref.sh` | `scripts/eval_ssr3dllm_scanrefer_multi3dref.sh` |
| Benchmark SSR3D-LLM readout | `scripts/run_benchmark_readout.sh` | `scripts/benchmark_ssr3dllm_readout.sh` |
| Export Mask3D features | `scripts/run_export_mask3d_features.sh` | `scripts/export_mask3d_feats_predbox_fullresfix.sh` |
| Build optional DINO sidecar cache | `scripts/run_build_dino_sidecar_cache.sh` | `scripts/build_dino_sidecar_cache.sh` |

## Compatibility

- This release keeps eval/demo aliases plus one optional local cache-export alias.
