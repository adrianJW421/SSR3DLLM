# SSR3DLLM pipeline (evaluation release)

This document summarizes the **mainline** SSR3D-LLM pipeline used by this release:

## Core concepts

- `<geom>`: routing token; enables the grounding branch.
- `<step1>..<stepK>`: implicit step-slot markers; hidden states at these positions act as spatial-state slots.
- Variable step length: a per-step mask `m_k∈{0,1}` freezes updates after the effective length `L` (masked steps are inert).
- Scoring: recursive refinement over candidates, step-by-step, conditioned on `(slot_k, memory_tokens)`.

## Main scripts (public)

- Appendix protocol eval:
  - `scripts/run_eval_appendix_examples.sh`
- Dialog demo eval:
  - `scripts/run_eval_dialog_demo.sh`
- Unified eval/ask entry:
  - `scripts/run_eval_unified.sh`
- Step-slot eval helpers:
  - `scripts/run_eval_stepslot_varlen.sh`
  - `scripts/run_eval_referit3d_suite.sh`

## Important consistency notes

- `<geom>` is **not** the state carrier; the state lives in `<stepk>` hidden states (+ memory tokens).
- Updates are **recursive** with freeze-gated masking, not a per-step ensemble.
- For paper-aligned evaluation, keep `K=4` and clip `L_used≤K` as used by the ViGOR-format referential orders.
