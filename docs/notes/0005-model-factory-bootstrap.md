# Model Factory bootstrap (2026-07-16)

## What landed

- Enforceable Stage 0–2 artifacts under `docs/model-factory/`
- Python package `opjax.model_factory` (scrub, canary, pre-upload gate, splits, audit)
- CLI: `opjax-model-factory` / `opjax model-factory -- …`
- Gated runbooks for Stages 3–10 (no private training uploads)
- Reference packs under `references/model-factory/`

## Binding constraints carried from the plan

- No managed-trainer upload without rights manifest + pre-upload gate
- DeepSWE / sealed SudarshanBench never used as training fuel for reported scores
- SFT kill condition restored
- Inkling/Prime/Fusion/OPD/PorTAL remain hypotheses until measured

## Wave A auth

Completed on the operator laptop (HF / Tinker / Prime). This cloud workspace does not assume those credentials.
