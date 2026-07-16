# Stage 3 — Base tournament (spec)

**Gate before Stage 5:** Pick the training base from measured $/quality, not novelty.  
**Prerequisite:** Stages 0–2 artifacts exist (this repo). Sealed set should be non-empty before scoring claims.

## Candidates (currently trainable)

| Base | Why included | Train path | Notes |
|------|--------------|------------|-------|
| Inkling (full) | Day-0 Tinker; product interest | Tinker LoRA | Costly serve; license accept on HF |
| Laguna XS.2 | Coding MoE; Prime allowlist stand-in | Prime / local | Do not treat Laguna wins as Inkling evidence |
| Qwen dense (e.g. Qwen3-8B class) | Cheap control; PorTAL-compatible family later | Tinker / Prime | Dense baseline |
| Inkling-Small | Contingent | TBD | **Only after** availability + train/serve smoke |

## Protocol

1. Freeze task semantics (SudarshanBench `dev` for tuning, never sealed for selection hacks).
2. Use **model-native** chat/tool renderers (portable recipes ≠ identical tokens).
3. Arms: each base under `prompt_rules` control first (no training).
4. Record pass rate, latency, USD/task, VRAM/serve class.
5. Optional: tiny public-model Tinker smoke (no private data) to verify trainer path.
6. Decision memo: chosen Stage-5 base + rejected alternatives with cost table.

## Explicit non-decision

Do not lock Fusion roles or PorTAL targets from this tournament.
