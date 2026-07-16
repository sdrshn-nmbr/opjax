# Stage 1 — One falsifiable claim

## Claim

A governed LoRA on **`thinkingmachines/Inkling`** (currently trainable on Tinker) improves
**tool-schema / tigerstyle sidekick** behavior over no-training controls, at fixed budget,
on a **sealed** split.

Concretely: on the sealed sidekick task family, LoRA wins on the primary rubric score
versus (1) base Inkling, (2) prompt + rules only, (3) few-shot/RAG static pack,
(4) static policy / linter-only baseline — under the Stage-0 private spend cap.

## Primary metric

- **Sidekick rubric pass rate** on sealed tasks (mechanical: schema-valid tool calls,
  allowlisted style invariants checkable by static guards where possible).
- Secondary: held-out NLL on train-split-held-out conversations; latency; cost.

## Kill condition (restored)

If Stage-5 LoRA fails to beat the best no-training control on the sealed primary metric
within the private budget → **stop weight-training claims**; fix data/task design or park.
Ambition does not override this kill.

## Statistics design

| Element | Policy |
|---------|--------|
| Seeds | `{0, 1, 2}` for claimed comparisons; smoke may use `{0}` only |
| Pairing | Same sealed task order / prompts across conditions |
| MDE | DRAFT: **+10 absolute points** rubric pass rate vs best control |
| CI | Report mean ± bootstrap 95% CI across seeds; no peeking at sealed during hill-climb |
| Dev | Rotating-dev for hill-climb only |
| Multiplicity | One primary metric; secondaries exploratory |

## What would falsify the claim

- LoRA ≤ best control on sealed primary metric under budget
- Gains only on train/dev contamination (sealed flat)
- Gains only from longer prompts / RAG, reproducible without weight updates

## Out of scope for this claim

Fusion dual-agent, OPD, PorTAL, DeepSWE public leaderboard chasing, Inkling-Small.

## Status

Hypothesis text frozen for v1 grind (2026-07-16). Sealed tasks may still be empty;
until sealed tasks exist, any LoRA result is labeled **`exploratory / plumbing`**, not a Stage-5 win.
