# Memo — Promote Stage 5 → unlock Stage 6

**Date:** 2026-07-19  
**Hypothesis ID:** Stage-1 sealed LoRA claim ([`../01-claim/hypothesis.md`](../01-claim/hypothesis.md))  
**Stage:** 5 → 6 gate

## What changed

Stage-5 Inkling LoRA (v2) completed under the Stage-5 spend cap and beat the best no-training control on sealed SudarshanBench.

## Metrics (sealed)

| Arm | Mean pass rate | Notes |
|-----|----------------|-------|
| `lora` | **1.00** | seeds `{0,1,2}` all 4/4 |
| Best control `fewshot_rag` | **0.58** | |
| Absolute lift | **+0.42** | ≥ +10 pp MDE |
| Kill triggered? | **No** | [`sealed-v2-summary.json`](sealed-v2-summary.json) |

Final sampler: `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final`  
Details: [`results-v2.md`](results-v2.md)

## Cost

- Period Tinker Usage: **$286.34** (reconcile: [`../agent-ops/spend-ledger-reconcile.md`](../agent-ops/spend-ledger-reconcile.md)).
- Stage-5 Inkling attribution est. **~$126** ≪ **$500** hard cap.

## Decision

**Promote** — unlock Stage 6 thin RL per [`../06-env-rl/runbook.md`](../06-env-rl/runbook.md).

### Caveats (binding)

1. Sealed n=4 micro-tasks at Stage-5 freeze — not SWE/agentic parity.
2. Stage 6 must re-baseline on the **hardened sealed set** (splits v2) before claiming RL lift.
3. Kill again if Stage-6 sealed does not improve vs Stage-5 LoRA under the Stage-6 budget.
4. Laguna / Prime Hosted Training wins ≠ Inkling evidence. Inkling RL stays on **Tinker**.

## Citations

- [`results-v2.md`](results-v2.md), [`runbook.md`](runbook.md)
- [`../03-tournament/decision.md`](../03-tournament/decision.md)
- [`../00-governance/spend-caps.md`](../00-governance/spend-caps.md)
