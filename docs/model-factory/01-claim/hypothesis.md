# Stage 1 — Falsifiable claim

## Claim (binding)

> A **governed LoRA** on a **currently trainable** base improves **one mechanical sidekick task family** over **prompt-only**, **few-shot/RAG**, and **static-policy** (linter/tool-guard) controls, at a **fixed USD / GPU-hour / wall-clock budget**, on a **sealed** SudarshanBench split.

## Scope of “mechanical sidekick”

First family (v1): **bounded implementation tasks** — apply a specified change in an allowlisted repo with tests as oracle (edit files, run tests, stop). Explicitly **not** open-ended product planning, Fusion routing, or multi-hour autonomy.

## Controls (must all be run)

| Arm | Description |
|-----|-------------|
| `base` | Frozen base, default system prompt |
| `prompt_rules` | Base + tigerstyle/philosophy rules in system prompt |
| `fewshot_rag` | Base + retrieved exemplars from train split only |
| `static_policy` | Base + linters/tool guards (no weight update) |
| `lora` | Governed LoRA/SFT under Stage-5 runbook |

## Kill condition (restored)

If Stage-5 `lora` fails to beat the best no-training control on the **sealed** split under the Stage-5 spend cap:

1. Stop weight-training claims for this task family.
2. Either raise data quality (Stage 4), change task family, or park the project.
3. Do **not** proceed to Stage 6–7 “because moonshot.”

Ambition strengthens falsification; it does not guarantee continuation.

## Non-claims

- Does not claim Composer/SWE-1.7 parity.
- Does not claim Inkling-Small superiority (contingent).
- Does not claim PorTAL / OPD / Fusion benefits.
