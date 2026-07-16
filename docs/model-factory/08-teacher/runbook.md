# Stage 8 — Teacher update

## Paths

| Path | When allowed | Method |
|------|--------------|--------|
| **OPD** | Teacher exposes logprobs on student tokens **and** token-compatible | Reverse-KL on-policy distill ([TML OPD](https://thinkingmachines.ai/blog/on-policy-distillation/)) |
| **Sequence distill** | Ordinary API teachers | Off-policy SFT on teacher sequences |
| **Forbidden** | Calling API text feedback “OPD” | — |

## Always

- Run **retention suite** (old sealed + train tasks) after any teacher update.
- Rollback checkpoint if retention regresses beyond pre-registered floor.
- Continual OPD that mutates weights conflicts with PorTAL frozen `z_t`/core — keep Path B separate ([09-portal](../09-portal/runbook.md)).
