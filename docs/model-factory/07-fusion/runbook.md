# Stage 7 — Fusion proof (late)

**Prerequisite:** Solo sidekick sealed win (Stage 5/6).

## Comparisons (infer roles from data)

| Arm | Description |
|-----|-------------|
| big-only | Lead/base alone |
| sidekick-only | Post-trained specialist alone |
| router-only | Router without dual persistent agents |
| dual | Fusion-style dual agent |

## Forbidden assumptions

- “Big plans, Small implements” is a **hypothesis**, not a lock.
- Shared tokenizer ≠ shared KV (see Stage 10).
- Do not start Fusion to rescue a failed Stage 5.
