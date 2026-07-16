# Stage 1 — Statistical design

**Decision:** Agentic outcomes are stochastic; report paired, repeated, cost-aware results.  
**Audit:** Red-team H12 — no seeds/MDE/CI was a critical gap.

## Design

| Element | Choice |
|---------|--------|
| Primary metric | Sealed-split task pass rate (tests green) |
| Secondary | Tigerstyle rubric score (0–1), latency p50, USD/task |
| Seeds | 3 evaluation seeds `{0,1,2}` for sampling arms |
| Pairing | Same task IDs across arms |
| Aggregation | Mean pass rate ± bootstrap 95% CI (10k resamples) over tasks |
| Multiple comparisons | Pre-register primary contrast: `lora` vs best control; secondaries exploratory |
| MDE | Target detectable absolute lift **+10 pp** sealed pass rate at ~80% power with n≈30 sealed tasks (revisit after task count freezes) |
| Early stop | No peeking at sealed during Stage 4–5 training; use `dev` only |

## Logging requirements

Every eval artifact must include: stage, arm, base model id, commit SHA, harness version, seed, cost USD, GPU-hours, wall time, split name (`sealed` / `dev` / …).

## Forbidden

- Reporting DeepSWE public leaderboard numbers from a model trained on that split.
- Selecting “best seed” post hoc for the primary claim.
