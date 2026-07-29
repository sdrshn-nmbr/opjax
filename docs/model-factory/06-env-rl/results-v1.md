# Stage 6 results v1 — thin RL

**Status:** thin GRPO v1b complete; sealed after-eval done; **Stage-6 kill triggered**.

## Gate checklist

| Gate | Status |
|------|--------|
| Stage 5 promote | Done — [`../05-controlled-lora/promote-to-stage6.md`](../05-controlled-lora/promote-to-stage6.md) |
| Sealed harden v2 | Done — n=8 sealed IDs |
| Solution scanners | **ok** — [`evals/solution-scan.json`](evals/solution-scan.json) |
| Verifier FP/FN probe | **fp=0 fn=0** — [`evals/verifier-probe.json`](evals/verifier-probe.json) |
| Stage-5 LoRA baseline on sealed v2 | **0.875** (7/8) — `data/model-factory/evals/sealed-v3-baseline-lora/` |
| Thin RL train | **v1b** — 5 steps, **1** trained step, then abort (fuel saturated) |
| Sealed after-eval + kill | **0.875** (7/8) — Δ **+0.000** → **kill** |

## Contrast (headline)

| Arm | Sealed v2 pass rate (seed 0) |
|-----|------------------------------|
| Stage-5 LoRA baseline | **0.875** |
| Stage-6 thin RL v1b | **0.875** |
| Absolute Δ | **+0.000** |
| Kill triggered? | **Yes** — no sealed lift under this thin config |

Failing sealed task both times: `sb-0013`.

## Train profile (v1b)

| Field | Value |
|-------|-------|
| Fuel | `train,dev` (sb-0001…0007); never sealed |
| Steps ran / trained | 5 / **1** |
| Mean reward curve | 1.0, 0.8125, 1.0, 1.0, 1.0 |
| Rollouts / datums | 80 / 4 |
| Abort | 3 consecutive idle steps (saturated fuel) |
| Wall | ~355 s |
| Artifacts | `data/model-factory/rl/thin-v1/{thin_rl_history.json,profile.json}` |

## Samplers

| Role | Path |
|------|------|
| Warm-start | `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/weights/final` |
| Post-RL sampler | `tinker://539b5033-b78b-51f6-9d0e-97e0262490f2:train:0/sampler_weights/final` |
| Post-RL weights | `tinker://539b5033-b78b-51f6-9d0e-97e0262490f2:train:0/weights/final` |

## Decision

**Stop Stage-6 weight claims** on this fuel/hparams. Do not promote to Stage 7 “because moonshot.”

Next options (operator): harder RL fuel / bench harden (more headroom), OPSD/teacher track (Stage 8), HF export of Stage-5 LoRA (still the sealed winner), or park weight training.

Climb ladder: [`climb-ladder.md`](climb-ladder.md) · `data/model-factory/evals/climb_ladder.json`
