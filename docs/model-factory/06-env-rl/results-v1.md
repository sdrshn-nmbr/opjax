# Stage 6 results v1 — thin RL

**Status:** scaffold — env-qual in progress; **no Inkling RL spend** until operator approves.

## Gate checklist

| Gate | Status |
|------|--------|
| Stage 5 promote | Done — [`../05-controlled-lora/promote-to-stage6.md`](../05-controlled-lora/promote-to-stage6.md) |
| Sealed harden v2 | Done — n=8 sealed IDs (`sb-0013`…`0016` added) |
| Solution scanners | **ok** — [`evals/solution-scan.json`](evals/solution-scan.json) |
| Verifier FP/FN probe | **fp=0 fn=0** — [`evals/verifier-probe.json`](evals/verifier-probe.json) |
| Stage-5 LoRA baseline on sealed v2 | **Blocked** — Tinker API connection error from agent host ([`evals/baseline-lora-v3-status.json`](evals/baseline-lora-v3-status.json)); re-run on connected host |
| Thin RL train | **Blocked** on spend approval (`thin-rl -- --dry-run` plan ready) |
| Sealed after-eval + kill | Pending |

## Planned contrast

| Arm | Metric |
|-----|--------|
| Stage-5 LoRA baseline (sealed v2) | pass rate |
| Stage-6 thin RL | pass rate |
| Kill | thin RL ≤ baseline under budget → stop weight claims |

## Sampler

- Warm-start: `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final`
- Post-RL sampler: _TBD_
