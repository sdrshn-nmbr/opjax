# Stage 6 — Artifact inventory (A / B / C)

Organized 2026-07-19 before env-qual / thin RL coding.

## A — Binding (use)

| Artifact | Path / ID | Why |
|----------|-----------|-----|
| Stage-5 final sampler | `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final` | Warm-start for thin RL |
| Stage-5 results | [`../05-controlled-lora/results-v2.md`](../05-controlled-lora/results-v2.md), [`../05-controlled-lora/sealed-v2-summary.json`](../05-controlled-lora/sealed-v2-summary.json) | Kill gate evidence |
| Promote memo | [`../05-controlled-lora/promote-to-stage6.md`](../05-controlled-lora/promote-to-stage6.md) | Stage 5→6 unlock |
| Sealed splits v2 | [`../02-sealed-eval/sudarshanbench/splits.json`](../02-sealed-eval/sudarshanbench/splits.json) | Headline eval IDs |
| v1 freeze archive | [`../02-sealed-eval/sudarshanbench/splits.v1-freeze.json`](../02-sealed-eval/sudarshanbench/splits.v1-freeze.json) | Historical Stage-5 sealed |
| Fixtures sealed | `../02-sealed-eval/sudarshanbench/fixtures/sb-0008`…`0011`, `0013`…`0016` | Pytest oracle |
| Eval harness | `src/opjax/model_factory/eval_sudarshanbench.py` | Before/after sealed runner |
| Stage-6 runbook | [`runbook.md`](runbook.md) | Qualification + training policy |
| Spend caps / ledger | [`../00-governance/spend-caps.md`](../00-governance/spend-caps.md), [`../agent-ops/spend-ledger-reconcile.md`](../agent-ops/spend-ledger-reconcile.md) | Budget |
| Rights manifest | [`../00-governance/rights-manifest.json`](../00-governance/rights-manifest.json) | Upload / scrub flags |
| Scrubbed upload JSONL | `data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl` (gitignored) | Stage-4 emit; do not re-upload casually |

## B — Smoke / substrate (use carefully)

| Artifact | Notes |
|----------|-------|
| Tinker Tutorial 104 / `rl_loop.py` | Minimal on-policy GRPO |
| Tinker `verifiers_rl` recipe | Prime Hub envs → Tinker backend |
| Tinker `code_rl` | SandboxFusion / Modal sandboxes |
| Prime Environments Hub + Laguna XS.2 | $0 smoke / env plumbing only |
| [`../03-tournament/decision.md`](../03-tournament/decision.md) | Laguna ≠ Inkling evidence |

## C — Defer

| Item | Why deferred |
|------|----------------|
| Harbor / agent-RL / multimodal scale | Blows thin budget; not one-repo first |
| Full `math_rl` defaults on Inkling | Token scale vs ~$65 wallet |
| Fusion / PorTAL / KV (Stages 7–10) | Downstream of Stage-6 kill |
| opjax.md Phase 4/6 GUI-agent GRPO | Different numbering / project |
| Prime Hosted Training for Inkling | **Not on allowlist** |

## Local Stage-6 modules (this branch)

| Module | Role |
|--------|------|
| `src/opjax/model_factory/reward_env.py` | One-repo pytest reward |
| `src/opjax/model_factory/solution_scanners.py` | Solution-channel scanners |
| `src/opjax/model_factory/verifier_probe.py` | Verifier FP/FN probe |
| `src/opjax/model_factory/thin_rl.py` | Thin GRPO loop (spend-gated) |
