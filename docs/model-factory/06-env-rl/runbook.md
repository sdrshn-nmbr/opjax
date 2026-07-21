# Stage 6 — Environment qualification → thin RL

**Prerequisite:** Stage 5 sealed win under kill condition.  
**Unlock memo:** [`../05-controlled-lora/promote-to-stage6.md`](../05-controlled-lora/promote-to-stage6.md)

## Qualification (before RL)

1. Measure verifier false-positive / false-negative rates on a labeled probe set.
   - CLI: `uv run opjax-model-factory verifier-probe --write docs/model-factory/06-env-rl/evals/verifier-probe.json`
2. Solution-channel scanners (git history, caches, package registries, web).
   - CLI: `uv run opjax-model-factory scan-solutions --write docs/model-factory/06-env-rl/evals/solution-scan.json`
3. Single-repo test-reward tasks first.
   - Module: `opjax.model_factory.reward_env` / `grade-solution`
4. Harden sealed set for headroom ([sealed-harden.md](sealed-harden.md)) and re-baseline Stage-5 LoRA once.

## Training policy

- Prefer on-policy with version pins.
- If async: document lag bounds, importance/replay, router replay — do not claim “on-policy” without this.
- **Inkling weight claim:** Tinker only (warm-start Stage-5 sampler).
- Candidate **env** substrates: Prime Lab/verifiers (smoke first), Modal, Fireworks remote — choose by evidence.
  - Inkling is **not** on Prime Hosted Training allowlist; Laguna wins ≠ Inkling evidence.
- Pedagogy from SWE-1.7 / Composer; do not copy scale.
- Thin defaults: `group_size` 2–4, ≤32 steps, `lr≈1e-5`, skip constant-reward groups.
- CLI planner (no spend): `uv run opjax-model-factory thin-rl -- --dry-run`
- Paid run requires operator OK: `… thin-rl -- --i-approve-spend …`

## Retention

Keep a replay buffer / forgetting suite when continuing from Stage 5 checkpoint.

## Kill

If sealed v2 pass rate does **not** improve vs Stage-5 LoRA baseline under the Stage-6 spend cap → stop Stage-6 weight claims (no Stage 7 “because moonshot”).

## Artifacts

See [artifacts.md](artifacts.md).
