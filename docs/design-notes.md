# Design Notes

Running ledger of in-flight decisions during implementation. Append-only.
For locked decisions and reasoning, see the plan file. This file is for
mid-execution micro-decisions: things discovered while writing the code that
weren't visible at planning time.

Each entry: date, scope, decision, why, alternatives considered, follow-ups.

---

## 2026-05-11 — Day 0 setup

**Decision:** Add `chex==0.1.91`, `jaxtyping==0.3.9`, `orbax-checkpoint==0.11.39`,
`wandb==0.26.1` as exact pins; mirror into `REMOTE_IMAGE_PACKAGES` for Modal image
reproducibility.

**Why:** Plan locks JAX hygiene stack on Day 0. Exact pins (`==`) match the rest of
the project's style — `>=` would let Modal builds drift from local. xprof deferred:
it's distributed as `tensorboard-plugin-profile` / via `pip install xprof-nightly`
inconsistently; install when we actually need a TPU/GPU profile in Phase 3.

**Alternatives considered:** `>=` constraints (rejected: drift between local and
Modal image bakes correctness bugs that only show up on remote). Adding xprof
preemptively (rejected: install when needed; the package landscape is messy and
will look different by Phase 3).

**Follow-ups:** confirm Modal image rebuild succeeds with new pins before any
Phase 1A pos-emb probe. Re-run `cpu_smoke` and `gpu_smoke_cli` end of Day 0.
