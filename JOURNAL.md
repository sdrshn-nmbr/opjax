# JOURNAL

Daily three-line minimum:
1. What shipped.
2. What broke.
3. What isn't understood.

During distillation-flavored phases (Phase 2, Phase 3.5, every cross-stage boundary):
4. Gini coefficient of per-token loss contributions on the latest step. Flag if >1.5x running median.

During Phase 3:
5. Per-specialist eval pass@1 on held-out tasks (or "no eval ran today").

---

## 2026-05-11 — Day 0

1. **Shipped:** Phase 0 floor confirmed (10 tests green, MaxText mirror present). Added `chex`, `jaxtyping`, `orbax-checkpoint`, `wandb` as exact pins to `pyproject.toml` and `REMOTE_IMAGE_PACKAGES`. Created `docs/design-notes.md` and this file.
2. **Broke:** Nothing.
3. **Don't understand yet:** What `cap_tokens` value will actually exercise the self-summary path during Phase 3 rollouts — won't know until specialists are training. Filed under Open Questions in the plan.

**Validation:** Modal `cpu_smoke` green (image rebuilt with new pins in 2.35s, cached layers held). `gpu_smoke_cli` green on H100 80GB HBM3, JAX 0.10.0, cuda:0. Both volumes (v2) mounted, all three secrets present. New deps did not break the image. App: ap-9QfjwhL80ixdnZvBa4EXw7.
