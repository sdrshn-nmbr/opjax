# Model Factory changelog

## 2026-07-16

- Landed living guide under `docs/model-factory/`.
- Stage 0–2 templates: rights, retention, spend, scrub/canary, hypothesis, sealed protocol, DeepSWE policy.
- Bootstrap: `hf auth login`; `tinker-cookbook[inkling]` / `tml_renderers`.
- Factory package `opjax.factory` for scrub → Tinker JSONL + preflight.
- **Public Inkling Tinker smoke SUCCESS** — see `runs/inkling-smoke.md` (3 LoRA steps, checkpoint saved).
- Private Stage-5 **blocked** on axport ingress — see `BLOCKED-axport-ingress.md`.
- Phase 6 citation pack: `references/factory/README.md` (gitignore exception).
- `opjax factory …` CLI dispatch; Stage-4 README stub.
- **Axport ingress unblocked** via R2; cursor.zip → 74-traj scrubbed slice + rights sign-off `20260716-axport-cursor-3ffdff36`.
- **Stage-5 exploratory LoRA SUCCESS** — `runs/inkling-axport-v1.md` (`067b00f3…:train:0`). Sealed eval still empty → not a scientific win.
