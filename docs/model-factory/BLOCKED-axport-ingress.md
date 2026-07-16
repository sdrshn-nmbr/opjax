# BLOCKED — Axport corpus ingress (2026-07-16)

## Status

**Hard stop before private Stage-5 LoRA.**

This cloud VM has:

- No `data/axport/raw` (or `data/factory/axport/raw`)
- No `~/.config/axport/r2.env`
- No `CLOUDFLARE_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` in the environment
- Generic `AWS_*` credentials present but **not** verified as Cloudflare R2 for `s3://axport/`

## What is unblocked

- Stage 0–2 docs
- `opjax.factory` scrub / rights / render / preflight
- Public Inkling Tinker smoke (`docs/model-factory/runs/inkling-smoke.md`)

## How to unblock

Pick one:

1. **Add R2 secrets** to Cursor Cloud Environment (`CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, optional `AXPORT_BUCKET=axport`) and re-run agent with axport sync.
2. **Upload** a local axport export (or scrubbed JSONL) into `data/axport/raw/` on the VM (gitignored).
3. Provide a **pre-scrubbed** train JSONL + filled `docs/model-factory/stage0/signoffs/<slice-id>.md` with Owner `APPROVE`.

Then: `python -m opjax.factory render-tinker …` → `preflight` (no `--allow-public-fixture`) → Stage-5 LoRA under private spend cap.

## Policy

Do **not** invent private trajectories or upload Conway/employer data without Stage-0 sign-off.
