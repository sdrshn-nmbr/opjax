# Stage 0 — Spend caps

**Decision:** Cap side-project spend before any Stage-5 training claim is funded.  
**Audit:** Red-team C8 — no budget ⇒ multi-TB Inkling ops can dominate before science.  
**Status:** Template active; fill real numbers before first paid GPU job.

## Caps (edit before funding Stage 3+)

| Bucket | Cap (USD) | Cap (GPU-hours) | Wall-clock | Notes |
|--------|-----------|-----------------|------------|-------|
| Stage 0–2 docs/tooling | 0 | 0 | — | Local only |
| Stage 3 base tournament smoke | 150 | 20 | 7 days | Inference + tiny smokes; no private upload |
| Stage 4 curation compute | 50 | 5 | 14 days | Local/CPU preferred |
| Stage 5 first LoRA | 500 | 80 | 14 days | **Hard stop** if sealed eval fails |
| Stage 6 thin RL | 750 | 120 | 21 days | Only if Stage 5 passes kill |
| Research (PorTAL/KV) | 300 | 40 | parallel | Separate ledger line; does not unlock Fusion |

**Total soft ceiling (Stages 3–6):** USD 1,450 unless explicitly raised in the spend ledger with a dated note.

## Rules

1. Every paid job logs provider, $, GPU-hours, stage, and hypothesis ID in [spend-ledger.example.json](spend-ledger.example.json) (copy to `data/model-factory/ledger/spend-ledger.json`, gitignored via `data/`).
2. Agents may **propose** runs; humans approve above USD 25 or any private-data upload.
3. `opjax-model-factory pre-upload-gate` requires `usd_remaining > 0`.
4. Full Inkling BF16 (~2 TB) / NVFP4 (~600 GB) inference is **out of budget** unless a dedicated amendment raises the Stage-3 row.

## Success metrics (economics)

- Success per dollar = sealed-task lift / USD spent on that stage.
- Success per GPU-hour = sealed-task lift / GPU-hours.
- Log latency p50/p95 for any served sidekick candidate.
