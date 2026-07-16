# Spend / compute caps (Stage 0)

**Status:** DRAFT defaults accepted for first grind (2026-07-16). Tighten when real invoices arrive.

## Soft ledger rules

1. Every Tinker / GPU run gets a memo under `docs/model-factory/runs/`.
2. Agent must stop at the first cap hit (USD, wall-clock, or steps).
3. Private uploads require Stage-0 sign-off **and** remaining budget under private cap.
4. Do not start a second private run while the first is open without explicit approval.

## DRAFT caps

| Class | Max USD | Max wall-clock | Max steps | Notes |
|-------|---------|----------------|-----------|-------|
| Public plumbing smoke | **50** | **30 min** | **50** | Fixture / public JSONL only |
| Stage-5 first private LoRA | **300** | **4 h** | **2000** | Scrubbed train slice only |
| Stage-3 tournament smoke | **100** | **2 h** | n/a | Inference-heavy; defer if smoke unpaid |

Env mirrors (optional): `FACTORY_SMOKE_MAX_USD`, `FACTORY_SMOKE_MAX_STEPS`, `FACTORY_PRIVATE_MAX_USD`.

## Inkling-specific notes

- LoRA LR must be set **manually** (cookbook `get_lr` raises `NotImplementedError`).
- Prefer small `batch_size` (8–32) and `lora_rank` 16–32 for first runs.
- Do not download full Inkling weights for Tinker training.

## Approval

Changing caps upward requires a one-line note in `CHANGELOG.md` with date + reason.
