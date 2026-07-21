# Spend ledger ops

- Template / reconciled snapshot: [../00-governance/spend-ledger.example.json](../00-governance/spend-ledger.example.json)
- Human-readable reconcile: [spend-ledger-reconcile.md](spend-ledger-reconcile.md)
- Live file: `data/model-factory/ledger/spend-ledger.json` (under gitignored `data/`)
- Update `usd_spent`, `usd_remaining`, and append `entries[]` after every paid job.
- Pre-upload gate reads `usd_remaining`.
- 2026-07-19: example ledger populated from Tinker Usage period total **$286.34** (Stage-5 Inkling est. $126 + unallocated Jul-13 residual).
