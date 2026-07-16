# Spend ledger ops

- Template: [../00-governance/spend-ledger.example.json](../00-governance/spend-ledger.example.json)
- Live file: `data/model-factory/ledger/spend-ledger.json` (under gitignored `data/`)
- Update `usd_spent`, `usd_remaining`, and append `entries[]` after every paid job.
- Pre-upload gate reads `usd_remaining`.
