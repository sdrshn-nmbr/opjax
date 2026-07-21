# Stage 4 — Audit metrics checklist

Before marking a slice `secrets_scrubbed` / proposing Stage 5:

- [x] `audit-jsonl` report saved under `data/model-factory/audits/`
- [x] complete_trajectories / n_records ≥ agreed threshold (record it)
- [x] duplicate_hashes investigated
- [x] recovery_segments > 0 or explicit waiver
- [x] license_tags populated or slice marked private-conway
- [x] scrub gate run; hits remediated
- [x] canaries embedded for any upload candidate
- [x] rights manifest slice flags updated by a human

## Closeout record (2026-07-19) — slice `axport-all-available`

| Field | Value |
|-------|-------|
| Source | Full R2 `export.zip` via Stage-4 max-yield filters (`allowlist.yaml` `axport-all`) |
| Kept sessions | **5015** |
| Single-turn SFT rows | **25386** (+ 3 canary rows in scrubbed upload) |
| Approx assistant tokens | ~14.6M (audit whitespace proxy) |
| Gate artifact | `data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl` (gitignored `data/`) |
| `pre-upload-gate` | **passed** for provider `tinker` (recorded in Stage-5 results-v2) |
| Rights | `rights_cleared: true`, `secrets_scrubbed: true` in [`../00-governance/rights-manifest.json`](../00-governance/rights-manifest.json) |
| Canaries | JSONL-safe canaries embedded; `canary_tested` flipped true on closeout |
| Recovery | `keep_recovery_segments: true` in allowlist; no success-only wipe |
| License | `mixed-axport` / `may-include-private-conway` (private-conway slice) |
| Rebuild? | **No** — 25k JSONL not rebuilt; audit showed no scrub failure / contamination that would force a refresh |

**Decision:** Stage 4 data job is closed for the Stage-5/6 claim chain. Further data work is a new dated slice, not a silent overwrite.
