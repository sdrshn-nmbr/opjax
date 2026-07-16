# Stage 4 — Audit metrics checklist

Before marking a slice `secrets_scrubbed` / proposing Stage 5:

- [ ] `audit-jsonl` report saved under `data/model-factory/audits/`
- [ ] complete_trajectories / n_records ≥ agreed threshold (record it)
- [ ] duplicate_hashes investigated
- [ ] recovery_segments > 0 or explicit waiver
- [ ] license_tags populated or slice marked private-conway
- [ ] scrub gate run; hits remediated
- [ ] canaries embedded for any upload candidate
- [ ] rights manifest slice flags updated by a human
