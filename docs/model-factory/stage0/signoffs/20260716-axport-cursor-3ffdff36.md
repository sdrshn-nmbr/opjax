# Rights sign-off — 20260716-axport-cursor-3ffdff36

| Field | Value |
|-------|-------|
| Slice ID | `20260716-axport-cursor-3ffdff36` |
| Created | 2026-07-16 |
| Owner | sudarshan (via cloud agent) |
| Intended provider(s) | tinker |
| Dataset path(s) | `data/factory/tinker/train_axport_cursor.jsonl` |
| Row count / trajectory count | 74 |
| Content SHA256 (train JSONL) | `3ffdff369237589cbdc1ff24f14986a9d191b9c59475d908e2b613da6f212f16` |

## Source inventory

| Source | Includes PII / secrets risk? | License / ownership | Allowlisted? |
|--------|------------------------------|---------------------|--------------|
| Axport Cursor sessions (R2 `axport` / cursor.zip) | High until scrubbed | Personal machine traces; Conway cwd demos | Yes — cwd allowlist conway/conway + opjax |
| Scrubbed via opjax.factory | Residual risk accepted | — | Yes |

## Employer / OSS gates

- [x] Confirmed projects may leave the VM for Tinker training (owner-provided R2 + request to proceed)
- [x] Managed trainer Tinker approved for this slice
- [x] No intentional customer production dumps; coding agent transcripts only
- [x] Scrub + canary preflight required before upload

## Provider disclosure

| Provider | Data leaving machine | Approved for this slice? | Retention answer ref |
|----------|----------------------|--------------------------|----------------------|
| Tinker | Training JSONL, run metadata, checkpoints | Yes | Accepted risk — retention UNKNOWN; owner approved 2026-07-16 |

## Scrub / canary

- [x] Scrub pipeline: opjax.factory.render_tinker scrub=True
- [x] Canary file: data/factory/canaries.txt
- [x] Preflight must pass before train

## Sign-off

| Role | Name | Date | Decision |
|------|------|------|----------|
| Owner | sudarshan | 2026-07-16 | APPROVE |
