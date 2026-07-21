# Rights manifest template (Stage 0)

Copy to `signoffs/<slice-id>.md` (or YAML sidecar) and fill before any managed-trainer upload.

## Slice identity

| Field | Value |
|-------|-------|
| Slice ID | `YYYYMMDD-<short-hash>` |
| Created | |
| Owner | |
| Intended provider(s) | e.g. `tinker` |
| Dataset path(s) | |
| Row count / trajectory count | |
| Content SHA256 (train JSONL) | |

## Source inventory

| Source | Includes PII / secrets risk? | License / ownership | Allowlisted? |
|--------|------------------------------|---------------------|--------------|
| Axport sessions (Cursor / Claude / …) | High until scrubbed | Personal / employer? **FILL** | |
| Conway repo traces | Employer IP | Employer policy **FILL** | |
| Allowlisted OSS demos | Low | SPDX **FILL** | |
| Synthetic fixtures (`tests/factory/fixtures`) | None | Apache-2.0 / project | Yes |

## Employer / OSS gates

- [ ] Confirmed which projects may leave the laptop / VM
- [ ] Confirmed managed trainer is an approved disclosure surface for this slice
- [ ] OSS licenses recorded for any third-party code in demonstrations
- [ ] No customer data / regulated data in slice (or explicit legal OK)

## Provider disclosure

| Provider | Data leaving machine | Approved for this slice? | Retention answer ref |
|----------|----------------------|--------------------------|----------------------|
| Tinker | Training JSONL, run metadata, checkpoints | | `provider-retention-checklist.md` |
| Hugging Face Hub | Adapter push (if any) | | |
| W&B / logs | Metrics, possibly samples | | |
| Anthropic/OpenAI | Grader prompts (if used) | | |

## Scrub / canary

- [ ] Scrub pipeline version / commit:
- [ ] Canary IDs planted and **absent** from upload artifact (preflight pass)
- [ ] Scrub recall notes: `stage0/scrub-and-canary.md`

## Sign-off

| Role | Name | Date | Decision |
|------|------|------|----------|
| Owner | | | APPROVE / REJECT |

**Gate:** `python -m opjax.factory preflight` must pass for provider=`tinker` using this signed manifest.
Without APPROVE, private upload is forbidden. Public fixture smoke does not require this file.
