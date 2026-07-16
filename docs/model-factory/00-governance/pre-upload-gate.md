# Stage 0 — Hard pre-upload gate

**Decision:** Fail closed before any managed-trainer upload.  
**Implementation:** `opjax.model_factory.pre_upload.run_pre_upload_gate` / CLI `pre-upload-gate`.

## Checks (all must pass)

1. Rights manifest exists and is readable JSON.
2. `providers.<provider>.upload_approved` and `retention_reviewed` are true.
3. Target `data_slices[<id>].rights_cleared` and `secrets_scrubbed` are true.
4. Spend ledger `usd_remaining > 0` (unless `--allow-no-spend-check` for dry public experiments).
5. Secret scrub runs; **any** scrub hit fails the gate (forces human re-review).
6. Canary tokens are embedded into the outgoing scrubbed artifact and persisted under the output dir.

## Usage

```bash
uv run opjax-model-factory pre-upload-gate \
  --source data/model-factory/audits/example.jsonl \
  --provider tinker \
  --rights-manifest docs/model-factory/00-governance/rights-manifest.example.json \
  --slice-id example-public-oss \
  --spend-ledger docs/model-factory/00-governance/spend-ledger.example.json \
  --output-dir data/model-factory/audits
```

Exit codes: `0` ok, `3` gate failed.

## Non-goals

- This gate does **not** prove legal compliance; it enforces process.
- Canaries detect crude leakage into later public text; they do not replace DPA review.
