# Secret scrub + canary leak tests (Stage 0)

Regex scrub ≠ legal gate. Upload is disclosure. This doc defines the **technical** hard gate implemented by `opjax.factory.scrub` / `preflight`.

## Goals

1. Remove high-confidence secrets from trajectories before any trainer upload.
2. Plant **canaries** in raw/dev copies; fail closed if canaries appear in upload artifact.
3. Report scrub recall on a held-out dirty fixture (unit tests).

## Patterns scrubbed (v1)

- AWS-style access keys (`AKIA…`)
- Anthropic / OpenAI / HF / generic `sk-` / `hf_` tokens (heuristic)
- PEM blocks (`BEGIN … PRIVATE KEY`)
- `.env` style `KEY=value` lines for known secret key names
- Bearer tokens in `Authorization:` headers
- Slack / GitHub classic PATs (heuristic)

False positives are acceptable; false negatives on canaries are not.

## Canary protocol

1. Generate canary strings: `CANARY_FACTORY_<uuid>` (and optional fake key-shaped canaries).
2. Inject into a **dev-only** copy of raw data or into unit fixtures — never into production secrets stores.
3. After scrub + render, scan upload JSONL:
   - Any canary hit → **preflight FAIL**
4. Record canary IDs in the slice sign-off (not the canary values in git if they look like keys; UUID canaries OK).

## Pre-upload gate

```bash
python -m opjax.factory preflight \
  --dataset data/factory/tinker/train.jsonl \
  --manifest docs/model-factory/stage0/signoffs/<slice-id>.md \
  --provider tinker \
  --canary-file data/factory/canaries.txt
```

Public fixture smoke may use `--allow-public-fixture` and skip manifest APPROVE.

## Metrics (Stage-4 audit)

- Canary detection rate on planted set (must be 100% for CI fixtures)
- Count of scrub substitutions
- Residual high-entropy token warnings (informational)

## Out of scope (v1)

- Perfect PII redaction of names/emails (best-effort optional)
- Legal classification of employer IP (rights manifest)
