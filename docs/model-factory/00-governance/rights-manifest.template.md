# Stage 0 — Rights manifest (template)

Copy to a dated JSON instance (see [rights-manifest.example.json](rights-manifest.example.json)) and fill before any upload.

## Required fields per data slice

| Field | Meaning |
|-------|---------|
| `id` | Stable slice identifier |
| `sources` | axport / OSS repo / synth / other |
| `employer_approval` | `required` / `obtained` / `n_a` for Conway private |
| `license_tags` | SPDX or `private-conway` / `unknown` |
| `pii_risk` | `low` / `medium` / `high` |
| `third_party_model_outputs` | Whether traces contain other vendors' generations |
| `rights_cleared` | Human attestation boolean |
| `secrets_scrubbed` | True only after scrub + manual review |
| `canary_tested` | True after canary embed + optional post-hoc leak check |

## Required fields per provider

| Field | Meaning |
|-------|---------|
| `upload_approved` | Explicit allow for this provider |
| `retention_reviewed` | DPA / retention notes read |
| `retention_summary` | Free text: how long data/logs are kept |
| `zero_retention_available` | If known |
| `contact` | Who approved |

## Hard gate

`opjax-model_factory pre-upload-gate` fails unless:

1. `providers.<name>.upload_approved` and `retention_reviewed` are true.
2. Matching `data_slices[].rights_cleared` and `secrets_scrubbed` are true.
3. Spend ledger has headroom (unless explicitly bypassed for dry-run public data).

Regex scrub alone never sets `rights_cleared`.
