# Stage 5 results v2 — full axport Inkling LoRA (2026-07-16 → 2026-07-17)

## Plan adherence

- Stage 0 gate: `pre-upload-gate` **passed** for Tinker + `axport-all-available` (JSONL-safe canaries).
- Stage 2: sealed IDs never trained; before controls then after `lora` (seeds `{0,1,2}`).
- Stage 4: full R2 `export.zip`, max-yield filters (operator: more data always).
- Stage 5: [hparams-v2.md](hparams-v2.md) / [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — all layers (Tinker defaults), rank 64 = Inkling max, constant LR, batch 16.
- Kill condition: `lora` vs best sealed control under Stage-5 spend cap — **not triggered**.

## Data

| Field | Value |
|-------|-------|
| Source | `data/model-factory/axport/export.zip` (6678 `.md`) |
| Kept sessions | **5015** |
| Single-turn SFT rows | **25386** (+ 3 canary rows in scrubbed upload) |
| Approx assistant tokens | ~14.6M (audit whitespace proxy) |
| Gate artifact | `data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl` |

## Before evals (sealed) — no weight update

| Arm | seed0 | seed1 | seed2 | Mean |
|-----|-------|-------|-------|------|
| `base` | 0/4 | — | — | 0.00 |
| `prompt_rules` | 0/4 | — | — | 0.00 |
| `fewshot_rag` | 1/4 | 3/4 | 3/4 | **0.583** |
| `static_policy` | 1/4 | 1/4 | 0/4 | 0.167 |

Best control on sealed: **`fewshot_rag` (~0.58 mean)**.

## Train

| Field | Value |
|-------|-------|
| Base | `thinkingmachines/Inkling` |
| LoRA | rank **64**, lr 1e-4 **constant**, batch **16**, max_length 4096, 1 epoch |
| Steps | **1586 / 1586** (resumed from `000800` after billing reload) |
| Train NLL | ~2.23 → **0.55** (final step) |
| Elapsed tokens | ~48.4M |
| Final sampler | `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final` |
| Logs | `logs/model-factory/inkling-sft-v2/` |

Paused mid-run on HTTP 402 (billing); resumed after operator reload from checkpoint `000800` (new train client id `21e391ab-…`).

## After evals (sealed) — LoRA

| Arm | seed0 | seed1 | seed2 | Mean |
|-----|-------|-------|-------|------|
| `lora` | **4/4** | **4/4** | **4/4** | **1.00** |

Dev after (sanity): `lora` 3/3.

## Primary contrast (kill check)

| Contrast | Result |
|----------|--------|
| `lora` mean sealed pass rate | **1.00** |
| Best control (`fewshot_rag`) mean | **0.58** |
| Absolute lift | **+0.42** (≥ +10 pp MDE target) |
| Kill triggered? | **No** |

Caveat: sealed n=4 micro-tasks; not SWE/agentic parity. Still passes Stage-5 falsification gate for this task family.

## Artifacts

- Before summaries: [evals-v2/](evals-v2/)
- After reports: `data/model-factory/evals/sealed-v2-after/` (gitignored `data/`; key JSON copied under `evals-v2/`)

## Spend

Reconcile USD against [Tinker Usage](https://tinker.thinkingmachines.ai/usage). Stay inside Stage-5 **$500** hard cap. Ledger entry marked pending reconcile.
