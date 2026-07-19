# Model Factory

Living experiment artifacts for the personalized coding-model factory.

**Source plan (do not edit from agents casually):** `/opt/cursor/artifacts/plans/inkling_coding_ft_experiment_269bb455.plan.md`

**Status:** Stages **0–2** enforceable. Stage **3** base locked (Inkling). Stage **4** closeout recorded. Stage **5** LoRA sealed win (v2). Stage **6** env-qual + thin-RL tooling in progress — no Inkling RL spend without operator OK. Stages **7–10** remain gated runbooks.

## Index

| Stage | Path | Gate |
|-------|------|------|
| 0 Governance | [00-governance/](00-governance/) ([live rights-manifest.json](00-governance/rights-manifest.json)) | Axport cleared; **Tinker** upload approved; other providers denied |
| 1 Claim | [01-claim/](01-claim/) | Written hypothesis + kill + stats |
| 2 Sealed eval | [02-sealed-eval/](02-sealed-eval/) | Disjoint splits; DeepSWE report never trains |
| 3 Tournament | [03-tournament/](03-tournament/) | Cost-normalized base pick |
| 4 Data factory | [04-data-factory/](04-data-factory/) — [audit closeout](04-data-factory/audit-metrics.md) | Audit metrics + scrubbed JSONL |
| 5 Controlled LoRA | [05-controlled-lora/](05-controlled-lora/) — [results-v2](05-controlled-lora/results-v2.md) — [promote](05-controlled-lora/promote-to-stage6.md) | Sealed win vs no-training controls |
| 6 Env RL | [06-env-rl/](06-env-rl/) — [artifacts](06-env-rl/artifacts.md) — [harden](06-env-rl/sealed-harden.md) | Verifier FP/FN + lag bounds |
| 7 Fusion | [07-fusion/](07-fusion/) | Solo sidekick sealed win first |
| 8 Teacher | [08-teacher/](08-teacher/) | OPD only if logprobs+token-compatible |
| 9 PorTAL | [09-portal/](09-portal/) | Qwen3→Gemma replication first |
| 10 KV | [10-kv/](10-kv/) | Same-checkpoint eng before neural research |
| Agent ops | [agent-ops/](agent-ops/) | Spend ledger + runbooks |
| References | [../../references/model-factory/](../../references/model-factory/) | Citation packs |

## Tooling

```bash
# Scrub secrets
uv run opjax-model-factory scrub path/to/file.txt --write /tmp/out.txt

# Hard pre-upload gate (fails closed)
uv run opjax-model-factory pre-upload-gate \
  --source path/to/slice.jsonl \
  --provider tinker \
  --rights-manifest docs/model-factory/00-governance/rights-manifest.example.json \
  --slice-id example-public-oss \
  --spend-ledger docs/model-factory/00-governance/spend-ledger.example.json \
  --output-dir data/model-factory/audits

# Audit trajectory JSONL
uv run opjax-model-factory audit-jsonl path/to/data.jsonl

# Validate sealed splits
uv run opjax-model-factory check-splits \
  --manifest docs/model-factory/02-sealed-eval/sudarshanbench/splits.json

# Stage 6 env-qual (local / $0)
uv run opjax-model-factory scan-solutions \
  --write docs/model-factory/06-env-rl/evals/solution-scan.json
uv run opjax-model-factory verifier-probe \
  --write docs/model-factory/06-env-rl/evals/verifier-probe.json

# Thin RL planner only (no spend)
uv run opjax-model-factory thin-rl -- --dry-run
```

Also: `uv run opjax model-factory -- scrub …`
