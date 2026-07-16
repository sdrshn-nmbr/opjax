# Stage 5 results v2 — full axport Inkling LoRA (2026-07-16)

## Plan adherence

- Stage 0 gate: `pre-upload-gate` **passed** for Tinker + `axport-all-available` (JSONL-safe canaries).
- Stage 2: sealed IDs untouched for training; before/after protocol used.
- Stage 4: full R2 `export.zip`, max-yield filters (operator: more data always).
- Stage 5: hparams from [hparams-v2.md](hparams-v2.md) / [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (all layers via Tinker defaults, rank 64 = Inkling max, constant LR, moderate batch).
- Kill condition: compare `lora` vs best **sealed** control under Stage-5 spend cap — **pending** (billing block before after-eval).

## Data

| Field | Value |
|-------|-------|
| Source | `data/model-factory/axport/export.zip` (R2 full dump, 6678 `.md`) |
| Kept sessions | **5015** |
| Single-turn SFT rows | **25386** (+ 3 canary rows in scrubbed upload) |
| Approx assistant tokens | ~14.6M (audit whitespace proxy) |
| Curation | Flatten **before** truncate; synthetic user for orphan assistants; no tool-use filter |
| Gate artifact | `data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl` (25389 parseable lines) |

## Before evals (sealed, seed 0) — no weight update

| Arm | Pass | Pass rate |
|-----|------|-----------|
| `base` | 0 / 4 | 0.00 |
| `prompt_rules` | 0 / 4 | 0.00 |
| `fewshot_rag` | 1 / 4 | **0.25** |
| `static_policy` | 1 / 4 | **0.25** |

Best control on sealed: **0.25**. Artifacts: `data/model-factory/evals/sealed-v2-before/`.

Dev before (iteration only): `fewshot_rag` 3/3, `static_policy` 1/3, base/prompt_rules 0/3 — `data/model-factory/evals/dev-v2-before/`.

## Train (partial — billing stop)

| Field | Value |
|-------|-------|
| Base | `thinkingmachines/Inkling` |
| LoRA | rank **64** (provider max), lr 1e-4 **constant**, batch **16**, max_length 4096, 1 epoch |
| Planned steps | 1586 |
| Completed | **~831 / 1586** (~52%) before pause |
| Train NLL | ~2.23 → ~0.69 (noisy; last logged ~0.69) |
| Run id | `tinker://818a811e-da7c-5c93-a956-26e1de68ff7f:train:0` |
| Last checkpoint | `…/weights/000800` · sampler `…/sampler_weights/000800` |
| Logs | `logs/model-factory/inkling-sft-v2/` |

### Billing block

At ~52% progress Tinker returned **HTTP 402**:

> Access blocked due to billing status. Add payment at https://tinker.thinkingmachines.ai/billing/balance

Sampling from the checkpoint also 402s until balance is restored. Local train process was stopped to avoid retry spam.

### Resume (after payment)

```bash
PYTHONPATH=src .venv/bin/python -m tinker_cookbook.recipes.chat_sl.train \
  log_path=logs/model-factory/inkling-sft-v2-resume \
  model_name=thinkingmachines/Inkling \
  load_checkpoint_path=tinker://818a811e-da7c-5c93-a956-26e1de68ff7f:train:0/weights/000800 \
  dataset=data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl \
  renderer_name=tml_v0 \
  lora_rank=64 learning_rate=1e-4 lr_schedule=constant \
  batch_size=16 max_length=4096 num_epochs=1 \
  save_every=50 eval_every=0 \
  behavior_if_log_dir_exists=delete wandb_project=null
```

Then after-eval:

```bash
PYTHONPATH=src .venv/bin/python -m opjax.model_factory.eval_sudarshanbench \
  --arms lora \
  --sampler-path 'tinker://<run>:train:0/sampler_weights/final' \
  --split sealed --seed 0 --stage 5 \
  --out-dir data/model-factory/evals/sealed-v2-after
```

Primary contrast: `lora` vs best control (0.25 sealed). Kill if `lora` does not beat that under the Stage-5 USD cap.

## After evals

**Not run** — API 402 blocks sampling. Fill here after resume.

## Spend

Reconcile against Tinker Usage after payment. v1 was **$5.52** (dashboard). v2 partial (~24M elapsed tokens at ckpt 800) will dominate; stay inside Stage-5 **$500** hard cap.
