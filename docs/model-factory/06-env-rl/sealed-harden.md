# Sealed harden gate (pre–Stage-6 RL)

**Date:** 2026-07-19  
**Why:** Stage-5 LoRA scored **1.00** on sealed v1 (n=4). Without headroom, Stage-6 cannot show sealed lift (or a real kill).

## What changed

| Item | Action |
|------|--------|
| Manifest | `splits.json` → **version 2** |
| Archive | [`../02-sealed-eval/sudarshanbench/splits.v1-freeze.json`](../02-sealed-eval/sudarshanbench/splits.v1-freeze.json) |
| New sealed IDs | `sb-0013` merge intervals, `sb-0014` roman→int, `sb-0015` group anagrams, `sb-0016` search insert |
| Never-train | All sealed v1 + v2 IDs remain forbidden for training |

Protocol satisfied: additions via **versioned manifest**, not by moving train/dev → sealed silently.

## Baseline command (Stage-5 LoRA arm, no hparam search)

```bash
python -m opjax.model_factory.eval_sudarshanbench \
  --arms lora \
  --sampler-path 'tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final' \
  --split sealed \
  --seed 0 \
  --stage 6 \
  --out-dir data/model-factory/evals/sealed-v3-baseline-lora
```

Copy key JSON summaries under `docs/model-factory/06-env-rl/evals/` when available.

## Baseline status (2026-07-19)

Attempted from cloud agent host → **`tinker.APIConnectionError`** (no egress to Tinker auth). Status artifact: [`evals/baseline-lora-v3-status.json`](evals/baseline-lora-v3-status.json).

Sealed **tasks** are hardened (n=8). Numeric LoRA baseline on sealed v2 is **pending** a host with Tinker connectivity. **Not waived.**

## Risk if baseline skipped

Written waiver would state: Stage-6 kill compares against Stage-5 LoRA on **sealed v2**, so a missing baseline makes the kill uninterpretable. **Do not waive** — run baseline once (sampling spend, typically ≪ $25).
