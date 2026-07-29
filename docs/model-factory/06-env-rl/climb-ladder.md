# Climb ladder — sealed pass-rate deltas

**Updated:** `2026-07-23T00:01:05.982286+00:00`

Headline metric = SudarshanBench **sealed** pytest pass rate. Never train on sealed.

| Rung | Stage | Arm | Split | n | Pass rate | Δ vs prior | Notes |
|------|-------|-----|-------|---|-----------|------------|-------|
| `stage5-before-best-control` | 5 | `fewshot_rag` | `sealed_v1` | 4 | **0.583** | — | Best no-training control on sealed v1 (seeds 0–2 mean) |
| `stage5-lora-sealed-v1` | 5 | `lora` | `sealed_v1` | 4 | **1.000** | +0.417 | Stage-5 LoRA claim win; kill not triggered |
| `stage6-lora-baseline-sealed-v2` | 6 | `lora` | `sealed_v2` | 8 | **0.875** | -0.125 | Pre-RL Stage-5 LoRA on hardened sealed v2; fail sb-0013 only |
| `stage6-thin-rl-sealed-v2` | 6 | `thin_rl` | `sealed_v2` | 8 | **0.875** | +0.000 | GRPO v1b after-eval seed0; fail sb-0013 only; Δ=+0.000 vs LoRA baseline → Stage-6 kill |

## Profile (latest RL run)

```json
{
  "fuel_task_ids": [
    "sb-0001",
    "sb-0002",
    "sb-0003",
    "sb-0004",
    "sb-0005",
    "sb-0006",
    "sb-0007"
  ],
  "fuel_splits": [
    "train",
    "dev"
  ],
  "n_steps_ran": 5,
  "n_steps_trained": 1,
  "mean_reward_curve": [
    1.0,
    0.8125,
    1.0,
    1.0,
    1.0
  ],
  "total_rollouts": 80,
  "total_datums": 4,
  "total_completion_tokens": 2509,
  "wall_s": 354.89,
  "aborted_reason": "no training signal for 3 consecutive steps (fuel likely saturated; expand tasks or raise temperature)",
  "baseline_sealed_v2_pass_rate": 0.875,
  "kill_rule": "post_rl_sealed_pass_rate <= 0.875 under budget \u2192 kill Stage-6",
  "final_sampler": "tinker://539b5033-b78b-51f6-9d0e-97e0262490f2:train:0/sampler_weights/final",
  "final_weights": "tinker://539b5033-b78b-51f6-9d0e-97e0262490f2:train:0/weights/final",
  "history_path": "data/model-factory/rl/thin-v1/thin_rl_history.json"
}
```
