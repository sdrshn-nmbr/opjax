# Stage 6 — Thin RL hparams (binding defaults)

| Knob | Value | Notes |
|------|-------|-------|
| Base / warm-start | Stage-5 sampler `tinker://21e391ab-…/sampler_weights/final` | Fresh optimizer for RL |
| Model | `thinkingmachines/Inkling` | Tinker allowlist |
| Algorithm | On-policy GRPO-style (mean-centered advantages) | Tinker Tutorial 104 / `importance_sampling` |
| `learning_rate` | `1e-5` | ~10× lower than Stage-5 SFT constant 1e-4 |
| `group_size` | `2` (thin) … `4` | Skip degenerate groups |
| `max_steps` | `15–20` first claim | Hard stop ≤32 without amendment |
| `max_tokens` | `512` | Completions |
| `kl_penalty_coef` | `0.05` | When using cookbook KL helpers |
| RL fuel split | `train` only | Never sealed |
| Eval split | sealed v2 (`splits.json` version 2) | Once after train |
| Cap | $750 / 21 days | Soft; wallet may be ~$65 until reload |

Spend gate: no paid loop without `--i-approve-spend` after operator approval.
