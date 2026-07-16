# Stage 2 — Sealed evaluation protocol

**Decision:** Split task families into train / rotating-dev / sealed-test / time-forward.  
**Gate:** Protocol + frozen sealed set before any Stage-5 training.

## Split families

| Split | Role | May train? | May tune prompts/hparams? | May report as headline? |
|-------|------|------------|---------------------------|-------------------------|
| `train` | SFT/RL fuel | Yes | Yes | No |
| `dev` | Rotating development | No weights on these IDs | Yes | No (dev only) |
| `sealed` | Held-out private truth | **Never** | **Never** | **Yes** (primary) |
| `time_forward` | Post-freeze tasks | Never | Never | Shadow / freshness (P3) |

Manifest: [sudarshanbench/splits.json](sudarshanbench/splits.json)  
Tooling: `uv run opjax-model-factory check-splits --manifest …`

## SudarshanBench layout

```
02-sealed-eval/sudarshanbench/
  splits.json          # task id → split
  tasks/               # task specs (prompt, repo pin, test cmd)
  train/ dev/ sealed/ time-forward/  # optional per-split mirrors
```

Sealed directory starts **empty** (`.gitkeep` only) until the first task freeze commit. After freeze, additions go to `time_forward` or a new versioned manifest — do not silently move IDs from sealed → train.

## DeepSWE policy

See [deepswe-policy.md](deepswe-policy.md). Public report split is listed under `deepswe_report_split` and is treated like sealed for contamination checks.

## Anti-cheat

1. Graders isolated from training machines when practical.
2. Solution-access scanners before RL (Stage 6).
3. No web search / git history shortcuts in eval sandboxes unless the task explicitly allows.
4. Generator ≠ primary grader for the sealed headline metric (tests are the oracle for pass/fail).

## Rubrics (secondary)

- Tigerstyle / philosophy checklist (model- or human-graded) — secondary only.
- Bad-delegation probes — Fusion Stage 7, not Stage 1 primary.
