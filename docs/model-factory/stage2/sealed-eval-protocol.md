# Stage 2 — Sealed evaluation protocol

## Splits

| Split | Purpose | May train? | May hill-climb? |
|-------|---------|------------|-----------------|
| `train` | SFT / LoRA fuel | Yes | n/a |
| `dev` (rotating) | Hyperparams, prompts, filters | No weights on sealed; may tune on dev | Yes |
| `sealed-test` | Report / kill decision | **Never** | **Never** |
| `time-forward` | Shadow freshness (P3) | Never | Observe only |

Paths (reserved even if empty):

```
data/factory/splits/train/
data/factory/splits/dev/
data/factory/splits/sealed/
data/factory/splits/time_forward/
```

## Freeze rule

Before any weight training that will be cited as Stage-5 evidence:

1. Create `data/factory/splits/sealed/.frozen` with timestamp + task list hash
   (empty task list is allowed for plumbing era — document emptiness).
2. Commit the freeze stub / task ids under `docs/model-factory/stage2/` when tasks exist.
3. Adding sealed tasks after a claimed run **invalidates** that claim (new freeze id).

**Current freeze (2026-07-16):** sealed set is **empty on purpose**.
File: `sealed-freeze-empty.md`. Any Inkling LoRA before tasks land = exploratory.

## SudarshanBench

Private sealed eval + rubrics (to build). Rules:

- Sealed portion never trains
- Graders isolated from generators where possible
- Solution-access scanners before RL (Stage 6)

## Anti-cheat (minimum)

- No sealed task text in train JSONL (preflight path check)
- No DeepSWE public report split in train (see `deepswe-policy.md`)
- Log prompt templates used for each control condition

## Controls for Stage-5 comparisons

1. Base model (no adapter)
2. Prompt + rules (no train)
3. Few-shot / RAG pack (no train)
4. Static policy / linters
5. LoRA (trained)

## Gate

Protocol documented + sealed freeze recorded before funded Stage-5 claim.
Plumbing smoke on public fixtures does not require sealed tasks.
