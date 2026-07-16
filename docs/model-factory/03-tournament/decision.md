# Stage 3 decision (operator deferred 2026-07-16)

## Chosen Stage-5 base

**`Qwen/Qwen3-8B` (or nearest Tinker-supported Qwen3 8B-class chat id)** via **Tinker LoRA**.

### Why

| Criterion | Choice |
|-----------|--------|
| Trainable **now** | Yes on Tinker (Wave A ready) |
| Cost | Far below full Inkling serve/train |
| Fairness | Dense baseline; PorTAL-compatible family later |
| Product path | Inkling remains the later product hypothesis — **not** evidenced by this first LoRA |

## Rejected for *first* LoRA

| Base | Why not first |
|------|----------------|
| Inkling full | Expensive; prove recipe on cheap base first |
| Inkling-Small | Not verified available |
| Laguna XS.2 | Good Stage 6 / Prime stand-in; don’t conflate with Tinker/Inkling claims |

## What this does *not* decide

Fusion roles, PorTAL targets, or “Inkling is best.” Re-run Stage 5 on Inkling only after Qwen recipe works.
