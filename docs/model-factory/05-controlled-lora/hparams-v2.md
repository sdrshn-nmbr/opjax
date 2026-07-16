# Stage 5 hparams v2 — aligned with TML LoRA Without Regret

Source: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman / Thinking Machines Lab, Sep 2025).

## Binding choices for full-axport SFT

| Knob | v2 choice | Why |
|------|-----------|-----|
| Layers | Tinker defaults: `train_mlp=True`, `train_attn=True`, `train_unembed=True` | Blog: apply LoRA to **all layers**, especially MLP/MoE; attention-only underperforms |
| Rank | **64** (Tinker max for Inkling; up from v1’s 16) | Prefer max capacity under provider cap; blog: higher rank delays capacity falloff on large SFT sets |
| Batch size | **16** | Keep moderate (blog: LoRA less tolerant of very large batches); 16 balances wall-clock vs dynamics on ~25k examples |
| LR | **1e-4**, schedule **constant** | Optimal LoRA LR ≈ independent of rank (with α/r); constant LR matches blog sweeps; ~10× FullFT heuristic for longer runs |
| Epochs | **1** | Match blog SFT protocol; avoid multi-epoch memorization before sealed read |
| Max length | **4096** | Same as v1; per-example caps at curation |
| Data | Full R2 `export.zip`, max recall (min 2 turns, no tool-use filter, synthetic user for orphan assistants) → single-turn flatten **before** session truncate | Operator: more data always; plan Stage 4 = hundreds–low thousands+ trajectories |

## Explicit non-choices

- No sealed-based hparam search (stats-design: peeking forbidden; use `dev` only while iterating).
- No attention-only LoRA.
- No FullFT (out of Stage-5 economics for Inkling).
