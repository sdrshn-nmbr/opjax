# Stage 5 hparams v2 — aligned with TML LoRA Without Regret

Source: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman / Thinking Machines Lab, Sep 2025).

## Binding choices for full-axport SFT

| Knob | v2 choice | Why |
|------|-----------|-----|
| Layers | Tinker defaults: `train_mlp=True`, `train_attn=True`, `train_unembed=True` | Blog: apply LoRA to **all layers**, especially MLP/MoE; attention-only underperforms |
| Rank | **64** (up from v1’s 16) | Larger corpus needs more adapter capacity; low ranks fall off the loss curve when capacity-bound |
| Batch size | **4** | LoRA is less tolerant of large batches than FullFT; keep small |
| LR | **1e-4**, schedule **constant** | Optimal LoRA LR ≈ independent of rank (with α/r); constant LR matches blog sweeps; ~10× FullFT heuristic for longer runs |
| Epochs | **1** | Match blog SFT protocol; avoid multi-epoch memorization before sealed read |
| Max length | **4096** | Same as v1; axport turns truncated at curation |
| Data | Full R2 `export.zip` under Stage-4 filters (`require_tool_use`, min turns) → single-turn flatten for `tml_v0` | Operator: use whatever axport provides; plan Stage 4 target = hundreds–low thousands trajectories |

## Explicit non-choices

- No sealed-based hparam search (stats-design: peeking forbidden; use `dev` only while iterating).
- No attention-only LoRA.
- No FullFT (out of Stage-5 economics for Inkling).
