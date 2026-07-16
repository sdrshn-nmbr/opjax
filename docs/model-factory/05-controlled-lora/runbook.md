# Stage 5 — First controlled LoRA (gated runbook)

**Do not start until:** Stages 0–2 complete, Stage 3 base chosen, Stage 4 slice cleared, sealed set frozen & non-empty.

**Base (locked):** `thinkingmachines/Inkling` on **Tinker LoRA** — [decision.md](../03-tournament/decision.md).

## Procedure

1. Pass `pre-upload-gate` for **tinker** + slice `axport-all-available` (or a scrubbed sub-slice).
2. Train LoRA/SFT on Inkling under Stage-5 spend cap ([spend-caps.md](../00-governance/spend-caps.md)).
3. Evaluate all control arms + `lora` on **dev** while iterating.
4. **Once:** evaluate on **sealed**; no further hparam changes.
5. Log train loss, held-out NLL, sealed metrics, cost, latency ([stats-design.md](../01-claim/stats-design.md)).

## Kill

If `lora` ≤ best control on sealed under budget → stop weight claims (see [hypothesis.md](../01-claim/hypothesis.md)).

## Transfer boundary

Qwen/Laguna ablations ≠ Inkling evidence. Primary Stage-5 claim is Inkling-only.
