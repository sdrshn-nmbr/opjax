# Stage 5 results v1 — Inkling LoRA (2026-07-16)

## Run

| Field | Value |
|-------|-------|
| Base | `thinkingmachines/Inkling` |
| Trainer | Tinker LoRA rank 16, lr 1e-4, batch 4, max_length 4096 |
| Data | 300 axport Conway sessions → 223 single-turn pairs (scrubbed; pre-upload-gate passed) |
| Steps | v1 crashed at 32 (multi-turn `tml_v0` limit); v1b resumed from `000030`, completed 30 steps |
| Final sampler | `tinker://ca64cb06-6f72-57e6-b469-444b0a5940df:train:0/sampler_weights/final` |
| Train NLL | ~1.96 → ~1.12 (v1b) |
| Logs | `logs/model-factory/inkling-sft-v1` / `inkling-sft-v1b` (local) |

## Sealed SudarshanBench (n=4 micro-tasks)

| Arm | Pass | Pass rate |
|-----|------|-----------|
| Base Inkling | 0 / 4 | 0.0 |
| LoRA (final) | 4 / 4 | **1.0** |

Base often emitted tool-call scaffolding instead of a code block (extraction failed). LoRA returned fixed `solution.py` that passed `pytest`.

## Kill condition

**Not triggered** — LoRA beat base on sealed under the Stage-5 budget line item (~USD 25 estimated in ledger).

## Caveats

- Sealed set is tiny in-repo micro-benchmarks, not full agentic SWE.
- Multi-turn axport traces required single-turn flattening for Inkling `tml_v0` SFT.
- Spend USD is estimated; confirm against Tinker billing.
- Do not treat this as Composer/SWE parity.

## Next

- Grow sealed set; add prompt_rules / static_policy controls.
- Optional: longer SFT / thin RL (Stage 6) only if sealed stays green as tasks harden.
