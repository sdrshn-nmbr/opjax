# Phase 1 Foundation Notes

## Source Facts

- `cua-bench` represents GUI actions as dataclasses such as `ClickAction`, `TypeAction`, `HotkeyAction`, `ScrollAction`, `WaitAction`, and `DoneAction`.
- `cua-bench` accepts snake-case action strings like `click(100, 200)`, but our training target starts in FunctionGemma function-call format so Phase 2 has less format drift.
- Gemma 4 vision uses an additive positional embedding at `VisionEntry.pos_emb_param` in `gemma/gm/nn/gemma4/vision/_layers.py`.
- The Tzafon positional-scaling probe should multiply that embedding before measuring click accuracy.

## Learning Hook

The first slice deliberately separates three things that agent code often mixes together:

- syntax: parsing model text into an action
- semantics: deciding whether an action means success in an environment
- supervision: repairing failed outputs into better training targets

Keeping those separate is what lets us swap a fake policy for Gemma 4, a deterministic verifier for cua-bench, and a fake repairer for Claude without rewriting the loop.
