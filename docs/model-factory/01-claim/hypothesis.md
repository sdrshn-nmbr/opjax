# Pallas Agent — Falsifiable claim

## Claim (binding)

> Starting from base `thinkingmachines/Inkling`, governed Pallas-specific
> training produces authentic Pallas kernels that compile on TPU v5e, are
> correct at full target shapes, and outperform the JAX/XLA baseline on unseen
> kernel families.

Success is lexicographic: compile, correct, authentic Pallas, stable timing, and
then faster than baseline. Failing an earlier condition prevents headline
credit from a later condition.

## Controls

| Arm | Starting point | Training |
|-----|----------------|----------|
| `A` | Base Inkling | None |
| `B` | Base Inkling | Verified Pallas SFT |
| `C` | Base Inkling | Kernel-domain adaptive LoRA, then B's identical SFT |
| `D` | Base Inkling | Kernel-domain adaptive LoRA only; diagnostic |

Public JAXBench is a comparator, not a globally sealed benchmark. A separate
private family-heldout evaluation is required for a generalization claim.

## Stop conditions

- If B does not improve correct authentic Pallas over A, audit SFT data,
  rendering, model capacity, and evaluation sensitivity before escalation.
- If C does not robustly improve private-family results over B, remove
  domain-adaptive LoRA from the main recipe.
- If improvement occurs only on public JAXBench, make no generalization claim.
- Do not run RL before supervised competence, verifier integrity, reward
  variance, and private-development headroom are demonstrated.

## Non-claims

- No Composer-equivalence or broad continual-pretraining claim.
- No general coding-agent claim.
- No GPU result presented as TPU evidence.
- No inference-time agent result presented as a weight-training result.
