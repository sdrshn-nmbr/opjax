# JAXBench — hard-Pallas prompt probe (Stage-5 LoRA)

**Takeaway:** Forcing Pallas in the prompt works for *emission* (15/17 priority workloads call `pl.pallas_call`), but **correctness collapses to 0/17** on the CPU grader. Soft prompt was 13/17 correct with **0** Pallas. Prompting alone does not get us usable kernels.

## Setup

- Sampler: Stage-5 LoRA `tinker://21e391ab-…/sampler_weights/final`
- Tier: priority (`1p`–`17p`)
- Flag: `--prompt-mode pallas_required`
- Out: `data/model-factory/evals/jaxbench-pallas-prompt-lora/`

## Soft vs hard prompt

| Mode | Correct | Mentions / uses Pallas |
|------|---------|------------------------|
| soft (prefer Pallas) | 13/17 (~0.76) | 0/17 |
| **pallas_required** | **0/17** | **15/17** (`pallas_call`) |

## What it actually wrote

After stripping sampler junk tokens (`<|end_message|>` etc.):

- 15/17 emit `pl.pallas_call`
- 9/17 import / use `pltpu`
- 7/17 use GPU-ish `pl.load` / `pl.store` (not the usual TPU BlockSpec style)
- 2/17 (`15p` RetNet, `16p` Mamba) ignored the hard rule and stayed pure JAX
- Most failures on CPU were runtime / API misuse; one remaining syntax fail (`5p`)

CPU grading is a weak signal here: real TPU Pallas often cannot run on laptop JAX anyway. Still, the generated kernels look like shallow / wrong Pallas (mixed Triton-style APIs, bad BlockSpecs), not gold `optimized.py` quality.

## Implication

Prompting can push Inkling *toward* Pallas syntax. It cannot replace Pallas training fuel or a TPU reward loop. Next lift is still SFT/RL on real Pallas (e.g. `pallasbench-unified` + JAXBench gold), then re-eval on v5e.
