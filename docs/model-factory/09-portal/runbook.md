# Stage 9 — PorTAL research (Path B)

**Separate from continual OPD / CL weight updates.**

## Milestone order

1. **Reproduce** published Qwen3→Gemma on [portallib](https://github.com/ramp-public/portallib) (MCQ).
2. Short coding **classification / next-action** on supported dense bases.
3. Only then consider agentic trajectories / MoE (Laguna, Inkling) as new research.

## Invalid first spike (do not do)

Laguna-XS.2 + Qwen3.5 MoE + long tool trajectories under portallib v0.1.

## Metrics

Recovered lift vs from-scratch LoRA on a **non-sealed** probe first; sealed only after method works.
