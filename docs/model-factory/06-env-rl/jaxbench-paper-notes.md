# JAXBench paper notes (local PDF)

**Source:** `/Users/sudarshan/Downloads/jaxbench.pdf`  
**Title:** JAXBench: Benchmarking Autonomous TPU Kernel Optimization (Tschand, Hong, et al., Google / Harvard / Berkeley / DeepMind, 2026)  
**Code:** https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/JAXBench

## What matters for our hillclimb

1. **Objective matches ours.** Suite is for AI-generated **Pallas** on TPUs vs XLA (+ Tokamax upper bound on 8 kernels). Plain JAX rewrites are not the win condition.
2. **Gates then metric:** compile → correct (`allclose` atol/rtol=1e-2 bf16) → **speedup over XLA** (device-side `jax.profiler`). Incorrect kernels count as 1× in aggregates.
3. **Context >> model scale on Pallas.** Curated TPU/Pallas docs (Autocomp-style) lifts per-sample correctness **5.8% → 37.3%** and solves **48/50** at 1.28× geomean. API misuse dominates failures without it.
4. **Search structure after correctness.** Autocomp beam search → **1.36×** geomean; recovers much of Tokamax **2.08×** upper bound on the 8 gold kernels (still trails paged/ragged attention).
5. **Paper hardware is v6e.** We use singular **v5e** for cost; same harness, smaller HBM — expect more OOMs on full CONFIG (already saw on LoRA). v6e later if v5e saturates claims.
6. **Implication for Stage-6b:** SFT/RL should ingest **Pallas API + TPU arch context** (not just kernels), and the env loop must return compile/correct/profile feedback (MaxKernel / iterative / OPD-OPSD). Soft prompt alone is exactly their weak Best-of-N regime (~13/50 correct, ~1.01×).

## Method ranking (Gemini 3 Flash, 50 workloads, ~144 samples)

| Method | Geomean vs XLA | Correct |
|--------|----------------|---------|
| Best-of-N | 1.01× | 13/50 |
| Iterative refinement | 1.18× | 32/50 |
| Iterative + Autocomp context | 1.28× | 48/50 |
| Autocomp | 1.36× | 45/50 |
