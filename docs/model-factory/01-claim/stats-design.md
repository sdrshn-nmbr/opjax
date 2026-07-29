# Pallas Agent — Statistical design

**Decision:** Kernel generation and timing are stochastic. Compare paired arms
under one frozen experiment contract and preserve component evidence rather
than collapsing the claim into one scalar.

## Design

| Element | Choice |
|---------|--------|
| Primary outcome | Correct authentic Pallas kernels faster than JAX/XLA |
| Entry conditions | TPU compile, full-shape correctness, authentic reachable Pallas, stable timing |
| Diagnostic outcomes | Parse, compile, plain-JAX correctness, Pallas emission, repair, failure distribution |
| Seeds | Sampling seeds `{0,1,2}` for every compared arm |
| Pairing | Identical task IDs, prompts, policies, and hardware across arms |
| Timing | 3 independent harness runs; each uses 3 warmups and 20 measured iterations |
| Stability | No headline credit when timing coefficient of variation exceeds 0.10 |
| Speed threshold | Headline faster-than-baseline threshold `>=1.05x` |
| Aggregation | Family-level rates and geometric-mean speedup over valid correct Pallas kernels |
| Primary contrasts | B vs A for SFT; C vs B for incremental DAPT value |
| Generalization | Requires a populated, family-disjoint private evaluation |

## Logging requirements

Every evaluation artifact includes the contract hash, repository revision and
tracked-dirty state, source revisions, kernel hashes, model lineage, arm, prompt
context and hash, sampling seed, JAX/JAXlib/libtpu versions, TPU target and
topology, correctness evidence, repeated timing evidence, and structured
failure reasons.

## Forbidden

- Training on JAXBench reference implementations or recognizable derivatives.
- Treating reference-visible prompts as scored.
- Selecting the best seed or timing run post hoc.
- Treating public-only improvement as private-family generalization.
- Treating GPU execution as TPU correctness or performance.
