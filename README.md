# Palinkle

Palinkle is an evidence-first effort to train Inkling Small to write correct,
real [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernels for
TPUs.

Palinkle combines **Pallas** with **Inkling**, the model family used for the
current training experiments.

## Sources and references

The project pins code and data sources by revision in
[`config/pallas/sources.json`](config/pallas/sources.json). The links below
state each source's role; inclusion here does not mean it is allowed in
training.

### Models

- **Inkling:** [release](https://thinkingmachines.ai/news/introducing-inkling/),
  [official model card](https://thinkingmachines.ai/model-card/inkling/), and
  [weights](https://huggingface.co/thinkingmachines/Inkling). This was the
  original base model and remains a historical comparison.
- **Inkling Small:**
  [release](https://thinkingmachines.ai/news/inkling-small/),
  [official model card](https://thinkingmachines.ai/model-card/inkling-small/),
  and [weights](https://huggingface.co/thinkingmachines/Inkling-Small). This is
  the active base model.

### Pallas code and training-source candidates

- **JAX and Pallas:** the
  [Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) and
  [pinned JAX source](https://github.com/jax-ml/jax/tree/aaf50c6a71d3bde4188c1836323f3a0ae9cb9e7f).
  Only allowlisted Pallas documentation, implementation, and test paths may
  enter the governed corpus.
- **Tokamax:**
  [pinned repository](https://github.com/openxla/tokamax/tree/b33bdfa64a78cc16193f3c77dd223bb040aeebf4),
  used for allowlisted production Pallas and Mosaic kernel candidates.
- **MaxText:**
  [pinned kernel tree](https://github.com/AI-Hypercomputer/maxtext/tree/17c7172720ca813b05e5ea248dedd78a0c64612e/src/maxtext/kernels),
  used for allowlisted production kernel candidates.
- **Hugging Face kernel sources:** the governed Hub discovery and admission
  pipeline records row-level licenses, repository revisions, duplicate checks,
  and split assignments. The final broad-kernel DAPT release spans 95 source
  repositories; its
  [manifest](data/pallas/runs/g3-hub-dapt-admission/manifest.json), rather than
  a search result page, defines membership.

### Evaluation and discovery-only sources

- **JAXBench:**
  [pinned benchmark](https://github.com/AI-Hypercomputer/accelerator-agents/tree/6b6c44293c43976032ba12d2f72d6bebeaf2394f/JAXBench),
  held out for public evaluation. Its implementations are forbidden in
  training.
- **PallasBench:**
  [pinned repository](https://github.com/Tyronita/PallasBench/tree/30a6ee07fd4923f3877906a94002d994e972d6fe)
  and the
  [unified Hugging Face dataset](https://huggingface.co/datasets/EvanOLeary/pallasbench-unified/tree/b0c928c21101a96ddee17682d897b8897fa27740).
  These are benchmark, mining, and discovery evidence—not training sources.

The agent harness also builds on
[DeepSWE](https://github.com/datacurve-ai/deep-swe),
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
[Chex](https://github.com/google-deepmind/chex), and
[Tinker](https://tinker-docs.thinkingmachines.ai/). These are implementation
references, not corpus rows.

The multi-turn kernel-training reference is
[Kevin](https://arxiv.org/abs/2507.11948); a local Markdown extraction is kept
in [`kevin32b.md`](kevin32b.md) beside the existing Composer 2 notes.

The project is deliberately narrow. A generated program only counts when it:

1. satisfies an independently defined numerical specification;
2. uses Pallas rather than a plain-JAX fallback;
3. lowers normally on a real TPU, without `interpret=True`;
4. runs safely at the declared full shape;
5. leaves compiler and profiler evidence; and
6. is compared with XLA only after correctness is established.

## Current result

G4.2 is the first credible positive checkpoint result in the Pallas program.
It is a narrow diagnostic win, not yet a broadly capable or fast kernel agent.

| Model | Valid kernels at 3 calls | Valid kernels at 6 calls |
|---|---:|---:|
| Inkling Small base | 0/12 | 4/12 |
| G4.1 SFT | 0/12 | 3/12 |
| G4.2 repair SFT | **7/12** | **7/12** |

These are 12 task-and-seed cells built from four tasks: add, matmul, RMSNorm,
and row sum. The three seeds are repeated samples, not separate tasks.

G4.2 beat the base model in 10 of 24 paired task/seed/horizon cells, lost none,
and introduced no task-family regression at six calls. It did not demonstrate a
meaningful speed improvement:

- median G4.2 speedup was `1.0039x` at three calls and `0.9998x` at six;
- the best verified sample reached `1.0825x`;
- row sum remained unsolved; and
- extra calls did not repair any G4.2 failures because successful trajectories
  had already submitted by the third call.

The honest conclusion is that verified repair-format SFT improved local Pallas
syntax, API use, and solution structure. It has not yet proved broad transfer,
learned repair reasoning, or optimization skill.

The aggregate evidence is in
[`g42-final-results.json`](data/pallas/runs/g42-final-results.json).

## How the experiment works

```text
licensed source code
        |
        v
governed corpus and contamination checks
        |
        v
Inkling Small LoRA training through Tinker
        |
        v
isolated coding workspace with a bounded call budget
        |
        v
captured Git patch
        |
        v
pristine TPU verifier
        |
        +-- numerical correctness at seeds 0, 1, and 2
        +-- real Pallas lowering
        +-- runtime safety
        +-- compiler and Perfetto evidence
        +-- candidate versus XLA timing
```

The G4.2 harness follows the useful boundary from DeepSWE-style evaluation:
the model works in a disposable repository, submits a patch, and cannot see the
authoritative verifier or reference solution. Curriculum tasks may return one
sanitized failure stage. Benchmark tasks return no hidden feedback during the
rollout.

The verifier gives:

- `1` for a correct, authentic, normally lowered, profiled Pallas kernel;
- `0` for a candidate failure; and
- `-1` for an infrastructure failure that cannot be blamed on the candidate.

Speed is recorded separately. A fast wrong answer never receives credit.

## Data boundaries

The project keeps training, development, and evaluation sources separate.

- **JAXBench is held-out evaluation.** Its implementations are not training
  material.
- **PallasBench is discovery and benchmark evidence, not training data.** Its
  rows are preserved for analysis but excluded from the training release.
- **The G4 SFT release contains 32 TPU-verified kernels across eight operation
  families.** This met the minimum experiment gate; it was not evidence that 32
  examples were broadly sufficient.
- **The G4.2 release contains 32 verified six-action trajectories and 192
  prefix-SFT rows.** Those 192 rows are prefixes of 32 trajectories, not 192
  independent repairs.
- **The broad kernel DAPT pool contains 830 authorized rows.** DAPT has not
  started, and the pool must remain an ablation rather than an assumed benefit.

Every promoted SFT kernel must pass full-shape correctness over fixed seeds,
normal TPU lowering, profile checks, license checks, duplicate checks, and
JAXBench contamination checks.

## Repository map

| Path | Purpose |
|---|---|
| [`config/pallas`](config/pallas) | Frozen experiment, source, split, harness, and evaluation contracts |
| [`src/opjax/pallas`](src/opjax/pallas) | Corpus, sampling, training, agent harness, verifier, and evaluation code |
| [`tests/pallas`](tests/pallas) | Local contract and adversarial tests |
| [`tests_tinker/pallas`](tests_tinker/pallas) | Tinker and agent-driver integration tests |
| [`environments/pallas-eval`](environments/pallas-eval) | Isolated TPU evaluation environment |
| [`data/pallas/runs`](data/pallas/runs) | Committed manifests and evidence for accepted runs |
| [`docs/model-factory`](docs/model-factory) | Earlier model-factory work and the experiments that led to the Pallas pivot |
| [`archive`](archive) | Superseded project plans, references, and work logs retained for provenance |

The active implementation is under `src/opjax/pallas`. Earlier broad plans in
[`archive/opjax.md`](archive/opjax.md), `composer2.md`, and the model-factory
documents are historical research context. They are not the current execution
plan.

## Local use

Palinkle uses Python 3.12 and `uv` for the local environment.

```bash
uv sync
uv run pytest -q
uv run opjax-pallas validate-contracts
```

Useful entry points include:

```bash
# Inspect all Pallas commands.
uv run opjax-pallas --help

# Validate the governed corpus and source contracts.
uv run opjax-pallas validate-corpus --help

# Inspect the bounded G4.2 agent driver.
uv run opjax-pallas-g42-agent --help

# Inspect the matched experiment runner.
uv run opjax-pallas-g42-experiment --help
```

TPU runs require the pinned cloud environment and produce immutable evidence
manifests. Local tests do not establish TPU correctness.

## What we learned

Several attractive results disappeared when the experiment became stricter.
Those reversals define the project more than the early scores do.

- Raw JAX correctness is not Pallas competence. A model can return the supplied
  baseline or plain JAX and still pass a numerical test.
- Reference-visible evaluation rewards copying. The earlier apparent advantage
  of the personal-trace LoRA disappeared under spec-only prompts and anti-copy
  scoring.
- `interpret=True` runs the kernel body but does not prove real TPU Pallas
  lowering. Five of six initially reported correct Gate 2 candidates used it.
- A training loss decrease does not prove executable code. The first direct
  Pallas SFT checkpoint learned the look of Pallas while still reversing the
  `BlockSpec` arguments and emitting incomplete kernels.
- A successful process is not necessarily a successful candidate. One TPU
  canary was initially blamed on the model before the evaluator's retained TPU
  lock was found and corrected.
- Row counts can overstate diversity. The 32-row SFT threshold authorized one
  small causal experiment; it did not establish broad data sufficiency.
- Multi-turn data helped, but the current trajectories are scripted around
  known verified solutions. The result does not isolate genuine
  feedback-conditioned repair from solution exposure and shell-format learning.
- Simple add and dense GEMM are useful mechanics checks but weak optimization
  benchmarks because XLA already handles them well.

The practical lesson is simple: preserve the claim, the intervention, and the
evidence as separate objects. When one changes, the result must be re-scored.

## Next boundary

Gate 5, the broad-kernel DAPT ablation, is eligible but has not started. It
should not start merely because the procedural gate is open.

The next useful work is:

1. freeze a larger independent evaluation with unseen operations and no shared
   task-generation code;
2. report task-level results separately from repeated model seeds;
3. reproduce checkpoint comparisons across training seeds;
4. separate complete-solution exposure, shell-action training, verifier
   language, and genuine repair feedback; and
5. build a performance suite around fusion, ragged computation, structured
   sparsity, MoE, and larger model subgraphs where XLA has measurable headroom.

The JAXBench v5e audit found one provisional example: corrected Megablox GMM
ran at a stable `1.147x` over XLA across three device-profiled repeats. The
other seven bundled optimized references were not valid default-shape v5e
comparisons under a single runtime contract. See the
[`headroom manifest`](data/pallas/runs/jaxbench-v5e-headroom/manifest.json).

## Worklog

This is the human-readable project log. New entries are appended; old entries
are not silently rewritten. If a result is overturned, a later entry records
the correction.

| Date | Work | Result |
|---|---|---|
| 2026-07-15 | Began as a personalized Inkling coding-sidekick experiment built from exported agent traces. The scope expanded into a general model factory, RL, serving, teacher transfer, and memory research. | The ambition was useful, but the project mixed several separate claims and lacked stable falsifiers. |
| 2026-07-16 | Added data rights, retention, scrubbing, upload gates, sealed splits, axport curation, and controlled Tinker training. | The first rank-64 Inkling LoRA reached 4/4 on four small mechanical tasks. This supported narrow format/task compliance, not agentic coding ability. |
| 2026-07-22 | Ran thin GRPO on the hardened eight-task set. | The score stayed 7/8 and the same task failed. The kill condition fired; the low-diversity training tasks had saturated. |
| 2026-07-23 to 2026-07-28 | Evaluated the trace LoRA and base Inkling on JAXBench, corrected asymmetric prompting and extraction, removed visible references, and added anti-copy scoring. | The apparent LoRA advantage reversed. Base Inkling retained more Pallas exploration. The project pivoted to correct, real, fast Pallas kernels. |
| 2026-07-29 | Froze Gate 0 contracts and built the Gate 1 evaluator. Ran Gate 2 over 50 JAXBench workloads and three seeds. | The first report said six correct Pallas candidates. Review found five used `interpret=True`; the corrected result was one normally lowered correct kernel out of 150 and no speed win. |
| 2026-07-30 | Added empirical lowering checks, isolated TPU processes, compiler markers, Perfetto traces, Chex runtime assertions, and fail-closed evidence. | The evaluator could now distinguish real Pallas execution from interpreted or plain-JAX behavior. |
| 2026-07-30 to 2026-08-02 | Built governed GitHub and Hugging Face discovery, scanned roughly 982,000 Hub datasets, corrected benchmark/source classification bugs, and TPU-verified the SFT release. | The final release reached 32 verified kernels across eight families. The broad DAPT pool reached 830 authorized rows, but DAPT remained untested. |
| 2026-08-02 | Switched the active training lineage to Inkling Small. Ran an upper-bound search on JAXBench GEMM. | The best searched Pallas GEMM reached `0.9993x`, effectively XLA parity. JAXPR was confirmed to be too high-level for explaining the final TPU performance difference. |
| 2026-08-04 | Trained direct Pallas SFT on the 32-row release. | Training completed, but all three TPU canaries failed. The model learned Pallas-shaped text, not executable kernels. |
| 2026-08-04 | Audited supervision and created G4.1 with explicit output contracts and at most three feedback attempts. | The renderer was sound, but prompts had hidden the exact interface. G4.1 recovered some modules through feedback yet produced no verified gain over base: 1/4 versus 1/4. |
| 2026-08-04 to 2026-08-05 | Built the DeepSWE-style G4.2 patch harness, pristine verifier, 32-task repair curriculum, six-action trajectories, and matched three-model evaluation. | G4.2 reached 7/12 at both horizons versus base at 0/12 and 4/12. This was the first positive checkpoint result, but speed stayed near XLA parity and the benchmark covered only four tasks. |
| 2026-08-05 | Audited all eight bundled optimized JAXBench references on a v5e before selecting performance tasks. | No strict subset worked under the project runtime. A corrected compatibility lane found stable `1.147x` Megablox GMM headroom; it remains provisional until its runtime and harness contract are frozen. |

The machine-readable manifests under [`data/pallas/runs`](data/pallas/runs)
remain the source of truth when this summary and an artifact disagree.
