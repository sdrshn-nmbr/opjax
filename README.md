# Palinkle

Palinkle trains Inkling Small to write correct and fast
[JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernels for TPUs.
The name combines Pallas and Inkling.

The project has one rule: a result only counts when the generated kernel is
correct, runs through real Pallas lowering on a TPU, and leaves enough evidence
for someone else to check it.

## Sources

[`config/pallas/sources.json`](config/pallas/sources.json) pins each code and
data source to a revision. A source listed here is not automatically approved
for training.

### Models

- **Inkling:** [release](https://thinkingmachines.ai/news/introducing-inkling/),
  [model card](https://thinkingmachines.ai/model-card/inkling/), and
  [weights](https://huggingface.co/thinkingmachines/Inkling). This was the
  original base model and is now a historical comparison.
- **Inkling Small:**
  [release](https://thinkingmachines.ai/news/inkling-small/),
  [model card](https://thinkingmachines.ai/model-card/inkling-small/), and
  [weights](https://huggingface.co/thinkingmachines/Inkling-Small). This is the
  current base model.

### Training sources

- **JAX and Pallas:**
  [documentation](https://docs.jax.dev/en/latest/pallas/index.html) and
  [pinned source](https://github.com/jax-ml/jax/tree/aaf50c6a71d3bde4188c1836323f3a0ae9cb9e7f).
  Only approved documentation, implementation, and test paths may enter the
  training data.
- **Tokamax:**
  [pinned repository](https://github.com/openxla/tokamax/tree/b33bdfa64a78cc16193f3c77dd223bb040aeebf4),
  used for approved Pallas and Mosaic kernel examples.
- **MaxText:**
  [pinned kernel directory](https://github.com/AI-Hypercomputer/maxtext/tree/17c7172720ca813b05e5ea248dedd78a0c64612e/src/maxtext/kernels),
  used for approved production kernel examples.
- **Hugging Face:** the data pipeline records each row's license, source
  revision, duplicate status, and split. The broad-kernel dataset contains 830
  approved rows from 95 repositories. Its
  [manifest](data/pallas/runs/g3-hub-dapt-admission/manifest.json) defines the
  dataset.

### Evaluation-only sources

- **JAXBench:**
  [pinned benchmark](https://github.com/AI-Hypercomputer/accelerator-agents/tree/6b6c44293c43976032ba12d2f72d6bebeaf2394f/JAXBench).
  Its implementations are held out from training.
- **PallasBench:**
  [pinned repository](https://github.com/Tyronita/PallasBench/tree/30a6ee07fd4923f3877906a94002d994e972d6fe)
  and
  [pinned dataset](https://huggingface.co/datasets/EvanOLeary/pallasbench-unified/tree/b0c928c21101a96ddee17682d897b8897fa27740).
  It is used for evaluation and data discovery, not training.

The coding harness uses
[DeepSWE](https://github.com/datacurve-ai/deep-swe),
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
[Chex](https://github.com/google-deepmind/chex), and
[Tinker](https://tinker-docs.thinkingmachines.ai/). The multi-turn training
reference is [Kevin](https://arxiv.org/abs/2507.11948), with a local extraction
in [`kevin32b.md`](kevin32b.md). `composer2.md` contains the Composer 2 notes.

## Current result

G4.2 is the first checkpoint that clearly improved Pallas kernel generation.
It is still a small result, not a general or fast kernel-writing agent.

| Model | Valid after 3 calls | Valid after 6 calls |
|---|---:|---:|
| Inkling Small base | 0/12 | 4/12 |
| G4.1 SFT | 0/12 | 3/12 |
| G4.2 repair SFT | **7/12** | **7/12** |

The 12 cases are four tasks—add, matrix multiplication, RMSNorm, and row
sum—sampled with three model seeds. They are not 12 independent tasks.

Across the same task, seed, and call-limit pairs, G4.2 beat the base model in
10 of 24 cases and lost none. No task family became worse after six calls.
Speed did not meaningfully improve:

- median speedup was `1.0039x` after three calls and `0.9998x` after six;
- the best valid sample reached `1.0825x`;
- row sum remained unsolved; and
- the extra three calls did not fix any failed G4.2 runs.

The result shows better Pallas syntax, API use, and solution structure. It does
not yet show broad transfer, learned repair, or kernel optimization skill. The
full result is in
[`g42-final-results.json`](data/pallas/runs/g42-final-results.json).

## What counts as a valid kernel

A generated kernel must:

1. match an independently written numerical specification;
2. use Pallas instead of falling back to ordinary JAX;
3. lower normally on a real TPU without `interpret=True`;
4. run safely at the full declared shape;
5. leave compiler and profiler evidence; and
6. be compared with XLA only after it passes the first five checks.

The model works in a temporary Git repository and submits a patch. It cannot
see the hidden verifier or reference solution. Training tasks may return one
short failure message. Evaluation tasks return no hidden feedback while the
model is working.

The verifier returns:

- `1` for a valid Pallas kernel;
- `0` when the candidate fails; and
- `-1` when the test system fails for a reason unrelated to the candidate.

Speed is recorded separately. A fast wrong answer gets no credit.

## Data rules

- JAXBench code never enters training data.
- PallasBench code never enters training data.
- The G4 SFT dataset has 32 TPU-verified kernels across eight operation
  families. This was enough to run one small experiment, not enough to claim
  broad coverage.
- The G4.2 dataset has 32 verified six-action runs and 192 training rows. The
  192 rows are prefixes of those 32 runs, not independent repairs.
- The broad-kernel dataset has 830 approved rows. Domain-adaptive pretraining
  (DAPT) has not started and remains an experiment, not an assumed improvement.

Every SFT kernel must pass full-shape correctness on fixed seeds, real TPU
lowering, profiling, license checks, duplicate checks, and JAXBench overlap
checks.

## Repository map

| Path | Contents |
|---|---|
| [`config/pallas`](config/pallas) | Experiment, source, split, harness, and evaluation settings |
| [`src/opjax/pallas`](src/opjax/pallas) | Data, training, agent, verifier, and evaluation code |
| [`tests/pallas`](tests/pallas) | Local and adversarial tests |
| [`tests_tinker/pallas`](tests_tinker/pallas) | Tinker and agent integration tests |
| [`environments/pallas-eval`](environments/pallas-eval) | Isolated TPU test environment |
| [`data/pallas/runs`](data/pallas/runs) | Run manifests and evidence |
| [`docs/model-factory`](docs/model-factory) | Earlier model-training experiments |
| [`archive`](archive) | Old plans, references, and work logs kept for provenance |

The current code is in `src/opjax/pallas`. The broad plan in
[`archive/opjax.md`](archive/opjax.md), `composer2.md`, and the model-factory
documents is historical context, not the current plan.

## Run locally

Palinkle uses Python 3.12 and `uv`.

```bash
uv sync
uv run pytest -q
uv run opjax-pallas validate-contracts
```

Useful commands:

```bash
uv run opjax-pallas --help
uv run opjax-pallas validate-corpus --help
uv run opjax-pallas-g42-agent --help
uv run opjax-pallas-g42-experiment --help
```

Local tests do not prove that a kernel works on a TPU. TPU runs use the pinned
cloud environment and produce evidence manifests.

## What we learned

- Passing a JAX correctness test does not prove Pallas skill. A model can copy
  the baseline or return ordinary JAX.
- Showing the reference implementation encourages copying. When references
  were removed and copied answers lost credit, the earlier LoRA advantage
  disappeared.
- `interpret=True` runs the kernel body as a JAX loop. It does not prove normal
  Pallas lowering on a TPU. Five of the first six reported successes used it.
- Lower training loss does not prove executable code. The first Pallas SFT
  model produced Pallas-looking code with reversed `BlockSpec` arguments and
  incomplete kernels.
- A failed run is not always the model's fault. One TPU failure came from a
  stale lock in the evaluator.
- Row count is not the same as data diversity. The 32-row threshold allowed a
  small test; it did not prove that the dataset was large enough.
- Multi-turn training helped, but the current runs were built around known
  solutions. The result does not yet separate real feedback-driven repair from
  seeing the solution pattern and learning the shell format.
- Add and dense matrix multiplication are useful basic tests but weak speed
  tests because XLA already handles them well.

The main lesson is simple: keep the claim, the training change, and the
evidence separate. If any one changes, score the result again.

## Next step

Gate 5 is the broad-kernel DAPT experiment. It is allowed to start, but the
open questions from G4.2 should be resolved first:

1. freeze a larger evaluation set with unseen operations and separate task
   generation code;
2. report results by task, not only by repeated model seeds;
3. repeat checkpoint comparisons across training seeds;
4. test solution exposure, shell-format training, verifier wording, and real
   repair feedback separately; and
5. add performance tasks where XLA has room to improve, including fusion,
   ragged computation, structured sparsity, mixture-of-experts routing, and
   larger model subgraphs.

The JAXBench v5e check found one possible performance task: a corrected
Megablox grouped matrix multiplication ran at `1.147x` XLA speed across three
profiled runs. The other seven optimized references did not run as fair
default-shape comparisons in one shared environment. This result remains
provisional until the runtime and test setup are frozen. See the
[`headroom manifest`](data/pallas/runs/jaxbench-v5e-headroom/manifest.json).

## Worklog

This table is the human-readable project log. New entries are appended. If a
result is overturned, a later entry records the correction. The manifests in
[`data/pallas/runs`](data/pallas/runs) are the source of truth.

| Date | Work | Result |
|---|---|---|
| 2026-07-15 | Started with a personalized Inkling coding model trained on exported agent sessions. The scope grew into a general model factory, RL, serving, teacher transfer, and memory research. | The broader research was useful, but it mixed separate claims and lacked stable pass/fail rules. |
| 2026-07-16 | Added data rights, retention, scrubbing, upload checks, sealed splits, agent-session curation, and controlled Tinker training. | The first rank-64 Inkling LoRA passed 4/4 small mechanical tasks. This showed narrow task compliance, not general coding skill. |
| 2026-07-22 | Ran a small GRPO experiment on the harder eight-task set. | The score stayed 7/8 and the same task failed. Training stopped because the small task set had saturated. |
| 2026-07-23 to 2026-07-28 | Compared the trace LoRA with base Inkling on JAXBench, fixed unequal prompts and extraction, hid references, and rejected copied answers. | The apparent LoRA advantage reversed. Base Inkling tried Pallas more often, so the project shifted to correct, real, fast Pallas kernels. |
| 2026-07-29 | Froze the first contracts, built the evaluator, and tested 50 JAXBench tasks with three model seeds. | The first report claimed six correct Pallas kernels. Five used `interpret=True`; the corrected result was one normally lowered kernel out of 150 and no speed win. |
| 2026-07-30 | Added real lowering checks, isolated TPU processes, compiler markers, Perfetto traces, Chex assertions, and strict evidence checks. | The evaluator could now separate normal Pallas execution from interpreted or ordinary JAX code. |
| 2026-07-30 to 2026-08-02 | Built GitHub and Hugging Face data discovery, scanned about 982,000 datasets, fixed source-classification bugs, and verified the SFT data on TPUs. | The SFT dataset reached 32 kernels across eight families. The broad DAPT dataset reached 830 approved rows; DAPT remained untested. |
| 2026-08-02 | Switched training to Inkling Small and searched for a faster JAXBench matrix multiplication kernel. | The best Pallas kernel reached `0.9993x`, effectively equal to XLA. JAXPR was too high-level to explain the final TPU speed difference. |
| 2026-08-04 | Trained direct Pallas SFT on the 32 verified kernels. | Training finished, but all three TPU checks failed. The model learned the shape of Pallas code, not working kernels. |
| 2026-08-04 | Fixed hidden interface details in the training prompts and allowed up to three feedback attempts in G4.1. | G4.1 recovered some kernels after feedback but did not beat the base model: 1/4 versus 1/4. |
| 2026-08-04 to 2026-08-05 | Built the G4.2 patch-based agent test, hidden verifier, 32-task repair dataset, six-action runs, and matched three-model comparison. | G4.2 reached 7/12 after both three and six calls, versus base at 0/12 and 4/12. This was the first positive checkpoint result, but speed stayed near XLA and the test covered only four tasks. |
| 2026-08-05 | Tested all eight optimized JAXBench references on a v5e before choosing speed tasks. | None worked as a strict shared-environment comparison. A corrected setup found stable `1.147x` Megablox headroom, which remains provisional. |
