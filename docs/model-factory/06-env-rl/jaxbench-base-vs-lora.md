# JAXBench — base Inkling vs Stage-5 axport LoRA

**Status:** Done. Official TPU v5e grades pulled; VM deleted.

## Kernel-track scoring (binding)

Only **correct + real Pallas** counts. Plain-JAX matches = diagnostic only → **0 credit**.

## Soft-prompt results

| Arm | CPU raw correct | TPU raw correct | Mentions Pallas (CPU) | **pallas_correct** |
|-----|-----------------|-----------------|-----------------------|--------------------|
| Stage-5 LoRA | 43/50 | **41/50** | 0/50 | **0/50** |
| Base Inkling | 11/50 | **9/50** | 22/50 | **0/50** |

LoRA wins the diagnostic “match JAX baseline” score. Both are **0** on the hillclimb metric. Base tries more Pallas-shaped code and mostly errors; LoRA stays plain JAX and passes more often. Max TPU speedup for base correct kernels ≈ 1.0 (same story as LoRA).

## Artifacts

- LoRA CPU: `data/model-factory/evals/jaxbench-baseline-lora/`
- Base CPU: `data/model-factory/evals/jaxbench-baseline-base/`
- LoRA TPU: `data/model-factory/evals/jaxbench-tpu-v5e-lora/`
- Base TPU: `data/model-factory/evals/jaxbench-tpu-v5e-base/`

## Why LoRA "beat" base — it is not JAX skill

Measured over all 50 generations per arm (`kernels/` + `summary.json`):

| Signal | Stage-5 LoRA | Base Inkling |
|--------|--------------|--------------|
| Median completion length | 464 chars (~116 tok) | 7,244 chars (~1,811 tok) |
| Hit / near the 2048-token cap | 0/50 | **15/50** |
| Extracted code / completion ratio | 0.80 | **0.19** |
| Syntax-valid extracted code | **50/50** | 15/50 |
| Contains `def workload` | **50/50** | 13/50 |
| Files with prose leaked into code | 0 | **30** |
| Median similarity of its `workload` to the given baseline | **0.846** (10 near-copies >0.9) | 0.478 |
| Mentions Pallas | 0/50 | 22/50 |

Two mechanisms, neither of which is JAX competence:

1. **Format compliance.** LoRA returns one short fenced function. Base emits long chain-of-thought prose, often never closes a code block inside the 2048-token budget, so the harness extracts reasoning text → `SyntaxError` / `no_code` / `no_workload_fn` (36 of its 39 failures are truncated outputs).
2. **Minimal-diff copying.** The prompt *contains* the correct JAX baseline. LoRA largely returns it back (median similarity 0.85). Passing needs no kernel knowledge.

Implications:

- The base arm was **scored unfairly** (token cap + no format enforcement + heuristic extraction). A fair rerun needs a larger `max_tokens`, strict fenced-output contract, and a retry on `no_code`.
- Base shows more *Pallas inclination* (22/50 attempts) than LoRA (0/50). Warm-starting Stage-6b from the axport LoRA (D21) may import an anti-Pallas prior: "satisfy the oracle with the smallest edit."
- Neither arm scores on the kernel metric: **pallas_correct = 0/50 for both.**

## Fair rerun (2026-07-27) — official v5e grade

The first comparison was invalid: base was capped at 2048 tokens with no output
contract and a prose-tolerant extractor. Harness changes, applied **identically
to both arms** (`eval_jaxbench_baseline.py`):

- `max_tokens` 2048 → **8192**.
- Explicit `ANSWER_CONTRACT` in the system prompt (reply must end in one closed
  ```python block defining `workload`).
- Renderer stop sequences passed to the sampler.
- Parse-aware extractor: candidate blocks are ranked by `ast.parse` validity, so
  prose is never handed to the grader as code; unclosed final fences recovered.
- Up to **2 retries** when no valid block comes back, with the rejection reason
  fed back. New `unparseable_code` status instead of silent `runtime_error`.
- Truncation and attempt counts recorded per workload.

Sampling on Tinker; grading on a **singular v5e** (`us-west4-a`, deleted after
the run) via `scripts/jaxbench_evaluate_all.sh`, merged with local Pallas
detection by `scripts/jaxbench_merge_tpu.py`.

### Official v5e result

| Signal | base | LoRA |
|--------|------|------|
| Correct (TPU official) | **37/50** | **42/50** |
| Attempts Pallas | 9/50 | 2/50 |
| **pallas_correct (headline)** | **1/50** | **0/50** |
| Correct kernels ≥1.05× baseline | **3/50** | **0/50** |
| Median speedup of correct kernels | 1.000 | 1.000 |
| **Best speedup across all 50** | **15.05×** | **1.000×** |

Answer-shape diagnostics (CPU pass, same generations):

| Signal | LoRA | base |
|--------|------|------|
| Median completion chars | 676 | 8,920 |
| Median code/completion ratio | 0.865 | 0.082 |
| Truncated on some attempt | 0/50 | 10/50 |
| Needed a retry | 0/50 | 6/50 |
| Syntactically valid / defines `workload` | 50/50 | **50/50** |
| Median similarity to the given baseline | **0.933** | 0.441 |
| Near-copies of baseline (>0.9) | **27/50** | 1/50 |

### Conclusions

1. **The original 41 vs 9 gap was ~90% harness artifact.** Base goes
   **9 → 37** correct once given budget, a format contract, and retries. The
   remaining honest gap is **42 vs 37**.
2. **LoRA passes by returning the baseline.** Median similarity 0.933, 27/50
   near-copies, and — decisively — **its best speedup over all 50 workloads is
   exactly 1.000×**. It never makes anything faster because it rarely changes
   anything. Base produces 3 kernels ≥1.05× and one 15.05×.
3. **Base has faint real Pallas ability; LoRA has none.** Base: 9 attempts,
   1 correct on TPU (`44k_Matmul_Divide_GELU`, real `pl.pallas_call`, though
   0.82× — correct but slow). LoRA: 2 attempts, 0 correct. Two further CPU-only
   artifacts confirmed: the local CPU pass auto-fails Pallas via tile-size
   asserts against shrunken dims and an older local `pl.load` API, so **CPU
   functional cannot score Pallas at all** — TPU is mandatory for this metric.
4. **The 15.05× is benchmark exploitation, not kernel skill.** In
   `26k_...InstanceNorm...`, InstanceNorm over 1×1 spatial dims is identically
   zero, so base folded the entire BMM branch away to `(y + in_bias) * y`.
   Correct and legal, but it is algebra on a degenerate workload.

### Consequence for D21 (warm-start)

Warm-starting Stage-6b from the axport LoRA now looks actively risky. The LoRA's
learned policy is "return the smallest edit that satisfies the oracle," which on
this benchmark means *reproduce the reference*: 0 speedups, 2 Pallas attempts,
0 pallas_correct. Base is worse at passing but retains exploration — 4.5× more
Pallas attempts, the only pallas_correct, and the only speedups. Since the
Stage-6b reward is speed-on-top-of-correct, the LoRA prior optimises against the
thing we want. **D21 should be revisited before Pallas SFT starts.**

## Copying probe (D29) — withhold the reference implementation

Same 12 priority workloads, same models, same settings. Only the prompt context
changes: `--prompt-context spec` supplies `CONFIG`, `create_inputs`, and the
`workload` signature + docstring, but **not** the reference body. Graded on CPU
(so Pallas is unscorable here — see above).

| | base | LoRA |
|--|------|------|
| Correct, **baseline shown** (same 12) | 4/12 | **9/12** |
| Correct, **spec only** | **3/12** | **3/12** |
| Median similarity to baseline, shown → spec | 0.441 → 0.169 | **0.933 → 0.29** |
| Near-copies (>0.9), shown → spec | 1 → 0 | **27/50 → 1/12** |
| Truncated / needed retry (spec) | 5 / 2 | 0 / 0 |
| Pallas attempts (spec) | 1/12 | 0/12 |

**The entire measured gap between the two models is the reference implementation
being in the prompt.** Remove it and they are identical at 3/12. The LoRA loses
6 of its 9 wins; base loses 1 of 4.

The LoRA is not incapable — given only a signature and docstring it writes a
clean, correct RMSNorm from scratch. It is *anchored*: with a reference present
it reproduces it rather than optimising. Its 9 spec-mode failures are ordinary
shape/einsum errors on the attention workloads.

This confirms the D28 read: on JAXBench the axport LoRA is measuring format
compliance and mimicry, not kernel ability.

## Harness change: the anti-mimicry gate

Implemented in `src/opjax/model_factory/jaxbench_scoring.py`, used by the eval
harness, the TPU merge, and (later) the Stage-6b reward, so credit rules live in
exactly one place.

- **`--prompt-context spec` is now the default.** The reference body is withheld,
  so a score cannot be earned by copying. `--prompt-context baseline` still works
  and still reports raw correctness, but the run is labelled `scorable: false` and
  prints a warning.
- **Similarity gate.** When the reference *was* shown, a candidate whose
  `workload` matches it at similarity ≥ 0.90 (or is a verbatim file copy) is
  marked `copied_reference`: no credit, zero reward. The gate is deliberately
  **inactive** in spec mode — with nothing to copy, resembling the reference is
  convergence on the right answer, not mimicry.
- Similarity is measured on the extracted `workload` function, not the whole
  file, so copied imports/`CONFIG` neither inflate nor mask the score.
- `uses_pallas` tightened from "the string `pallas` appears" to an actual
  `pallas_call(` launch.
- **Reward shape** (`reward()`): 0.0 for incorrect *or* copied; 0.3 for
  correct-and-original; +0.3 for real Pallas; +up to 0.4 scaled by speedup,
  saturating at 2×. The speed term is gated behind Pallas on purpose — otherwise
  base's 15.05× algebraic fold (0.7) would outrank its one genuinely correct
  Pallas kernel (0.6), which inverts the stated objective.
- 11 unit tests in `tests/test_model_factory_jaxbench_scoring.py` pin the gate,
  the spec-mode exemption, and the reward ordering.

### Effect on the recorded results

Re-scoring the fair v5e run under the gate (both arms were reference-shown, so
the gate applies):

| Signal | base | LoRA |
|--------|------|------|
| Correct, raw | 37/50 | **42/50** |
| Copies gated | 1/50 | **27/50** |
| **Credited (correct, not a copy)** | **36/50** | **16/50** |
| pallas_correct | **1/50** | 0/50 |
| Mean reward | **0.222** | 0.096 |

The gate inverts the apparent ranking: the LoRA's headline advantage was 26
workloads of reproduction.
