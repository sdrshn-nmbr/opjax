# Operator-from-Scratch in JAX
**A multi-model GUI agent system trained end-to-end with multi-stage GRPO, multi-expert specialization, on-policy distillation, three-tier memory (RL-learned token-level compression + analytical inter-model KV transfer + amortized perceiver-based KV compaction), and latent inter-model communication, in pure JAX**

A deadline-free, learning-maximizing project. The goal is JAX/Flax/Optax/Tunix/Pallas mastery via building a research-grade GUI agent system. Each phase produces a shippable artifact and stands on its own; later phases compose previous ones. Every phase boundary includes an explicit on-policy cross-stage distillation step to prevent capability regression — the technique GLM-5.1 and DeepSeek V4 converged on independently in April 2026.

The terminal architecture, when fully realized:

```
Screenshot → [Gemma 4 multimodal: perception + planning + reasoning]
                      │
                      │ (Latent Briefing: KV cache compaction)
                      ▼
              [FunctionGemma fine-tune: structured action emission]
                      │
                      ▼
                   cua sandbox
                      │
                      ▼
                   reward signal → joint GRPO update on both models
```

With Composer 2 self-summarization layered on Gemma 4 for long-horizon coherence, reverse distillation from FunctionGemma into Gemma 4 as a pre-training phase to bake in format discipline, multi-expert specialization in Phase 3 merged via on-policy distillation, and cross-stage distillation between every phase boundary to protect earlier-phase capabilities.

## Why this shape

Every component has a justified role:

- **Gemma 4 multimodal** does what only a frontier multimodal model can: visual perception of arbitrary GUIs, multi-step planning, language-conditioned reasoning. JAX-native via `google-deepmind/gemma`.
- **FunctionGemma** does what specialized small models do best: deterministic structured output with low schema-violation rate. JAX-native, 270M, runs in milliseconds.
- **Latent Briefing** does what no other technique does: transfer model state between agents without re-tokenizing. The multi-agent setup created by Gemma 4 + FunctionGemma is the natural home Latent Briefing was waiting for.
- **Composer 2 self-summarization** does what RAG cannot: learns task-aware *token-level* context compression as a behavior, with reward signal propagating through the summary tokens. The discrete coarse layer of memory.
- **Latent Briefing** does inter-model KV transfer: analytical compaction (attention-magnitude scoring + top-K + ridge regression) optimized per-context at handoff time. The bridge between the planner and the actor.
- **STILL** (perceiver-based learned KV compaction) does what neither does: amortizes per-context compaction into a learned forward pass via a 7M-param perceiver per LLM layer, runs in milliseconds, scales to unbounded context iteratively. The continuous fine-grained layer of memory. Together with self-summary and Latent Briefing, this gives the agent a three-tier memory hierarchy: continuous compression as it runs, RL-learned summary checkpoints, analytical handoff between models.
- **Reverse distillation** transfers FunctionGemma's format discipline into Gemma 4's output distribution at the logit level, even though the weights cannot transfer directly.
- **Multi-expert specialization + distillation merge** (DeepSeek V4 pattern): solves multi-domain interference. Three specialists per domain trained independently, then merged into one student via on-policy distillation.
- **Cross-stage on-policy distillation** between phase boundaries (GLM-5.1 + DeepSeek V4 pattern): prevents capability regression as the model moves through sequential training stages.
- **Dr. GRPO loss + k1 KL estimator + nonlinear length penalty + GRMs + RFT**: the distilled set of post-training algorithmic refinements from Composer 2, Kimi K2.6, and the broader frontier post-training community.
- **Tunix** is the JAX-native scaffolding that makes all of this composable. Its SFT, GRPO, and distillation trainers are read as production references and used where reinventing would be busywork.

This is a project at the actual frontier of public agent research. The full architecture as described does not exist publicly anywhere.

## Day 1 actions

1. **Claim TRC quota**. TRC invitation is in hand (April 27, 2026 — 30-day free trial). To activate:
   - `gcloud projects create <project-id>` to create the GCP project
   - `gcloud services enable tpu.googleapis.com --project=<project-id>` to turn on the Cloud TPU API
   - Submit the project ID via the TRC form linked in the welcome email
   - Wait for confirmation email from TRC support before creating any TPUs (otherwise charges may bill to your account)
   - Don't burn the 30-day clock. The trial starts when TRC confirms registration. Use the first few days for setup and the early phases on local GPU; spin up TPU only when actually training. Specifically: Phase 1 (SFT floor), most of Phase 2 (reverse distillation can run on a single GPU at small batch), and Phase 3 specialist warm-up should fit on local hardware. Move to TPU when the multi-expert GRPO runs in Phase 3 and Phase 3.5 distillation merge become the bottleneck.
2. Skim JAX 101 end to end.
3. Read: GRPO paper (DeepSeek-Math), R1 paper, Composer 2 technical report (especially Section 4.1 + Dr. GRPO + k1 KL + length penalty), Latent Briefing blog post, STILL paper (Baseten "Towards infinite context windows" April 2026 — read carefully, especially the three architectural fixes), Ramp Labs "How we built Steer" (April 2026 — Gemma-family layer-selection findings), Rimsky/Panickssery on Contrastive Activation Addition (arXiv 2312.06681), FunctionGemma model card and formatting guide, Tunix's GRPO and distillation trainer source, MaxText `gemma4-26b` config and `Run_Gemma4.md`, SGLang-Jax architecture blog post (LMSYS, October 29 2025), vLLM-TPU redesign blog post (October 2025), JAX Scaling Book chapters 7-10 (transformer inference, serving Llama 3 on TPUs, profiling, programming TPUs in JAX), Gemma 4 Flax `transformer.py`, Tzafon's "Training VLM for CUA" blog post, FDM-1 blog post, GLM-5 paper Section on cross-stage distillation, DeepSeek V4 paper post-training section, Kimi K2.5 paper PARL/GRM/RFT sections.
4. Set up environment: `uv venv`, install `jax[cuda12]`, `flax`, `optax`, `google-tunix`, `maxtext`, `cua`, `cua-bench`, `transformers`, `datasets`, `wandb`.
5. Pull `AI-Hypercomputer/maxtext`, `google-deepmind/gemma`, `google/tunix`, `sgl-project/sglang-jax`, `trycua/cua` repos. Skim READMEs and example notebooks.
6. Run hello-worlds: cua sandbox screenshot/click/shell, Tunix GRPO demo notebook, FunctionGemma function-call inference, MaxText `gemma4-26b` inference, SGLang-Jax serving Gemma 4.
7. Once TRC confirms registration, run a final hello-world: connect to your TPU VM via `gcloud compute tpus tpu-vm ssh`, install JAX, run a multi-device sanity check (`jax.device_count()` should return 8 for v3-8). This confirms the multi-host orchestration plumbing works before you depend on it.

## TPU inference and serving stack

The 2024-era complaint that "TPU = stuck reimplementing everything" is genuinely obsolete as of late 2025/early 2026. There are three production-grade TPU inference stacks with published benchmarks, all native JAX, all integrating cleanly with the rest of the project's tooling. Adopting them rather than writing equivalents from scratch is the right call — they solve infrastructure that isn't on the JAX-learning critical path, freeing time for the components that genuinely teach (GRPO trainer, self-summary, Latent Briefing, STILL perceiver, Pallas kernels).

**MaxText as the reference codebase**. MaxText is Google's flagship JAX/Flax LLM training and inference codebase, designed for high MFU and "optimization-free" performance via XLA's whole-program analysis. As of April 2, 2026, MaxText supports Gemma 4 multi-modal models (`gemma4-26b` and `gemma4-31b` configs). It integrates with Tunix for post-training, Orbax for checkpointing, Optax for optimization, Grain for dataloading. Real-world validation: Kakao reported 2.7x throughput on production LLMs migrating to the JAX AI Stack (MaxText + Tunix + JetStream); Lightricks broke through scaling walls on a 13B video model; Escalante achieved 3.65x perf/$ on protein design.

We adopt MaxText as our primary reference codebase rather than writing model code from scratch. The JAX-learning value comes from *reading and modifying* a production codebase that already implements optimal sharding, vocabulary tiling, and MFU-tuned configs for Gemma 4 — not from re-deriving these patterns ourselves.

**SGLang-Jax for rollout serving from Phase 3+**. Released October 29, 2025 (LMSYS). Native JAX TPU inference engine with continuous batching, RadixCache prefix caching, custom Pallas kernels for attention and MoE, tensor and expert parallelism, speculative decoding. Their overlap scheduler reduces the prefill-to-decode gap from approximately 12ms to **38µs** on Qwen3-32B by pipelining CPU prep of batch N+1 while TPU processes batch N. For a project where rollout serving is the bottleneck (cua interaction is I/O-bound, the model forward is compute-bound), this is exactly the optimization we cannot afford to write ourselves but absolutely can pip-install.

**vLLM-TPU as alternative inference backend**. October 2025 redesign with `tpu-inference` provides PyTorch and JAX support via a unified JAX→XLA lowering path. PagedAttention, continuous batching, "nearly 5x more performant than the first TPU prototype back in Feb 2025." Tunix integrates with vLLM-TPU for fast rollout. SGLang-Jax and vLLM-TPU are alternatives — pick whichever has better support for the specific Gemma 4 config we use and switch if needed.

**JetStream + Pathways for multi-host scaling in Phase 6+**. JetStream achieves 1703 tokens/sec on Llama 3.1 405B via Pathways multi-host disaggregated serving. Disaggregated serving separates prefill (compute-bound) from decode (memory-bound) into independent pools, improving utilization on both. Production-grade, configurable rather than codable. Becomes relevant if Phase 6 joint multi-model GRPO exceeds single-host capacity.

**CPU host offloading (Intel/Google recipe, April 2026)**. For Phase 3.5 multi-expert merge where memory pressure is highest, the `save_and_offload_only_these_names` recipe lets us offload Q/K/V projection weights to CPU memory during fine-tuning, leveraging the host's 512GB+ RAM rather than rematerializing during the backward pass. Free memory headroom — direct quote of relevance: "modern host machines have much larger memory size than accelerators (512GB or more) and can offer extra compute power." Trades compute for memory in our favor on a v5p-8 / v3-8.

**Where we still need our own Pallas kernels**. The published TPU inference stacks above are optimized for transformer forward passes. Our project introduces operations they don't have built-in:
- Latent Briefing attention-magnitude scoring + top-K (Phase 5)
- STILL perceiver cross-attention (Phase 5.5, optional)
- Composer 2 self-summary KV cache surgery (Phase 3)

These are the natural Pallas targets and they're where systems-ML expertise gets built. ThunderKittens 2.0 (Stanford Hazy Research, February 2026) supports TPU. JAX Scaling Book Chapter 10 ("Programming TPUs in JAX") is the canonical reference. Pallas itself has matured significantly through 2025-2026.

**The honest framing of what we write vs adopt**:

| Component | Approach |
|---|---|
| Gemma 4 model code | Adopt MaxText `gemma4-26b` or `google-deepmind/gemma` |
| SFT trainer | Adopt Tunix PeftTrainer |
| Inference serving for rollouts | Adopt SGLang-Jax or vLLM-TPU |
| Multi-host scaling | Adopt JetStream + Pathways (when needed) |
| Sharding / FSDP | Adopt MaxText conventions |
| Checkpointing | Adopt Orbax via MaxText |
| GRPO trainer | **Write** (alongside Tunix's reference) |
| Composer 2 self-summary | **Write** |
| Latent Briefing | **Write** + Pallas kernel |
| STILL perceiver | **Write** + the three architectural fixes |
| Distillation merge | **Write** (alongside Tunix's reference) |
| Activation steering hooks | **Write** in Flax NNX |
| Custom Pallas kernels | **Write** (the project's systems-ML core) |

Adopting infrastructure we don't need to learn from is not cheating — it's resource discipline. The components in the "Write" column are the ones that teach JAX. The components in the "Adopt" column are infrastructure we'd be poorly reimplementing if we tried.

## Phase 1: Single-model SFT floor

Get a Gemma 4 SFT policy working on cua trajectories. Establish the eval harness, the data pipeline, the trajectory collection.

Core work:
- Gemma 4 E4B Flax inference working on GPU
- cua sandbox wrapped with `reset()` / `step(action)` semantics
- Structured action parser (initially regex-based; will be replaced)
- 100-1000 expert trajectories collected on a narrow task family
- Trajectory dataset published to HF Hub
- Tunix PeftTrainer for SFT, LoRA fine
- Eval pipeline: pass@1 on held-out task instances

Trajectory data quality via iSFT (Baseten, October 2025): rather than training on raw expert trajectories, run the iterative SFT loop. For each prompt, generate an initial completion with the base model, score with a strong grader (Sonnet 4.6 or Gemini 3 Pro with thinking), use grader feedback to iteratively refine the completion until it passes, save the perfected version as the training target. Each example contains O(T) bits of dense supervision rather than the O(1) bit of a binary reward, and the resulting dataset can outperform any fixed teacher because compute-as-supervision lets the grader spend extra reasoning tokens beyond what the teacher policy alone could produce.

iSFT vs RFT — complementary, not duplicate. We use both, in different phases, for different purposes:
- **iSFT (Phase 1)**: REPAIRS failed attempts via grader feedback into perfect outputs, then trains on those. Critical when base success rate is low — at 5% success, RFT requires ~20 samples per training example, iSFT requires ~2-3 refinement rounds. RFT scales exponentially worse as the task gets harder.
- **RFT (Phase 6)**: KEEPS successful rollouts from GRPO and treats them as fresh SFT data. Useful after the policy has been RL-shaped and produces high-quality rollouts naturally; consolidates that behavior back into a stable SFT base.

The mental model: iSFT lifts a weak base policy via compute-as-supervision before RL begins. RFT consolidates a strong RL-trained policy into stable weights. They live at opposite ends of the training arc and never overlap.

Rationale-Guided Training (RGT, Baseten, October 2025): for the iSFT-refined dataset, we already have grader feedback explaining *why* each refinement was needed. Distill those rationales into compact strategic rules using Claude Haiku, then create two training pairs per example:
- Pair A: `prompt → refined_output`
- Pair B: `prompt → [THINK] strategy [/THINK] refined_output`

The model learns `P(y|x,z)` rather than marginalizing `P(y|x)`. At inference, prepend empty think tags; the model performs nearly as well as with explicit rationales because the strategy is now internalized in weights. Baseten reports 10x sample efficiency vs standard SFT and 1.5-1.7x vs iSFT alone, with the marginal cost being only the rationale distillation step (which uses the grader feedback we're already generating). The format `[THINK] z [/THINK] output` also composes cleanly with Composer 2 self-summarization in Phase 3 — both are explicit reasoning traces with reward signal flowing through them.

Tzafon insert (1 day, before any training):
- Identify the additive positional embedding tensor in Gemma 4's vision tower
- Measure click-accuracy on a small target-test set (clicking a colored ball at known coordinates)
- Multiplicatively scale the positional embedding by {1.5, 2, 3, 5, 10}, re-measure
- If a scaling factor yields meaningful improvement, use that as default for the rest of the project

Tzafon reports 40% → 80% click accuracy on Qwen3-VL-4B by scaling 3×. If the same effect exists on Gemma 4, this is a free 30-40 point absolute lift before any training. If it doesn't, you've still learned something concrete about Gemma 4's representation layout.

End of phase: a working, evaluable SFT-trained Gemma 4 policy that does GUI tasks at non-zero success rate, trained on iSFT-refined trajectories with RGT strategy tokens.

Gaps filled: pytrees, PRNG, Flax nnx state vs params, Gemma 4 multimodal token formatting, cua's async API, JIT recompilation, Tunix config conventions, qwix LoRA integration, Grain dataset patterns, vision-tower internals, iSFT refinement loops, RGT strategy distillation patterns, dense vs sparse supervision tradeoffs.

## Phase 2: Reverse distillation from FunctionGemma

Bake FunctionGemma's function-call format discipline into Gemma 4's output distribution. This is the "bake in tendencies" step.

- FunctionGemma loaded as teacher (270M, fits trivially alongside Gemma 4)
- Generate function-calling prompt set covering common API surfaces (a few thousand examples is plenty)
- For each prompt, compute FunctionGemma's output distribution
- Train Gemma 4 with KL divergence loss against FunctionGemma's distribution on the same prompts
- Use Tunix's distillation trainer (Logit Strategy) as reference; either configure it directly or write your own version alongside as a learning exercise
- Measure: does Gemma 4 now emit cleaner function-call format on held-out prompts?

End of phase: Gemma 4 with format discipline distilled in, before any cua-specific training.

Gaps filled: reverse distillation in JAX, KL divergence on logits across vocabulary, Tunix distillation trainer source, weighted multi-task loss composition.

## Phase 2 → Phase 3 boundary: cross-stage distillation

Every phase boundary from here on includes an explicit on-policy cross-stage distillation step. Mechanism (GLM-5.1 + DeepSeek V4 pattern):
- Previous-phase final checkpoint becomes the teacher
- New-phase student is trained on its own on-policy rollouts
- Loss = task reward + λ·KL(student || teacher), full-vocabulary KL
- stop_gradient on teacher logits, use the inference engine to fetch teacher logits

Without this, sequential RL phases cause cumulative regression of earlier capabilities. With it, the pipeline becomes monotone in capability.

For this specific boundary: Phase 2's distilled-format-discipline Gemma 4 is the teacher. Phase 3's GRPO student starts from Phase 2's weights, but its training loss includes a KL regularization term against Phase 2's logits. As GRPO pushes the student toward task success, distillation pulls it back toward Phase 2's clean format discipline. Net effect: Phase 3 starts from a strong base and doesn't lose it.

## Phase 3: Multi-expert GRPO with self-summary and the full algorithmic toolkit

Goal: three Gemma 4 specialists, each trained independently with hand-written GRPO + Composer 2 self-summary on a specific cua task family. This is the JAX RL learning core, expanded to multi-domain.

The multi-expert pivot (DeepSeek V4 pattern): instead of training one Gemma 4 generalist on the union of cua task families, train three separate specialists — file-ops expert, web-form expert, settings-toggle expert. Each gets its own GRPO loop with task-specific rewards. Solves multi-domain interference (specialists don't pollute each other's gradients during specialization). End artifact is three independently-evaluable expert checkpoints plus measurable per-domain expert performance.

For each specialist, the training loop includes:

GRPO core:
- Read Tunix's GRPO trainer line by line, write annotation document
- Run Tunix GRPO demo on a tiny problem (Qwen 0.5B math) and reproduce their curve
- Write your own GRPO trainer from primitives, alongside Tunix's
- Roll out N completions per prompt, score with cua verifier, group-relative advantages, policy gradient with KL regularization to frozen reference

Composer 2 self-summarization:
- Rollouts can chain across `<summary>...</summary>` boundaries
- Reward propagates through summary tokens
- KV cache surgery for the self-summary path

Algorithmic refinements (Composer 2 + R1-zero analysis):
- **Dr. GRPO loss form**: no length standardization, no group std normalization. Removes length bias and degenerate within-group amplification. ~10 LOC change.
- **k1 KL estimator**: `k1 = -log(r)` instead of `k3 = (r-1) - log(r)`. Lower variance at large KL divergence, which dominates early training. ~2 LOC change.
- **Nonlinear length penalty**: $C_{length}(x) = \frac{(1+kx)^{1-q}-1}{k(1-q)}$ where x is a weighted combination of thinking/tool-call/output tokens. Concave-down and increasing — penalizes verbosity on easy tasks while still allowing long thinking on hard tasks. cua tasks vary wildly in difficulty, so this fits. ~50 LOC.

Reward shaping:
- **Tzafon failure-recovery rewards**: detect mid-trajectory action failure (verifier signal that expected element didn't appear), give a small auxiliary reward when the agent then tries a *different* action that succeeds, give a small penalty for repeating the failed action. Tzafon's insight: "recovery from failures > click accuracy" because at 95% per-step accuracy, a 32-step trajectory has 19.5% success.
- **Generative Reward Models (GRMs) with multiple rubrics** (Kimi K2.6 pattern): alongside the binary cua verifier, add LLM-judged trajectory-quality rubrics — efficient action selection, correct intent inference, low redundancy in tool calls. Use a smaller frozen model (Gemma 3 12B?) as the GRM. Rotate among 2-3 rubrics per training step to prevent the policy from overfitting to one rubric. ~200 LOC + a frozen judge model.
- **RGT rationale bank reuse**: the strategic rules distilled from Phase 1 grader feedback (the `[THINK] z [/THINK]` rule library) double as a starting taxonomy for the GRM rubrics here. Each RGT rule ("don't repeat the same failed action," "verify form fields before submitting," "scroll before claiming an element doesn't exist") becomes a candidate rubric the GRM can score against. This reuses Phase 1's compute investment and gives GRM rubrics that are already validated against the cua task family rather than guessed at.

Infrastructure pattern (small-scale slime):
- **Decoupled rollout/training workers**: one process runs cua rollouts continuously into a buffer, another consumes batches and runs training updates, weights synced periodically (every K steps). At our scale we don't need cross-region clusters, but the *pattern* is valuable because cua rollouts are I/O-bound (browser interaction) and training is compute-bound — separating them enables overlap. ~400 LOC.
- **APRIL (Active Partial Rollouts)**: preempt long-tail rollouts at a soft deadline, salvage partial trajectories with a flag indicating incompleteness. cua has long-tail behavior (some tasks complete in 3 actions, some take 30). ~150 LOC of timeout + partial-reward bookkeeping.

Empirical fact useful for debugging (Fireworks observation, March 2026): more than 98% of bf16 weights remain bit-identical between adjacent RL checkpoints. Intuition: small RL learning rates plus sparse RL gradient signal mean most fp32 movements never cross the bf16 representation threshold. If your RL run looks like nothing is moving, that's expected behavior, not a bug. The "Understanding and Exploiting Weight Update Sparsity" paper (arXiv 2602.03839) provides the theoretical foundation.

Comparison and triangulation:
- Compare your trainer's curve to Tunix's GRPO on the same task. They should be close.
- Tinker baseline: same task, same data, LoRA SFT and GRPO on Qwen3-VL-30B-A3B-Instruct via Tinker (now generally available with vision input as of Dec 2025). Three-way comparison: Tinker SFT (PyTorch), Tunix GRPO (JAX library), your hand-written GRPO with all the bells and whistles (JAX hand-written).

Pallas:
- Write one custom kernel on the hot path: logprob computation over the vocab, GRPO advantage normalization, or KV cache update for the self-summary path. Benchmark vs vanilla.

TPU:
- TRC v3-8 is the working baseline. Port the GRPO trainer to TPU once local GPU runs are stable. shard_map for rollouts, FSDP-style param sharding when the policy gets bigger via MaxText conventions. Profile early and often using `jax.profiler` + TensorBoard (JAX Scaling Book Chapter 9 is the canonical reference). The 30-day TRC clock is real — don't waste cycles on broken JIT recompilation loops or debug runs that should have happened on local GPU.

Autoresearch ratchet:
- Claude Code as agent, your trainer as substrate, task-success rate as metric. Run overnight on a Saturday.

End of phase: three Gemma 4 expert checkpoints, each beating SFT on its specific task family. Plus a hand-written GRPO trainer, a Pallas kernel, three baselines on a chart, and an end-to-end TPU training run.

Phase 3 by-product (cheap, do it while you have the trajectories): for each specialist, derive a behavioral steering vector via Contrastive Activation Addition (CAA, Rimsky et al.). For each (specialist, task) pair, run forward passes on contrasting trajectories (specialist-style vs generalist-style on the same prompts), capture residual stream activations at multiple layers, take the mean difference. Save per-layer vectors to disk. ~50 LOC plus storage. These vectors get reused in Phase 7 for runtime mode-switching on the merged model — meaning even after Phase 3.5 collapses three specialists into one, you can re-apply specialist behavior at inference via vector addition rather than weight swapping. Cost is essentially zero given the trajectories already exist.

Steer's empirical findings (Ramp Labs, March 2026) give us strong priors for Gemma 4 since it's the same family as Gemma 3 27B-IT they tested on:
- Best single layer is approximately 66% depth (layer 41 of 62 for Gemma 3 27B; on Gemma 4 E4B with its 26 layers, target around layer 17)
- Sparse 5-layer global-attention configuration outperforms dense local-attention configurations
- Magnitude calibration is non-monotonic — binary search starting at α=1.0
- Different concepts have different natural magnitudes; expect to recalibrate per vector

Gaps filled: scan over multi-segment rollouts, stop_gradient on reference logprobs, batch-mode KL approximation, Mesh + NamedSharding + PartitionSpec, host-callback for env interaction (since cua isn't JIT-compatible), KV cache surgery for self-summary segments, Pallas BlockSpec, Tinker API, GRM evaluation patterns, contrastive activation addition, residual stream hook patterns in Flax NNX.

## Phase 3.5: Multi-expert merge via on-policy distillation

Take the three specialists from Phase 3 and merge them into a single Gemma 4 generalist via on-policy distillation. This is the DeepSeek V4 mechanism: student samples its own trajectories, three specialists provide dense full-vocabulary logit signal, student learns to integrate distinct proficiencies into one model.

Empirical foundation (Baseten "Dense, on-policy, or both?", March 2026): the design of this phase is not arbitrary. Baseten ran a controlled comparison of post-training signals on a constitutional alignment task: off-policy iSFT, on-policy iSFT, vanilla GRPO, off-policy self-distillation, and on-policy self-distillation (OPSD). The cleanly separable result: only the *combination* of on-policy rollouts AND dense supervision generalizes out-of-distribution. Off-policy distillation with the same teacher and constitution fails completely. RL with sparse scalar rewards fails. The mechanism is DAgger-style: training on the learner's own state distribution avoids compounding error from teacher-student distribution mismatch. Phase 3.5 is OPSD scaled to multi-teacher merge — three specialists provide dense logit signal on rollouts the student itself generated.

Mechanism:
- Initialize student from one of the specialists (or from Phase 2's checkpoint)
- Student generates rollouts on a mixed prompt distribution (drawn from all three task families)
- For each token, compute KL divergence from each specialist's output distribution
- Loss = weighted sum of per-specialist KL divergences. Per-specialist weighting is tunable; start equal, then upweight specialists whose domains the student is weakest at.
- KL is computed over the full vocabulary, not a token-level estimate, to stabilize gradients when specialists disagree on a token.

This is smaller than Phase 3 because it's distillation rather than full GRPO. End artifact: one merged Gemma 4 generalist that performs well across all three task families, plus the three specialist checkpoints as separate artifacts.

End of phase: four checkpoints (three specialists + one merged generalist), four lines on the per-domain eval chart, a clean ablation for the writeup ("how much does multi-expert + merge beat single-policy training on the same total compute?").

Gaps filled: full-vocabulary KL in JAX without OOM, on-policy distillation infrastructure, multi-teacher loss composition, weighted distillation strategies, mode-seeking reverse KL dynamics (the mechanism behind why OPSD preserves prior capabilities — see Baseten "Continual learning and the post monolith AI era" Appendix on RL vs SFT for the theory).

## Phase 3.5 → Phase 4 boundary: cross-stage distillation

Merged Phase 3.5 model is the teacher for Phase 4's routing-specific Gemma 4 student. Same mechanism as before.

## Phase 4: Introduce FunctionGemma as actor + simple routing

Split the single-model architecture into Gemma 4 (planner) + FunctionGemma (actor), connected by text-level handoff. No Latent Briefing yet.

- Fine-tune FunctionGemma on cua action schema using Tunix
- Inference pipeline: Gemma 4 sees screenshot, emits intent in natural language; FunctionGemma takes intent text, emits structured action
- Compare end-to-end task success vs the Phase 3.5 single-model setup
- Latency comparison: how much does the routing setup speed up action emission?

End of phase: a two-model GUI agent with cleaner separation of concerns and faster action emission.

Gaps filled: multi-model serving in JAX, FunctionGemma's chat format and special tokens, fine-tuning a small specialist alongside a big generalist, latency profiling.

## Phase 4 → Phase 5 boundary: cross-stage distillation

Phase 4 model is teacher for Phase 5's latent-briefing student. Same mechanism.

## Phase 5: Latent Briefing replaces text handoff

Replace the text-level Gemma-4-to-FunctionGemma handoff with KV cache compaction. This is the highest-leverage Pallas kernel target in the project.

- Implement attention-based KV cache compaction: score keys by attention magnitude against task query, top-K selection, ridge regression for value reconstruction of dropped keys
- Memory optimizations: in-place softmax, CPU offload of stale cache, chunked prefill
- Single-agent variant first (within Gemma 4 itself, for long rollouts)
- Then multi-agent variant: compact Gemma 4's trajectory KV, transfer to FunctionGemma's cache space
- Pallas kernel for the attention-magnitude scoring + top-K selection (this is the natural Pallas target)
- Compare token usage and latency vs Phase 4 text handoff

Implementation requirements (Baseten "Repeated KV cache for long-running agents", March 2026): three findings from Baseten's incremental compaction experiments are load-bearing and prevent silent failures. Apply these from day one rather than discovering them via debugging:

- **Chunked compaction is a structural requirement, not an optimization**: Baseten reports a 49 percentage point gap between monolithic and chunked compaction at the same compression ratio on heterogeneous multi-document context. The mechanism: cua trajectories are heterogeneous (different apps, different screenshots, different verifier signals across steps), which produces approximately block-diagonal attention. A monolithic least-squares fit over the entire concatenated sequence is geometrically ill-posed because each chunk's queries load on nearly orthogonal subspaces. The fix is to fit each trajectory segment independently (split at app/task boundaries or at fixed step intervals), then stitch results. Each sub-problem has dense, well-conditioned design matrix.

- **Re-compaction has JPEG-of-a-JPEG dynamics**: when compacting an already-compacted cache, error amplifies through the recurrence $e_{r+1} \leq ||A_r|| \cdot e_r + \eta_r$ where $||A_r||$ is the attention weight matrix's local condition number. A handful of poorly-conditioned heads drive worst-case error. Baseten reports a 4-16% accuracy drop at moderate compression ratios beyond fresh compaction. For our project this means: prefer fresh re-prefill when feasible (every few turns), and when forced into re-compaction, use nonuniform head budgets to allocate more capacity to sensitive heads.

- **Fresh compression degrades gracefully**: at chunked-fresh 8x compression, accuracy stays above 80% on dense factual QA; the marginal cost of additional compression decreases logarithmically. The design choice is therefore: build the API so re-prefill from raw context is cheap to invoke periodically, rather than relying on indefinite re-compaction.

End of phase: a working multi-agent GUI system with latent inter-model communication. Real research artifact.

Gaps filled: custom KV cache management in JAX, attention manipulation outside the standard forward pass, Pallas kernel for sparse attention scoring, ridge regression on reconstructed values, chunk-boundary detection in heterogeneous trajectories, condition-number diagnostics for attention heads.

## Phase 5.5: STILL — learned amortized KV cache compaction

This phase is optional but high-value, particularly for the JAX-learning goal. It is positioned after Latent Briefing because the two are temporally distinct rather than redundant: Latent Briefing handles single inter-model handoffs analytically; STILL handles continuous in-context compression as the agent runs. Together they form a complete memory hierarchy.

The relationship STILL has to Latent Briefing is exactly what SAEs have to per-input sparse coding. Latent Briefing optimizes per-context at inference time. STILL trains a fixed perceiver encoder once, then compacts any cache in a forward pass. After this phase the agent has both: STILL for continuous rolling compression during long rollouts, Latent Briefing as a fallback / inference-time tool for inter-model handoffs.

Why this is unusually rich JAX learning. The three architectural fixes Charles O'Neill et al. describe are each non-trivial JAX engineering and each silent-failure-prone:

- **RoPE-aware un-rotate / re-rotate pipeline**: apply inverse RoPE to strip positional encoding, run the perceiver with its own internal RoPE on `linspace(0, T-1, num_latents)`, re-rotate compact keys at evenly-spaced positions. This teaches you exactly how RoPE decomposes and recomposes at the tensor level. Their early MoE run spent 235 steps with broken positional encoding before this was identified.
- **Identity initialization with biased Q/K projections**: value pathway as identity chain (v_proj = identity, out_proj = identity, W_key/W_value = identity patterns), Q/K bias terms set so that projected keys are content-independent at init and the perceiver's RoPE is the dominant attention signal, zero-init residuals for self-attention output. This unlocks scaling from 128 to 8192 latents (the prior ceiling was an optimization failure, not an architectural one).
- **Removing the final RMSNorm**: standard perceivers normalize before output heads; here that destroys the natural norm variation in real KV entries.

Each fix is a 10-50 line code change but each requires a deep understanding of why it works. From a JAX-learning density perspective, this phase is the single highest-leverage chunk in the project.

Mechanism for our setting:

1. Adapt STILL to Gemma 4 multimodal. Gemma 4 E4B has 26 layers; per-layer perceiver = 7M params × 26 = ~180M trainable params. Plenty of room on a v3-8 TPU.
2. Training data: generate extractive MCQs over GUI trajectories. "After step 12, what was the value in the username field?" / "Which menu was open at step 18?" / "Did the agent click submit?" Use Claude Sonnet 4.6 to generate questions and distractors separately following Baseten's two-stage pipeline.
3. KL distillation training: teacher = LLM + full cache, student = LLM + STILL-compacted cache, loss = KL on answer tokens. Frozen Gemma 4 throughout. Only the perceiver trains.
4. Use Tunix's distillation trainer as the scaffolding (it's the same Logit Strategy pattern). Write the perceiver itself from scratch — that's the JAX learning.
5. Train at 8x compression (1024 latents from 8K context) as the headline ratio. Sweep down to 64x for the writeup curve.

Iterative compaction (the Phase 5.5 stretch): compact the first 8K of trajectory, prepend to the next 8K, compact again, etc. Train on randomized number of passes per training step (1, 2, 4, 8) so the perceiver sees both fresh and previously-compacted input. This unlocks unbounded GUI agent rollouts with fixed memory.

Composing STILL with the rest of the stack:

- During cua rollouts in Phase 6 (joint GRPO), STILL runs continuously: every K tokens, drop the oldest chunk through STILL, prepend the compact result.
- Composer 2 self-summary still triggers at coarse RL-learned checkpoints (token-level discrete summaries). The summary's KVs themselves can be STILL-compressed.
- When the orchestrator hands off to FunctionGemma, Latent Briefing handles the inter-model transfer. STILL gives a smaller starting cache to compress.
- Cross-stage distillation between Phase 5 and Phase 5.5: the perceiver is initialized with identity init regardless, but the *student LLM* during STILL training should match the Phase 5 final checkpoint's behavior.

The most ambitious sub-stretch within this phase: integrate STILL into the agent's action space, so the model can invoke it as a tool ("compress the older trajectory but keep the recent screenshots at full fidelity") and learn the compression policy via GRPO. This is what Charles O'Neill describes as capability (2) in the Baseten roadmap — learned context management. Doing it in pure JAX before any frontier lab publishes the equivalent would be a real research contribution.

End of phase: a learned KV-cache compactor for Gemma 4, evaluated at multiple compression ratios, with iterative compaction extending to unbounded GUI agent rollouts. Plus a Pallas kernel for the perceiver cross-attention if you want a second one. Plus a dataset of GUI-trajectory MCQs published to HF Hub.

Gaps filled: RoPE decomposition and recomposition at tensor level, perceiver architectures in JAX, identity-init coordination across multi-block networks, KL distillation infra against frozen LLM logits, training data generation with Claude as a labeling tool, iterative-compaction training schedules.

Cost is honest: this phase is 1.5-3 months of work on its own. It's optional. The descope ladder accommodates skipping it.

## Phase 5.5 → Phase 6 boundary: cross-stage distillation

## Phase 6: Joint end-to-end GRPO

Train both Gemma 4 and FunctionGemma together with task reward propagating through both. Genuinely novel territory.

- Reward attribution: how does task success at the cua level credit-assign across two models?
- Two coupled policy gradients, with shared advantage signal
- Off-policy management: which model lags behind in update frequency?
- Training stability: this is fragile, expect lots of debugging
- Rejection-sampling fine-tuning (RFT) loop (Kimi pattern): after GRPO converges, extract high-reward trajectories from the rollout buffer, use them as fresh SFT data for both models, iterate. Each loop produces new checkpoints with successful behavior consolidated.

End of phase: a jointly-trained multi-model agent system. Probably the most ambitious training run in the project.

Gaps filled: multi-policy GRPO, shared-reward credit assignment, joint training stability tricks, RFT data pipeline.

## Phase 7: Stretches and writeup

Pick any subset:

- **SAE interp**: train SAEs in JAX on Gemma 4 and FunctionGemma residual streams during task execution. Compare features that fire on visual perception (Gemma 4) vs action emission (FunctionGemma). Compare to Goodfire's R1 work and Ramp Labs' Steer.
- **Activation steering for runtime mode-switching on the merged model**: take the per-specialist steering vectors derived as a Phase 3 by-product. Apply them at inference to the Phase 3.5 merged model. Demonstrate that runtime vector addition recovers specialist behavior on the merged model, eliminating the need to deploy three separate checkpoints. Sweep magnitudes following Steer's binary-search protocol; expect non-monotonic effects (Taimeskhanov et al.). The deliverable is a clean ablation: merged-model + runtime-vector vs deployed-specialist on each task family. If the gap is small, you've shown that activation steering is a viable deployment-time replacement for multi-expert serving.
- **SAE-detect + steer-actuate closed-loop governance**: this is the Phantom-thesis governance prototype. Goodfire-style SAE probes detect high-level conditions on the agent's residual stream during rollout (privileged-mode operation, hallucinated tool input, PII emission, off-policy exploration). When a probe fires above threshold, apply a precomputed steering vector that nudges the activations toward the corrective behavior (cautious-mode, grounded-mode, redacted-output, exploitation-mode). Detection and actuation both happen at the activation level, before token generation completes. This closes the sense-act loop at the mechanistic level rather than via output filtering or post-hoc moderation. Implement at inference time on the Phase 3.5 merged model. Frame this as the Phantom governance prototype in the writeup — mechanistic governance, fused at the activation level, not bolted on. Real research artifact.

  Published precedent (Baseten + ANU, "Do transformers notice their own mistakes?", May 2025, arXiv:2507.23221): this is not speculative. The paper demonstrates the exact pattern. A logistic-regression linear probe on Gemma-2's residual stream at layers 12-30 detects contextual hallucinations with F1 > 0.95 on news data and > 0.70 on logical contradictions. The probe is generator-agnostic (the observer model didn't write the text) and runs in a single forward pass. They then prove causality: injecting the normalized hallucination vector into a generator's residual stream during writing produces a 51 percentage point swing in hallucination rate (35% at α=−60 vs 86% at α=+60). Detection via linear probe + actuation via single steering vector is the published mechanism we extend to a multi-condition sense-act loop. They release the ContraTales benchmark; we extend the same approach to GUI-agent-specific failure modes (privileged mode, tool input hallucination, PII emission). Cite this paper as the foundation; our contribution is the multi-condition + multi-vector extension on a multimodal agent in pure JAX.
- **Tzafon-style cosine similarity layer probing**: pick a steering vector for a semantic concept ("submit button"), probe each layer's patch embeddings for cosine similarity, plot information-density vs depth. Compare pre- and post-fine-tuned models. Perfect SAE companion.
- **MCTS over reasoning**: at inference, MCTS over Gemma 4's reasoning chains before delegating to FunctionGemma. AlphaProof for GUIs. Recovers the MCTS-in-JAX practice.
- **OSWorld scale-up**: from narrow task family to full OSWorld benchmark.
- **Multi-host training**: TRC's standard grant is v3-8 (single host); if a larger grant becomes available (v3-32 or Trillium tier), scale Phase 6 joint GRPO across hosts via JetStream + Pathways. Disaggregated serving becomes meaningful at this scale.
- **Quantization**: NVFP4 / MXFP8 training following the Composer 2 recipe.
- **IDM in JAX (FDM-1 inspired)**: train an IDM on cua's deterministic ground-truth (screenshot_t, screenshot_t+1) → action pairs using a masked diffusion architecture. Apply to unlabeled screen recordings (YouTube tutorials, ScreenSpot demos). Mix into Phase 1 SFT data for retraining. Genuine data-augmentation strategy beyond cua's deterministic API.

Then writeup. Multi-plot blog post. Open-source repo. Trajectory dataset, all model checkpoints (specialists, merged, FunctionGemma actor, jointly-trained), the SAE if you trained one, the IDM if you trained one, and the Pallas kernel as standalone artifacts on HF Hub.

## Descope ladder (for any external pressure)

This is now a graceful-degradation ladder rather than a sprint constraint:

- **Floor**: Phase 1 only. SFT'd Gemma 4 on cua. A real artifact.
- **Reasonable stop**: Phases 1-3. Multi-expert GRPO with self-summary, Tinker baseline, Pallas kernel. Already a serious project.
- **Strong stop**: Phases 1-3.5. Multi-expert + merge via on-policy distillation. Research-grade on its own.
- **Stronger stop**: Phases 1-5. Multi-model agent with Latent Briefing. Frontier-relevant.
- **Memory-frontier stop**: Phases 1-5.5. Add learned amortized KV compaction (STILL) on top of the above. Three-tier memory hierarchy in pure JAX. Genuinely novel.
- **Full**: All phases including joint GRPO and Phase 7 stretches. Paper-grade contribution.

## Tool integration (final, ambitious)

| Tool | Role | Phase |
|---|---|---|
| Gemma 4 (Flax) | Multimodal planner / reasoner | 1+ |
| MaxText | Production JAX/Flax reference codebase for Gemma 4; supplies optimal sharding, vocabulary tiling, MFU configs; integrates Tunix for post-training | 1+ |
| Tunix | SFT, distillation, GRPO scaffolding + reference impls | 1+ |
| SGLang-Jax | Native JAX TPU inference engine: continuous batching, RadixCache prefix caching, custom Pallas kernels for attention + MoE, overlap scheduler (12ms→38µs prefill-decode gap) | rollout serving from Phase 3+ |
| vLLM-TPU (via tpu-inference) | Alternative inference backend with PagedAttention and continuous batching; 5x perf vs Feb 2025 prototype | rollout serving from Phase 3+ |
| JetStream + Pathways | Multi-host disaggregated serving for Phase 6+ scale (when joint multi-model training exceeds single host) | 6+ multi-host |
| FunctionGemma (270M) | Distillation teacher, then inference actor | 2, 4+ |
| cua + cua-bench | RL env, verifier, trajectory collection | 1+ |
| HF Datasets | Trajectory streaming and publication | 1+ |
| CPU host offloading (Intel/Google recipe) | Free memory headroom for Phase 3.5 multi-expert merge | 3.5 |
| iSFT (Baseten) | Repair-failed-attempts data refinement; lifts Phase 1 trajectory quality via grader-feedback iteration | 1 |
| RGT / Rationale-Guided Training (Baseten) | 10x sample efficiency over SFT via `[THINK] strategy [/THINK]` rationales distilled from grader feedback | 1 (and reusable in Phase 3 for GRM rubric distillation) |
| OPSD / On-Policy Self-Distillation (Baseten "Dense, on-policy, or both?") | Empirical foundation for Phase 3.5 multi-teacher merge: dense + on-policy is necessary and sufficient for OOD generalization | 3.5 |
| Composer 2 self-summary | Long-horizon context management | 3+ |
| Dr. GRPO + k1 KL + length penalty | Algorithmic GRPO refinements | 3+ |
| Tzafon failure-recovery rewards | Reward shaping for retry behavior | 3+ |
| Generative Reward Models (Kimi K2.6) | Multi-rubric trajectory quality scoring | 3+ |
| Decoupled rollout/training (slime pattern) | Small-scale rollout/training overlap | 3+ |
| APRIL (Active Partial Rollouts) | Long-tail trajectory preemption | 3+ |
| Multi-expert + on-policy distillation merge (DeepSeek V4) | Phase 3.5 specialist consolidation | 3.5 |
| Cross-stage on-policy distillation (GLM-5.1 + DeepSeek V4) | Capability preservation between phase boundaries | every phase boundary |
| Latent Briefing | Multi-agent KV cache transfer + Pallas kernel | 5+ |
| Repeated KV cache findings (Baseten) | Chunked compaction structural requirement, JPEG-of-a-JPEG re-compaction dynamics, fresh-prefill graceful degradation — all govern Latent Briefing implementation | 5 |
| STILL (perceiver-based learned KV compaction) | Continuous in-context compression, the amortized version of Latent Briefing | 5.5 |
| Tinker (with Qwen3-VL via GA + vision input) | PyTorch baseline for triangulation | 3 |
| RFT (rejection-sampling fine-tuning) | Self-improving data loop | 6 |
| SAM 3.1 | Optional vision preprocessor (ablation) | 3 or 7 |
| Karpathy autoresearch (pattern) | Overnight ratchet on trainer | 3+ |
| HF ml-intern | Workflow accelerator for ablations | 2+ |
| Goodfire R1 SAEs / Ramp Steer | Reference for Phase 7 SAE work | 7 |
| Activation steering (CAA, Rimsky et al.) | Per-specialist behavioral vectors as Phase 3 by-product; runtime mode-switching and governance actuation in Phase 7 | 3 (derive), 7 (apply) |
| Linear hallucination probe + steering vector (Baseten + ANU, arXiv:2507.23221) | Published precedent for Phase 7 SAE-detect + steer-actuate; 0.95+ F1 detection, 51pt swing under causal steering | 7 |
| SAE-detect + steer-actuate governance loop | Phantom-thesis mechanistic governance prototype | 7 |
| Tzafon layer cosine-sim probing | Phase 7 interpretability companion | 7 |
| FDM-1 IDM in JAX | Phase 7 data augmentation stretch | 7 |
| Prime Intellect Env Hub | Fallback env source if cua wrapping stalls | fallback |

## Rejected with stated reasoning

For completeness, here are techniques that surfaced during planning but did not make it in, with reasoning:

- **FDM-1 paradigm pivot** (video-only foundation model): different project, abandons Gemma 4. Cited in writeup as paradigm pointer, not adopted as architecture.
- **OpenTinker as scaffolding**: PyTorch + verl. Wrong stack for JAX learning. Tunix already supports multi-turn agentic RL.
- **PARL scale-up to 300 sub-agents** (Kimi K2.6): irrelevant at our scale. We have one cua sandbox per training step.
- **slime as a framework**: PyTorch + Megatron + SGLang. Wrong stack. We adopt the *decoupling pattern*, not the framework.
- **Manifold-Constrained Hyper-Connections** (DeepSeek V4): pre-training architectural change. Out of scope.
- **Muon optimizer**: pre-training thing. Tunix uses Optax AdamW. Switching mid-project is a rabbit hole.
- **Router replay for MoE**: Gemma 4 isn't MoE.
- **Three reasoning effort modes** (DeepSeek V4): serving-time feature, not RL.
- **PipelineRL in-flight weight updates at production scale**: too much engineering for marginal benefit at our scale. Tunix's periodic weight sync is the right primitive.
- **Delta-compressed cross-region S3 weight sync**: wrong scale. We're on one node. Fireworks's "Frontier RL is Cheaper Than You Think" (March 2026) is the canonical articulation of this pattern, and even they explicitly note this approach matters less when the model fits comfortably on one node, which is our setting.
- **GAD / Generative Adversarial Distillation** (Baseten "Distillation without the dark", February 2026): only useful when the teacher is closed-source and you have no logit access, since GAD constructs an on-policy reward model via a co-evolving discriminator to substitute for the missing KL signal. We have white-box access to all teachers in our pipeline (FunctionGemma in Phase 2, Phase 3 specialists in Phase 3.5, Phase N model in Phase N→N+1 cross-stage distillation). Direct logit-level KL is the right objective; the GAD machinery is unnecessary overhead. Worth knowing as a backup if we ever distill from GPT-5.4 or Claude Opus 4.7.
- **MuZero/AlphaZero board-game project**: original plan. Pivoted because the multimodal Gemma 4 + cua project teaches JAX in realistic boundary-crossing context, which was the stated goal.

## Gap-filling resources

- **JAX**: docs > tutorials. Autodiff Cookbook, parallelism notebook, Pallas docs.
- **Flax nnx**: official docs.
- **Gemma 4 Flax**: `google-deepmind/gemma` repo, `flax/examples/gemma`.
- **Tunix**: `google/tunix` repo, GRPO and distillation demo notebooks.
- **GRPO**: DeepSeek-Math paper, R1 paper, Dr. GRPO (Liu et al. 2025) for the loss form refinements.
- **Composer 2 self-summary + algorithmic refinements**: Cursor Composer 2 technical report Section 4.1.
- **Cross-stage distillation**: GLM-5 paper Section on cross-stage distillation, DeepSeek V4 paper post-training section. Theoretical foundation: Baseten "Continual learning and the post monolith AI era" Appendix on RL vs SFT (mode-seeking reverse KL preserves prior capabilities; mode-covering forward KL causes catastrophic forgetting — the mechanism behind why on-policy distillation works as a phase-boundary stabilizer).
- **Multi-expert merge**: DeepSeek V4 paper post-training section. Empirical justification: Baseten "Dense, on-policy, or both?" (March 2026) — controlled comparison showing that only on-policy + dense supervision generalizes OOD.
- **iSFT (Phase 1 data refinement)**: Baseten "Iterative SFT (iSFT): dense reward learning" (October 2025). The DAgger connection in the footnote is the key insight — iSFT uses compute-as-supervision to produce targets that exceed any fixed expert policy.
- **RGT (Phase 1 sample efficiency)**: Baseten "Upweight the strategy, not the tokens" (October 2025). 10x sample efficiency, marginal cost free given iSFT grader feedback. Connects to Thinking Machines on-policy distillation as theoretical precedent for dense per-token supervision.
- **GRMs + RFT**: Kimi K2.5 paper, philschmid.de's "How Kimi, Cursor, and Chroma Train Agentic Models with RL" overview.
- **Latent Briefing**: Ramp Labs blog post (April 10, 2026). Implementation requirements: Baseten "Repeated KV cache for long-running agents" (March 2026) — chunked-vs-monolithic structural finding (49pt gap), JPEG-of-a-JPEG re-compaction dynamics, fresh-prefill graceful degradation.
- **STILL**: Baseten Research "Towards infinite context windows: neural KV cache compaction" (April 1, 2026). Read alongside the Cartridges paper (Eyuboglu et al., 2025) and Attention Matching (Zweiger et al., 2026) which STILL builds on. Perceiver IO (Jaegle et al., 2022) is also useful background.
- **Activation steering**: Ramp Labs Steer blog post "How we built Steer, our interpretability playground" (April 2, 2026) — the practical layer-selection findings on Gemma family. Turner et al. "Steering Language Models With Activation Engineering" (arXiv 2308.10248). Rimsky et al. / Panickssery et al. "Steering Llama 2 via Contrastive Activation Addition" (arXiv 2312.06681) — the canonical CAA construction. Persona Vectors (Chen et al., arXiv 2507.21509) for character-trait steering. Subhadip Mitra's "Activation Steering in 2026: A Practitioner's Field Guide" for the magnitude non-monotonicity finding (Taimeskhanov et al.). IBM activation-steering library as a reference implementation. steering-vectors Python library.
- **Phase 7 governance loop**: Baseten + ANU "Do transformers notice their own mistakes? Finding a linear hallucination detector inside LLMs" (May 2025, arXiv:2507.23221). Linear probe + single steering vector achieves generator-agnostic detection (F1 > 0.95) and causal control (51pt swing). Direct published precedent for the SAE-detect + steer-actuate pattern.
- **Continual learning thesis**: Baseten "Continual learning and the post monolith AI era" (February 2026). Strategic framing for why specialist + composition (our Phase 3 + 3.5 + cross-stage distillation pipeline) is the right shape, contra monolithic generalists.
- **FunctionGemma**: model card, formatting guide, Mobile Actions fine-tuning notebook in `google-gemini/gemma-cookbook`.
- **Tzafon insights**: Tzafon "Training VLM for CUA" blog post (Feb 26, 2026).
- **FDM-1 IDM (if attempted)**: si.inc FDM-1 blog post (Feb 23, 2026).
- **Tinker GA + vision**: Thinking Machines blog, December 12, 2025.
- **cua**: cua.ai/docs, cua-bench guide, OSWorld paper.
- **TPU**: cloud.google.com/tpu/docs, TRC Slack.

## Principles for the project

- **Demo over spec**: every Friday, something runnable, even if scrappy.
- **Write the parts that teach, configure the parts that don't**: Tunix SFT and Orbax checkpointing get configured. Your GRPO trainer, your self-summary integration, your Pallas kernel, your Latent Briefing implementation, your distillation merge get written.
- **Read the parts that teach**: Gemma 4 Flax source, Tunix GRPO source, FunctionGemma format guide, Composer 2 paper, GLM-5 + DeepSeek V4 post-training sections. Annotate them.
- **Top-down**: each technique gets pulled in when you actually need it. Don't pre-learn shard_map, learn it when your rollout is slow.
- **Confrontational testing**: try to break your verifier weekly. Reward hacking, degenerate self-summaries, format-collapse in the actor, GRM rubric overfitting are all real.
- **Compounding mastery**: each phase makes the next phase tractable. Phase 5 Latent Briefing only makes sense after Phase 4 routing exists. Don't skip ahead.
- **Cross-stage distillation as discipline**: every phase boundary gets an explicit distillation step. Do not skip these even if they feel like overhead. They are what make sequential training stable.
- **Respect the TRC budget**: the 30-day TRC trial is the project's most expensive constraint. Local GPU first, TPU only when the work genuinely requires it (Phase 3 multi-expert GRPO and beyond). Profile every TPU run with `jax.profiler` before assuming it's working — JIT recompilation, tracer leaks, and silent data-loading bottlenecks routinely waste hours on TPU. If a v3-32 grant becomes available later, treat it as a separate budget for Phase 6 joint training.
- **Journal daily**: three lines, every night. What shipped, what broke, what isn't understood.

## What success looks like

Whenever you stop, you have:

1. A working artifact at the phase you stopped at.
2. Real fluency in JAX, Flax NNX, Optax, and Tunix from reading their actual production source and writing your own components alongside.
3. Hands-on experience with techniques that most engineers in the field have only read about: iSFT data refinement (compute-as-supervision), Rationale-Guided Training (10x sample efficiency via strategy tokens), GRPO from primitives with Dr. GRPO + k1 KL + length penalty, Composer-style learned token-level compression, multi-expert specialization with OPSD-justified on-policy distillation merge, cross-stage distillation between phases (mode-seeking reverse KL theory), Latent Briefing analytical inter-model KV transfer (with chunked-compaction structural fix and JPEG-of-a-JPEG re-compaction discipline), STILL-style learned amortized KV compaction with the three architectural fixes (RoPE-aware un-rotate/re-rotate, identity initialization, removed final norm), reverse distillation, multi-model joint RL, GRMs with multi-rubric scoring, RFT data loops, decoupled rollout/training architecture, contrastive activation addition for runtime mode-switching, linear-probe + steering-vector closed-loop governance (extending the Baseten/ANU hallucination-detector precedent to multi-condition GUI-agent failure modes).
4. A direct prototype of the agent training stack relevant to your Phantom thesis.
5. A portfolio of model checkpoints: three specialists, one merged generalist, one FunctionGemma actor, one jointly-trained multi-model system, the STILL perceiver if Phase 5.5 was done, plus optionally an SAE and an IDM. A trajectory dataset and (if Phase 5.5) a GUI-trajectory MCQ dataset on HF Hub.
6. The credibility to talk shop, in detail, with anyone at any frontier lab doing post-training of multimodal agentic models.

The project is its own reward. Ship it at any phase. Each phase is a real artifact.

## References

Every link, paper, repo, and resource accumulated through the planning sessions for this project.

### Foundation models and reference implementations

- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma documentation hub: https://ai.google.dev/gemma/docs
- google-deepmind/gemma (production Flax reference): https://github.com/google-deepmind/gemma
- FunctionGemma overview: https://ai.google.dev/gemma/docs/functiongemma
- FunctionGemma 270M-it on Hugging Face: https://huggingface.co/google/functiongemma-270m-it
- FunctionGemma formatting and best practices: https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
- FunctionGemma fine-tuning notebook (Mobile Actions): https://github.com/google-gemini/gemma-cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb
- Mobile Actions dataset: https://huggingface.co/datasets/google/mobile-actions
- DeepSeek-V4-Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

### JAX-native training and inference infrastructure

- Tunix (Google's JAX-native post-training library): https://github.com/google/tunix
- Tunix on PyPI: https://pypi.org/project/google-tunix/
- MaxText (production JAX/Flax LLM codebase, AI-Hypercomputer): https://github.com/AI-Hypercomputer/maxtext
- MaxDiffusion (analog for diffusion models): https://github.com/AI-Hypercomputer/maxdiffusion
- SGLang-Jax repo: https://github.com/sgl-project/sglang-jax
- SGLang-Jax announcement (LMSYS, October 29 2025): https://www.lmsys.org/blog/2025-10-29-sglang-jax/
- vLLM-TPU redesign blog (vLLM blog, October 2025): https://blog.vllm.ai/2025/10/16/vllm-tpu.html
- JetStream documentation: https://github.com/google/JetStream
- AI Hypercomputer inference updates blog (Google Cloud, May 2025): https://cloud.google.com/blog/products/compute/ai-hypercomputer-inference-updates-for-google-cloud-tpu-and-gpu
- JetStream on GKE with Pathways tutorial: https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-multihost-tpu-jetstream
- Pathways multi-host inference docs: https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/multihost-inference
- "Building production AI on Google Cloud TPUs with JAX" (Google Developers Blog, November 2025 — Kakao 2.7x, Lightricks, Escalante examples): https://developers.googleblog.com/building-production-ai-on-google-cloud-tpus-with-jax/
- "Leveraging CPU memory for faster, cost-efficient TPU LLM training" (Google Open Source Blog, April 2026 — Intel/Google host offloading recipe): https://opensource.googleblog.com/2026/04/leveraging-cpu-memory-for-faster-cost-efficient-tpu-llm-training.html
- Cloud TPU JAX AI Stack documentation: https://docs.cloud.google.com/tpu/docs/jax-ai-stack
- JAX Scaling Book ("How to Scale Your Model"): https://jax-ml.github.io/scaling-book/
- ThunderKittens 2.0 (Stanford Hazy Research, February 2026 — TPU support): https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
- ParallelKittens (Sul et al., November 2025): https://arxiv.org/abs/2511.13940
- JAX documentation hub: https://jax.readthedocs.io/
- JAX 101 tutorial: https://jax.readthedocs.io/en/latest/jax-101
- Pallas (JAX kernel DSL): https://jax.readthedocs.io/en/latest/pallas
- Flax NNX basics: https://flax.readthedocs.io/en/latest/nnx_basics.html
- Orbax (JAX checkpointing): https://github.com/google/orbax
- Cloud TPU documentation: https://cloud.google.com/tpu/docs
- TPU Research Cloud (TRC): https://sites.research.google/trc

### RL post-training papers and frameworks

- DeepSeek-Math (GRPO): https://arxiv.org/abs/2402.03300
- Dr. GRPO (Liu et al., 2025): https://arxiv.org/abs/2503.20783
- DAPO (Yu et al., 2025): https://openreview.net/forum?id=2a36EMSSTp
- Better KL estimation (Amini et al., NeurIPS 2025): https://openreview.net/forum?id=um9kHMof0c
- KL approximation (Schulman blog): http://joschu.net/blog/kl-approx.html
- GLM-5 technical report: https://arxiv.org/html/2602.15763v1
- GLM-5 GitHub: https://github.com/zai-org/GLM-5
- GLM-5.1 announcement: https://z.ai/blog/glm-5.1
- slime (RL framework behind GLM-5): https://github.com/THUDM/slime
- slime documentation: https://thudm.github.io/slime/
- LMSYS slime overview: https://www.lmsys.org/blog/2025-07-09-slime/
- Kimi K2.5 paper (PARL, GRMs, RFT): https://arxiv.org/html/2602.02276v1
- Kimi K2.6 release: https://www.kimi.com/blog/kimi-k2-6
- Cursor Composer 2 technical report: https://cursor.com/resources/Composer2.pdf
- Cursor self-summarization (Composer 1.5): https://cursor.com/blog/self-summarization
- "How Kimi, Cursor, and Chroma Train Agentic Models with RL" overview: https://www.philschmid.de/kimi-composer-context
- PipelineRL: https://arxiv.org/abs/2509.19128
- AceReason-Nemotron 1.1: https://arxiv.org/abs/2506.13284
- Tinker general availability: https://thinkingmachines.ai/news/tinker-general-availability/
- Tinker homepage: https://thinkingmachines.ai/tinker/
- Tinker cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- OpenTinker: https://github.com/open-tinker/OpenTinker
- Prime Intellect Lab announcement: https://www.primeintellect.ai/blog/lab
- prime-rl: https://github.com/PrimeIntellect-ai/prime-rl
- Fireworks "Frontier RL Is Cheaper Than You Think" (March 23, 2026): https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think
- "Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL" (theoretical foundation for the 98% sparsity observation, arXiv 2602.03839): https://arxiv.org/pdf/2602.03839
- AReaL (asynchronous RL framework, arXiv 2505.24298): https://arxiv.org/abs/2505.24298
- Kimi checkpoint-engine: https://moonshotai.github.io/checkpoint-engine/
- MiniMax Forge (scalable agent RL framework): https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm
- Federico Cassano on Composer 2 multi-cluster training: https://x.com/ellev3n11/status/2034778708163404102

### Computer-use agents and environments

- cua repo: https://github.com/trycua/cua
- cua macOS internals blog post: https://github.com/trycua/cua/blob/main/blog/inside-macos-window-internals.md
- cua documentation: https://cua.ai/docs
- OSWorld benchmark paper: https://arxiv.org/abs/2404.07972
- Tzafon "Training VLM for CUA": https://www.tzafon.ai/blog/training-vlm-for-cua
- FDM-1 (Standard Intelligence): https://si.inc/posts/fdm1/
- Standard Intelligence posts: https://si.inc/posts/
- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- HuggingFace ml-intern: https://github.com/huggingface/ml-intern
- HuggingFace Datasets: https://huggingface.co/docs/datasets

### Memory, context compression, and continual learning

- Latent Briefing (Ramp Labs): https://labs.ramp.com/ (post dated April 10, 2026, "Latent Briefing: Efficient Memory Sharing for Multi-Agent Systems via KV Cache Compaction")
- Ramp Labs research blog: https://labs.ramp.com/
- STILL / "Towards infinite context windows" (Baseten): https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/
- Repeated KV cache for long-running agents (Baseten, March 2026): https://www.baseten.co/research/repeated-kv-cache-for-long-running-agents/
- Distillation without the dark (Baseten, February 2026): https://www.baseten.co/research/distillation-without-the-dark/
- Dense, on-policy, or both? (Baseten, March 2026): https://www.baseten.co/research/dense-on-policy-or-both/
- Iterative SFT (Baseten, October 2025): https://www.baseten.co/research/iterative-sft/
- Upweight the strategy, not the tokens / RGT (Baseten, October 2025): https://www.baseten.co/research/upweight-the-strategy-not-the-tokens-faster-training-with-explicit-reasoning-thro/
- Continual learning and the post monolith AI era (Baseten position piece, February 2026 — the strategic frame for specialist models + on-policy mode-seeking KL): https://www.baseten.co/research/continual-learning/
- Cartridges (Eyuboglu et al., 2025): https://arxiv.org/abs/2506.06266
- Attention Matching (Zweiger et al., 2026): https://arxiv.org/abs/2602.16284
- Perceiver (Jaegle et al., 2021): https://arxiv.org/abs/2103.03206
- Perceiver IO (Jaegle et al., 2022): https://arxiv.org/abs/2107.14795
- Perceiver Resampler / Flamingo (Alayrac et al., 2022): https://arxiv.org/abs/2204.14198
- Knowledge distillation (Hinton et al., 2015): https://arxiv.org/abs/1503.02531
- DAgger (Ross et al., 2011): https://arxiv.org/pdf/1011.0686
- On-policy self-distillation / Thinking Machines: https://thinkingmachines.ai/blog/on-policy-distillation/

### Vision and segmentation

- SAM 3.1 (Meta): https://ai.meta.com/sam (segment-anything official site, root for SAM 3.1)
- Qwen3-VL on Hugging Face: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct
- Qwen3-VL-30B-A3B-Instruct on Hugging Face: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
- DINOv2 paper: https://arxiv.org/pdf/2304.07193

### Interpretability and activation steering

- Goodfire R1 interpretability repo: https://github.com/goodfire-ai/r1-interpretability
- Ramp Labs Steer (interpretability playground, April 2, 2026 post): https://labs.ramp.com/
- Ramp Labs Steer announcement on X: https://x.com/RampLabs/status/2039726632886235648
- Steer AI live demo: https://labs.ramp.com/steer-ai
- Turner et al. "Steering Language Models With Activation Engineering" (arXiv 2308.10248): https://arxiv.org/abs/2308.10248
- Panickssery / Rimsky et al. "Steering Llama 2 via Contrastive Activation Addition" (arXiv 2312.06681): https://arxiv.org/abs/2312.06681
- Templeton / Conerly / Marcus et al. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (Anthropic Transformer Circuits Thread, 2024): https://transformer-circuits.pub/2024/scaling-monosemanticity
- Karvonen et al. "Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers" (arXiv 2512.15674): https://arxiv.org/abs/2512.15674
- Chen / Arditi / Sleight / Evans / Lindsey "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (arXiv 2507.21509): https://arxiv.org/abs/2507.21509
- Jayasekara et al. "Do transformers notice their own mistakes? Finding a linear hallucination detector inside LLMs" (Baseten + ANU, May 2025, arXiv 2507.23221): https://www.baseten.co/research/do-transformers-notice-their-own-mistakes/ — published precedent for detect-and-steer governance loops
- ContraTales benchmark: released alongside the linear hallucination detector paper
- steering-vectors Python library: https://github.com/steering-vectors/steering-vectors
- IBM activation-steering library (ICLR 2025): https://github.com/IBM/activation-steering
- Subhadip Mitra "Activation Steering in 2026: A Practitioner's Field Guide": https://subhadipmitra.com/blog/2026/activation-steering-field-guide/
- Modal GPU Memory Snapshots (used by Steer for cold-start mitigation): https://modal.com/blog/gpu-mem-snapshots

### Background and related papers

- Behavior Cloning from Observation: https://arxiv.org/abs/1805.01954
- VPT (Video PreTraining, OpenAI Minecraft): https://arxiv.org/abs/2206.11795
- VideoAgentTrek: https://arxiv.org/abs/2510.19488
- Masked diffusion language models (Sahoo et al.): https://s-sahoo.com/mdlm
- V-JEPA: https://arxiv.org/abs/2404.08471
- Ring Attention (Liu et al., 2024): https://openreview.net/forum?id=WsRHpHH4s0
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- DeepEP: https://github.com/deepseek-ai/DeepEP
- ZeRO / FSDP: https://arxiv.org/abs/2304.11277
- Megatron-LM: https://arxiv.org/abs/1909.08053
- ThunderKittens: https://openreview.net/forum?id=0fJfVOSUra
- ThunderKittens GEMM kernels: https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm
- Flash Attention 4 PR (Colfax / Cursor collaboration): https://github.com/Dao-AILab/flash-attention/pull/2270
- SWE-bench: https://openreview.net/forum?id=VTF8yNQM66
- Terminal-Bench: https://openreview.net/forum?id=a7Qa4CcHak
- METR "Measuring AI ability to complete long tasks": https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
- Cursor "Third era of software": https://cursor.com/blog/third-era
- Cursor MXFP8 kernels blog: https://cursor.com/blog/kernels

### RL-as-a-service and orchestration

- verl (Volcano Engine RL): https://github.com/volcengine/verl
- Ray: https://www.usenix.org/conference/osdi18/presentation/nishihara
- DBOS durable workflows: referenced in user context

### Tooling

- uv (Python packaging): https://github.com/astral-sh/uv
- Weights & Biases: https://wandb.ai
- vLLM (PyTorch original): https://github.com/vllm-project/vllm
- SGLang (PyTorch original): https://github.com/sgl-project/sglang

### Editorial and analyst coverage referenced

- DeepSeek-V4 deep dive (Kingy AI): https://kingy.ai/uncategorized/deepseek-v4-a-deep-dive-into-the-open-weight-frontier-model-rewriting-the-economics-of-million-token-context/
- DeepSeek-V4 review (Andrew Lukyanenko / Medium): https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1
- DeepSeek V4 two-stage post-training analysis (BSWEN): https://docs.bswen.com/blog/2026-04-25-deepseek-v4-two-stage-post-training/
- GLM-5.1 vs frontier comparison (WaveSpeed): https://wavespeed.ai/blog/posts/glm-5-1-vs-claude-gpt-gemini-deepseek-llm-comparison/
- GLM-5.1 review (Renovate QR): https://renovateqr.com/blog/glm-5-1-review-z-ai-coding-benchmark-2026
- Kimi K2.6 complete guide (AImadeTools): https://www.aimadetools.com/blog/kimi-k2-6-complete-guide/
- Kimi K2.6 Agent Swarm tutorial (AImadeTools): https://www.aimadetools.com/blog/kimi-k2-6-agent-swarm-tutorial/
- Kimi K2.6 Agent Swarm guide (Verdent): https://www.verdent.ai/guides/kimi-k2-6-agent-swarm
- Kimi K2.5 Agent Swarm tutorial (DataCamp): https://www.datacamp.com/tutorial/kimi-k2-agent-swarm-guide
- Kimi K2.5 InfoQ writeup: https://www.infoq.com/news/2026/02/kimi-k25-swarm/
- Kimi K2.6 release announcement (MarkTechPost): https://www.marktechpost.com/2026/04/20/moonshot-ai-releases-kimi-k2-6-with-long-horizon-coding-agent-swarm-scaling-to-300-sub-agents-and-4000-coordinated-steps/

### Personal project context

- The user's Phantom thesis (autonomous agents perceiving and modifying enterprise environments within governable constraints) provided as background; no public link.
- Earlier user explorations of Holo3, Goodfire SAE probes for visual triage / behavioral guardrails / output quality, Tailscale Aperture, Absurd workflow orchestration, VOID interaction-aware object removal — provided as user context, no canonical public links collected here.
