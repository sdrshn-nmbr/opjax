# Training backend migration

This document maps Palinkle's completed Tinker experiments to the pinned Miles
and SGLang source trees. It is a migration contract, not evidence that the GPU
backend has reproduced a result yet.

## Decision

Palinkle remains the source of truth for task construction, split isolation,
agent workspaces, trajectories, rewards, remote TPU verification, profiling,
and frozen evaluation. The training runtime becomes replaceable.

- **Miles is the only active training runtime.** The pinned source has
  native Inkling and Inkling Small model code, LoRA, rendering, Megatron
  training, SGLang rollout, GRPO, and OPD.
- **SGLang is the only active rollout runtime.** The `sglang-miles` branch is
  the serving half of the Miles Inkling contract: it renders Inkling messages,
  serves the base and LoRA adapters, returns rollout log probabilities and MoE
  routed-expert IDs, and receives updated adapter tensors.
- **Palinkle keeps its current harness and verifier.** Migrating the trainer
  does not authorize replacing the isolated Git workspace, patch snapshots,
  remote TPU verifier, or evidence schemas.

PRIME-RL and other Prime Intellect products are deferred. Their OPD/OPSD
implementations may be read later, but they are not dependencies, submodules,
or execution targets for this migration.

## Pinned inspection surfaces

| Surface | Revision | Relevant contract |
|---|---|---|
| Tinker SDK | `0.24.0` | Managed LoRA client, forward/backward, optimizer, sampler materialization, checkpoint state |
| Tinker Cookbook | `0.5.3` | Inkling tokenizer/renderer, supervised datum construction, rollout and RL recipes |
| Miles | `b1860dd264e17c96d5d92da96c957d88cfd3a1f8` | Inkling Small LoRA, SGLang rollout, Megatron trainer, GRPO, OPD |
| SGLang `sglang-miles` | `cb05a44f35a7c9e27e46d74112cc841ca674ef43` | Inkling model, renderer, LoRA serving, routed-expert capture, dynamic adapter loading |

Run the contract audit after checkout, dependency changes, or submodule
updates:

```bash
git submodule update --init --recursive
uv run --no-default-groups --group tinker python scripts/audit_training_backends.py
```

The audit uses live Python reflection for Tinker and AST inspection for the
pinned source trees. It fails on Tinker version drift, missing git revisions,
missing files, or renamed critical symbols.

## Existing Tinker contract

| Palinkle stage | Current Tinker mechanism | Provider-neutral meaning |
|---|---|---|
| G4 SFT | Cookbook renderer and `conversation_to_datum`; rank-64 LoRA; cross-entropy; Adam | Render messages exactly, mask only intended assistant tokens, apply one optimizer update per frozen batch |
| G5 DAPT | Raw source tokenization; deterministic lane-aware packing; next-token cross-entropy; DAPT state continued through identical G4.2 SFT | Preserve token order, boundaries, EOS policy, loss mask, lane weights, optimizer state, and parent checkpoint identity |
| G6 GRPO | Materialize sampler weights; collect bounded agent trajectories; TPU-verify patches; construct behavior-logprob, token-mask, and advantage tensors; importance-sampling loss | Sample from the exact policy checkpoint, retain token-level behavior probabilities, assign group-relative reward only to sampled response tokens, then update the same policy |
| Evaluation | Separate immutable task package and remote TPU verifier | Keep evaluation outside the training provider and never return hidden benchmark feedback during a rollout |
| Evidence | Preparation hash, config hash, row IDs, token counts, per-step events, checkpoint identity, sampler identity, TPU artifacts | Every backend must emit enough data to reconstruct the update and attribute failures |

The Tinker-specific code remains in `src/opjax/pallas/training.py`,
`g5_training.py`, `g6_rollout.py`, and `g6_training.py`. These files define the
behavior to reproduce; they are not the abstraction boundary for a new
backend.

## Miles translation

| Palinkle object or operation | Miles surface | Required adaptation |
|---|---|---|
| Inkling Small model and rank-64 LoRA | `miles_plugins/models/inkling/model.py`, `lora.py`, and `scripts/run_inkling.py` | Convert the official checkpoint, set rank and alpha explicitly, and prove which linear modules correspond to Tinker's attention, MLP, and unembedding targets |
| `tml_v0` rendered messages | `render_inkling_messages_to_ids` | Compare token IDs and supervised masks for a frozen corpus; semantic similarity is insufficient |
| SFT `Datum` | Miles `Sample` plus `sft_loss_function` and SFT rollout | Map prompt/response tokens and loss mask without re-rendering or truncation |
| Packed DAPT sequence | Miles `Sample` with a full next-token loss mask | Add a raw-token data source that bypasses chat rendering and preserves G5 packing exactly |
| Tinker sampler materialization | `RolloutManager` and SGLang rollout | Record the actor version served by every rollout and reject stale weight versions outside the frozen policy-lag rule |
| G6 trainable turn | Miles `Sample` | Map tokens, response length, reward, loss mask, rollout log probabilities, and metadata one-for-one |
| GRPO advantage and update | `compute_advantages` and `policy_loss_function` | Freeze group construction, normalization, clipping, KL, and token aggregation before comparison |
| Gate 7 OPD | `on_policy_distillation.py` and OPD loss | Add the teacher endpoint, teacher-token log probabilities, top-k policy, and reverse-KL settings to the run manifest |
| Checkpoint and sampler export | Megatron checkpoint plus Inkling LoRA export | Hash both training state and served adapter; prove a reload produces identical logits on canary prompts |

The pinned Miles script states that full Inkling Small and its LoRA mode were
validated upstream on 32 H200 GPUs. This is an upstream capability claim, not
a Palinkle result. The four-layer CI checkpoint is also not proof that the
official full checkpoint converts or trains correctly in our environment.

## SGLang translation

| Palinkle object or operation | SGLang surface | Required adaptation |
|---|---|---|
| `tml_v0` generation prompt | `render_inkling_messages` and `InklingTokenizer` | Compare exact token IDs, reasoning-effort framing, tool-call encoding, assistant prefix, and stop behavior |
| Inkling Small base policy | `InklingForConditionalGeneration` | Verify checkpoint identity and base logits before enabling an adapter |
| Rank-64 policy adapter | `LoRAAdapter`, `LoRAManager`, and Inkling-specific LoRA layers | Load the exact Miles-exported names and shapes; reject partial adapter loads |
| G6 generation request | `GenerateReqInput` | Request token IDs, log probabilities, routed experts, and the exact adapter version for every sampled turn |
| MoE routing replay | routed-expert capturer and Miles `rollout_routed_experts` | Validate row count against the rendered and media-expanded sequence; retain the complete top-k expert trace |
| Policy update | Miles dynamic adapter loading through the SGLang engine | Pause generation, switch one complete adapter version, flush affected caches, and prove the served version before resuming |
| Rollout evidence | token output and response metadata | Preserve prompt tokens, response tokens, stop reason, behavior log probabilities, route trace, adapter version, and server configuration |

The joint [SGLang and Miles Inkling report](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support/)
explains why ordinary rollout parity is insufficient for this MoE. Their
contract adds training-side kernels aligned with SGLang arithmetic, Rollout
Routing Replay for selected expert IDs, and adapter-only synchronization. The
reported train-rollout KL near `1e-3` is a useful diagnostic target, not a pass
condition copied into Palinkle without measurement.

## Conformance gates

No result is compared with Tinker until the applicable lower gate passes.

1. **Checkout audit:** both Miles and SGLang gitlinks resolve to the revisions
   above; the source-contract audit passes with clean submodules.
2. **Renderer parity:** frozen prompts produce identical token IDs, stop rules,
   assistant loss masks, and truncation decisions.
3. **Forward parity:** the converted base checkpoint produces sufficiently
   close logits and next-token rankings on frozen short and long canaries.
4. **SFT parity:** one rank-64 LoRA update on one frozen batch matches trainable
   parameter coverage, token-weighted loss, optimizer settings, and direction
   of logit change.
5. **DAPT parity:** the frozen G5 packs conserve tokens and masks; one update
   reduces held-out DAPT NLL without chat rendering.
6. **Rollout parity:** the same checkpoint and seed preserve task visibility,
   turn accounting, action parsing, patch hashes, behavior log probabilities,
   and TPU reward classification.
7. **GRPO reproduction:** rerun the G6 S0 lane and its frozen 16-task
   evaluation before changing the algorithm.
8. **Distillation:** run Miles OPD from the reproduced S0 state. OPSD and RMSD
   remain separate future algorithm ports rather than dependencies of this
   migration.

Each gate must produce a machine-readable manifest that identifies the source
revisions, environment, model conversion, renderer, tokenizer, data release,
optimizer, rollout policy, verifier, and checkpoint hashes.

## Known gaps

- Miles has OPD but no native OPSD symbol at this revision.
- Neither Miles nor SGLang exposes RMSD as a named algorithm.
- Miles' Inkling LoRA target set is not yet proven equivalent to Tinker's
  `train_mlp=True`, `train_attn=True`, and `train_unembed=True` contract.
- Tinker's managed checkpoint and sampler identities do not directly map to a
  Megatron checkpoint plus SGLang adapter; reload and logit parity are required.
- SGLang's routed-expert trace must be proven aligned with Miles after token
  rendering, packing, and any media expansion.
- The official full Inkling Small checkpoint conversion has not been run in
  this repository.

Gate 7 is therefore paused at backend conformance. The next executable probe
is Miles renderer and base-logit parity, followed by a one-batch SFT canary.
This changes the runtime, not the task distribution, reward, or evaluation.
