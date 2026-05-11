# Design Notes

Running ledger of in-flight decisions during implementation. Append-only.
For locked decisions and reasoning, see the plan file. This file is for
mid-execution micro-decisions: things discovered while writing the code that
weren't visible at planning time.

Each entry: date, scope, decision, why, alternatives considered, follow-ups.

---

## 2026-05-11 — Day 0 setup

**Decision:** Add `chex==0.1.91`, `jaxtyping==0.3.9`, `orbax-checkpoint==0.11.39`,
`wandb==0.26.1` as exact pins; mirror into `REMOTE_IMAGE_PACKAGES` for Modal image
reproducibility.

**Why:** Plan locks JAX hygiene stack on Day 0. Exact pins (`==`) match the rest of
the project's style — `>=` would let Modal builds drift from local. xprof deferred:
it's distributed as `tensorboard-plugin-profile` / via `pip install xprof-nightly`
inconsistently; install when we actually need a TPU/GPU profile in Phase 3.

**Alternatives considered:** `>=` constraints (rejected: drift between local and
Modal image bakes correctness bugs that only show up on remote). Adding xprof
preemptively (rejected: install when needed; the package landscape is messy and
will look different by Phase 3).

**Follow-ups:** confirm Modal image rebuild succeeds with new pins before any
Phase 1A pos-emb probe. Re-run `cpu_smoke` and `gpu_smoke_cli` end of Day 0.

---

## 2026-05-11 — Phase 1A target selection (HF discovery deep-dive)

**Decision:** Phase 1A target is **`google/gemma-4-E2B-it`** (smallest accessible
multimodal Gemma 4, plan-locked as "smallest accessible"). Phase 1D SFT target
remains `gemma-4-E4B-it` per plan.

**Discovery summary:**
- HF discovery returned 38 Gemma candidates. Gemma 4 nano-family (`gemma-4-E2B`,
  `E4B`, plus `-it` and `-it-assistant` variants) are tagged `any-to-any` (text +
  vision + audio), *not* `image-text-to-text` — discovery filter must include both.
- All Gemma 4 variants are `gated=False` (no manual approval). All Gemma 3 variants
  are `gated=manual`. Gemma 4 wins on access friction.
- E2B-it HF manifest: 10.28 GB total, single `model.safetensors` shard (10.25 GB
  bf16), `processor_config.json` + `chat_template.jinja` present, library=transformers.
- E4B-it HF manifest: 16.03 GB total, otherwise identical structure.

**Two loader paths exist:**

| Path | Storage | Format | Size | Loader |
|---|---|---|---|---|
| **A — GCS Orbax** | `gs://gemma-data/checkpoints/gemma4-e2b-it` | Orbax (OCDBT) | 18.25 GB (mostly f32, bf16-cast on load) | `gemma.gm.ckpts.load_params(path)` |
| **B — HF safetensors** | `huggingface.co/google/gemma-4-E2B-it` | safetensors (PyTorch) | 10.25 GB (bf16) | `transformers.Gemma4ForConditionalGeneration.from_pretrained(...)` |

**GCS bucket validation:** `https://storage.googleapis.com/storage/v1/b/gemma-data/o?prefix=checkpoints/gemma4-e2b-it/` returned 24 objects unauthenticated — bucket is public-read. Orbax checkpoint structure confirmed (`_CHECKPOINT_METADATA`, `_METADATA`, `manifest.ocdbt`, `ocdbt.process_0` ~18 GB single OCDBT blob).

**Recommendation: Path A (GCS Orbax + `references/gemma` Flax NNX).** Reasoning:
- Aligns with plan's JAX-purist goal; no PyTorch→JAX converter to write.
- Loader (`load_params` in `references/gemma/gemma/gm/ckpts/_checkpoint.py`) handles
  NESTED / FLAT / STACKED / KAULDRON layouts, MM-aware (`text_only=True` skips
  vision encoder), bf16-casts on load with explicit f32 retention on MM embedder
  params. Uses `epath.Path('gs://...')` for transparent GCS reads.
- The vision tower's `pos_emb_param` lives at `params['vision_encoder']` (nested
  layout) — exactly the Tzafon scaling target identified in
  `gemma4/vision/_layers.py`.
- "Cost" of installing `gemma` + `kauldron` on Modal is itself JAX learning per the
  plan's "read the parts that teach" principle.

**Why not Path B:** the recipe-mismatch tax framing applies here too — using a
PyTorch loader to validate a JAX-purist learning project's first claim gate would
be a recipe-style mismatch. Faster time-to-signal, but the signal we want is
"does the Tzafon scaling effect reproduce in our JAX stack?" not "does it
reproduce anywhere?"

**Follow-ups:**
- See revised decision below — Phase 1A target upgraded from E2B-it to 26B-A4B-it
  and loader switched to Tunix `params_safetensors` after the user surfaced
  `tunix/models/gemma4/` and the MoE memory math came in favorable on H100 80GB.

---

## 2026-05-11 — REVISED Phase 1A target: upgrade to gemma-4-26B-A4B-it

**Decision:** Phase 1A target upgraded to **`google/gemma-4-26B-A4B-it`**. Loader
switched to **Tunix `tunix/models/gemma4/params_safetensors.py`** (Flax NNX,
direct from HF safetensors). Supersedes both Path A (GCS Orbax via
`references/gemma`) and Path B (HF + PyTorch transformers) from the prior entry.

**Why the upgrade:**
- 26B-A4B is the production-grade Gemma 4 (MoE: 128 experts, top_k=8, expert_dim=704,
  dense shared MLP 2112, 30 layers). E-variants are the nano/edge tier ("E" =
  effective/edge), designed for on-device deployment, not the right base for an
  RL-trained agent that will run on cloud GPU/TPU.
- MoE inference FLOPs = ~4B active params/token (same as E4B-dense), so probe-time
  cost is ~4× E2B but still cheap on H100 (Tzafon sweep ~1 hour).
- Quality tier matches the plan's "research-grade agent" ambition. Doing Phase 1A
  on a model we'd then have to abandon for Phase 1+ training would be wasted
  motion.

**Why Tunix loader instead of `references/gemma`:**
- `tunix/models/gemma4/params_safetensors.py` ships a complete torch-key →
  Flax NNX-key regex mapping for all four variants (E2B/E4B with Per-Layer
  Embeddings, 26B-A4B with MoE, 31B dense). Already mirrored locally at
  `references/tunix/tunix/models/gemma4/`.
- Loads from HF safetensors (smaller than GCS Orbax: 52 GB vs 96 GB for 26B-A4B),
  no precision-conversion roundtrip.
- Tunix-native ⇒ trainers (SFT PeftTrainer, GRPO `grpo_learner.py`, distillation
  `distillation_trainer.py`) work without integration glue.
- Includes `splash_attention_kernel` (Pallas TPU) for Phase 3+ TPU port; `shard_map`
  + `PartitionSpec` sharding pre-wired.
- Avoids the `kauldron` heavy install path that `references/gemma` would force.

**HF manifest for `google/gemma-4-26B-A4B-it`:**
- gated=False, library=transformers, pipeline=image-text-to-text
- Total 51.64 GB; weights 51.61 GB across 2 shards
  (`model-00001-of-00002.safetensors` 49.9 GB + `model-00002-of-00002.safetensors` 1.7 GB)
- `processor_config.json`, `chat_template.jinja`, `tokenizer.json` (32 MB),
  `model.safetensors.index.json` shard manifest, `generation_config.json` present.

**Architecture (from `references/gemma/gemma/gm/nn/gemma4/_gemma4.py:238-301`):**
- num_layers=30, embed_dim=2816, num_heads=16, head_dim=256, num_kv_heads=8,
  num_global_kv_heads=2, vocab=262144
- Attention pattern: 5 LOCAL_SLIDING + 1 GLOBAL, repeating × 5
- sliding_window_size=1024, global_rope_proportion=0.25 (global has shorter RoPE
  to extend effective context), local_rope_proportion=1.0
- MoE: 128 experts, top_k_experts=8, expert_dim=704, dense MLP hidden_dim=2112
- Vision: SigLIP-like, d_model=1152, 27 layers, num_heads=16, ffw_hidden=4304,
  output_length=280, use_bidirectional_attention=True

**H100 80GB memory math (single device):**

Phase 1A inference (Tzafon probe, ~2k context):
- Weights bf16: 52 GB
- KV cache @ 4k: 252 MB (5/6 layers sliding-window capped at 1024 tokens)
- Activations + JAX overhead: ~5 GB
- **Total: ~57 GB, 23 GB headroom ✓**

Phase 1D SFT + LoRA (forward + backward, short sequences):
- Frozen weights bf16: 52 GB
- LoRA adapters + AdamW state: ~250 MB
- Activations: ~10 GB (no recomputation)
- KV + JAX: ~3 GB
- **Total: ~65 GB, 15 GB headroom ✓**

Phase 3 GRPO rollouts (group size 8, 4k context):
- Weights shared across rollouts
- 8 × 252 MB KV cache = 2 GB
- **Fits comfortably ✓**

Phase 3.5 OPSD merge (3 frozen teachers + 1 student forward pass, single H100):
- 4 × 52 GB bf16 = **208 GB ❌ won't fit**
- 4 × 26 GB int8 = 104 GB ❌
- 4 × 13 GB fp4  =  52 GB ✓ (28 GB headroom)
- **FP4 has load-bearing role at Phase 3.5 that the plan didn't see.** Without
  FP4: CPU host offload teachers (Modal supports 256+ GB host RAM), or jump to
  TRC TPU v3-8/v5p-8 with sharded teachers, or descend specialists to E4B.

**128k context analysis (Gemma 4 max window):**
- KV cache @ 128k: only 1.6 GB for 26B-A4B (sliding-window caps local-attn KV; only
  the 5-of-30 global layers carry full-context KV). Vast improvement over
  dense-attention architectures.
- Inference at full 128k still fits with comfortable headroom.

**Quantization landscape in JAX:**
- `references/gemma/gemma/gm/ckpts/_quantization.py` only does QAT *structure*
  conversion (wraps kernels in `_SimulateQuantizedEinsum`) — no actual weight
  quantization. The comment in `_checkpoint.py:277-279` says: "this do not
  quantize the weights, but just refactor the params to the QAT structure."
- For real FP4 in JAX: **`aqt` (Accurate Quantized Training, Google JAX-native)**
  is the canonical library. Supports int4/int8 + MXFP8/NVFP4 on H100 via
  Transformer Engine. Hand-rolled Pallas FP4 kernels are an alternative but
  more work.
- Promote **FP4 from Phase 7 stretch to Phase 3.5 enabler-stretch** in the plan.
  Investigation: profile `aqt` integration ahead of Phase 3.5 implementation;
  measure quality degradation on 26B-A4B-it for vision + text tasks separately
  (vision-tower KV-cache and softmax pathways are quantization-sensitive).

**Storage on Modal:**
- HF safetensors pull: 52 GB to `opjax-hf-cache-v2`. One-time download ~20-30 min
  at typical HF Hub speeds. Volume is configured `create_if_missing=True`.
- HF cache layout: `/mnt/hf-cache/hub/models--google--gemma-4-26B-A4B-it/...`
- After first cache hit, all Modal containers see the model immediately via the
  shared volume.

**Phase 1A iteration speed estimate:**
- Per forward pass on H100: ~4B active FLOPs/token. For a 2k-token click prompt:
  ~800 ms inference (rough estimate based on H100 ~989 TFLOPs bf16 throughput,
  ~10% utilization realistic for single-batch inference).
- Tzafon sweep: 6 scaling factors × 50 click tasks per tier × 3 tiers = 900 forwards
  × 0.8s ≈ 12 minutes pure compute. Plus model load (~30s) + warmup. Under an hour.

**Follow-ups:**
- Add `tunix` package to `pyproject.toml` and `REMOTE_IMAGE_PACKAGES`. Investigate
  whether Tunix can be installed from PyPI or must be installed from
  `references/tunix` (editable). Probably needs `flax==<latest nnx>`, `optax`,
  `grain`, `qwix` as transitive deps.
- Add a `gemma4_load_probe` Modal entrypoint: load `google/gemma-4-26B-A4B-it` via
  `tunix.models.gemma4.params_safetensors.load_params`, report param tree shape
  summary + presence of vision_encoder.pos_emb with its concrete shape.
- Stage HF safetensors to `opjax-hf-cache-v2` via `huggingface_hub.snapshot_download`
  on first run; subsequent loads hit cache.
- Update plan: FP4/AQT investigation added to Phase 3.5 pre-work; CPU host
  offloading kept as primary fallback; TRC TPU as secondary fallback.
- Investigate whether Tunix `gemma4/model.py` works on GPU (it uses TPU-Pallas
  `splash_attention_kernel`; need to confirm GPU fallback path exists or wire one).
