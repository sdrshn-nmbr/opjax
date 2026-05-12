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

**Follow-ups:** see "Phase 1A loader pivot" entry below — the Tunix gemma4 module
turned out to be text-only, so we switched to `references/gemma` (DeepMind Linen
canonical impl) for the probe.

---

## 2026-05-11 — Phase 1A loader pivot: Tunix gemma4 is text-only

**Discovery:** Tunix's `tunix/models/gemma4/` ships **without a vision encoder**.
The `model.py` references `vision_proj` / `vision_soft_emb_norm_weight` only in
`ShardingConfig` (placeholder for future intent). `params_safetensors.py` has
zero vision mappings. `sampling_example.ipynb` is text-only. By contrast,
`tunix/models/gemma3/` is fully multimodal — has `vision.py` with NNX SigLIP
encoder, `merge_embeddings.py`, and `vision_encoder = vision.SigLiP(...)`
integration into `Gemma3`.

**Recon agent finding (issue #1380, 2026-04-09):** maintainer `s-noghabi` stated
2026-04-28 that vision support for Gemma 4 is "on our roadmap for this quarter."
Internal Google work (copybara) is likely underway and invisible to us until a
PR drops. External-PR throughput is dire (Qwen3-VL PR #1177 stalled 10+ weeks),
making upstream contribution lower-EV than originally framed.

**Loader switched: use `gemma` (from `github.com/google-deepmind/gemma`, Flax
Linen) for Phase 1A.** Tunix kept installed for trainers (SFT PeftTrainer, GRPO,
distillation) — it doesn't need to host the model architecture, just consume a
Flax model. Two model frameworks coexist for now; reconcile when/if Tunix ships
gemma4 vision.

**Action on Tunix vision support:** **Do NOT write the port.** Comment on issue
#1380 with a +1 / use case, offer help with a graceful out so "no thanks we've
got it" is the easy answer. Monitor `copybara-service[bot]` PRs touching
`tunix/models/gemma4/vision.py`. 2-3 week clock before reconsidering.

---

## 2026-05-11 — Probe failure analysis: GCS Orbax stores f32, loader doesn't cast

**Failure:** First `gemma4_load_probe` attempt (GPU H100, called
`gemma.gm.ckpts.load_params`) OOM'd at ~6 min in: "Allocator (GPU_0_bfc) ran out
of memory trying to allocate 1.89 GiB."

**Root cause:** I had documented earlier (in this same file) that the GCS Orbax
loader "bf16-casts on restore." That is **wrong** — the bf16 cast in
`_checkpoint.py:295-321` only applies to the multimodal embedder f32 retention
hack at the very end. The bulk of the param tree restores at whatever dtype
the checkpoint stored, which for GCS Orbax is **float32** (confirmed: all 619
leaves in `gemma4-26b-a4b-it` are f32 on disk).

Numerical confirmation: GCS bucket 95.7 GB ÷ 25.8B params = 3.71 bytes/param.
That's f32 (4 bytes) for most leaves with a thin layer of bf16 metadata —
matches the f32 leaf count exactly.

**Mitigation:** added `gemma4_metadata_probe` (CPU, manifest-only, no weight
load) that reads the schema in ~15s. For the eventual sweep, `gemma4_load_probe`
v2 was patched to bf16-cast `meta.item_metadata.tree` before `ckpt.restore` —
casts happen *before* materialization, so f32 never lands on device.

**Lesson:** separate metadata probes from weight probes. Schema discovery is
cheap, never needs HBM; weight loading is expensive, needs careful precision
management. Conflating them produces the f32-OOM failure mode.

---

## 2026-05-11 — Phase 1A schema lock: pos_emb located

**Tzafon scaling target confirmed:**
- Path: `vision_encoder.entry.pos_emb`
- Shape: `[10240, 2, 1152]`, dtype float32 on disk
- Param count: 23,592,960 (~24M) — tiny fraction of the 25.8B total

**Shape interpretation (factorized pos_emb):** The `(G=10240, axis_pair=2,
d_model=1152)` decomposes spatial positions: `pos_emb[g, 0, :]` is the
embedding for the x-coordinate at grid position g; `pos_emb[g, 1, :]` is the
y-coordinate embedding. Lookup is additive (`lookup_x + lookup_y`), and bilinear
interpolation handles input grids smaller than G. The `assert
pos_emb_shape_yx[-1] == 2` in `gemma/gm/nn/gemma4/vision/_layers.py:53`
documents this contract.

**Why factorized:** dramatic param savings. Full 2D pos_emb at G=10240 would be
`10240^2 × 1152 ≈ 121B params`. Factorized is `10240 × 2 × 1152 ≈ 24M`. 5000x
smaller. Tzafon's claim is that *multiplicative scaling of this 24M-param tensor*
shifts the vision tower's spatial-prior weight versus content-content attention,
yielding the 30-40pt click-accuracy lift on Qwen3-VL — we now have the
parameter to test that on Gemma 4.

**Follow-ups:**
- Next probe step: load weights at bf16 via `gemma4_load_probe` v2; confirm
  bf16 cast resolves the OOM and the resulting param tree exposes
  `vision_encoder.entry.pos_emb` with the expected shape on device.
- After that: wire a single forward-pass smoke test (one click task → model
  emits something) before any sweep.
- Then: Tzafon sweep — multiplicatively scale `params['vision_encoder']['entry']
  ['pos_emb']` by k ∈ {1.0, 1.5, 2, 3, 5, 10}, measure click accuracy across
  the 3 synthetic tiers (target / distractors / button).

---

## 2026-05-11 — End-to-end multimodal click inference works (H200, 7-fix chain)

**Result:** `smoke_image` v7 returned `ok=true` on H200. First confirmed
multimodal click inference through Gemma 4 26B-A4B-it on Modal.

**The seven-fix chain (in order discovered):**

1. **`dialog` pinned from git source.** PyPI `dialog==1.0.0` is missing the
   `Format` enum that `gemma.gm.text.ChatSampler` imports. Symptom:
   `AttributeError: module 'dialog' has no attribute 'Format'`. Fix: `uv add
   "dialog @ git+https://github.com/google-deepmind/dialog.git"` and add a
   matching pin to `REMOTE_IMAGE_PACKAGES`. Also added `gemma_import_smoke`
   (22.75s CPU dep check) so future Python-dep mismatches surface before any
   GPU spin-up.

2. **`_safe_probe_error` un-masked.** My HF-discovery helper had been
   rewriting ANY exception containing the substring "token" as a fake HF auth
   failure. That swallowed the real Gemma 4 image-marker error on smoke_image
   v1 and routed me on a 20-minute false-trail. Narrowed the rewrite to
   `hf_gemma4_discovery` context with explicit "401" check; preserve raw
   message + traceback otherwise.

3. **`<|image|>` marker (not `<start_of_image>`).** Gemma 3's
   `<start_of_image>` token is deprecated in Gemma 4's chat template. Symptom:
   `ValueError: token <start_of_image> is deprecated; use <|image|> for Gemma
   4`. Fix: change one string in `smoke_image`.

4. **`images=[image]` list-wrap.** Gemma 4's `_preprocess_images` iterates
   over the kwarg, so a bare PIL `Image` raises `TypeError: 'Image' object is
   not iterable`. Fix: pass `images=[image]`.

5. **`Gemma4_26B_A4B(text_only=False)`.** The class field
   `_Gemma4Base.text_only: bool = True` defaults the model to text-only
   construction (no vision encoder built at all). Symptom: `AssertionError:
   vision_encoder is None` deep inside the forward. Fix: explicit
   `text_only=False` at construction.

6. **`ChatSampler(cache_length=2048, max_out_length=256)`.** Default
   `cache_length=4096, max_out_length=2048` produced a ~12 GiB allocation at
   decode on H100. Reducing both helped — but not enough; v6 still OOM'd on
   H100 with the same 12.29 GiB allocator failure. The buffer reduction is
   load-bearing for headroom but not the sole win.

7. **`GPU_TYPE = "H200"`.** Multimodal prefill on Gemma 4 26B-A4B with vision
   tower + global-attn KV creates an activation peak the H100's 80 GB cannot
   absorb under JIT. H200 has 141 GB HBM3e — same architecture (Hopper), no
   code changes required beyond the type string. The container loaded weights
   (612.6s cold from GCS) and ran one inference end-to-end (168.62s including
   prefill JIT).

**Model output observation (drives parser change below):**
```
<|channel>thought
<channel|><start_function_call>call:click{x:251,y:102}<end_function_call><turn|>
```

Two notable things:
- Model natively emits a `<start_function_call>...<end_function_call>` wrapper
  without any SFT exposure. Pretraining attractor. Encouraging.
- It uses **bare comma-separated args** (`x:251,y:102`), not the canonical
  FunctionGemma escape-wrapped form (`x:<escape>251<escape>,y:<escape>102
  <escape>`). The `<escape>` token is FunctionGemma-specific; base Gemma 4
  doesn't know to emit it.
- Reasoning channel (`<|channel>thought ... <channel|>`) and turn boundary
  (`<turn|>`) are surfaced verbatim by ChatSampler. Production would post-process
  these; for the probe they don't matter.

**Parser extension:** added `_BARE_ARG_RE` fallback to `actions.parse_function_call`.
Strict escape-wrapped form remains the primary regex (canonical, the Phase 1
SFT target); if it yields no arguments AND the body isn't empty, try bare-form
parsing. Two new tests pin the v7 output literal and a whitespace-tolerant
bare-form case. All 12 tests green.

**Why fallback rather than permissive:** bare-form parsing on values with
embedded commas (e.g., `type{text:"hello, world"}`) is ambiguous; escape
tokens disambiguate. Keeping escape-form privileged means Phase 1 SFT
targets the unambiguous grammar even though the un-SFT'd base emits a
simpler one. This matches the plan's framing of Phase 2 reverse-distillation
as "cleanup of format discipline" rather than first-time exposure.

**Click accuracy at k=1.0 (no scaling):** target_xy=(478, 68), model predicted
(251, 102). Distance ~227 pixels, target radius 29 — miss. Single task, no
scaling, no SFT — the prior is poor as expected. The tzafon_sweep is the
next call.

**Follow-ups:**
- Run `tzafon_sweep` on the warm H200 container (1.0, 1.5, 2, 3, 5, 10 ×
  6 sample tasks × 3 tiers = 108 inferences). The container should stay warm
  per `scaledown_window=1800` *if no code edits happen between runs*.
- Don't bake any clean-up of `<|channel>thought` / `<turn|>` into the probe —
  let those land in JOURNAL as raw, deal with them at Phase 1D when we're
  shaping SFT data.
