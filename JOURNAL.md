# JOURNAL

Daily three-line minimum:
1. What shipped.
2. What broke.
3. What isn't understood.

During distillation-flavored phases (Phase 2, Phase 3.5, every cross-stage boundary):
4. Gini coefficient of per-token loss contributions on the latest step. Flag if >1.5x running median.

During Phase 3:
5. Per-specialist eval pass@1 on held-out tasks (or "no eval ran today").

---

## 2026-05-11 — Day 0

1. **Shipped:** Phase 0 floor confirmed (10 tests green, MaxText mirror present). Added `chex`, `jaxtyping`, `orbax-checkpoint`, `wandb` as exact pins to `pyproject.toml` and `REMOTE_IMAGE_PACKAGES`. Created `docs/design-notes.md` and this file.
2. **Broke:** Nothing.
3. **Don't understand yet:** What `cap_tokens` value will actually exercise the self-summary path during Phase 3 rollouts — won't know until specialists are training. Filed under Open Questions in the plan.

**Validation:** Modal `cpu_smoke` green (image rebuilt with new pins in 2.35s, cached layers held). `gpu_smoke_cli` green on H100 80GB HBM3, JAX 0.10.0, cuda:0. Both volumes (v2) mounted, all three secrets present. New deps did not break the image. App: ap-9QfjwhL80ixdnZvBa4EXw7.

**Probe execution:**
- First attempt (`gemma4_load_probe` v1, GPU H100, full `gemma.gm.ckpts.load_params`) **OOM'd at ~6 min**: Allocator ran out of memory trying to allocate 1.89 GiB. Root cause: GCS Orbax stores f32 (95.7 GB total / 25.8B params ≈ 3.7 bytes/param). The loader's "bf16 cast" comment is aspirational — actually only applies to MM embedder f32 retention hack at the end. f32 weights don't fit on H100 80GB.
- Pivoted to `gemma4_metadata_probe` (CPU, Orbax manifest only, no weight load). **14.8s end-to-end.** No GPU needed for schema discovery.
- Patched `gemma4_load_probe` v2 with explicit bf16 cast of `metadata.item_metadata.tree` before `ckpt.restore` — cast happens BEFORE materialization, so f32 never lands on device. Should resolve OOM; deferred until needed (the sweep) since metadata probe is enough for now.

**Pre-stage abandoned (premature optimization).** Started `gemma4_prestage_to_volume` to mirror 96 GB GCS → `/mnt/hf-cache/orbax/gemma4-26b-a4b-it`, expecting subsequent loads to be ~3-5 min instead of ~20 min. Reality: urllib serial downloads at ~1 GB/min vs load_probe's tensorstore-parallel streaming at ~5 GB/min. After 53 min, 56% done (18/32 chunks). Killed and removed partial checkpoint. Decision: rely on `@modal.enter` keeping the model resident in HBM across method calls — load cost amortized across the entire Tzafon sweep. If we later want true fast reloads for many independent container starts, rewrite the stager to use `concurrent.futures.ThreadPoolExecutor` or `tensorstore.copy()`.

**`gemma4_load_probe` v2 succeeded:** bf16-cast restore loaded 51.6 GB resident on H100 cuda:0 in 19.79 min (first-time GCS stream of 96 GB). `vision_encoder.entry.pos_emb` confirmed at `bfloat16` `[10240, 2, 1152]` on device — bf16 cast happens during materialization, f32 never lands in HBM. Plan-anticipated risk ("MaxText 26B doesn't fit on H100") mitigated without descending to fallbacks.

**Probe results (gemma-4-26B-A4B-it):**
- Total params: 25,806,083,662 (matches "26B"). bf16 size: 51.6 GB. All 619 leaves f32 on disk.
- Top-level keys: `embedder`, `final_norm`, `layer_0..layer_29`, `vision_encoder`. Vision subtree: `vision_encoder.{entry,standardize,transformer}`.
- **Tzafon scaling target locked: `vision_encoder.entry.pos_emb`, shape `[10240, 2, 1152]`, dtype f32, 23.6M params.**
- Shape is factorized (`G=10240`, axis_pair=2, `d_model=1152`). `pos_emb[g, 0, :]` = x-coord embedding, `pos_emb[g, 1, :]` = y-coord embedding, additive lookup, bilinear interp for grids smaller than G.
- Vision encoder d_model=1152 matches the `_gemma4.py:287` config for 26B-A4B vision tower.

**HF discovery deep-dive (revised after MoE memory math):**
- **Phase 1A target upgraded: `google/gemma-4-26B-A4B-it`** (replacing initial E2B-it pick). 26B-A4B MoE has ~4B active params/tok inference cost (same as E4B-dense) but 26B-tier quality. User's instinct to skip the nano-tier variants was correct.
- **Loader switched to Tunix `tunix/models/gemma4/params_safetensors.py`** (Flax NNX, direct from HF safetensors). Supersedes both prior options: faster than GCS Orbax (52 GB vs 96 GB pull), all-JAX (vs HF transformers PyTorch), no kauldron heavy install, Tunix-native ⇒ trainers work without glue.
- **H100 80GB inference budget: 26B-A4B at bf16 = 52 GB weights, 28 GB headroom. KV @ 128k = 1.6 GB (sliding-window caps local-attn KV).**
- **FP4 promoted from Phase 7 stretch to Phase 3.5 enabler:** 4-model OPSD merge at bf16 = 208 GB (won't fit single H100); at fp4 = 52 GB (fits with 28 GB headroom). Real integration needs `aqt` library (Google's JAX-native quantization). Fallbacks: CPU host offload teachers, TRC TPU.
- Architecture confirmed: 30 layers, embed=2816, heads=16, kv_heads=8 (global=2), head_dim=256, MoE 128 experts top_k=8 expert_dim=704, sliding_window=1024, 5L+1G attention pattern.
- All Gemma 4 variants ungated; HF safetensors stored bf16; GCS Orbax stored mostly f32.
- Decision and full rationale in `docs/design-notes.md`.
