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

**HF discovery deep-dive (revised after MoE memory math):**
- **Phase 1A target upgraded: `google/gemma-4-26B-A4B-it`** (replacing initial E2B-it pick). 26B-A4B MoE has ~4B active params/tok inference cost (same as E4B-dense) but 26B-tier quality. User's instinct to skip the nano-tier variants was correct.
- **Loader switched to Tunix `tunix/models/gemma4/params_safetensors.py`** (Flax NNX, direct from HF safetensors). Supersedes both prior options: faster than GCS Orbax (52 GB vs 96 GB pull), all-JAX (vs HF transformers PyTorch), no kauldron heavy install, Tunix-native ⇒ trainers work without glue.
- **H100 80GB inference budget: 26B-A4B at bf16 = 52 GB weights, 28 GB headroom. KV @ 128k = 1.6 GB (sliding-window caps local-attn KV).**
- **FP4 promoted from Phase 7 stretch to Phase 3.5 enabler:** 4-model OPSD merge at bf16 = 208 GB (won't fit single H100); at fp4 = 52 GB (fits with 28 GB headroom). Real integration needs `aqt` library (Google's JAX-native quantization). Fallbacks: CPU host offload teachers, TRC TPU.
- Architecture confirmed: 30 layers, embed=2816, heads=16, kv_heads=8 (global=2), head_dim=256, MoE 128 experts top_k=8 expert_dim=704, sliding_window=1024, 5L+1G attention pattern.
- All Gemma 4 variants ungated; HF safetensors stored bf16; GCS Orbax stored mostly f32.
- Decision and full rationale in `docs/design-notes.md`.
