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
- Add `gemma` (from `references/gemma`) + transitive deps (`kauldron`,
  `etils[edc,enp,epath,epy,etree]`, `einops`, `sentencepiece`, `dialog`) to
  `pyproject.toml` and `REMOTE_IMAGE_PACKAGES`. Heavy install; Modal image rebuild
  will be slow the first time.
- Add a `gemma4_gcs_probe` Modal entrypoint: imports `gemma`, calls
  `load_params('gs://gemma-data/checkpoints/gemma4-e2b-it', text_only=False)`
  with no params (lets Orbax create the shape-dtype-struct tree), reports param
  tree shape summary + presence of `vision_encoder.pos_emb`. No weights pre-cached
  to volume yet — see if direct GCS read is fast enough.
- If direct GCS read is slow / unreliable, stage to `opjax-hf-cache-v2` via
  `gsutil -m cp -r` and load from local path.
- Pre-Phase-1A test: confirm `params['vision_encoder']` contains a key path ending
  in `pos_emb` with shape `(H_grid, W_grid, d_model)` so the sweep can address it
  unambiguously.
