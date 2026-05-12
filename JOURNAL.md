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

**`smoke_text` passed end-to-end.** After pinning `dialog @ git+https://github.com/google-deepmind/dialog.git` (PyPI v1.0.0 was missing `Format` enum), re-ran. Container did the ~20-min GCS load + bf16 cast in `@modal.enter`, then `sampler.chat("Hello, who are you?")` returned in 125.59s for 64 tokens. Model identified itself as Gemma 4. **First confirmed end-to-end inference through Gemma 4 26B-A4B-it on Modal H100.** Added `gemma_import_smoke` CPU function (22.75s) to catch this class of dep bug pre-load in future.

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

**`smoke_image` v7 passed on H200 — first multimodal click inference.** After a
seven-fix chain over six attempts, `smoke_image` returned `ok=true` on Gemma 4
26B-A4B-it. Load 612.6s cold from GCS, one click inference 168.62s wall-clock.
Model output: `<|channel>thought\n<channel|><start_function_call>call:click
{x:251,y:102}<end_function_call><turn|>`. The seven fixes (in order
discovered): (1) `dialog @ git` instead of PyPI for the `Format` enum;
(2) un-mask `_safe_probe_error` so the real Gemma 4 image-marker error
surfaced; (3) `<|image|>` marker (Gemma 3's `<start_of_image>` is deprecated);
(4) `images=[image]` list-wrap (Gemma 4 iterates the kwarg); (5)
`Gemma4_26B_A4B(text_only=False)` (the class field defaults True, strips
vision encoder); (6) `ChatSampler(cache_length=2048, max_out_length=256)`
buffer reduction; (7) `GPU_TYPE = "H200"` — H100's 80 GB couldn't absorb
multimodal prefill activation peak under JIT, even after buffer reduction.
H200's 141 GB HBM3e fits comfortably. Full chain documented in design-notes.

**Parser extension (`actions.parse_function_call`):** base Gemma 4 emits bare
comma-separated args (`x:251,y:102`) without the canonical FunctionGemma
`<escape>` tokens. Added `_BARE_ARG_RE` fallback — strict escape-wrapped form
still primary regex, bare form only used if strict yields nothing AND body is
non-empty. Two new tests added, all 12 pass. Phase 1 SFT remains the canonical
escape-wrapped target; this is just a recovery hatch for the un-SFT'd base.

3. **Don't understand yet:** Whether the warm container will actually persist
across separate `modal run` invocations (no edits between) given Modal's
scale-down semantics — the design intent is `scaledown_window=1800`, but it
will be empirical. Also: the model picked (251, 102) for a Submit button at
(478, 68) — distance ~227 px on a 640×480 image. That's WAY off, but at
k=1.0 (no scaling), no SFT, this is the un-improved baseline. The Tzafon
sweep's signal is *whether some k makes this distance shrink*.

4. **What's next:** run `tzafon_sweep` on the warm container — 6 scaling
factors × ≤3 tasks × 3 tiers = up to 54 inferences. ~2 minutes per inference
at the v7 rate ⇒ ~2 hour wall-clock. Outputs JSON-per-(scale,tier,task) with
verification.distance_px, plus a summary plot if time permits.

---

**`tzafon_sweep` v1 first attempt failed cleanly on all 18 inferences with the
same Gemma 3 deprecated-token error.** Cold container init 605.7s (Modal had
scaled the v7 warm container down), then every inference short-circuited in
0.06-0.36s. Root cause: the seven-fix chain's fix #3 (`<|image|>` marker)
was only applied to `smoke_image`'s prompt path — line 659 — but
`tzafon_sweep`'s `_prompt_for` helper at line 756 still emitted Gemma 3's
`<start_of_image>`. The original commit `b58ead5` claimed the fix touched
"both methods" but since that commit had an empty diff, only my session's
roll-up actually wrote the change, and I only inspected the smoke_image
path. Lesson: when consolidating multi-commit fix chains, grep for the OLD
pattern in the whole file before testing, not just the path you're
exercising.

Also seen on first container attempt: `Runner segmentation fault (SIGSEGV),
exit code 139` before Modal retried successfully on second attempt. First
SIGSEGV we've seen on H200; not reproducing on the retry. Filed under "watch
for again" — could be cold-init driver race, GCS stream timing, or
container-specific transient. No action.

Fix landed in modal_app.py line 756. Re-launching the lean sweep now;
container is cold again (the failed v1 run scaled down).

---

**Phase 1A close-out — claim gate "no clear scaling effect; lock k=1.0".**
`tzafon_sweep` v2 ran 18 inferences end-to-end on H200 (cold load 434.8s
after two SIGSEGV retries; sweep 439.8s). JIT cache worked beautifully:
first scale's 3 inferences took 365.3s (compile + run), every subsequent
scale ran 3 inferences in ~15s — 5s/inference, 34× faster than v7. The
dict-spread params rebuild + fresh ChatSampler per scale did NOT trigger
recompilation because shape/dtype unchanged.

Result: 0/3 success at every scale on every tier. Per-task distance_px
across scales (same seed = same image, only pos_emb scaling varies):

  target/seed=1000:   k=1.0:26.9  1.5:53.9  2.0:56.8  3.0:35.4  5.0:56.4  10.0:46.3
  distractors/2000:   k=1.0:49.8  1.5:39.2  2.0:49.0  3.0:49.0  5.0:60.4  10.0:48.7
  button/3000:        k=1.0:49.7  1.5:56.1  2.0:48.7  3.0:115.5 5.0:176.1 10.0:48.2

Reading: target/distractors essentially flat across scales (model picks
attractor positions ~(180,225) and (~145,270) regardless of k). Button
tier shows real y-drift at k=3/k=5 (194 → 312 → 374) but DRIFTS AWAY
from target, snaps back at k=10. So multiplicative pos_emb scaling DOES
modulate Gemma 4 26B-A4B's spatial prior — but not in a Tzafon-helpful
direction. No scaling factor improves accuracy.

Decision per plan: log finding, lock k=1.0, proceed. The 30-40pt lift
Tzafon reported on Qwen3-VL does NOT transfer to Gemma 4 26B-A4B. Several
plausible reasons (any combination): (a) MoE+sliding-window arch responds
differently to entry pos_emb perturbation than Qwen's dense arch; (b)
bf16-cast vision tower has less headroom for value amplification than
fp32; (c) base Gemma 4 has weak native click-grounding (Phase 1 SFT
target); (d) prompt template too curt — model defaults to attractor click
positions. None of these are worth investigating now — the answer to "do
we need to scale pos_emb during inference?" is "no", which is the only
decision Phase 1A needed to gate.

Also seen: two consecutive SIGSEGV (exit 139) on container startup before
third attempt succeeded. Pattern now reproducible across v1 (1 segfault)
and v2 (2 segfaults). Modal's automatic retry masks it but the load
cost is paid each time. Filed as known-flake. If it ever stops retrying
or burns three attempts before success, escalate.

5. **What's next:** Phase 1B — dual loader compatibility gate. Identify
the canonical SFT entrypoint for Phase 1D. Working hypothesis (carries
forward from Phase 1A debug knowledge): `references/gemma` (Flax Linen,
full multimodal) + Tunix PeftTrainer + qwix LoRA. MaxText probe pending
decision below.

---

## 2026-05-12 — Phase 1B pivot: port Gemma 4 vision to Tunix NNX (in fork)

**Decision:** Skip the MaxText half of Phase 1B. Instead, port Gemma 4
vision encoder to Tunix NNX in a personal fork (`sdrshn-nmbr/tunix`,
branch `opjax/gemma4-vision-port`) tracked as a submodule at
`references/tunix`. This collapses the "which SFT loader?" decision into
"can we make Tunix PeftTrainer accept multimodal Gemma 4?" — answered by
doing the work.

Why this beats both the formal Phase 1B (probe MaxText) and the
short-circuit (declare references/gemma canonical and stay Linen-only):
- **Highest JAX-learning density chunk available right now.** Touches
  Flax NNX vs Linen state, factorized pos_emb lookup invariants, RoPE
  coordination across modalities, qwix LoRA composition, soft-embedding
  merge, Orbax checkpoint key mapping — five conceptually rich axes in
  one feature. The "Sudarshan writes function bodies on conceptually
  rich code" pair-contract pegs this as a max-density target.
- **No SFT-loader fork-in-the-road later.** If we port, Phase 1D becomes
  configuration; otherwise it's another design-decision sub-phase.
- **PR pathway preserved.** The fork lives at github.com/sdrshn-nmbr/tunix
  on branch opjax/gemma4-vision-port. If our port matures, offering it
  back to Google via Tunix issue #1380 is just `gh pr create`.

**Mechanical setup landed in this commit:**
1. Forked `google/tunix` → `sdrshn-nmbr/tunix` via `gh repo fork`.
2. Branch `opjax/gemma4-vision-port` pushed to fork.
3. `references/tunix` converted from a read-only mirror clone to a git
   submodule pointing at the fork branch. `.gitmodules` registered at
   opjax root.
4. `.gitignore` adjusted: `references/*` excludes contents broadly, with
   `!references/tunix` as the one tracked exception. The "references is
   read-only" convention is intentionally inverted for this submodule;
   other reference mirrors (gemma, maxtext, cua, composer2, bstn, tm)
   remain ignored.
5. `pyproject.toml`: `google-tunix` source switched to editable path
   `references/tunix` — local changes resolve immediately in `uv sync`.
6. `REMOTE_IMAGE_PACKAGES`: Modal image now installs google-tunix from
   `git+https://github.com/sdrshn-nmbr/tunix.git@opjax/gemma4-vision-port`
   so cloud runs hit our fork's branch, not upstream.

**Sub-task 1 scaffolded in this commit** (port `_layers.py`):
- `references/tunix/tunix/models/gemma4/vision.py`: VisionEntry,
  VisionExit, Standardize NNX skeletons. Helpers (`factorized_posemb`,
  `patchify`) and `Standardize` fully implemented (mechanical
  translations). VisionEntry and VisionExit have TODO bodies with
  ★ Insight blocks at each slot explaining the design tradeoff to
  internalize before writing the body.
- `references/tunix/tests/models/gemma4/vision_test.py`: TDD baseline.
  6 tests pass immediately (helpers + Standardize + VisionEntry init
  shape). 6 tests RED (VisionEntry.__call__, VisionExit pooling) —
  these GREEN when the user writes the bodies.

**Sub-task sequence** (each lands its own commit on the fork branch,
each pushable as its own PR if we go upstream):
1. ✓ Scaffold + TDD baseline (this commit, fork SHA 2b039f44)
2. VisionEntry.__call__ + VisionExit pooling bodies (user implements)
3. Port `_transformer.py` + `_modules.py` + `_norms.py` (vision
   transformer body)
4. Port `_encoder.py` (top-level VisionEncoder wiring)
5. Extend `tunix/models/gemma4/model.py` Gemma4 with vision_encoder
   field + encode_images() + multimodal __call__ path
6. Extend `params_safetensors.py` with vision-encoder torch → NNX key
   mapping for HF safetensors load
7. End-to-end: PeftTrainer.train_step on one synthetic click task with
   qwix LoRA applied. Loss finite, no XLA recompile spam.

**Phase 1C status:** unchanged. Glue-code work (Gemma4Inference →
synthetic.click → Claude repairer → JSONL → HF Hub). Independent of the
port; produces a dataset that feeds whichever Phase 1D path we end up
on. Will be wired in parallel with the port. Sudarshan writes vision
port bodies; Claude wires Phase 1C.

**Out-of-scope:** MaxText probe (the plan-literal other half of Phase
1B). Deferred to the Phase 2 → 3 boundary where MaxText's TPU sharding
patterns actually start paying off. The MaxText decision is no longer
load-bearing for getting a working multimodal SFT pipeline.
