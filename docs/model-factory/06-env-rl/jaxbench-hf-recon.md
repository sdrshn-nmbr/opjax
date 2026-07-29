# JAXBench + Hugging Face recon — JAX / Pallas kernel training fuel

**Date:** 2026-07-23  
**Scope:** Model Factory Stage 6+ — find HF (and JAXBench GitHub) resources to improve Inkling on **JAX** and **Pallas** kernel writing. No training performed.

**Auth:** `hf auth whoami` → `user=sdrshn-nmbr` (orgs: Purdue, Conway-AI, humanitys-last-hackathon). No auth gaps for public datasets.

---

## Executive summary

| Source | On HF? | Pallas/JAX? | Volume | Verifiable tests? | Best use |
|--------|--------|-------------|--------|-------------------|----------|
| **JAXBench** (GitHub) | **No** | Yes (8 gold Pallas, 50 JAX baselines) | 50 tasks | Yes (harness) | **Sealed eval** + small gold SFT |
| **EvanOLeary/pallasbench-unified** | Yes | Yes | ~1,100 rows | Correctness flags | **#1 SFT fuel** |
| **EvanOLeary/pallasbench-robust-gpu-a100** | Yes | Yes | 45 rows | Yes | Gold Pallas pairs + profiling |
| **ScalingIntelligence/KernelBench** | Yes | No (PyTorch refs) | 270 problems | Via harness (CUDA) | Translation / prompt volume |
| **SakanaAI/AI-CUDA-Engineer-Archive** | Yes | No (CUDA) | ~30.6k kernels | Archive metadata | CUDA→Pallas teacher / distillation |
| **allenanie/kernelbench_with_prompts** | Yes | No | ~250 L1–3 | Harness | Prompt templates for kernel gen |
| **BonnieWang/KernelBenchX** | Yes | No (PyTorch + tests) | 176 tasks | `test_code` column | Triton-adjacent eval patterns |
| **Infatoshi/kernelbench-*-traces** | Yes | No (CUDA/Triton agents) | n&lt;1K JSONL files | Roofline grading | Defer for Pallas (wrong DSL) |

**Bottom line:** There is **no JAXBench dataset on Hugging Face**. The only HF datasets with real **Pallas kernel code** are **EvanOLeary/pallasbench-***. For TPU-targeted sealed eval, clone **JAXBench from GitHub**. For SFT volume, combine **pallasbench-unified** (primary) with a **rendered SudarshanBench-Pallas** family from JAXBench GitHub + harness.

---

## JAXBench (GitHub — primary reference)

**Repo:** [AI-Hypercomputer/accelerator-agents/JAXBench](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/JAXBench)  
**License:** Apache-2.0 (parent `accelerator-agents` repo)  
**HF presence:** **None** — API/MCP search for `JAXBench`, `jaxbench`, `accelerator agents` returns zero dataset/model/space hits.

### What it is

- 50 curated **JAX/TPU kernel workloads** with a production eval harness (`python -m JAXBench evaluate|run|list`).
- **17 priority kernels** (`1p`–`17p`): production ops from Llama-3.1, DeepSeek-V3, Mixtral, Mamba-2, RetNet, AlphaFold2, etc.
- **33 KernelBench L2 workloads** (`18k`–`50k`): fused matmul/conv patterns adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench).
- **8 hand-optimized Pallas references** (`optimized.py`): Flash Attention, GQA, MLA, Sparse Attention, Paged/Ragged Paged Attention, Megablox GMM, GEMM.

### Task shape

Each workload directory has:

```python
CONFIG = {...}                              # metadata
def create_inputs(dtype=jnp.bfloat16): ...  # deterministic inputs (seed=42)
def workload(*inputs): ...                  # baseline JAX implementation
def get_flops(): ...                        # optional
```

Agent/kernel submission exports only `workload(*inputs)`; harness checks correctness (`np.allclose`, atol/rtol=1e-2), benchmarks via `jax.profiler.trace()`, reports speedup vs baseline XLA and Pallas reference.

### MaxKernel adapted dataset (same repo, not on HF)

`MaxKernel/evaluation/jaxbench_adapted_dataset/` — **50 tasks**, each with:

- `kernel_task.yaml` — `task_id`, `description`, `input_gen_code`
- `reference.py` — `get_inputs()` + `computation()` JAX reference

Useful for **prompt rendering** and MaxKernel-style agent eval; still GitHub-only.

### Train fuel vs sealed eval

| Role | Suitability | Notes |
|------|-------------|-------|
| **Sealed eval** | **Excellent** | Fixed 50 tasks, deterministic inputs, TPU correctness harness. Treat like SudarshanBench sealed — never train on held-out prompt variants drawn from same task IDs. |
| **SFT fuel** | **Small but high quality** | 50 baseline references + 8 Pallas optimized = **&lt;60 gold pairs**. Can synthesize many prompt→kernel rows, but raw volume is tiny. |
| **RL fuel** | **Strong verifier** | `evaluate --json` gives correct/speedup signal; needs TPU VM (v5e/v6e). |

### Contamination risks

- JAXBench **is** the eval benchmark — any SFT row whose prompt embeds workload names, CONFIG dims, or baseline source from these 50 tasks **contaminates sealed eval**.
- The 33 KernelBench-derived workloads overlap **ScalingIntelligence/KernelBench level-2** naming — keep JAXBench sealed disjoint from KernelBench-derived training prompts or accept overlap.
- `optimized.py` files are **reference solutions** — fine for SFT targets on train split, never leak into eval prompts.

---

## HF CLI notes (hf 1.23.0)

### Worked

```bash
hf auth whoami

# Hub API search (hf has no `datasets search` subcommand)
curl -s "https://huggingface.co/api/datasets?search=kernelbench&limit=20&sort=downloads&direction=-1"

# Dataset metadata / files
hf datasets card EvanOLeary/pallasbench-unified --metadata
hf datasets list ScalingIntelligence/KernelBench
hf datasets info ScalingIntelligence/KernelBench   # large YAML dump

# SQL over parquet (reliable for row counts + samples)
hf datasets sql "SELECT COUNT(*) AS n FROM read_parquet('https://huggingface.co/datasets/ScalingIntelligence/KernelBench/resolve/main/data/level_1-00000-of-00001.parquet')"

# Download specific files
hf download --repo-type dataset ScalingIntelligence/KernelBench data/level_1-00000-of-00001.parquet
hf download --repo-type dataset EvanOLeary/pallasbench-robust-gpu-a100 pallasbench_sakana_format.jsonl
```

### Did not work / gaps

```bash
hf datasets search jax          # Error: No such command 'search'
hf download ScalingIntelligence/KernelBench ...   # fails without --repo-type dataset
hf datasets parquet <repo>      # often returns no URLs; use direct parquet URLs in sql instead
```

**MCP:** `hub_repo_search` on `plugin-huggingface-skills-huggingface-skills` worked for `kernelbench`, `pallas`, `jax` (with noise).

---

## Ranked HF dataset shortlist

### Tier A — use for Pallas/JAX skill (top picks)

#### 1. `EvanOLeary/pallasbench-unified` ⭐

| Field | Value |
|-------|-------|
| License | Not in `--metadata` blob; sibling `pallasbench-robust-gpu-a100` is **apache-2.0** |
| Rows | **633 + 317 + 150 = 1,100** (level_1/2/3 parquet splits) |
| Modality | `Pallas_Code`, `JAX_Code_*`, IR dumps (Jaxpr, StableHLO, PTX, SASS), timing, correctness |
| Updated | 2026-06-01 |
| Sample | `Op_Name=add`, `Correct=True/False`, `Pallas_Code` ~500–1000 chars; multiple generations per op |

**Why:** Only large HF dataset with **real Pallas kernel source**, JAX baselines, correctness bit, and profiling metadata. Multiple LLM generations per task → good SFT + RL preference pairs.

**Caveats:** GPU-focused (Triton backend via Pallas); not identical to TPU Mosaic. Filter `Correct=True` for SFT; use incorrect rows for RL negatives. May overlap conceptually with JAXBench KernelBench L2 patterns.

#### 2. `EvanOLeary/pallasbench-robust-gpu-a100` ⭐

| Field | Value |
|-------|-------|
| License | **apache-2.0** |
| Rows | **45** kernels (+ `pallasbench_sakana_format.jsonl`, kernel source trees) |
| Updated | 2026-05-30 |

**Why:** Curated **pass/fail** Pallas kernels with `Pallas_Code` + `Pallas_Code_Original` + `JAX_Baseline_Code`, compile-time and fix metadata. Highest **per-row quality** for teaching GPU-safe Pallas tiling.

**Caveats:** Tiny; eval-only overlap risk if same ops appear in sealed JAXBench tasks. Pair with unified for volume.

#### 3. `EvanOLeary/pallasbench-archive` / `pallasbench-robust`

| Field | Value |
|-------|-------|
| Rows | archive: **1K–10K** category; robust: **&lt;1K** |
| Notes | Supersets / earlier eval runs; prefer **unified** unless you need historical generations |

#### 4. JAXBench GitHub (not HF) — render to JSONL ⭐ for sealed + gold SFT

| Field | Value |
|-------|-------|
| License | **Apache-2.0** |
| Rows | 50 baselines, 8 Pallas optimized |
| Tests | Harness correctness + timing |

**Why:** **TPU-native** target distribution; aligns with accelerator-agents / MaxKernel roadmap (CUDA→Pallas). Best **sealed eval** anchor for Model Factory.

---

### Tier B — strong adjacent fuel (translation / volume)

#### 5. `ScalingIntelligence/KernelBench`

| Field | Value |
|-------|-------|
| License | Not in metadata (paper/repo; cite Ouyang et al.) |
| Rows | **270** (L1:100, L2:100, L3:50, L4:20) |
| Columns | `code` (PyTorch Module), `level`, `name`, `problem_id` |
| Updated | 2025-07-21 |

**Why:** JAXBench includes 33 L2 problems from here; good for **“given PyTorch fused op → write Pallas”** prompts. Largest **structured** kernel benchmark on HF.

**Caveats:** **CUDA/Triton target**, not JAX. High public benchmark exposure → contamination if you eval on KernelBench without held-out levels. JAXBench overlap on L2 subset.

#### 6. `allenanie/kernelbench_with_prompts`

| Field | Value |
|-------|-------|
| License | **mit** |
| Rows | ~100 per level × 3 levels × 2 variants (cuda + orgex prompts) |
| Format | JSONL with explicit `prompt` + `code` |

**Why:** Ready-made **prompt strings** for kernel generation SFT; saves prompt engineering.

**Caveats:** CUDA/Triton only; duplicates KernelBench content.

#### 7. `SakanaAI/AI-CUDA-Engineer-Archive`

| Field | Value |
|-------|-------|
| License | **cc-by-4.0** |
| Rows | L1: **12,157**; L2: **12,938**; L3: **5,520** (~30.6k total) |
| Columns | `Op_Name`, `CUDA_Code`, runtime/speedup metadata |

**Why:** Massive **verified CUDA** corpus; MaxKernel explicitly cites this lineage. Use for **CUDA→Pallas** translation SFT (pairs with JAX/Pallas targets from pallasbench or JAXBench).

**Caveats:** Wrong DSL for direct Pallas eval; CC-BY attribution required.

#### 8. `BonnieWang/KernelBenchX`

| Field | Value |
|-------|-------|
| License | **apache-2.0** |
| Rows | **176** tasks + **110** corpus rows |
| Columns | `reference_code`, `test_code`, `description`, `difficulty` |

**Why:** Includes **`test_code`** in parquet — closest HF analog to SudarshanBench pytest verifiers for fused PyTorch ops.

**Caveats:** PyTorch/CUDA eval stack; Triton-focused paper; not JAX.

#### 9. `ScalingIntelligence/kernelbench-samples`

| Field | Value |
|-------|-------|
| License | **apache-2.0** |
| Content | Agent run artifacts (`baseline_eval/`, `iterative_refinement/`, `repeated_sampling/`) |

**Why:** Shows **successful agent trajectories** on KernelBench (paper reproduction).

**Caveats:** Unstructured dirs; CUDA; small vs traces corpora.

---

### Tier C — agent traces (defer for Pallas-first)

#### 10. `Infatoshi/kernelbench-hard-traces` / `kernelbench-mega-traces`

| Field | Value |
|-------|-------|
| License | **mit** |
| Format | Per-run JSONL agent sessions (Claude, Codex, GLM, etc.) |
| Downloads | hard-traces ~1.7k; mega-traces ~2.4k |
| Updated | 2026-07-22 |

**Why:** Rich **frontier agent behavior** on hard CUDA kernels (FP8 GEMM, paged attention, MoE).

**Caveats:** **CUDA/Triton output**, not Pallas; trace format needs heavy rendering; high risk of teaching wrong DSL. Use only after Pallas base works, for cross-domain tool-use patterns.

#### 11. `Infatoshi/kernelbench-hard-problems` / `hard-submissions` / `v3-*`

| Field | Value |
|-------|-------|
| License | MIT / CC-BY-4.0 (runs) |
| Rows | 7 hard problems; submissions parquet |

**Why:** Problem definitions + winning CUDA submissions.

**Caveats:** Tiny problem count; CUDA; eval overlap with public leaderboard.

#### 12. `rtferraz/cuda-kernel-engineering`

Tutorial-style repo (vector-add → tiled GEMM chapters), not a flat training corpus. Useful pedagogy, poor JSONL yield.

#### 13. `winglian/gpu-mode-triton-augment-sample`

| Rows | 10 synthetic Triton pairs |
| License | distilabel synthetic |

**Caveats:** Trivial volume; Triton not Pallas.

---

## Datasets to avoid or defer

| Dataset | Reason |
|---------|--------|
| HF `jax` search hits (`whisper-jax-*`, `HumanPhenotype_JAX`, `jaxmetaverse/*`) | Wrong “JAX” — audio, bio, usernames |
| `CyberHarem/pallas_arknights`, `FinchResearch/pallas_splitted_18c`, `open-llm-leaderboard-old/details_Mihaiii__Pallas-*` | Character art / unrelated LLM named Pallas |
| `anonymous-gpu-kernel/anonymous-gpu-kernel` | datasets-server 500 on parquet; unclear provenance |
| `makora-ai/triton-gpu-latency`, `ttt-ttt9/robust_kbench-*` | Triton/CUDA latency tables; no JAX |
| Infatoshi traces (for now) | Teach CUDA agent loops before Pallas syntax is stable |
| KernelBench HF rows as **sealed eval** while training on same `problem_id` | Direct contamination |

---

## Contamination matrix (train vs eval)

| Eval anchor | Do not train on |
|-------------|-----------------|
| **JAXBench 50** (sealed) | Same task IDs, CONFIG strings, baseline/optimized source, MaxKernel `jaxbench_adapted_dataset` prompts |
| **SudarshanBench sealed_v2** | Keep disjoint from new Pallas family IDs |
| **KernelBench L4** (20) | Hold out as cross-benchmark sealed; train L1–L3 only if using KernelBench |
| **pallasbench-unified** | If used for eval, hold out `Task_ID`/`Op_Name` splits |

---

## Recommended Model Factory next steps

Stage 6 thin GRPO **killed** on toy SudarshanBench (sealed Δ=0). For **JAX/Pallas kernel skill**, do not bolt Pallas onto existing `sb-000*` fuel — stand up a **new bench family**:

### 1. `SudarshanBench-Pallas` (draft)

| Split | Source | Size |
|-------|--------|------|
| **sealed** | JAXBench 50 workloads via harness (GitHub) | 50 |
| **dev** | 10 held-out `Op_Name` from pallasbench-unified | ~10 |
| **train** | `pallasbench-unified` `Correct=True` minus dev; JAXBench 8 `optimized.py` pairs; optional KernelBench L1→Pallas synthetic | ~500–900 |

Render to Tinker JSONL: `(prompt: reference JAX + spec) → (completion: Pallas kernel)` using existing Stage-4/5 pipeline.

### 2. Download + render script (no training yet)

```bash
hf download --repo-type dataset EvanOLeary/pallasbench-unified data/
hf download --repo-type dataset EvanOLeary/pallasbench-robust-gpu-a100 pallasbench_sakana_format.jsonl
git clone --depth 1 https://github.com/AI-Hypercomputer/accelerator-agents.git /tmp/accelerator-agents
# render: docs/model-factory/04-data-factory/ (new renderer jaxbench_pallas_render.py — TBD)
```

### 3. RL verifier

Wire **JAXBench harness** `evaluate_kernel` as reward (correctness binary + optional speedup shaping) on TPU — mirrors Stage-6 thin RL but with real kernel signal.

### 4. Optional volume pass (later)

- `SakanaAI/AI-CUDA-Engineer-Archive` → CUDA→Pallas translation pairs (CC-BY)
- `ScalingIntelligence/KernelBench` L1 prompts → Pallas completions via teacher or self-distill

### 5. Do **not** spend Tinker budget until

- Sealed **JAXBench subset** (n≥20) probed with current LoRA baseline (expect ~0 without Pallas fuel)
- Train/dev JSONL scrubbed + rights manifest updated for apache-2.0 / cc-by-4.0 sources

---

## Sample rows (verified 2026-07-23)

**ScalingIntelligence/KernelBench** L1: `problem_id=100`, `name=100_HingeLoss`, PyTorch `code` len 566.

**EvanOLeary/pallasbench-robust-gpu-a100**: `Op_Name=relu`, `Correct=True`, `Backend=triton`, `Pallas_Code` uses `pl.pallas_call`, `BlockSpec`, `program_id`.

**EvanOLeary/pallasbench-unified** L1: 633 rows; mixed `Correct` True/False; includes full IR captures.

**BonnieWang/KernelBenchX**: `fused_bmm_rmsnorm_gelu_dropout_sub` with `reference_code` + CUDA `test_code`.

---

## References

- JAXBench README: https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/JAXBench
- KernelBench HF: https://huggingface.co/datasets/ScalingIntelligence/KernelBench
- PallasBench unified: https://huggingface.co/datasets/EvanOLeary/pallasbench-unified
- Climb ladder (current sealed metric): `docs/model-factory/06-env-rl/climb-ladder.md`
