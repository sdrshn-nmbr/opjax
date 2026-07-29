# JAXBench — Stage-5 Inkling LoRA on TPU v5e (official harness)

**Takeaway:** Official JAXBench `evaluate` on a singular **v5litepod-1** scores Stage-5 LoRA kernels **41/50 (82%)** correct. No meaningful speedups vs baseline (kernels are plain JAX rewrites; median speedup ≈ 1.0). Confirms the CPU functional story and adds HBM / dtype / dynamic-shape failures that only show at full CONFIG on 16 GB HBM.

## Hardware / recipe

From Codex session `019f8aa4-4888-7162-983b-a5e17fe93abc` (GCP CLI, project `astral-medley-465922-b2`):

```bash
export PROJECT_ID=astral-medley-465922-b2
export ZONE=us-central1-a
export TPU_NAME=opjax-jaxbench-v5e

gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --accelerator-type=v5litepod-1 \
  --version=v2-tpuv5-litepod

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" --worker=0 \
  --command='pip3 install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'

# after upload of JAXBench + kernels:
python3 -m JAXBench evaluate --workload 8p_GEMM --kernel kernels/8p_GEMM.py --tpu v5e --json

gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" --quiet
```

- TPU VM: `opjax-jaxbench-v5e` · `v5litepod-1` · `us-central1-a` · **deleted after run**
- JAX: 0.6.2 + libtpu · `jax.device_count() == 1`
- Kernels: Stage-5 LoRA gens from `data/model-factory/evals/jaxbench-baseline-lora/kernels/`
- Warmup/iters: 3 / 20 (timing stable enough for speedup claims)

## Scores

| Slice | Correct |
|-------|---------|
| All | **41/50 (0.82)** |
| Priority (`*p_*`) | 12/17 (~0.71) |
| KernelBench-L2 (`*k_*`) | 29/33 (~0.88) |

Status mix: `correct` 41 · `error` 3 · `incorrect` 3 · `runtime_error` 3.

Speedups vs JAXBench baseline: max **1.0**, one regressor (`26k` speedup 0.87). No Pallas in candidate kernels (same as CPU baseline).

## Failures

| Workload | Status | Note |
|----------|--------|------|
| `2p_GQA_Attention` | error | HBM OOM (~16.5G / 15.75G) |
| `3p_MLA_Attention` | runtime_error | shape broadcast (same family as CPU) |
| `11p_Megablox_GMM` | error | traced/dynamic shape under `jit` |
| `15p_RetNet_Retention` | incorrect | numerical |
| `16p_Mamba2_SSD` | error | HBM OOM |
| `37k_Matmul_Swish_Scaling` | incorrect | also failed CPU |
| `44k_Matmul_Divide_GELU` | runtime_error | bf16/f32 dtype mix |
| `47k_…` | runtime_error | rank mismatch (also CPU) |
| `48k_…` | incorrect | also failed CPU |

**vs CPU functional baseline (43/50, shrunk dims):** TPU-only new fails = GQA/Mamba OOM, Megablox, RetNet, `44k`. CPU fails that **pass** on TPU full CONFIG = `4p` Sparse, `6p` Paged, `7p` Ragged (shrink path was harsher than full-size bf16 TPU).

## Artifacts

- `data/model-factory/evals/jaxbench-tpu-v5e-lora/summary.json`
- `data/model-factory/evals/jaxbench-tpu-v5e-lora/tpu_eval_results.json`
- `data/model-factory/evals/jaxbench-tpu-v5e-lora/eval.log`

## Implication

Stage-5 LoRA already matches baseline JAX on most workloads at full TPU sizes, but does **not** emit faster Pallas. Next lift needs Pallas-capable training fuel (e.g. `pallasbench-unified`) or an RL/env loop that rewards `pltpu` / speedup, not just correctness.
