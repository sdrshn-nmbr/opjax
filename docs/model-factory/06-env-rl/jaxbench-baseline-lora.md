# JAXBench baseline — Stage-5 Inkling LoRA (CPU functional)

**Takeaway:** On functional correctness vs JAXBench baselines (CPU, scaled dims), Stage-5 LoRA scores **43/50 (86%)**. It **never emitted Pallas** (0/50) — it rewrites plain JAX that often matches the reference. Hard fails cluster on attention variants (MLA / sparse / paged / ragged). This is **not** an official TPU JAXBench speedup score.

## Methodology

Tinker sample Stage-5 LoRA; CPU JAX correctness vs baseline; CONFIG ints capped via shrink to max_dim=128. Not official TPU JAXBench timing.

## By status

```json
{
  "correct": 43,
  "runtime_error": 4,
  "incorrect": 3
}
```

## Per workload

| Workload | Tier | Status | Correct | Pallas |
|----------|------|--------|---------|--------|
| `1p_Flash_Attention` | priority | correct | True | False |
| `2p_GQA_Attention` | priority | correct | True | False |
| `3p_MLA_Attention` | priority | runtime_error | False | False |
| `4p_Sparse_Attention` | priority | runtime_error | False | False |
| `5p_Flex_Attention` | priority | correct | True | False |
| `6p_Paged_Attention` | priority | runtime_error | False | False |
| `7p_Ragged_Paged_Attention` | priority | incorrect | False | False |
| `8p_GEMM` | priority | correct | True | False |
| `9p_SwiGLU_MLP` | priority | correct | True | False |
| `10p_Sparse_MoE` | priority | correct | True | False |
| `11p_Megablox_GMM` | priority | correct | True | False |
| `12p_RMSNorm` | priority | correct | True | False |
| `13p_Cross_Entropy` | priority | correct | True | False |
| `14p_Ragged_Dot` | priority | correct | True | False |
| `15p_RetNet_Retention` | priority | correct | True | False |
| `16p_Mamba2_SSD` | priority | correct | True | False |
| `17p_Triangle_Multiplication` | priority | correct | True | False |
| `18k_Conv2D_ReLU_BiasAdd` | kernelbench_l2 | correct | True | False |
| `19k_Matmul_Subtract_Multiply_ReLU` | kernelbench_l2 | correct | True | False |
| `20k_Gemm_Multiply_LeakyReLU` | kernelbench_l2 | correct | True | False |
| `21k_Gemm_Divide_Sum_Scaling` | kernelbench_l2 | correct | True | False |
| `22k_Conv2d_InstanceNorm_Divide` | kernelbench_l2 | correct | True | False |
| `23k_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp` | kernelbench_l2 | correct | True | False |
| `24k_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish` | kernelbench_l2 | correct | True | False |
| `25k_Conv3d_GroupNorm_Mean` | kernelbench_l2 | correct | True | False |
| `26k_BMM_InstanceNorm_Sum_ResidualAdd_Multiply` | kernelbench_l2 | correct | True | False |
| `27k_Matmul_Mish_Mish` | kernelbench_l2 | correct | True | False |
| `28k_ConvTranspose3d_LayerNorm_GELU_Scaling` | kernelbench_l2 | correct | True | False |
| `29k_Matmul_Swish_Sum_GroupNorm` | kernelbench_l2 | correct | True | False |
| `30k_Matmul_Scaling_ResidualAdd` | kernelbench_l2 | correct | True | False |
| `31k_Gemm_BatchNorm_GELU_ReLU` | kernelbench_l2 | correct | True | False |
| `32k_Gemm_Sigmoid_LogSumExp` | kernelbench_l2 | correct | True | False |
| `33k_Conv3d_Mish_Tanh` | kernelbench_l2 | correct | True | False |
| `34k_Conv2d_Activation_BatchNorm` | kernelbench_l2 | correct | True | False |
| `35k_Gemm_Scaling_Hardtanh_GELU` | kernelbench_l2 | correct | True | False |
| `36k_Matmul_Sigmoid_Sum` | kernelbench_l2 | correct | True | False |
| `37k_Matmul_Swish_Scaling` | kernelbench_l2 | incorrect | False | False |
| `38k_Matmul_Dropout_Softmax` | kernelbench_l2 | correct | True | False |
| `39k_Conv2d_GELU_GlobalAvgPool` | kernelbench_l2 | correct | True | False |
| `40k_Gemm_GroupNorm_Min_BiasAdd` | kernelbench_l2 | correct | True | False |
| `41k_Gemm_Add_ReLU` | kernelbench_l2 | correct | True | False |
| `42k_Gemm_Max_Subtract_GELU` | kernelbench_l2 | correct | True | False |
| `43k_Gemm_BatchNorm_Scaling_Softmax` | kernelbench_l2 | correct | True | False |
| `44k_Matmul_Divide_GELU` | kernelbench_l2 | correct | True | False |
| `45k_Gemm_GroupNorm_Swish_Multiply_Swish` | kernelbench_l2 | correct | True | False |
| `46k_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp` | kernelbench_l2 | correct | True | False |
| `47k_Matmul_Add_Swish_Tanh_GELU_Hardtanh` | kernelbench_l2 | runtime_error | False | False |
| `48k_Matmul_BatchNorm_BiasAdd_Divide_Swish` | kernelbench_l2 | incorrect | False | False |
| `49k_Matmul_AvgPool_GELU_Scale_Max` | kernelbench_l2 | correct | True | False |
| `50k_Matmul_GELU_Softmax` | kernelbench_l2 | correct | True | False |
