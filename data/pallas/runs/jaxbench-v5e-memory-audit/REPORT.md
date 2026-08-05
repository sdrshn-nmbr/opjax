# Calibrated JAXBench v5e headroom results

This report rerenders the eight bundled optimized references after separating
physical v5e capacity from Pallas scoped VMEM and applying a fail-closed scoring
contract. A speedup is shown only when both arms use the same inputs, pass an
independent full-shape oracle, use normal Pallas lowering, and are timed on the
device after compilation and warmup.

| Workload | XLA median | Pallas median | Speedup | Result |
|---|---:|---:|---:|---|
| `1p_Flash_Attention` | 29.3753 ms | — | — | Excluded: optimized reference is incorrect |
| `2p_GQA_Attention` | — | 27.0965 ms | — | Excluded: naive XLA oracle exceeds HBM |
| `3p_MLA_Attention` | 39.7046 ms | — | — | Excluded: optimized reference is incorrect |
| `4p_Sparse_Attention` | 7.5757 ms | 12.1039 ms | — | Excluded: arms use different batch shapes and FLOPs |
| `6p_Paged_Attention` | 11.8045 ms | — | — | Excluded: no valid paired oracle after API repair |
| `7p_Ragged_Paged_Attention` | 35.2304 ms | — | — | Excluded: no valid paired oracle after API and VMEM repair |
| `8p_GEMM` | 19.8461 ms | 19.9141 ms | **0.9966x** | Valid; correct Pallas at XLA parity, no headroom |
| `11p_Megablox_GMM` | 4.1864 ms | 3.6496 ms | **1.1470x** | Valid; correct Pallas headroom |

## Result

- Scorable coverage: **2/8**.
- Correct among scorable references: **2/2**.
- References beating XLA: **1/2**.
- References meeting the frozen 1.05x headroom threshold: **1/2**.
- Performance subset: **`11p_Megablox_GMM`**.
- Correctness/parity diagnostic: **`8p_GEMM`**.
- No suite-wide aggregate is reported because six references remain excluded.

The calibrated result is therefore not “five kernels do not fit on v5e.” It is:
one demonstrated fusion-rich Pallas win, one valid GEMM at XLA parity, and six
benchmark references that require oracle, correctness, shape, or API repair
before they can make performance claims.
