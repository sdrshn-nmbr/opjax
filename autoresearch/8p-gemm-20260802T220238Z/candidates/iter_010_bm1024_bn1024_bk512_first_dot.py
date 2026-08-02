"""8p_GEMM autoresearch candidate 010: first-dot accumulator initialization."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 1024
BN = 1024
BK = 512


def _kernel(x_ref, y_ref, out_ref, acc_ref):
    dot = jnp.dot(
        x_ref[...],
        y_ref[...],
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == 0)
    def _first():
        acc_ref[...] = dot

    @pl.when(pl.program_id(2) != 0)
    def _rest():
        acc_ref[...] = acc_ref[...] + dot

    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@jax.jit
def workload(x, y):
    m, k = x.shape
    n = y.shape[1]
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((BM, BK), lambda i, j, kk: (i, kk)),
                pl.BlockSpec((BK, BN), lambda i, j, kk: (kk, j)),
            ],
            out_specs=pl.BlockSpec((BM, BN), lambda i, j, kk: (i, j)),
            grid=(m // BM, n // BN, k // BK),
            scratch_shapes=[pltpu.VMEM((BM, BN), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(x, y)
