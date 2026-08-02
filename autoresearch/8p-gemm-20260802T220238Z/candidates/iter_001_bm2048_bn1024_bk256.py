"""8p_GEMM autoresearch candidate 001: bm=2048, bn=1024, bk=256."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 2048
BN = 1024
BK = 256


def _kernel(x_ref, y_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        x_ref[...],
        y_ref[...],
        preferred_element_type=jnp.float32,
    )
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
