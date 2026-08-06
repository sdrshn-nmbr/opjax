import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

def workload(*inputs):
    a, b = inputs
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    def matmul_kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        b = pl.load(b_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        o = jnp.dot(a, b)
        pl.store(o_ref, (pl.dslice(0, 128), pl.dslice(0, 128)), o)

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape[:-1] + b.shape[-1:], a.dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((128, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(M // 128, N // 128),
    )(a, b)
