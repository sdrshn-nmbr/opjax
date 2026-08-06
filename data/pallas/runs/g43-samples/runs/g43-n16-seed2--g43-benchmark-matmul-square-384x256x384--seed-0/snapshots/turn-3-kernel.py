import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def matmul_kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        b = pl.load(b_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        o = jnp.dot(a, b)
        pl.store(o_ref, (pl.dslice(0, 128), pl.dslice(0, 128)), o)
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((384, 384), jnp.float32),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((128, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(3, 3),
    )(a, b)
