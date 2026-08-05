import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def matmul_kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0), pl.dslice(0)))
        b = pl.load(b_ref, (pl.dslice(0), pl.dslice(0)))
        o = jnp.dot(a, b)
        pl.store(o_ref, (pl.dslice(0), pl.dslice(0)), o)
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        in_specs=[
            pl.BlockSpec((128, 256), lambda i, j: (i, 0)),
            pl.BlockSpec((256, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(1, 1),
    )(a, b)
