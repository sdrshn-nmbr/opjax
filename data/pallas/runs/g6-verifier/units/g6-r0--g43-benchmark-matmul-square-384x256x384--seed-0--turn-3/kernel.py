import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        b = pl.load(b_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
        o = jnp.dot(a, b)
        pl.store(o_ref, (pl.dslice(0, 128), pl.dslice(0, 128)), o)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(3, 3),
    )(a, b)
