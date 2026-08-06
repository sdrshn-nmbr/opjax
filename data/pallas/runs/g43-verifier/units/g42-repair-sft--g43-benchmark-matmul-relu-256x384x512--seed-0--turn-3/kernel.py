import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0, 128), pl.dslice(0, 384)))
        b = pl.load(b_ref, (pl.dslice(0, 384), pl.dslice(0, 128)))
        o = jnp.dot(a, b)
        o = jnp.maximum(o, 0.0)
        pl.store(o_ref, (pl.dslice(0, 128), pl.dslice(0, 128)), o)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec((128, 384), lambda i, j: (i, 0)),
            pl.BlockSpec((384, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(2, 4),
    )(a, b)
