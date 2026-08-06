import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(a_ref, b_ref, o_ref):
        a = pl.load(a_ref, (pl.dslice(0, 256), pl.dslice(0, 384)))
        b = pl.load(b_ref, (pl.dslice(0, 384), pl.dslice(0, 512)))
        o = jnp.dot(a, b)
        o = jnp.maximum(o, 0.0)
        pl.store(o_ref, (pl.dslice(0, 256), pl.dslice(0, 512)), o)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec((256, 384), lambda i, j: (0, 0)),
            pl.BlockSpec((384, 512), lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((256, 512), lambda i, j: (0, 0)),
        grid=(1, 1),
    )(*inputs)
