import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((384, 384), jnp.float32),
        in_specs=[
            pl.BlockSpec((384, 384), lambda i, j: (i, j)),
            pl.BlockSpec((384, 384), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((384, 384), lambda i, j: (i, j)),
        grid=(1, 1),
    )(*inputs)
