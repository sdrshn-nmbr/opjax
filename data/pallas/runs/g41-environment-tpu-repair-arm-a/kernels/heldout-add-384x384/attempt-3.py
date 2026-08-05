import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x, y = inputs
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct((384, 384), jnp.float32),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(3, 3),
        interpret=False,
    )(x, y)
