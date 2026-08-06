import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(inputs[0].shape, inputs[0].dtype),
        in_specs=[
            pl.BlockSpec((768, 384), lambda i, j: (i, j)),
            pl.BlockSpec((768, 384), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((768, 384), lambda i, j: (i, j)),
        grid=(1, 1),
    )(*inputs)
