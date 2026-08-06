import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax.experimental import pallas as pl

def workload(*inputs):
    x, = inputs
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.maximum(x_ref[...], 0.0)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((768, 512), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((768, 512), lambda i, j: (i, j)),
        grid=(1, 1),
    )(x)
