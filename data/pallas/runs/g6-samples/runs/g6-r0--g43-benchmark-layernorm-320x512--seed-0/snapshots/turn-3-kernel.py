import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        y = (x - mean) / jnp.sqrt(var + 1e-5)
        o_ref[...] = y
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((320, 512), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((320, 512), lambda i: (i, 0)),
        grid=(1,),
    )(x)
