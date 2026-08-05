import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        row = x_ref[...]
        s = jnp.sum(row, axis=-1)
        o_ref[...] = s
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        in_specs=[pl.BlockSpec((1, 256), lambda i, j: (i, 0))],
        out_specs=pl.BlockSpec((1,), lambda i, j: (i,)),
        grid=(128, 1),
    )(x)
    return (out,)
