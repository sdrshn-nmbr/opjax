import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=-1)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        in_specs=[pl.BlockSpec((128, 256), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128,), lambda i, j: (i,)),
        grid=(1, 1),
    )(x)
