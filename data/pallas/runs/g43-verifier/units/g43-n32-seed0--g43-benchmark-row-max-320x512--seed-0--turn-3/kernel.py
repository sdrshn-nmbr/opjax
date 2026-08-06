import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.max(x_ref[...], axis=1, keepdims=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((320, 1), jnp.float32),
        in_specs=[pl.BlockSpec((320, 512), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((320, 1), lambda i: (0, 0)),
        grid=(1,),
    )(x)
