import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1, keepdims=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 1), jnp.float32),
        in_specs=[pl.BlockSpec((128, 256), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((128, 1), lambda i: (0, 0)),
        grid=(1,),
    )(x)
