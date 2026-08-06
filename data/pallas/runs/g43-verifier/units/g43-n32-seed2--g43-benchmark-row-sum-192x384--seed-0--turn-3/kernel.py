import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1, keepdims=True)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((192, 1), jnp.float32),
        in_specs=[pl.BlockSpec((192, 384), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((192, 1), lambda i: (0, 0)),
        grid=(1,),
    )(x)
