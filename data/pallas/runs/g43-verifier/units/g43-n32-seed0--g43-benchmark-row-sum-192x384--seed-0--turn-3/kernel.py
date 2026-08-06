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
        out_shape=jax.ShapeDtypeStruct(x.shape[:1] + (1,), x.dtype),
        in_specs=[pl.BlockSpec((32, 384), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((32, 1), lambda i: (i, 0)),
        grid=(x.shape[0] // 32,),
    )(x)
