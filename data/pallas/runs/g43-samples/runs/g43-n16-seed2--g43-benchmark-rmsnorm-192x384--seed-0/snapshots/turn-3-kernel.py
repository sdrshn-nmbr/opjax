import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def rmsnorm_kernel(x_ref, o_ref):
        x = x_ref[...]
        mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + 1e-6)
        o_ref[...] = x / rms
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((192, 384), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((192, 384), lambda i: (i, 0)),
        grid=(1,),
    )(x)
