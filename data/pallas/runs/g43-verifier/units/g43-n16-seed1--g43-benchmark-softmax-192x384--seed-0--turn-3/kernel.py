import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        max_x = jnp.max(x, axis=-1, keepdims=True)
        exp_x = jnp.exp(x - max_x)
        sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
        o_ref[...] = exp_x / sum_exp
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((192, 384), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((192, 384), lambda i: (0, 0)),
        grid=(1,),
    )(x)
