import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    x, = inputs

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.exp(x_ref[...])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((640, 384), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((640, 384), lambda i: (0, 0)),
        grid=(1,),
    )(x)
