import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.abs(x).T

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape[::-1], x.dtype),
        in_specs=[pl.BlockSpec((384, 640), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((640, 384), lambda i: (0, 0)),
        grid=(1,),
    )(x)
