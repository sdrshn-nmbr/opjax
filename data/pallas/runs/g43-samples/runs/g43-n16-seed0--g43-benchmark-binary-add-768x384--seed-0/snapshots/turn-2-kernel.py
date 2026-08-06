import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((768, 384), jnp.float32),
        in_specs=[
            pl.BlockSpec((768, 384), lambda i: (0, 0)),
            pl.BlockSpec((768, 384), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((768, 384), lambda i: (0, 0)),
        grid=(1,),
    )(*inputs)
