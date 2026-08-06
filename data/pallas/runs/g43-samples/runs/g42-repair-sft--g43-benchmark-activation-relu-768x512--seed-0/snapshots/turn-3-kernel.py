import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.maximum(x, 0)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((768, 512), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((768, 512), lambda i: (0, 0)),
        grid=(1,),
    )(x)
