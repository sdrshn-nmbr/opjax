import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.transpose(x_ref[...])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape[::-1], x.dtype),
        in_specs=[pl.BlockSpec((256, 512), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((512, 256), lambda i: (0, 0)),
        grid=(1,),
    )(x)
