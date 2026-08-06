import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.transpose(x_ref[...])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape[::-1], x.dtype),
        in_specs=[pl.BlockSpec((256, 512), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((512, 256), lambda i: (0, 0)),
        grid=(1,),
    )(x)
