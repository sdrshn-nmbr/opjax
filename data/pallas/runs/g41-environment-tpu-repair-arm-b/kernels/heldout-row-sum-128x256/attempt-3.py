import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        o_ref[0] = jnp.sum(x_ref[0, :])
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        in_specs=[pl.BlockSpec(lambda i: (i, 0), (1, 256))],
        out_specs=pl.BlockSpec(lambda i: (i,), (1,)),
        grid=(128,),
    )(x)
    return (out,)
