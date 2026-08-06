import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[...]
        y = y_ref[...]
        o_ref[...] = jax.nn.gelu(x * y)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((640, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec((640, 512), lambda i: (0, 0)),
            pl.BlockSpec((640, 512), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((640, 512), lambda i: (0, 0)),
        grid=(1,),
    )(*inputs)
