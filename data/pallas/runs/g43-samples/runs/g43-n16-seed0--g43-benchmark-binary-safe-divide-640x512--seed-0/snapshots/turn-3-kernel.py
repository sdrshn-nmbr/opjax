import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def safe_divide(x, y):
    return jnp.where(y == 0, jnp.zeros_like(x), x / y)

def workload(*inputs):
    x, y = inputs
    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[...]
        y = y_ref[...]
        o_ref[...] = jnp.where(y == 0, jnp.zeros_like(x), x / y)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((640, 512), lambda i, j: (i, j)),
            pl.BlockSpec((640, 512), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((640, 512), lambda i, j: (i, j)),
    )(x, y)
