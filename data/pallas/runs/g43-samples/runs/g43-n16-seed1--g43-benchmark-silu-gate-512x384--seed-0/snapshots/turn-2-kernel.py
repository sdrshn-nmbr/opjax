import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, z_ref):
        x = x_ref[...]
        y = y_ref[...]
        z = z_ref[...]
        gate = jax.nn.sigmoid(x)
        out = y * gate * (1.0 + jnp.exp(x))  # approximate SiLU gate semantics
        z_ref[...] = out

    x, y = inputs
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((512, 384), lambda i, j: (i, j)),
            pl.BlockSpec((512, 384), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((512, 384), lambda i, j: (i, j)),
        grid=(1, 1),
    )(x, y)
