import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[...]
        y = y_ref[...]
        o_ref[...] = jnp.where(y == 0, jnp.zeros_like(x), x / y)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(inputs[0].shape, inputs[0].dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(inputs[0].shape[0] // 128, inputs[0].shape[1] // 128),
    )(*inputs)
