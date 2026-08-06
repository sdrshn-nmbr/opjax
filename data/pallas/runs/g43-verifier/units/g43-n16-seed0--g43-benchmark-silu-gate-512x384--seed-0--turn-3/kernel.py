import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, z_ref):
        x = x_ref[...]
        y = y_ref[...]
        z_ref[...] = jax.nn.silu(x) * y

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(inputs[0].shape, inputs[0].dtype),
        in_specs=[
            pl.BlockSpec((512, 384), lambda i, j: (i, j)),
            pl.BlockSpec((512, 384), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((512, 384), lambda i, j: (i, j)),
        grid=(1, 1),
    )(*inputs)
