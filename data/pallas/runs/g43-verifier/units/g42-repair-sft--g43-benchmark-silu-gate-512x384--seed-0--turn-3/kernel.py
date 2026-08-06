import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

def workload(*inputs):
    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[...]
        y = y_ref[...]
        o_ref[...] = x * jax.nn.silu(y)

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
