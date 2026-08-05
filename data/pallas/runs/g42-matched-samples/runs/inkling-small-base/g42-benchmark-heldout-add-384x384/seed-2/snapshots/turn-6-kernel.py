import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    a, b = inputs
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = a_ref[...] + b_ref[...]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(3, 3),
    )(a, b)
