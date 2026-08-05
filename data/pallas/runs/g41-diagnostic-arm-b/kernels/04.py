import jax
from jax.experimental import pallas as pl

def workload(*inputs):
    x, y = inputs
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1, 1),
        in_specs=[
            pl.BlockSpec(lambda i, j: (0, 0), x.shape),
            pl.BlockSpec(lambda i, j: (0, 0), y.shape),
        ],
        out_specs=pl.BlockSpec(lambda i, j: (0, 0), x.shape),
    )(x, y)
