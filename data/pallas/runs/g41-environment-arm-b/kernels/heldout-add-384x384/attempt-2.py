import jax
from jax.experimental import pallas as pl

def workload(*inputs):
    x, y = inputs
    def kernel(x_ref, y_ref, o_ref):
        x_val = pl.load(x_ref, (slice(None), slice(None)))
        y_val = pl.load(y_ref, (slice(None), slice(None)))
        pl.store(o_ref, (slice(None), slice(None)), x_val + y_val)
    block_shape = x.shape
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(block_shape, x.dtype),
        grid=(1, 1),
        in_specs=[
            pl.BlockSpec(block_shape, lambda i, j: (0, 0)),
            pl.BlockSpec(block_shape, lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec(block_shape, lambda i, j: (0, 0)),
    )(x, y)
