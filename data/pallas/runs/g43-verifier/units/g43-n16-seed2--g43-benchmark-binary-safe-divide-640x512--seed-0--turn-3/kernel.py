import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl

def safe_divide_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = jnp.where(y == 0, jnp.zeros_like(x), x / y)

def workload(*inputs):
    x, y = inputs
    block_shape = (32, 32)
    def index_map(i, j):
        return (i, j)
    return pl.pallas_call(
        safe_divide_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(block_shape, index_map),
            pl.BlockSpec(block_shape, index_map),
        ],
        out_specs=pl.BlockSpec(block_shape, index_map),
        grid=(x.shape[0] // block_shape[0], x.shape[1] // block_shape[1]),
    )(x, y)
