import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = jnp.where(y == 0, jnp.zeros_like(x), x / y)

def safe_divide(x, y):
    return jax.jit(
        pl.pallas_call(
            workload,
            out_shape=jax.ShapeDtypeStruct.like(x),
            grid=(x.shape[0], x.shape[1]),
            in_specs=[
                pl.BlockSpec((1, 1), lambda i, j: (i, j)),
                pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
        )
    )(x, y)
