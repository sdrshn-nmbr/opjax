import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, o_ref):
    o_ref[...] = jnp.sum(x_ref[...], axis=1)

def row_sum(x):
    return pl.pallas_call(
        workload,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        grid=(x.shape[0],),
        in_specs=[pl.BlockSpec((x.shape[1],), lambda i: i)],
        out_specs=pl.BlockSpec((1,), lambda i: i),
    )(x)
