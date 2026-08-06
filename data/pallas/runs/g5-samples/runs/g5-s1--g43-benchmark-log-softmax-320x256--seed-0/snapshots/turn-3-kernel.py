import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, o_ref):
    o_ref[...] = jax.nn.log_softmax(x_ref[...], axis=-1)

def log_softmax(x):
    return pl.pallas_call(
        workload,
        out_shape=jax.ShapeDtypeStruct.like(x),
        grid=(x.shape[0],),
        in_specs=[pl.BlockSpec((1, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, x.shape[1]), lambda i: (i, 0)),
    )(x)
