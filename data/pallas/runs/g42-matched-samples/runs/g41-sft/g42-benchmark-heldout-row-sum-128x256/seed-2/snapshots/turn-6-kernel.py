import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        o_ref[0] = jnp.sum(x_ref[...], axis=1)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape[0], x.dtype),
        in_specs=[pl.BlockSpec((1, x.shape[1]), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
        grid=(x.shape[0],),
    )(x)
