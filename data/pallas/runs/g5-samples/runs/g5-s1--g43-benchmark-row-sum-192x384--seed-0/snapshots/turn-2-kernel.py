import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    x, = inputs

    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        grid=(x.shape[0],),
        in_specs=[pl.BlockSpec((x.shape[1],), lambda i: i)],
        out_specs=pl.BlockSpec((1,), lambda i: i),
    )(x)
