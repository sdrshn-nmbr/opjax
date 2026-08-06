import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.exp(x_ref[...])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        grid=(),
        in_specs=[pl.BlockSpec((640, 384), lambda: (0, 0))],
        out_specs=pl.BlockSpec((640, 384), lambda: (0, 0)),
    )(x)
