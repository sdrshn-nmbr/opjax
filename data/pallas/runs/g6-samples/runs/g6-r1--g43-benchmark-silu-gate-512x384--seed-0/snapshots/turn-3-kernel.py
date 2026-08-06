import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x, y):
    return pl.pallas_call(
        lambda x_ref, y_ref, o_ref: o_ref.__setitem__(slice(None), jax.nn.silu(x_ref[...]) * y_ref[...]),
        out_shape=jax.ShapeDtypeStruct((512, 384), jnp.float32),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((512, 384), lambda i: (0, 0)),
            pl.BlockSpec((512, 384), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((512, 384), lambda i: (0, 0)),
    )(x, y)
