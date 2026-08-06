import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, o_ref):
    o_ref[...] = jnp.transpose(x_ref[...], (1, 0))

def transpose_square(x):
    return pl.pallas_call(
        workload,
        out_shape=jax.ShapeDtypeStruct((512, 256), jnp.float32),
        grid=(256,),
        in_specs=[pl.BlockSpec((1, 512), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, 512), lambda i: (i, 0)),
    )(x)
