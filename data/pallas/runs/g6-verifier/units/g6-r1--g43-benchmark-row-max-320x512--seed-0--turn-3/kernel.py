import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.max(x_ref[...], axis=0)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((512,), x.dtype),
        grid=(320,),
        in_specs=[pl.BlockSpec((1, 512), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((512,), lambda i: (0,)),
    )(x)
