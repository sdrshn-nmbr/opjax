import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (256, 256)

def _kernel(x_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...], 0.0)

def workload(x):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, x.dtype),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
