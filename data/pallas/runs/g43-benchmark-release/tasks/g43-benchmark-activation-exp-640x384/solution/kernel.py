import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (640, 384)

def _kernel(x0_ref, o_ref):
    o_ref[...] = jnp.exp(x0_ref[...])

def workload(x0):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=(spec,),
        out_specs=spec,
    )(x0)
