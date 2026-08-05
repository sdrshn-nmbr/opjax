import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
SHAPE = (256, 384)

def _kernel(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.minimum(x_ref[...], y_ref[...])

def workload(x, y):
    spec = pl.BlockSpec(lambda i, j: (i, j), (128, 128))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, x.dtype), grid=(SHAPE[0] // 128, SHAPE[1] // 128), in_specs=(spec, spec), out_specs=spec)(x, y)
