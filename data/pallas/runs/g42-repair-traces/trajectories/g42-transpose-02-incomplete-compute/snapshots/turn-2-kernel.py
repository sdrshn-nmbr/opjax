import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
SHAPE = (256, 384)

def _kernel(x_ref, o_ref):
    o_ref[...] = 0.0 * jnp.transpose(jnp.square(x_ref[...]))

def workload(x):
    in_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (j, i))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct((SHAPE[1], SHAPE[0]), x.dtype), grid=(SHAPE[0] // 128, SHAPE[1] // 128), in_specs=(in_spec,), out_specs=out_spec)(x)
