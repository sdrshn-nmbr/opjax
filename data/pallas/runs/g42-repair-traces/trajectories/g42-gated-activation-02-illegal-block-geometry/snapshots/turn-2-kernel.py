import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
SHAPE = (512, 256)

def _kernel(x_ref, gate_ref, o_ref):
    o_ref[...] = jax.nn.gelu(x_ref[...]) * gate_ref[...]

def workload(x, gate):
    spec = pl.BlockSpec((7, 7), lambda i, j: (i, j))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, x.dtype), grid=(SHAPE[0] // 128, SHAPE[1] // 128), in_specs=(spec, spec), out_specs=spec)(x, gate)
