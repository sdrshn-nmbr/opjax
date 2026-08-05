import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
SHAPE = (128, 128)

def _kernel(x_ref, o_ref):
    values = x_ref[...].astype(jnp.float32)
    maximum = jnp.max(values, axis=-1, keepdims=True)
    numerator = jnp.exp(values - maximum)
    o_ref[...] = numerator / jnp.sum(numerator, axis=-1, keepdims=True)

def workload(x):
    spec = pl.BlockSpec(lambda i: (i, 0), (8, SHAPE[1]))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32), grid=(SHAPE[0] // 8,), in_specs=(spec,), out_specs=spec)(x)
