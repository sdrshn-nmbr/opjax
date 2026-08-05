import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
SHAPE = (128, 256)

def _kernel(x_ref, o_ref):
    values = x_ref[...].astype(jnp.float32)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    o_ref[...] = 0.0 * ((values - mean) * jax.lax.rsqrt(variance + 1e-05))

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32), grid=(SHAPE[0] // 8,), in_specs=(spec,), out_specs=spec)(x)
