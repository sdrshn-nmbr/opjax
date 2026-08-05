import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (256, 128)

def _kernel(x_ref, o_ref):
    values = x_ref[...].astype(jnp.float32)
    mean_square = jnp.mean(jnp.square(values), axis=-1, keepdims=True)
    o_ref[...] = values * jax.lax.rsqrt(mean_square + 1e-5)

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32), grid=(SHAPE[0] // 8,), in_specs=(spec,), out_specs=spec)(x)
