import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (320, 256)

def _kernel(x_ref, o_ref):
    values = x_ref[...].astype(jnp.float32)
    maximum = jnp.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    o_ref[...] = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 8,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
