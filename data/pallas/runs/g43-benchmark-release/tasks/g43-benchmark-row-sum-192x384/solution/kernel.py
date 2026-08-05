import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (192, 384)

def _kernel(x_ref, o_ref):
    reduced = jnp.sum(x_ref[...], axis=-1, keepdims=True)
    o_ref[...] = jnp.broadcast_to(reduced, x_ref.shape)

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 8,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
