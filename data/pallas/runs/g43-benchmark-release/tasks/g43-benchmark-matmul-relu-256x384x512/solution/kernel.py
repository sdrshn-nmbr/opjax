import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

M, K, N = 256, 384, 512

def _kernel(x_ref, y_ref, o_ref):
    values = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = jnp.maximum(values, 0.0)

def workload(x, y):
    x_spec = pl.BlockSpec((128, K), lambda i, j: (i, 0))
    y_spec = pl.BlockSpec((K, 128), lambda i, j: (0, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // 128, N // 128),
        in_specs=(x_spec, y_spec),
        out_specs=out_spec,
    )(x, y)
