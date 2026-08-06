import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    x, gate = inputs

    def kernel(x_ref, gate_ref, out_ref):
        x = x_ref[...]
        gate = gate_ref[...]
        out_ref[...] = jnp.where(gate > 0, x * gate, 0.5 * x * jnp.expm1(gate))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(x.shape[0] // 128, x.shape[1] // 128),
    )(x, gate)
