import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def matmul_kernel(a_ref, b_ref, o_ref):
        a = a_ref[...]
        b = b_ref[...]
        o_ref[...] = jnp.dot(a, b)
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape[:-1] + b.shape[1:], a.dtype),
        in_specs=[
            pl.BlockSpec((128, 256), lambda i, j: (i, 0)),
            pl.BlockSpec((256, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(1, 1),
    )(a, b)
