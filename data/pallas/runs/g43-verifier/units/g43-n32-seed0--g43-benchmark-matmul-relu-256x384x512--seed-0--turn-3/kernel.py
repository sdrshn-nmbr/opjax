import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl

def workload(*inputs):
    A, B = inputs
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    def matmul_kernel(a_ref, b_ref, o_ref):
        a = a_ref[...]
        b = b_ref[...]
        o_ref[...] = jnp.dot(a, b)

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((128, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        grid=(M // 128, N // 128),
    )(A, B)
