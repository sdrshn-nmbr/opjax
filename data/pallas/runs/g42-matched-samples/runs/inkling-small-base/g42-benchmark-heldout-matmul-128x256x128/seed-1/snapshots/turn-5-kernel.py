import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    a, b = inputs
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    block_m = 32
    block_n = 32

    def kernel(a_ref, b_ref, c_ref):
        c_ref[...] = jnp.dot(a_ref[...], b_ref[...])

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        in_specs=[
            pl.BlockSpec((block_m, K), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((K, block_n), lambda i, j: (0, j * block_n)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        grid=(M // block_m, N // block_n),
    )(a, b)
    return out
