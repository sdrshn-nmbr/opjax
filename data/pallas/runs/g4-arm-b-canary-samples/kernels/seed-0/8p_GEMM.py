import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    A, B = inputs
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    bm, bn = 128, 128

    def kernel(a_ref, b_ref, c_ref):
        a = a_ref[...]
        b = b_ref[...]
        c_ref[...] = jnp.dot(a, b).astype(c_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        grid=(M // bm, N // bn),
        in_specs=[
            pl.BlockSpec(lambda i, j: (i, 0), (bm, K)),
            pl.BlockSpec(lambda i, j: (0, j), (K, bn)),
        ],
        out_specs=pl.BlockSpec(lambda i, j: (i, j), (bm, bn)),
    )(A, B)
