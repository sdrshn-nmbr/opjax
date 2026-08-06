import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(lhs, rhs):
    m, k = lhs.shape
    k2, n = rhs.shape
    assert k == k2

    def kernel(lhs_ref, rhs_ref, out_ref):
        pl.dot(lhs_ref, rhs_ref, out_ref, allow_tf32=False)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // 128, n // 128),
        in_specs=[
            pl.BlockSpec((128, 256), lambda i, j: (i, 0)),
            pl.BlockSpec((256, 128), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
    )(lhs, rhs)
