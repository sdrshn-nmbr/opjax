import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(a, b):
    m, k = a.shape
    k2, n = b.shape
    assert k == k2

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m, n),
        in_specs=[
            pl.BlockSpec((1, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, 1), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
    )(a, b)
