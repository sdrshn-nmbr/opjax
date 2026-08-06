import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(lhs, rhs):
    m, k = lhs.shape
    k2, n = rhs.shape
    assert k == k2

    def kernel(lhs_ref, rhs_ref, out_ref):
        acc = jnp.zeros((m, n), dtype=jnp.float32)
        for i in range(k):
            acc += lhs_ref[:, i][:, None] * rhs_ref[i, :]
        out_ref[...] = jnp.maximum(acc, 0)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(),
        in_specs=[
            pl.BlockSpec((m, k), lambda: (0, 0)),
            pl.BlockSpec((k, n), lambda: (0, 0)),
        ],
    )(lhs, rhs)
