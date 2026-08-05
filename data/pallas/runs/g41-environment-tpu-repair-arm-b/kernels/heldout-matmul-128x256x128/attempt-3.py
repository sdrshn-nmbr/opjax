import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    M, K = a.shape
    _, N = b.shape

    block_m = 32
    block_n = 128
    block_k = 32

    def kernel(a_ref, b_ref, c_ref):
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        for k in range(0, K, block_k):
            a_tile = pl.load(a_ref, (pl.dslice(0, block_m), pl.dslice(k, block_k)))
            b_tile = pl.load(b_ref, (pl.dslice(k, block_k), pl.dslice(0, block_n)))
            acc += jnp.sum(a_tile[:, :, None] * b_tile[None, :, :], axis=1)
        pl.store(c_ref, (pl.dslice(0, block_m), pl.dslice(0, block_n)), acc)

    out_shape = jax.ShapeDtypeStruct((M, N), jnp.float32)
    grid = (M // block_m, N // block_n)

    in_specs = [
        pl.BlockSpec((block_m, K), lambda i, j: (i * block_m, 0)),
        pl.BlockSpec((K, block_n), lambda i, j: (0, j * block_n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n))

    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=False,
    )(a, b)
