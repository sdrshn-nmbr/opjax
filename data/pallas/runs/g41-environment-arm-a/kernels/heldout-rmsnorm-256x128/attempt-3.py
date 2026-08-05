import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    BLOCK_M = 1
    BLOCK_N = 128

    def rmsnorm_kernel(x_ref, o_ref):
        x_tile = pl.load(x_ref, (pl.dslice(0), pl.dslice(0)))
        sq_acc = jnp.zeros((BLOCK_M, 1), dtype=jnp.float32)
        for k in range(BLOCK_N):
            col = x_tile[:, k:k+1]
            sq_acc = sq_acc + col * col
        mean_sq = sq_acc / BLOCK_N
        eps = 1e-6
        rms = jnp.sqrt(mean_sq + eps)
        y = x_tile / rms
        pl.store(o_ref, (pl.dslice(0), pl.dslice(0)), y)

    out = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
        grid=(x.shape[0] // BLOCK_M, x.shape[1] // BLOCK_N),
        interpret=False,
    )(x)
    return (out,)
