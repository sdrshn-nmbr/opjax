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

    in_specs = [pl.BlockSpec(lambda i, j: (i, j), (BLOCK_M, BLOCK_N))]
    out_specs = pl.BlockSpec(lambda i, j: (i, j), (BLOCK_M, BLOCK_N))
    grid = (x.shape[0] // BLOCK_M, x.shape[1] // BLOCK_N)

    out = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
        interpret=False,
    )(x)
    return (out,)
