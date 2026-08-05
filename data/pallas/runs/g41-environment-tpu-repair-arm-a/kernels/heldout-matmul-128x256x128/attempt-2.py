import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    M, K = a.shape
    K2, N = b.shape
    # We assume full shapes as per spec
    block_M = 16
    block_N = 16
    block_K = 256
    
    def matmul_kernel(a_ref, b_ref, c_ref):
        # Load full K dimension for this M/N tile
        a_block = pl.load(a_ref, (pl.dslice(0, block_M), pl.dslice(0, block_K)))
        b_block = pl.load(b_ref, (pl.dslice(0, block_K), pl.dslice(0, block_N)))
        c_block = pl.dot(a_block, b_block)
        pl.store(c_ref, (pl.dslice(0, block_M), pl.dslice(0, block_N)), c_block)
    
    grid = (M // block_M, N // block_N)
    a_spec = pl.BlockSpec((block_M, block_K), lambda i, j: (i * block_M, 0))
    b_spec = pl.BlockSpec((block_K, block_N), lambda i, j: (0, j * block_N))
    c_spec = pl.BlockSpec((block_M, block_N), lambda i, j: (i * block_M, j * block_N))
    
    out = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        in_specs=[a_spec, b_spec],
        out_specs=c_spec,
        grid=grid,
    )(a, b)
    return out
