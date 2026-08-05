import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    M, K = a.shape
    K2, N = b.shape
    # We assume M=128, K=256, N=128
    block_m = 32
    block_n = 32
    block_k = 32
    
    def kernel(a_ref, b_ref, c_ref):
        # Initialize local accumulator
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        # Loop over K tiles
        for k in range(0, K, block_k):
            # Load A tile: relative to windowed a_ref
            a_tile = pl.load(a_ref, (pl.dslice(0, block_m), pl.dslice(k, block_k)))
            # Load B tile: relative to windowed b_ref
            b_tile = pl.load(b_ref, (pl.dslice(k, block_k), pl.dslice(0, block_n)))
            # Accumulate
            acc += jnp.dot(a_tile, b_tile)
        # Store to windowed c_ref
        pl.store(c_ref, (pl.dslice(0, block_m), pl.dslice(0, block_n)), acc)
    
    out_shape = jax.ShapeDtypeStruct((M, N), jnp.float32)
    grid = (M // block_m, N // block_n)
    
    # Block specs: block_shape first, index_map second
    in_specs = [
        pl.BlockSpec((block_m, K), lambda i, j: (i * block_m, 0)),
        pl.BlockSpec((K, block_n), lambda i, j: (0, j * block_n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n))
    
    result = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=False,
    )(a, b)
    return result
