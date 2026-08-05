import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    # Define kernel
    def matmul_kernel(a_ref, b_ref, c_ref):
        # Load full M-tile of A and N-tile of B
        a_block = a_ref[...]  # (32, 128)
        b_block = b_ref[...]  # (128, 32)
        # Initialize accumulator
        acc = jnp.zeros((32, 32), dtype=jnp.float32)
        # Loop over K tiles
        for k in range(4):
            a_tile = a_block[:, k*32:(k+1)*32]
            b_tile = b_block[k*32:(k+1)*32, :]
            acc += jnp.dot(a_tile.astype(jnp.float32), b_tile.astype(jnp.float32))
        # Store result
        pl.store(c_ref, acc.astype(jnp.bfloat16))
    
    # Block specs
    block_a = pl.BlockSpec((32, 128), lambda i, j: (i * 32, 0))
    block_b = pl.BlockSpec((128, 32), lambda i, j: (0, j * 32))
    block_c = pl.BlockSpec((32, 32), lambda i, j: (i * 32, j * 32))
    
    # Grid
    grid = (4, 4)
    
    # Call
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[block_a, block_b],
        out_specs=block_c,
        grid=grid,
    )(a, b)
