import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    # a: [128, 256], b: [256, 128]
    # Define block sizes
    block_m = 32
    block_n = 32
    block_k = 32
    
    # Grid
    grid = (a.shape[0] // block_m, b.shape[1] // block_n)
    
    def kernel(a_ref, b_ref, c_ref):
        # Get program ids
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)
        
        # Initialize accumulator
        # We can use jnp.zeros((block_m, block_n), dtype=jnp.float32)
        # But inside kernel, we should use pl.load? Actually we can create local arrays with jnp.zeros.
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Loop over K dimension
        for k in range(0, a.shape[1], block_k):
            # Load A block: [block_m, block_k]
            a_block = pl.load(a_ref, (pl.dslice(pid_m * block_m, block_m), pl.dslice(k, block_k)))
            # Load B block: [block_k, block_n]
            b_block = pl.load(b_ref, (pl.dslice(k, block_k), pl.dslice(pid_n * block_n, block_n)))
            
            # Compute dot product of a_block and b_block
            # We can use jnp.dot or manual loop.
            # Let's use jnp.dot for simplicity.
            acc += jnp.dot(a_block, b_block)
        
        # Store result
        pl.store(c_ref, (pl.dslice(pid_m * block_m, block_m), pl.dslice(pid_n * block_n, block_n)), acc)
    
    # Create output
    c = jnp.zeros((a.shape[0], b.shape[1]), dtype=jnp.float32)
    
    # Call pallas
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(c.shape, c.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec(lambda i, j: (i, 0), (block_m, block_k)),  # Wait, need to specify block spec correctly.
            pl.BlockSpec(lambda i, j: (0, j), (block_k, block_n)),
        ],
        out_specs=pl.BlockSpec(lambda i, j: (i, j), (block_m, block_n)),
        interpret=False,
    )(a, b)
    return out
