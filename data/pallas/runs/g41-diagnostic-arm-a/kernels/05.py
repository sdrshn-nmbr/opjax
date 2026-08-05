import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
  a, b = inputs
  # Ensure float32
  a = jnp.asarray(a, dtype=jnp.float32)
  b = jnp.asarray(b, dtype=jnp.float32)
  
  M, K = a.shape
  K2, N = b.shape
  assert M == 128 and K == 256 and K2 == 256 and N == 128
  
  block_M = 32
  block_N = 32
  block_K = 256
  
  def matmul_kernel(a_ref, b_ref, c_ref):
    # Load blocks
    a_block = pl.load(a_ref, (pl.dslice(0, block_M), pl.dslice(0, block_K)))
    b_block = pl.load(b_ref, (pl.dslice(0, block_K), pl.dslice(0, block_N)))
    # Compute
    c_block = pl.dot(a_block, b_block)
    # Store
    pl.store(c_ref, (pl.dslice(0, block_M), pl.dslice(0, block_N)), c_block)
  
  grid = (M // block_M, N // block_N)
  
  # Block specs
  a_spec = pl.BlockSpec(lambda i, j: (i * block_M, 0), (block_M, block_K))
  b_spec = pl.BlockSpec(lambda i, j: (0, j * block_N), (block_K, block_N))
  c_spec = pl.BlockSpec(lambda i, j: (i * block_M, j * block_N), (block_M, block_N))
  
  # Call pallas
  c = pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    in_specs=[a_spec, b_spec],
    out_specs=c_spec,
    grid=grid,
    interpret=False,
  )(a, b)
  
  return c
