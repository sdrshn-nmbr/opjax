import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = jax.nn.silu(x) * y
