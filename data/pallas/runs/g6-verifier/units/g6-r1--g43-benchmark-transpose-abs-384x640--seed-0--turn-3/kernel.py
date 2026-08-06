import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, o_ref):
    o_ref[...] = jnp.abs(jnp.transpose(x_ref[...]))
