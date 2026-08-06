import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jax.nn.softmax(x, axis=-1)
