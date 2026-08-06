import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        m = jnp.max(x, axis=-1, keepdims=True)
        s = jnp.sum(jnp.exp(x - m), axis=-1, keepdims=True)
        o_ref[...] = x - m - jnp.log(s)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((320, 256), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((320, 256), lambda i: (0, 0)),
        grid=(1,),
    )(x)
