import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    x, = inputs

    def kernel(x_ref, o_ref):
        x = x_ref[...]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        o_ref[...] = (x - mean) * jax.lax.rsqrt(var + 1e-5)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        grid=(),
        in_specs=[pl.BlockSpec((320, 512), lambda: (0, 0))],
        out_specs=pl.BlockSpec((320, 512), lambda: (0, 0)),
    )(x)
