import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        x = x_ref[...]
        mean_sq = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + 1e-5)
        o_ref[...] = x / rms.astype(x.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        grid=(),
        compiler_params=pl.CompilerParams(dimension_semantics=("parallel",)),
    )(x)
