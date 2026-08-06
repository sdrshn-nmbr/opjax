import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.maximum(x_ref[...], 0)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        grid=(),
        compiler_params=pl.CompilerParams(dimension_semantics=("parallel",)),
    )(x)
