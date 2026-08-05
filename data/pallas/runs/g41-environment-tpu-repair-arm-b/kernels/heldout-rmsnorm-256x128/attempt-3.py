import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]

    def rmsnorm_kernel(x_ref, y_ref):
        x = x_ref[:, :]
        sq = x * x
        sum_sq = jnp.sum(sq, axis=-1, keepdims=True)
        mean_sq = sum_sq / 128.0
        rms = jnp.sqrt(mean_sq + 1e-6)
        y_ref[:, :] = x / rms

    y = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i * 8, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i * 8, 0)),
        grid=(32,),
    )(x)
    return y
