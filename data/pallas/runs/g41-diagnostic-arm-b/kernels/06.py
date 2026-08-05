import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def rmsnorm_kernel(x_ref, y_ref):
        # Load the full (1, 128) block for this row
        x = x_ref[:, :]
        sq = x * x
        sum_sq = jnp.sum(sq, axis=-1, keepdims=True)
        mean_sq = sum_sq / 128.0
        eps = 1e-6
        rms = jnp.sqrt(mean_sq + eps)
        y_ref[:, :] = x / rms
    y = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(lambda i: (i, 0), (1, 128))],
        out_specs=pl.BlockSpec(lambda i: (i, 0), (1, 128)),
        grid=(256,),
    )(x)
    return y
