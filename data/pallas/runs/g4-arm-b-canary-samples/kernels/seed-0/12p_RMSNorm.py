import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

CONFIG = {
    'name': 'llama3_70b_rmsnorm',
    'model': 'Llama-3.1-70B',
    'operator': 'rms_norm',
    'batch': 8,
    'seq_len': 4096,
    'emb_dim': 8192,
    'epsilon': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    scale = jax.random.normal(k2, (D,), dtype=dtype) * 0.1 + 1.0
    return x, scale

def workload(*inputs):
    x, scale = inputs
    B, S, D = x.shape
    eps = CONFIG['epsilon']

    def kernel(x_ref, scale_ref, out_ref):
        x_local = x_ref[...]
        s_local = scale_ref[...]
        mean_sq = jnp.mean(jnp.square(x_local), axis=-1, keepdims=True)
        out_ref[...] = x_local * lax.rsqrt(mean_sq + eps) * s_local

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(lambda b, s: (b, s, 0), (1, 1, D)),
            pl.BlockSpec(lambda b, s: (0,), (D,)),
        ],
        out_specs=pl.BlockSpec(lambda b, s: (b, s, 0), (1, 1, D)),
        grid=(B, S),
    )(x, scale)
