import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

INPUT_SHAPE = (256, 512)
OUTPUT_SHAPE = (512, 256)

def _kernel(x_ref, o_ref):
    o_ref[...] = jnp.transpose(jnp.square(x_ref[...]))

def workload(x):
    input_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    output_spec = pl.BlockSpec((128, 128), lambda i, j: (j, i))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(OUTPUT_SHAPE, jnp.float32),
        grid=(INPUT_SHAPE[0] // 128, INPUT_SHAPE[1] // 128),
        in_specs=(input_spec,),
        out_specs=output_spec,
    )(x)
