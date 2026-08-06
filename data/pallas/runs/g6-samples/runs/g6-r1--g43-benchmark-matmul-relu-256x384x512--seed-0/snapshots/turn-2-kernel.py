import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(lhs, rhs):
    def kernel(lhs_ref, rhs_ref, out_ref):
        pl.dot(lhs_ref, rhs_ref, out_ref, allow_tf32=False)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 512), jnp.float32),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((256, 384), lambda i: (0, 0)),
            pl.BlockSpec((384, 512), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((256, 512), lambda i: (0, 0)),
    )(lhs, rhs)
