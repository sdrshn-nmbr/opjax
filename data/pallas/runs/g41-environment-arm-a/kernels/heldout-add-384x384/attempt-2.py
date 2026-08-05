def workload(*inputs):
    x, y = inputs
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((384, 384), jnp.float32),
        in_specs=[
            pl.BlockSpec((384, 384), lambda i, j: (0, 0)),
            pl.BlockSpec((384, 384), lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((384, 384), lambda i, j: (0, 0)),
        grid=(1, 1),
        interpret=False,
    )(x, y)
