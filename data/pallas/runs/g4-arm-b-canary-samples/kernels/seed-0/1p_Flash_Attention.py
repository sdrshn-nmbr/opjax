def workload(query, key, value):
    B, H, S, D = query.shape
    BLOCK_M = 64
    BLOCK_N = 64
    # Define kernel with captured constants? But kernel needs to be a pure function.
    # We can define kernel inside workload to capture BLOCK_M, BLOCK_N, S, D.
    # However, Pallas kernel must be traceable. Defining inside is okay.
    def kernel(q_ref, k_ref, v_ref, o_ref):
        ...
    # Call pallas_call
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        in_specs=[
            pl.BlockSpec(lambda b, h, m: (b, h, m * BLOCK_M, 0), (1, 1, BLOCK_M, D)),
            pl.BlockSpec(lambda b, h, m: (b, h, 0, 0), (1, 1, S, D)),
            pl.BlockSpec(lambda b, h, m: (b, h, 0, 0), (1, 1, S, D)),
        ],
        out_specs=pl.BlockSpec(lambda b, h, m: (b, h, m * BLOCK_M, 0), (1, 1, BLOCK_M, D)),
        grid=(B, H, S // BLOCK_M),
        interpret=False,
    )(query, key, value)
    return out
