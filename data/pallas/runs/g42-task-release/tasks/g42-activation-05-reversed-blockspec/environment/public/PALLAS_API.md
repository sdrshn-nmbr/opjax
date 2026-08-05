# Pallas task API

Edit `kernel.py`. It must define one complete `workload(*inputs)` implementation.
Use `pl.BlockSpec(block_shape, index_map)` in that order. The kernel must use a
reachable `pl.pallas_call`, must not use `interpret=True`, and must not include a
plain-JAX fallback. Run `python dev_check.py kernel.py` for public static feedback.
The final TPU verifier is separate and hidden.
