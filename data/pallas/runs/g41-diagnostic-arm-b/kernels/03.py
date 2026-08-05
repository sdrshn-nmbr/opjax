<|message_model|><|content_thinking|>We need to implement an authentic normal-lowering JAX Pallas kernel for row-sum-64x128 operation. Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by operation name. The callable must accept inputs with shapes [[64, 128]] and dtypes ['float32']. Must match independent semantic oracle at full declared shapes. Do not use interpret mode or plain-JAX fallback. Return exactly one self-contained Python module defining workload(*inputs). Use current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second. Include every required import and complete kernel body. Do not return incomplete kernel. Return only Python module, no prose.

We need to write a Pallas kernel that does row sum: input shape (64,128), output shape (64,) or maybe (64,1)? The operation name "row-sum-64x128" suggests sum over columns, output shape (64,). But "output semantics defined independently by the operation name" - we need to infer. Usually row sum of matrix yields vector of length 64. Let's assume output shape (64,). But we need to be careful: The callable workload(*inputs) should probably return the result. We need to define a pallas_call with grid, block specs.

Since input is 64x128 float32, we can process each row as a block? But Pallas operates on blocks. We can set block_shape for input as (1, 128) and output as (1,) with index_map mapping (i,j) -> (i,0) maybe. Or block_shape (64, 128) and do reduction inside kernel? But Pallas kernel is per block; if we use block_shape (64,128) then grid is (1,1) and kernel sums over the whole block and writes to output block of shape (64,). That is simpler.

Let's do: grid = (1,1). Input block spec: pl.BlockSpec((64,128), lambda i,j: (0,0)). Output block spec: pl.BlockSpec((64,), lambda i,j: (0,)). But output is 1D; block_shape (64,). Index map returns (0,). Actually for 1D output, index_map should return (i,) where i is grid index. Since grid is (1,1), return (0,). But we need to match semantics: output shape (64,). Let's define output shape as (64,). The kernel receives input_ref of shape (64,128) and output_ref of shape (64,). It computes sum over axis 1.

But Pallas kernel functions receive refs; we can use jnp.sum over axis. However inside kernel we should use lax or jnp? In Pallas, we can use jnp operations on refs by indexing? Actually in Pallas kernel, the arguments are pl.MemoryRef or similar; we can read via input_ref[...] which returns a JAX array (traced). We can do jnp.sum(input_ref[...], axis=1) and write to output_ref[...]. That should work.

But we need to ensure normal lowering, not interpret mode. We just define pallas_call with out_shape=jax.ShapeDtypeStruct((64,), jnp.float32). Actually we can infer from inputs.

Let's design workload(*inputs):

def workload(*inputs):
    x, = inputs
    # x shape (64,128)
    def kernel(x_ref, o_ref):
        # x_ref shape (64,128)
        # o_ref shape (64,)
        o_ref[...] = jnp.sum(x_ref[...], axis=1)
    # But we need to specify block specs.
    # Since we want full block, grid=(1,1)
    # Input block spec: (64,128), index_map=lambda i,j: (0,0)
    # Output block spec: (64,), index_map=lambda i,j: (0,)
    # Actually for 1D output, block_shape is (64,) and index_map returns (0,).
    # But index_map takes grid indices; if grid is (1,1), it gets (i,j). We return (0,).
    # Wait for 1D array, the index is a tuple of length 1? In BlockSpec, index_map should return a tuple of ints of length equal to array ndim? Actually index_map returns the starting indices for each dimension. For 1D array, return (0,). But if we pass lambda i,j: (0,) it's okay.
    # However the grid is 2D (1,1). The lambda receives two args.
    # Let's use grid=(1,1).
    # But maybe better to use grid=(64,) and block_shape=(1,128) for input and (1,) for output. That is more "authentic" and uses normal lowering with multiple blocks. But either works.
    # The prompt says "Implement an authentic normal-lowering JAX Pallas kernel for the row-sum-64x128 operation." It might expect a kernel that processes the whole thing in one block or multiple. But "authentic" suggests using pallas_call with proper block specs.
    # Let's do grid=(64,) with block_shape=(1,128) for input and (1,) for output. Then index_map for input: lambda i: (i, 0). For output: lambda i: (i,). This is clean.
    # But the API: pl.BlockSpec(block_shape, index_map). So block_shape first, index_map second.
    # We need to import jax, jax.numpy as jnp, jax.experimental.pallas as pl.
    # The kernel function signature: def kernel(x_ref, o_ref): ...
    # We need to use pl.pallas_call.
    # Let's write:
    # out_shape = jax.ShapeDtypeStruct((64,), jnp.float32)
    # But we can also derive from x: out_shape = jax.ShapeDtypeStruct((x.shape[0],), x.dtype)
    # Actually if input is (64,128), output is (64,).
    # Let's implement.
    # One issue: In Pallas, when using grid with 1D, the index_map lambda takes one argument (the grid index). So lambda i: (i, 0) for input (1,128) block. For output (1,) block: lambda i: (i,).
    # Let's do that.
    # Kernel:
    # def kernel(x_ref, o_ref):
    #     # x_ref is (1,128)
    #     # o_ref is (1,)
    #     o_ref[...] = jnp.sum(x_ref[...], axis=1)
    # Wait axis=1 sums over the second dimension (128), yielding shape (1,). Good.
    # But we need to be careful: jnp.sum(x_ref[...], axis=1) returns shape (1,). Then assign to o_ref[...] which is (1,). Good.
    # Actually if block_shape is (1,128), x_ref[...] is (1,128). Sum over axis=1 gives (1,). Good.
    # If we use block_shape (64,128), sum over axis=1 gives (64,). Good.
    # Let's choose grid=(1,1) with block_shape (64,128) and (64,). That is simpler and avoids loop.
    # But is it "authentic"? Yes.
    # However some Pallas versions require that block_shape divides the array shape? Not necessarily, but it's okay if block_shape equals array shape.
    # Let's do grid=(1,1).
    # Input BlockSpec: pl.BlockSpec((64, 128), lambda i, j: (0, 0))
    # Output BlockSpec: pl.BlockSpec((64,), lambda i, j: (0,))
    # Wait for 1D output, index_map should return tuple of length 1? Actually the array is 1D, so index_map should return (0,) not (0,0). But lambda i,j: (0,) is okay because it ignores j and returns a 1-tuple.
    # But does BlockSpec expect index_map to return a tuple of length equal to ndim? Yes. So (0,) is correct.
    # However the lambda takes two arguments because grid is 2D. That's fine.
    # Let's check: pl.BlockSpec((64,), lambda i, j: (0,)). That should work.
    # But some versions might require index_map to take exactly as many args as grid dimensions. So lambda i, j: (0,) is okay.
    # Alternatively, we can make grid 1D: grid=(1,). Then index_map takes one arg.
    # For input: pl.BlockSpec((64,128), lambda i: (0,0)). For output: pl.BlockSpec((64,), lambda i: (0,)).
    # That is cleaner.
    # Let's do grid=(1,).
    # Kernel:
    # def kernel(x_ref, o_ref):
    #     o_ref[...] = jnp.sum(x_ref[...], axis=1)
    # This is perfect.
    # But wait: The output semantics: "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." The operation name is "row-sum-64x128". So output is row sums: shape (64,). Good.
    # We must not use interpret mode or plain-JAX fallback. So we must call pl.pallas_call with interpret=False? Actually default is False? In some versions, interpret=False is default. But to be safe, we can pass interpret=False explicitly? The prompt says "Do not use interpret mode". So we should not pass interpret=True. We can omit it or pass False.
    # We need to return exactly one self-contained Python module defining workload(*inputs).
    # So the module should have imports and define workload.
    # Let's write:
    # import jax
    # import jax.numpy as jnp
    # from jax.experimental import pallas as pl
    # def workload(*inputs):
    #     x = inputs[0]
    #     def kernel(x_ref, o_ref):
    #         o_ref[...] = jnp.sum(x_ref[...], axis=1)
    #     return pl.pallas_call(
    #         kernel,
    #         out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
    #         grid=(1,),
    #         in_specs=[pl.BlockSpec((x.shape[0], x.shape[1]), lambda i: (0, 0))],
    #         out_specs=pl.BlockSpec((x.shape[0],), lambda i: (0,)),
    #     )(x)
    # Wait: The block_shape for input is (64,128). But if x is always (64,128), we can hardcode or use x.shape. Hardcoding is okay but using x.shape is more robust.
    # However the prompt says "The callable must accept inputs with shapes [[64, 128]] and dtypes ['float32']." So we can assume that.
    # But we need to make sure the kernel is correct.
    # One subtlety: In Pallas, when using jnp.sum inside kernel, it might not be allowed to use arbitrary jnp functions? Actually Pallas supports a subset of JAX primitives. jnp.sum is supported.
    # Another subtlety: The output block spec: pl.BlockSpec((64,), lambda i: (0,)). But the lambda takes one argument because grid is (1,). Good.
    # Wait: The in_specs is a list of BlockSpec. The out_specs is a single BlockSpec or list? In pl.pallas_call, out_specs can be a single BlockSpec or sequence. If out_shape is a single array, out_specs can be a single BlockSpec. Let's check API: pl.pallas_call(kernel, out_shape, grid, in_specs, out_specs, ...). out_specs is optional; if not provided, it uses default. But we should provide it.
    # Actually the signature: pallas_call(call, out_shape, grid_spec, in_specs=..., out_specs=..., ...). Wait need to check exact current API.
    # The prompt explicitly says: "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second."
    # It doesn't specify pallas_call signature exactly, but we need to use current API.
    # In recent JAX (0.4.30+), pallas_call is:
    # pl.pallas_call(
    #     f,
    #     out_shape,
    #     grid,
    #     in_specs=...,
    #     out_specs=...,
    #     interpret=...,
    #     ...
    # )
    # Actually I think the arguments are: pallas_call(call, out_shape, grid_spec, in_specs=..., out_specs=..., ...). Wait "grid_spec" might be a tuple or GridSpec.
    # Let's check memory: In jax.experimental.pallas, pallas_call signature is:
    # def pallas_call(
    #     call,
    #     out_shape,
    #     grid_spec,
    #     in_specs=None,
    #     out_specs=None,
    #     interpret=False,
    #     ...
    # ):
    # Where grid_spec can be a tuple (like (1,)) or a GridSpec.
    # Actually I recall that in some versions, the second argument is grid, not grid_spec. Let's be safe and use keyword arguments where possible.
    # The prompt says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." It doesn't restrict pallas_call usage.
    # We can write:
    # return pl.pallas_call(
    #     kernel,
    #     out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
    #     grid=(1,),
    #     in_specs=[pl.BlockSpec((64, 128), lambda i: (0, 0))],
    #     out_specs=pl.BlockSpec((64,), lambda i: (0,)),
    # )(x)
    # But is "grid" a keyword? I think yes, pallas_call accepts grid as a positional or keyword arg. Actually looking at source: def pallas_call(call, out_shape, grid_spec, ...). So the third positional arg is grid_spec. We can pass (1,) as third positional arg.
    # Let's write:
    # return pl.pallas_call(
    #     kernel,
    #     jax.ShapeDtypeStruct((64,), jnp.float32),
    #     (1,),
    #     in_specs=[pl.BlockSpec((64, 128), lambda i: (0, 0))],
    #     out_specs=pl.BlockSpec((64,), lambda i: (0,)),
    # )(x)
    # This should work.
    # But we need to be careful: The lambda for BlockSpec takes the grid indices. If grid is (1,), lambda i: (0,0) is fine.
    # However, some versions require index_map to be a function that takes the grid indices as separate args, not a tuple. So lambda i: ... is correct for 1D grid.
    # Let's verify with 2D grid: lambda i, j: ... . For 1D: lambda i: ... .
    # Good.
    # Now, is there any issue with using jnp.sum inside kernel? In Pallas, the kernel is compiled to a GPU/TPU kernel. jnp.sum is translated to a reduction. It should work.
    # But sometimes inside Pallas kernel, you need to use pl.program_id or pl.load? No, for simple operations, using jnp on refs is okay.
    # Actually in Pallas, the kernel receives refs that are "memory references". You can index them with slices or integers to get arrays. Doing x_ref[...] gives the whole block as a JAX array. Then jnp.sum works.
    # Let's write the complete module.
    # We must include every required import.
    # Imports: import jax, import jax.numpy as jnp, from jax.experimental import pallas as pl.
    # We might also need from jax import ShapeDtypeStruct? But jax.ShapeDtypeStruct is fine.
    # Let's write the module.
    # One more thing: The prompt says "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." It might mean that the operation is "row-sum-64x128" and the output is a scalar? No, "row-sum" implies sum per row. But "output semantics defined independently by the operation name" could mean that the operation name defines the output shape, which is (64,). Let's stick with (64,).
    # Wait, could it be that the operation is "row-sum" and the input is 64x128, so output is (64,). Yes.
    # Let's produce the final answer: just the Python code.
    # Double-check: The callable workload(*inputs) must accept inputs with shapes [[64,128]] and dtypes float32. It returns the result.
    # We should not use interpret mode. We won't pass interpret=True.
    # We should not use plain-JAX fallback. We are using pallas_call.
    # The kernel body is complete.
    # Let's write it.
    # One potential issue: The block_shape for output is (64,). But the output array is 1D. The index_map lambda i: (0,) returns the start index for the first dimension. That's correct.
    # Actually, does BlockSpec for 1D array expect index_map to return a tuple of length 1? Yes.
    # Let's write the code.
    # Wait: In some versions, pallas_call requires out_specs to be a sequence if out_shape is a sequence, but if out_shape is a single array, out_specs can be a single BlockSpec. Let's assume yes.
    # To be safe, we can pass out_specs=[pl.BlockSpec(...)]? Actually if out_shape is a single array, out_specs should be a single BlockSpec, not a list. But passing a list might also work if it treats it as sequence of specs for each output. Since there's one output, [spec] might be okay. But let's stick to single spec.
    # Let's check documentation: "out_specs: a sequence of BlockSpec or a single BlockSpec". Actually I think it can be either. But to be safe, we can pass a list: out_specs=[pl.BlockSpec((64,), lambda i: (0,))]. Wait if out_shape is a single array, passing a list of specs might be interpreted as specs for multiple outputs. That would be wrong. So pass single spec.
    # Actually looking at examples:
    # pl.pallas_call(
    #     kernel,
    #     out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    #     grid=(8,),
    #     in_specs=[pl.BlockSpec((1, 128), lambda i: (i, 0))],
    #     out_specs=pl.BlockSpec((1, 128), lambda i: (i, 0)),
    # )(x)
    # So out_specs is a single BlockSpec when one output.
    # Good.
    # Let's finalize.
    # Wait: The input block_shape is (64,128). The index_map lambda i: (0,0) ignores i. That's okay because grid is (1,). It just says always start at (0,0).
    # The kernel will process the whole input.
    # Let's write the code.
    # Actually, I want to make sure that the kernel uses the correct axis. If input block is (64,128), sum over axis=1 yields (64,). Good.
    # Code:
    # import jax
    # import jax.numpy as jnp
    # from jax.experimental import pallas as pl
    # def workload(*inputs):
    #     x = inputs[0]
    #     def kernel(x_ref, o_ref):
    #         o_ref[...] = jnp.sum(x_ref[...], axis=1)
    #     return pl.pallas_call(
    #         kernel,
    #         jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
    #         (1,),
    #         in_specs=[pl.BlockSpec((x.shape[0], x.shape[1]), lambda i: (0, 0))],
    #         out_specs=pl.BlockSpec((x.shape[0],), lambda i: (0,)),
    #     )(x)
    # This is clean.
    # But is x.shape[0] always 64? Yes per spec.
    # Let's hardcode to be explicit? Not necessary.
    # However, the prompt says "Compute sum for full-shape inputs [[64, 128]]". It might be okay to use x.shape.
    # Let's consider if there is any hidden requirement: "authentic normal-lowering JAX Pallas kernel". This might mean that the kernel should be lowered to a GPU kernel, not interpreted. Using pallas_call with default interpret=False achieves that.
    # One more thing: We might need to import jax.experimental.pallas as pl, but also maybe import jax.lax? Not needed.
    # Let's provide the module.
    # Wait: The prompt says "Return exactly one self-contained Python module defining workload(*inputs)." So the code should be a module. We just output the code block.
    # I will include a docstring? Not needed, but okay. Just code.
    # Let's write it.<|end_message|><|message_model|><|content_text|>import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.sum(x_ref[...], axis=1)
    return pl.pallas_call(
        kernel,
        jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        (1,),
        in_specs=[pl.BlockSpec((x.shape[0], x.shape[1]), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((x.shape[0],), lambda i: (0,)),
    )(x)<|end_message|><|content_model_end_sampling|>