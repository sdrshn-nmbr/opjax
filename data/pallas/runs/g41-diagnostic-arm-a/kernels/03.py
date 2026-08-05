<|message_model|><|content_thinking|>We need to implement an authentic normal-lowering JAX Pallas kernel for row-sum-64x128 operation. Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by operation name. The callable must accept inputs with shapes [[64, 128]] and dtypes ['float32']. Must match independent semantic oracle at full declared shapes. Do not use interpret mode or plain-JAX fallback. Return exactly one self-contained Python module defining workload(*inputs). Use current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second. Include every required import and complete kernel body. Do not return incomplete kernel. Return only Python module, no prose.

We need to write a Pallas kernel that does row sum: input shape (64,128), output shape (64,) or maybe (64,1)? The operation name "row-sum-64x128" suggests sum over columns, output shape (64,). But "output semantics defined independently by the operation name" - we need to infer. Usually row-sum yields shape (64,). Let's assume output is (64,). But could be (64,1). Let's think: The problem says "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." The operation name is "row-sum-64x128". So likely output is [64] (one per row). We need to match semantic oracle. We should define workload(*inputs) that takes a single input array of shape (64,128) float32 and returns sum over axis=1, shape (64,). Use pallas_call.

We need to use pl.BlockSpec(block_shape, index_map). The kernel should be normal-lowering, i.e., use pallas_call with grid, and inside kernel use pl.load, pl.store, etc.

Since input is 64x128, we can process each row as a block. For simplicity, we can have grid = (64,) and each block processes a row of 128 elements. But block_shape for input could be (1, 128) and output block_shape (1,). Or we could process multiple rows per block. Let's do grid = (64,) with block_shape (1, 128) for input and (1,) for output.

But we need to use pl.BlockSpec correctly: pl.BlockSpec(block_shape, index_map). The index_map is a function that takes grid indices and returns start indices.

So:

def workload(*inputs):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    import jax.experimental.pallas as pl

    x = inputs[0]

    def kernel(x_ref, o_ref):
        # x_ref is (1,128) block
        # o_ref is (1,) block
        # compute sum over last axis
        # We can load full block and sum
        val = pl.load(x_ref, (pl.dslice(0), pl.dslice(0)))  # but need to load whole block? Actually x_ref is already a block.
        # In Pallas, refs are block references; we can do arithmetic directly on them? Usually we load into a scalar or array.
        # We can do: row = pl.load(x_ref, (slice(None), slice(None))) but pl.load expects indices.
        # Actually in Pallas, you can do: val = x_ref[...]? Not exactly.
        # Let's use pl.load with (pl.dslice(0), pl.dslice(0))? That might load a scalar.
        # Better: use jnp.sum on the loaded array.
        # We can load the whole block: val = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))? But block_shape is (1,128), so indices are relative to block.
        # Actually pl.load(ref, idx) where idx is tuple of slices or ints. If we pass (slice(None), slice(None)) it loads the whole block.
        # Let's do: val = pl.load(x_ref, (slice(None), slice(None)))
        # Then sum: s = jnp.sum(val, axis=1, keepdims=True) or axis=-1.
        # Then store: pl.store(o_ref, (slice(None),), s) but o_ref is 1D block (1,). Actually block_shape for output is (1,). So index is (slice(None),).
        # Wait, if output is (64,), block_shape (1,) means each block is 1 element. So store at (slice(None),) is fine.
        # But s should be shape (1,). So s = jnp.sum(val, axis=-1, keepdims=True) gives (1,1) if val is (1,128). Actually val is (1,128). Sum over axis=-1 gives (1,). Good.
        # So s = jnp.sum(val, axis=-1)
        # Then pl.store(o_ref, (slice(None),), s) but s is scalar? Actually s is array of shape (1,). We can store with (slice(None),).
        # Let's do: pl.store(o_ref, (pl.dslice(0),), s[0])? But s is (1,). We can just do pl.store(o_ref, (slice(None),), s) if pl.store supports array assignment.
        # Actually pl.store(ref, idx, value) where value is scalar or array matching the slice.
        # If idx is (slice(None),) and value is array of shape (1,), it should work.
        # Alternatively, we can compute directly without load: use pl.dot? No.
        # Let's just load and sum.

    # Define block specs
    block_shape_in = (1, 128)
    block_shape_out = (1,)

    def index_map_in(i, j):
        return (i, 0)

    def index_map_out(i, j):
        return (i,)

    # But grid is 1D: (64,). So index_map takes one argument? Actually index_map is called with grid indices. If grid is (64,), then index_map takes (i,). But pl.BlockSpec expects index_map to take the grid indices as arguments. If grid is 1D, index_map takes one arg.
    # However, sometimes index_map takes (i, j) even for 1D grid? Actually the number of arguments matches grid dimensions.
    # So for grid=(64,), index_map_in(i) -> (i, 0). index_map_out(i) -> (i,).

    # But the problem says: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second. So we must use that order.

    # Let's define:
    in_specs = pl.BlockSpec((1, 128), lambda i: (i, 0))
    out_specs = pl.BlockSpec((1,), lambda i: (i,))

    # Then pallas_call
    kernel_fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        in_specs=[in_specs],
        out_specs=[out_specs],
        grid=(64,),
    )

    return kernel_fn(x)

Wait, need to check API: pl.pallas_call takes kernel, out_shape, in_specs, out_specs, grid, etc. The out_shape can be a tuple of ShapeDtypeStruct or just a shape? Usually out_shape=jax.ShapeDtypeStruct(shape, dtype). We need to import jax.

Also, the kernel function signature: def kernel(x_ref, o_ref): ... But with in_specs and out_specs, the kernel receives refs in order: first inputs, then outputs. Actually in pallas_call, the kernel is called with input refs and output refs. The order is: for each input spec, a ref; for each output spec, a ref. So if we have one input and one output, kernel(x_ref, o_ref) is correct.

But we need to be careful: The kernel should not use interpret mode. We are using pallas_call which lowers to GPU/TPU.

However, the problem says "authentic normal-lowering JAX Pallas kernel". It might require using pl.program_id, pl.load, pl.store, etc.

Let's refine the kernel to be robust.

In Pallas, inside kernel, we can access the block via x_ref. To load the entire block, we can do:

val = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))

But since block_shape is (1,128), the indices are relative to the block. Actually pl.load takes indices into the full array? No, pl.load takes indices relative to the block? Let's recall Pallas semantics.

In Pallas, a reference (ref) represents a block of the array. When you do pl.load(ref, idx), idx is a tuple of slices/ints that index into the block? Actually I think pl.load(ref, idx) indexes into the underlying array, but the ref is a window. Wait, need to be precise.

In Pallas, refs are "memory references" that point to a block of the array. The kernel operates on these blocks. When you load from a ref, you can load a sub-block or scalar. The indices are relative to the block? Let's check documentation.

From JAX Pallas docs: "pl.load(ref, idx) loads a slice of the array referenced by ref at index idx." The index is relative to the array, but since ref is a block, maybe it's relative to the block? Actually I think the index is relative to the block's origin in the full array? No, the ref is a view; indexing into it is relative to the view.

Wait, let's think: If we have a block spec that maps grid index i to (i, 0) with block_shape (1,128), then the ref passed to kernel is a 1x128 array (the block). If we do pl.load(x_ref, (0, 0)), we get the first element of the block. If we do pl.load(x_ref, (slice(None), slice(None))), we get the whole block. So indexing is relative to the block.

Thus we can do:

val = pl.load(x_ref, (slice(None), slice(None)))

Then sum:

s = jnp.sum(val, axis=-1)  # shape (1,)

Then store:

pl.store(o_ref, (slice(None),), s)

But is pl.store expecting the index to be relative to the block? Yes. So (slice(None),) selects the whole output block (size 1).

However, some versions require using pl.dslice for dynamic slices. But static slices should work.

Let's check if jnp.sum is allowed inside kernel. Yes, we can use JAX operations inside the kernel; they are traced.

But we must be careful: The kernel is compiled with pallas, and using jnp.sum is fine.

Alternatively, we can compute sum manually with a loop, but jnp.sum is simpler.

Now, about the output semantics: If the oracle expects output shape (64, 128) or something else? The operation name is "row-sum-64x128". Usually row sum reduces over columns, giving (64,). But sometimes "row-sum" could mean sum of each row, output (64,). Let's assume that.

But what if the oracle expects output shape (64, 1)? The problem says "output semantics defined independently by the operation name". It might be that the operation is "row-sum" and the input is 64x128, so output is 64. Let's stick with (64,).

However, to be safe, maybe we should check if there's any standard benchmark. The problem mentions "authentic normal-lowering JAX Pallas kernel for the row-sum-64x128 operation". This sounds like a benchmark from a suite (maybe "pallas-bench" or "jax-pallas-benchmarks"). The operation might be defined as: input [[64,128]] -> output [[64]] (sum over axis 1). Or maybe output [[64,1]]? Let's think.

In many ML contexts, row sum of a matrix yields a vector of length 64. If they want to keep dimensions, they might do keepdims=True. But the problem says "output semantics defined independently by the operation name". The operation name is "row-sum-64x128". It doesn't specify keepdims. Usually "sum" without keepdims reduces dimension. So output is (64,).

Let's write the module accordingly.

We need to include every required import: import jax, import jax.numpy as jnp, from jax.experimental import pallas as pl.

We need to define workload(*inputs). It should accept inputs (a tuple of arrays). It should return the result.

We must not use interpret mode or plain-JAX fallback. So we must call pl.pallas_call.

One issue: pallas_call requires a backend? By default it uses the default backend (GPU/TPU). If running on CPU, it might fall back to interpret? Actually Pallas on CPU is supported via a CPU backend? Not always. But the problem says "authentic normal-lowering", so we assume it runs on a supported backend.

Let's write the code.

Wait, the problem says: "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So we must use pl.BlockSpec((1,128), lambda i: (i,0)) etc.

Also, "Include every required import and a complete kernel body. Do not return an incomplete kernel."

So the module should be:

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        # Load the full block
        val = pl.load(x_ref, (slice(None), slice(None)))
        # Sum over the last axis (columns)
        s = jnp.sum(val, axis=-1)
        # Store to output block
        pl.store(o_ref, (slice(None),), s)
    # Block specs
    in_spec = pl.BlockSpec((1, 128), lambda i: (i, 0))
    out_spec = pl.BlockSpec((1,), lambda i: (i,))
    # Call
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        in_specs=[in_spec],
        out_specs=[out_spec],
        grid=(64,),
    )(x)

Wait, does pl.pallas_call return a callable? Yes, it returns a function that takes inputs and returns outputs. So we can do:

f = pl.pallas_call(...)
return f(x)

Or directly return pl.pallas_call(...)(x).

But we need to make sure that the lambda for index_map is correct. For grid=(64,), the index_map receives one argument i (an integer scalar representing the grid index). It should return a tuple of ints: (i, 0). That is correct.

However, some versions of Pallas require index_map to be a function that takes the grid indices as separate arguments, but also might pass them as a tuple? Actually the docs say: "index_map: a callable that takes the grid indices and returns the start indices of the block." So lambda i: (i, 0) is fine.

But wait: The block_shape for input is (1,128). The index_map returns (i, 0). That means for grid index i, the block starts at row i, col 0. That covers all 64 rows.

For output, block_shape (1,), index_map lambda i: (i,). That covers all 64 outputs.

Now, is there any issue with using slice(None) inside pl.load? I think pl.load supports Python slices. But sometimes it expects pl.dslice for dynamic slices. For static slices, slice is okay.

Let's check Pallas source: pl.load(ref, idx) where idx is a tuple of int, slice, or pl.dslice. So slice(None) is okay.

But we need to be careful: The kernel might be compiled with XLA, and using Python slices inside pl.load might be okay.

Alternatively, we can avoid pl.load and just do arithmetic on the ref directly? Actually refs support indexing with slices? Not directly. We must use pl.load.

Another approach: Use pl.dot? No.

Let's consider if we need to handle the case where the input is not exactly 64x128? The problem says "must accept inputs with shapes [[64, 128]]". So we assume exact.

Now, one more thing: The problem says "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." It might be that the output is also [[64, 128]] but with row sums broadcast? No, "row-sum" clearly reduces.

But let's double-check: Could "row-sum" mean sum of rows, i.e., sum over axis 0, giving (128,)? Actually "row sum" usually means sum of elements in each row, which is over columns (axis 1). "Column sum" is over rows (axis 0). So output is (64,).

Let's write the module.

Wait, the problem says "Return exactly one self-contained Python module defining workload(*inputs)." So we just output the code.

We should also include a check or just rely on pallas_call.

One potential issue: If the environment doesn't have GPU/TPU, pallas_call might raise an error. But the problem explicitly asks for authentic normal-lowering, so it's okay.

Let's refine the kernel to be more explicit and avoid any potential issues with slice indexing.

We can load with explicit indices:

val = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))

But pl.dslice(start, size) creates a dynamic slice. Since block_shape is (1,128), using pl.dslice(0,1) and pl.dslice(0,128) loads the whole block. That might be more robust.

Then sum:

s = jnp.sum(val, axis=-1)  # val is (1,128) -> s is (1,)

Then store:

pl.store(o_ref, (pl.dslice(0, 1),), s)

Wait, o_ref is (1,). So pl.dslice(0,1) selects the whole block. And s is (1,). That should work.

But does pl.store accept pl.dslice? Yes.

Let's use pl.dslice for clarity.

So kernel:

def kernel(x_ref, o_ref):
    val = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))
    s = jnp.sum(val, axis=-1)
    pl.store(o_ref, (pl.dslice(0, 1),), s)

Wait, pl.load with pl.dslice(0,1) and pl.dslice(0,128) might load a (1,128) array. Good.

But is pl.dslice needed? It might be fine.

However, some versions of Pallas require that the indices passed to pl.load are within the block. Since block is (1,128), pl.dslice(0,1) is okay.

Let's check if pl.dslice is imported from pl. Yes, pl.dslice.

Now, about the output: If s is shape (1,), and we store with pl.dslice(0,1), it expects a scalar or array of shape (1,). It should work.

Alternatively, we can compute sum without load by using pl.dot with a vector of ones. But load is simpler.

Let's consider if we can do it with a single scalar load and accumulate in a loop. That would be more "kernel-like". But jnp.sum is fine.

Wait, the problem says "authentic normal-lowering JAX Pallas kernel". It might be that using jnp.sum inside the kernel is okay because it lowers to XLA. But sometimes they want manual accumulation to show kernel body. However, jnp.sum is a JAX primitive that lowers correctly.

But to be safe and avoid any issues with jnp.sum inside pallas (though it should work), we can manually accumulate:

def kernel(x_ref, o_ref):
    # Initialize accumulator
    acc = jnp.zeros((1,), dtype=jnp.float32)
    # Loop over columns? But loop over 128 is okay.
    # Actually we can load the whole block and sum with jnp.sum.
    val = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))
    acc = jnp.sum(val, axis=-1)
    pl.store(o_ref, (pl.dslice(0, 1),), acc)

This is fine.

But wait: The kernel receives refs that are already blocks. If we load with pl.dslice(0,1) and pl.dslice(0,128), we are loading the entire block. That's okay.

However, there is a subtlety: In Pallas, the block_shape is the shape of the block passed to the kernel. So x_ref has shape (1,128). If we do pl.load(x_ref, (pl.dslice(0,1), pl.dslice(0,128))), we are loading a sub-block of size (1,128) from the block, which is the whole block. Good.

Now, what about the output? o_ref has shape (1,). We store with pl.dslice(0,1).

One more thing: The problem says "The callable must accept inputs with shapes [[64, 128]] and dtypes ['float32']." So we should not enforce shape inside, but pallas_call will handle it.

Let's write the final module.

Wait, I need to make sure that the import of pl is correct: from jax.experimental import pallas as pl.

Also, we might need to import jax.experimental.pallas as pl.

Now, is there any requirement to use pl.program_id? Not necessarily.

Let's check the exact API for pl.pallas_call:

pl.pallas_call(
    kernel,
    out_shape,
    in_specs=None,
    out_specs=None,
    grid=None,
    ...
)

Yes.

So the code:

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        # Load full 1x128 block
        block = pl.load(x_ref, (pl.dslice(0, 1), pl.dslice(0, 128)))
        # Sum over columns
        row_sum = jnp.sum(block, axis=-1)
        # Store 1-element block
        pl.store(o_ref, (pl.dslice(0, 1),), row_sum)
    in_spec = pl.BlockSpec((1, 128), lambda i: (i, 0))
    out_spec = pl.BlockSpec((1,), lambda i: (i,))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        in_specs=[in_spec],
        out_specs=[out_spec],
        grid=(64,),
    )(x)

Wait, does pl.pallas_call require out_specs to be a list of BlockSpec? Yes.

One potential issue: The lambda for index_map might need to accept a tuple of grid indices? Actually for a 1D grid, the index_map is called with a single integer (or scalar array). So lambda i: (i, 0) is correct.

But some versions pass the grid index as a tuple (i,) even for 1D? Let's check docs.

From JAX docs: "The index_map is a callable that takes the grid indices and returns the start indices of the block." It says "grid indices" plural, but for 1D grid it's a single index. Usually it's called as index_map(*grid_indices) or index_map(grid_indices)? Actually looking at examples:

pl.BlockSpec((128, 128), lambda i, j: (i, j))

So for 2D grid, it takes two args. For 1D grid, it takes one arg.

Thus lambda i: (i, 0) is correct.

But wait: The index_map might be called with scalar arrays, not Python ints. But returning (i, 0) where i is a scalar array is okay; it will be used to compute offsets.

Now, is there any issue with using jnp.sum inside the kernel? It should lower to XLA.

But let's think: The kernel is executed on accelerator. jnp.sum is a JAX primitive that is compiled by XLA. Inside pallas, the kernel is traced with abstract values representing the block shapes. So jnp.sum should work.

However, some Pallas kernels require using pl.dot or manual loops because some JAX operations are not supported inside pallas? Actually most are supported because pallas uses JAX tracing to build a JAXPR, which is then compiled by XLA. So jnp.sum is fine.

Let's consider if we need to specify the dtype of the accumulator. jnp.sum on float32 yields float32.

Now, one more subtlety: The problem says "normal-lowering". That means the kernel should lower to a standard XLA computation, not use interpret mode. Using pl.pallas_call does that.

But is there any requirement to use pl.program_id? Not necessarily.

Let's check if the output shape should be (64,) or (64, 1). The operation name "row-sum-64x128" suggests input is 64x128, output is 64. Let's assume that.

But to be absolutely safe, maybe we should output (64, 1) because sometimes row sum is kept as a column vector? However, "row-sum" as an operation in linear algebra usually yields a vector of length 64. If the oracle expects (64, 1), our answer would be wrong. How to decide?

Let's think about the phrase: "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." The operation name is "row-sum-64x128". If we parse "row-sum", it could mean "sum of rows" (i.e., sum over axis 0) giving (128,). But "row-sum" is ambiguous. Actually "row sum" of a matrix is the sum of elements in each row, which is a vector of length equal to number of rows (64). "Column sum" is sum of elements in each column, vector of length 128.

But "row-sum-64x128" could mean the operation is "row sum" applied to a 64x128 matrix. So output is 64.

Alternatively, it could mean "sum along rows", i.e., reduce rows, output 128. But "sum along rows" is usually "sum over axis 0". The phrase "row-sum" is more likely sum of each row.

Let's check common naming conventions in benchmarks. For example, in the "Pallas" or "Triton" benchmarks, there are operations like "matmul", "softmax", "layernorm", "row-sum". Usually "row-sum" means sum across columns for each row.

Thus output (64,) is correct.

But what if the benchmark expects output shape (64, 128) with each row being the sum of that row? No, that doesn't make sense.

Let's consider if the output should be (64, 1) to preserve dimensions for broadcasting. Some frameworks do that. But the problem says "output semantics defined independently by the operation name". It doesn't mention keepdims. So default is no keepdims.

Thus we will output (64,).

However, to be safe, we could make the output shape (64,) and also handle (64,1) if needed? No, we must match the oracle exactly.

Let's think if there's any clue in the problem statement: "Compute sum for full-shape inputs [[64, 128]] with output semantics defined independently by the operation name." The input is given as [[64, 128]] (a list of shapes). The output semantics are defined by the operation name. The operation name is "row-sum-64x128". If the output were (64,1), the operation might be named "row-sum-64x128-keepdims" or something. So (64,) is likely.

Let's proceed with (64,).

One more check: The problem says "The callable must accept inputs with shapes [[64, 128]] and dtypes ['float32']." It says inputs plural, but only one input. So workload(*inputs) unpacks to one array.

Now, let's consider if we need to handle multiple inputs? No.

Let's write the final code.

Wait, I want to double-check the pl.BlockSpec usage. The problem explicitly says: "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So we must write pl.BlockSpec((1, 128), lambda i: (i, 0)). Good.

Now, is there any import missing? We need jax, jax.numpy, jax.experimental.pallas.

Let's write the module.

One more thing: The kernel might need to be defined outside workload? It can be inside.

Let's produce the final answer.

Wait, I should consider if using pl.load with pl.dslice is correct. Actually, pl.dslice is used for dynamic slicing where the start index is dynamic (depends on program_id). But here we are loading the whole block, so the start is always 0 relative to the block. We could just use slice(None). But pl.dslice(0, 1) is also okay because 0 is static.

However, some versions of Pallas might require that pl.load indices are static slices or dynamic slices, but pl.dslice with constant start is okay.

To be safe, let's use slice(None) because it's simpler and definitely supported.

So:

val = pl.load(x_ref, (slice(None), slice(None)))

But does pl.load accept slice(None) for both dimensions? I think yes.

Let's check: In Pallas, pl.load(ref, idx) where idx is a tuple of int, slice, or pl.dslice. So slice(None) is okay.

Then:

s = jnp.sum(val, axis=-1)

pl.store(o_ref, (slice(None),), s)

This is clean.

But wait: If val is (1,128), jnp.sum(val, axis=-1) gives (1,). Good.

If o_ref is (1,), pl.store(o_ref, (slice(None),), s) stores s into the whole block.

This should work.

Let's write it.

One final check: The problem says "Do not use interpret mode or a plain-JAX fallback." We are using pl.pallas_call, which is correct.

We should not include any extra text.

So the final output is just the Python module.

Let's write it.<|end_message|><|message_model|><|content_text|>import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]

    def kernel(x_ref, o_ref):
        val = pl.load(x_ref, (slice(None), slice(None)))
        s = jnp.sum(val, axis=-1)
        pl.store(o_ref, (slice(None),), s)

    in_spec = pl.BlockSpec((1, 128), lambda i: (i, 0))
    out_spec = pl.BlockSpec((1,), lambda i: (i,))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64,), jnp.float32),
        in_specs=[in_spec],
        out_specs=[out_spec],
        grid=(64,),
    )(x)<|end_message|><|content_model_end_sampling|>