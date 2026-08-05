<|message_model|><|content_thinking|>The user wants an authentic normal-lowering JAX Pallas kernel for an add-256x256 operation. The inputs are full-shape [[256, 256], [256, 256]] with dtypes float32, float32. Output semantics defined independently by operation name (add). Must use Pallas (pl) with BlockSpec(block_shape, index_map). Must not use interpret mode or plain-JAX fallback. Must return exactly one self-contained Python module defining workload(*inputs). Must include every required import and complete kernel body.

We need to write a Pallas kernel that does element-wise addition of two 256x256 float32 matrices.

In Pallas, we define a kernel function that takes refs (input and output). We use pl.pallas_call with grid, in_specs, out_specs.

Since it's full-shape inputs, we can process the whole thing in one block or tile it. The user says "authentic normal-lowering JAX Pallas kernel". Usually for 256x256, we can use a single block of (256, 256) or tile into smaller blocks like (128,128) etc. But to be safe and simple, we can use block_shape=(256, 256) and index_map=lambda i,j: (i,j) with grid=(1,1). However, the user might want tiling? Not necessarily. But "normal-lowering" implies using Pallas correctly.

Let's design:

import jax
import jax.numpy as jnp
from jax import random
import jax.experimental.pallas as pl

def workload(*inputs):
    # inputs are two arrays of shape (256,256) float32
    # We define kernel
    def add_kernel(x_ref, y_ref, o_ref):
        # Load full blocks
        x = pl.load(x_ref, ...)
        # Actually in Pallas kernel, refs are block references. We can do:
        # o_ref[...] = x_ref[...] + y_ref[...]
        # But need to use pl.load? Actually direct indexing on refs works in Pallas.
        # Let's do: o_ref[...] = x_ref[...] + y_ref[...]
        pass

Wait, in Pallas, inside the kernel, the refs represent the local block. We can do:
o_ref[...] = x_ref[...] + y_ref[...]

But we need to be careful: x_ref and y_ref are pl.Ref objects. Direct assignment with slicing works.

Let's write:

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

Then define workload:

def workload(*inputs):
    x, y = inputs
    # Define block specs
    block_shape = (256, 256)
    # Since full shape is 256x256, grid is (1,1)
    # index_map for input and output: lambda i, j: (i, j) but since grid is (1,1), i and j are 0.
    # Actually index_map takes grid indices.
    # For block_shape (256,256) and grid (1,1):
    # index_map = lambda i, j: (i*256, j*256) but since block_shape is 256, and grid is 1, it's (0,0).
    # But more generally: lambda i, j: (i, j) if block_shape equals full shape? No, index_map maps grid indices to array indices (start indices).
    # If block_shape is (128,128) and grid is (2,2), index_map = lambda i,j: (i*128, j*128).
    # If block_shape is (256,256) and grid is (1,1), index_map = lambda i,j: (0,0) or lambda i,j: (i*256, j*256).
    # The user explicitly says: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second.
    # So we must use pl.BlockSpec(block_shape, index_map).

    # Let's choose block_shape = (128, 128) and grid = (2, 2) to show tiling, which is more "authentic".
    # But the user says "Compute add for full-shape inputs [[256, 256], [256, 256]]". It doesn't specify tiling.
    # However, using a single block is fine. But maybe they want to see index_map.
    # Let's use block_shape = (64, 64) and grid = (4, 4) to be safe? Actually any works.
    # But the kernel must match independent semantic oracle at full declared shapes.
    # As long as output is correct, it's fine.

    # Let's use block_shape = (256, 256) and grid = (1, 1) for simplicity.
    # index_map = lambda i, j: (0, 0) or lambda i, j: (i*256, j*256). The latter is more standard.
    # Actually if grid is (1,1), i and j are always 0, so (0,0) is fine. But to be general, use (i*256, j*256).

    # Wait: The user says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second."
    # So we must write pl.BlockSpec((256, 256), lambda i, j: (i * 256, j * 256)).

    # But if block_shape is (256,256) and array is (256,256), then index_map should return (0,0) for i=j=0. If we use (i*256, j*256), for i=0,j=0 it's (0,0). Good.

    # However, if we use grid=(1,1), the lambda receives i=0, j=0.

    # Let's define:
    block_shape = (256, 256)
    grid = (1, 1)
    in_specs = [
        pl.BlockSpec(block_shape, lambda i, j: (i * block_shape[0], j * block_shape[1])),
        pl.BlockSpec(block_shape, lambda i, j: (i * block_shape[0], j * block_shape[1])),
    ]
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * block_shape[0], j * block_shape[1]))

    # But wait: The index_map for output should also map to output indices. Since output is same shape, same.

    # Actually in Pallas, the index_map for inputs and outputs can be different if needed, but here same.

    # Let's write the pallas_call:
    result = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x, y)

    return result

But we need to make sure the kernel is correct. In Pallas, the kernel signature is:
def kernel(x_ref, y_ref, o_ref):
    ...

We can directly assign:
o_ref[...] = x_ref[...] + y_ref[...]

This is valid.

However, sometimes Pallas requires using pl.load and pl.store for more complex cases, but direct indexing is allowed for simple cases.

Wait, is direct indexing on refs allowed? Yes, in Pallas, refs support slicing and assignment.

But let's double-check: In the Pallas documentation, inside a kernel, you can do:
def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1

Yes.

So the module should be:

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    x, y = inputs
    assert x.shape == (256, 256) and y.shape == (256, 256)
    assert x.dtype == jnp.float32 and y.dtype == jnp.float32

    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    block_shape = (256, 256)
    grid = (1, 1)

    def index_map(i, j):
        return (i * block_shape[0], j * block_shape[1])

    in_specs = [
        pl.BlockSpec(block_shape, index_map),
        pl.BlockSpec(block_shape, index_map),
    ]
    out_specs = pl.BlockSpec(block_shape, index_map)

    out = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x, y)

    return out

Wait, the user says "Compute add for full-shape inputs [[256, 256], [256, 256]] with output semantics defined independently by the operation name." The operation name is add. So output is x+y.

The user also says "The callable must accept inputs with shapes [[256, 256], [256, 256]] and dtypes ['float32', 'float32']. It must match the independent semantic oracle at the full declared shapes."

So the workload function should just take *inputs and return the result.

One thing: The user explicitly says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So we must use that exact order.

Also "Include every required import and a complete kernel body. Do not return an incomplete kernel."

We should include imports: jax, jax.experimental.pallas as pl, maybe jax.numpy as jnp.

We must not use interpret mode or plain-JAX fallback. So no pl.pallas_call(..., interpret=True) or fallback.

We should ensure that the kernel is "normal-lowering". That just means it uses the standard Pallas lowering path.

Potential issue: If we use block_shape=(256,256) and grid=(1,1), the kernel runs once. That's fine.

But is there any issue with using lambda inside BlockSpec? No.

Let's refine the index_map: It should take grid indices. If grid is (1,1), it takes (0,0). The lambda should return the start index in the array. So (i * 256, j * 256) is correct.

Wait, but if block_shape is (256, 256), then the block covers the whole array. The index_map should return (0,0) for the first (and only) block. If we use (i*256, j*256), with i=0, j=0, it's (0,0). Good.

But what if the user expects tiling? Not required.

However, sometimes Pallas requires that the block_shape divides the array shape evenly if using grid > 1. Here grid=1, so fine.

Let's consider if we should use a smaller block to demonstrate tiling, e.g., block_shape=(128,128), grid=(2,2). That might be more "authentic" because it shows index_map actually doing something. But either works.

Let's use block_shape=(128, 128) and grid=(2, 2) to be safe and show proper tiling. The index_map would be lambda i, j: (i*128, j*128). This is standard.

But wait: The user says "Compute add for full-shape inputs [[256, 256], [256, 256]]". It doesn't say we must tile. But using tiling is okay as long as output is correct.

However, if we use block_shape=(128,128), the kernel will be called 4 times. The kernel body is the same.

Let's decide: Use block_shape=(64, 64) and grid=(4,4)? No need. Let's use (128,128) and grid=(2,2). It's a nice middle ground.

But is there any hidden requirement that the block_shape must match the full shape? No.

Actually, let's think about "authentic normal-lowering JAX Pallas kernel". In many examples, they use block_shape that is smaller than the array to allow parallelism. But for a 256x256 array, using 128x128 is fine.

However, the simplest is block_shape=(256,256), grid=(1,1). It avoids any potential indexing errors.

But let's check Pallas documentation: The index_map is a function that takes the grid indices and returns the start indices of the block in the array. If block_shape equals array shape, index_map should return (0,0) for all grid indices (but grid is 1,1). So (i*256, j*256) works.

Wait, is there a requirement that index_map must be pure and not capture variables? It can capture block_shape.

Let's write the code clearly.

One more thing: The user wants "Return exactly one self-contained Python module defining workload(*inputs)." So the module should define workload. It can also define helper functions inside or outside.

We should not include any extra text, just the module.

Let's draft:

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(*inputs):
    x, y = inputs
    # Ensure correct shapes/dtypes? Not strictly required but good.
    # The kernel
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

    block_shape = (256, 256)
    grid = (1, 1)

    def index_map(i, j):
        return (i * block_shape[0], j * block_shape[1])

    in_specs = [
        pl.BlockSpec(block_shape, index_map),
        pl.BlockSpec(block_shape, index_map),
    ]
    out_specs = pl.BlockSpec(block_shape, index_map)

    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x, y)

Wait, does pl.pallas_call return a callable that takes inputs? Yes. So we can do:
return pl.pallas_call(...)(x, y)

But we need to make sure that the output shape is correct. We can use jax.ShapeDtypeStruct((256, 256), jnp.float32) or derive from x.

Deriving from x is better: jax.ShapeDtypeStruct(x.shape, x.dtype).

One subtlety: The user says "with output semantics defined independently by the operation name." That just means the operation is add, so output = x + y.

Now, is there any issue with using direct assignment o_ref[...] = ...? In Pallas, the refs are mutable. Yes.

But sometimes for float32, we might need to use pl.load and pl.store? No, direct indexing is fine.

Wait, I recall that in Pallas, inside the kernel, you access the block via refs, and you can do arithmetic directly on them? Actually, x_ref[...] returns a jax array (the block). Then x_ref[...] + y_ref[...] produces a new array. Then assigning to o_ref[...] writes it back. This is correct.

Let's verify with a minimal example mentally:

import jax.experimental.pallas as pl
def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2

This is standard.

So the code is correct.

But let's consider if the user expects the kernel to use pl.load and pl.store explicitly. The prompt says "complete kernel body". Direct assignment is complete.

However, to be absolutely safe and explicit, we could use pl.load and pl.store:

def add_kernel(x_ref, y_ref, o_ref):
    x = pl.load(x_ref, (slice(None), slice(None)))
    y = pl.load(y_ref, (slice(None), slice(None)))
    pl.store(o_ref, (slice(None), slice(None)), x + y)

But pl.load and pl.store are used for more complex indexing. For full block, direct is simpler. But using pl.load/pl.store might be more "authentic" for a kernel that explicitly loads and stores.

Wait, the prompt says "authentic normal-lowering JAX Pallas kernel". It doesn't specify load/store. But let's check the Pallas API: pl.load(ref, idx) loads from ref at index idx. If idx is omitted or is (slice(None), ...), it loads the whole block. Actually, pl.load takes a ref and an index tuple.

But direct indexing is also part of the API.

To avoid any issues with pl.load syntax (which might require specific index formats), direct assignment is safer.

However, I want to ensure that the kernel is "normal-lowering". The lowering path is triggered by pl.pallas_call.

One more thing: The user says "Do not use interpret mode or a plain-JAX fallback." So we must not pass interpret=True to pallas_call, and we must not have a try/except that falls back to jnp.add.

Our code doesn't do that.

Let's consider if we need to handle the case where inputs are not exactly 256x256. The prompt says "must accept inputs with shapes [[256, 256], [256, 256]]". It doesn't say it must reject others, but it's okay.

Now, about the module: It should be self-contained. We can include everything.

Let's write the final code.

Wait, one more check: The user says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So we must write pl.BlockSpec((256, 256), lambda i, j: (i*256, j*256)). We must not swap them.

Also, "Return exactly one self-contained Python module defining workload(*inputs)." So the file/module should define workload.

Let's produce the final answer with just the code block.

I'll use block_shape=(128, 128) and grid=(2,2) to make index_map non-trivial. Actually, let's stick with (256,256) and grid=(1,1) to avoid any tiling issues. But is (1,1) grid okay? Yes.

Wait, I just realized: If block_shape is (256, 256) and the array is (256, 256), then the index_map for the output must also be correct. It is.

But some Pallas versions might require that the block_shape is a tuple of ints, and index_map is a callable.

Let's write:

def workload(*inputs):
    x, y = inputs
    def kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    bs = (256, 256)
    grid = (1, 1)
    imap = lambda i, j: (i * bs[0], j * bs[1])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(bs, imap), pl.BlockSpec(bs, imap)],
        out_specs=pl.BlockSpec(bs, imap),
        grid=grid,
    )(x, y)

This is clean.

But wait: The lambda captures bs. That's fine.

One final check: The user wants "Compute add for full-shape inputs [[256, 256], [256, 256]]". The inputs are passed as *inputs, so inputs is a tuple of two arrays.

The output should be a single array.

Everything seems correct.

I'll include imports: import jax, import jax.experimental.pallas as pl.

I might not need jax.numpy if I don't use jnp explicitly, but I can use jax.ShapeDtypeStruct which is in jax. Actually, jax.ShapeDtypeStruct is available as jax.ShapeDtypeStruct. I don't need jnp.

But to be safe, I'll import jax.numpy as jnp and use jnp.float32 if needed, but I can just use x.dtype.

Let's write the final module.

One more thought: Is it possible that the kernel needs to be decorated or defined in a specific way? No.

I'll provide the code.

Wait, I should check if pl.pallas_call requires the kernel to have a specific number of arguments matching in_specs + out_specs. Yes: kernel(x_ref, y_ref, o_ref) matches 2 inputs and 1 output.

Yes.

Final code:

import jax
import jax.experimental.pallas as pl

def workload(*inputs):
    x, y = inputs
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    block_shape = (256, 256)
    grid = (1, 1)
    def index_map(i, j):
        return (i * block_shape[0], j * block_shape[1])
    in_specs = [
        pl.BlockSpec(block_shape, index_map),
        pl.BlockSpec(block_shape, index_map),
    ]
    out_specs = pl.BlockSpec(block_shape, index_map)
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x, y)

This is perfect. It uses the exact API requested. It has a complete kernel body. It doesn't use interpret mode. It defines workload.

I'll output this.<|end_message|><|message_model|><|content_text|>import jax
import jax.experimental.pallas as pl

def workload(*inputs):
    x, y = inputs
    def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]
    block_shape = (256, 256)
    grid = (1, 1)
    def index_map(i, j):
        return (i * block_shape[0], j * block_shape[1])
    in_specs = [
        pl.BlockSpec(block_shape, index_map),
        pl.BlockSpec(block_shape, index_map),
    ]
    out_specs = pl.BlockSpec(block_shape, index_map)
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x, y)<|end_message|><|content_model_end_sampling|>