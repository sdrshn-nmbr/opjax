<|message_model|><|content_thinking|>The user wants an authentic normal-lowering JAX Pallas kernel for add-256x256 operation. Inputs are two float32 arrays of shape [256, 256]. Output is add (element-wise addition). Must use Pallas (pl) with BlockSpec(block_shape, index_map). Must not use interpret mode or plain-JAX fallback. Must return exactly one self-contained Python module defining workload(*inputs). Must match independent semantic oracle at full declared shapes.

We need to write a Pallas kernel that does element-wise add on 256x256 tiles. Since it's full-shape inputs, we can use a single block that covers the whole array, or multiple blocks. But to be safe and efficient, we can tile with block_shape (128, 128) or (256, 256). If we use (256, 256), index_map is lambda i, j: (i, j) but since grid is (1,1), index_map can be lambda i, j: (0, 0) or just lambda i, j: (i, j) with grid (1,1). Actually with block_shape (256, 256) and grid (1,1), index_map is lambda i, j: (i, j) but i and j are 0. That's fine.

But the user says "Compute add for full-shape inputs [[256, 256], [256, 256]] with output semantics defined independently by the operation name." So output is just a + b.

We need to define workload(*inputs) that takes the two inputs and returns the result using pallas_call.

Let's construct:

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    # Define kernel
    def kernel(a_ref, b_ref, o_ref):
        # Load full blocks? Since block_shape is (256, 256), refs are 256x256.
        # But we can just do o_ref[...] = a_ref[...] + b_ref[...]
        # However in Pallas, we need to use pl.load? Actually with refs, we can index.
        # For simplicity: o_ref[...] = a_ref[...] + b_ref[...]
        # But sometimes direct assignment to o_ref[...] works.
        # Let's use pl.load and pl.store? Actually in Pallas kernel, refs support indexing.
        # We can do: o_ref[...] = a_ref[...] + b_ref[...]
        # But to be safe with normal lowering, maybe use pl.load and pl.store.
        # Actually the standard is: o_ref[...] = a_ref[...] + b_ref[...]
        # Let's do that.
        o_ref[...] = a_ref[...] + b_ref[...]
    
    # Block specs
    block_shape = (256, 256)
    # Grid
    grid = (1, 1)
    # Index maps
    # For inputs and output, same mapping.
    # Since grid is (1,1), index_map can be lambda i, j: (0, 0) or lambda i, j: (i, j).
    # But block_shape is 256, so index_map should return the start index.
    # If grid is (1,1), i=0, j=0, so (0,0) is correct.
    # However if we want to be generic, we can use lambda i, j: (i * 256, j * 256) but grid is (1,1).
    # Let's just use lambda i, j: (i, j) with grid (1,1) and block_shape (256,256) - but that would mean start at (0,0) and (1,1) which is out of bounds? Actually index_map returns the start index in the array. If grid is (1,1), i and j are 0. So (i, j) = (0,0). That's fine.
    # But if block_shape is (128,128) and grid (2,2), index_map should be lambda i,j: (i*128, j*128).
    # To be safe and simple, let's use block_shape (128, 128) and grid (2, 2) with index_map lambda i, j: (i * 128, j * 128).
    # That is more "authentic" tiling.
    # Let's do that.
    block_shape = (128, 128)
    grid = (2, 2)
    index_map = lambda i, j: (i * 128, j * 128)
    
    # But the user wants full-shape inputs. Tiling is okay.
    # We need to define block specs for a, b, o.
    # For output, same.
    # We must use pl.BlockSpec(block_shape, index_map) exactly: first arg block_shape, second index_map.
    # So: pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128))
    # Actually the lambda captures nothing.
    # Let's write it clearly.
    
    # However, the kernel must handle the block references correctly.
    # If block_shape is (128,128), then a_ref is 128x128.
    # The kernel just does o_ref[...] = a_ref[...] + b_ref[...]
    # That works.
    
    # Let's implement.
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[
            pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
            pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
        ],
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
        grid=(2, 2),
    )(a, b)
    return out

Wait: The user says "The callable must accept inputs with shapes [[256, 256], [256, 256]] and dtypes ['float32', 'float32']." So workload(*inputs) should work.

But we need to make sure we don't use interpret mode. By default pallas_call uses the default backend (GPU/TPU). If running on CPU, it might need interpret? Actually Pallas on CPU might require interpret mode or specific backend. But the user explicitly says "Do not use interpret mode or a plain-JAX fallback." So we just call pl.pallas_call without interpret=True.

We need to include every required import.

Let's refine the kernel. In Pallas, the kernel function signature is kernel(*refs). For multiple inputs and one output, it's kernel(a_ref, b_ref, o_ref). The refs are pl.MemoryRef.

We can write:

def kernel(a_ref, b_ref, o_ref):
    o_ref[...] = a_ref[...] + b_ref[...]

This is valid.

But sometimes for float32, we need to ensure no type promotion issues. It's fine.

Now, about BlockSpec: The user explicitly says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second."

So we must write pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)).

We need to define workload(*inputs). It should unpack inputs.

Let's write the module.

One concern: If the environment doesn't have GPU/TPU, pallas_call might fail. But the user asks for authentic normal-lowering kernel, so it's okay.

Let's produce the final code.

Wait, the user says "Return exactly one self-contained Python module defining workload(*inputs). Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second. Include every required import and a complete kernel body. Do not return an incomplete kernel. Return only the Python module, with no prose."

So the output should be just the Python code, no markdown explanation.

Let's write:

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    assert a.shape == (256, 256) and b.shape == (256, 256)
    assert a.dtype == jnp.float32 and b.dtype == jnp.float32

    def add_kernel(a_ref, b_ref, o_ref):
        o_ref[...] = a_ref[...] + b_ref[...]

    block_shape = (128, 128)
    grid = (2, 2)

    def index_map(i, j):
        return (i * 128, j * 128)

    out = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[
            pl.BlockSpec(block_shape, index_map),
            pl.BlockSpec(block_shape, index_map),
        ],
        out_specs=pl.BlockSpec(block_shape, index_map),
        grid=grid,
    )(a, b)
    return out

Wait: The index_map is a function that takes grid indices. In pl.BlockSpec, index_map is called with the grid indices. So lambda i, j: (i*128, j*128) is correct.

But is it okay to define index_map as a named function? Yes.

However, the user wants "pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So using named variables is fine.

One more thing: The kernel might need to use pl.load and pl.store for some backends? Actually direct indexing of refs is supported in Pallas. For example, in many examples:

def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1

So it's fine.

But let's consider if we should use (256, 256) block_shape with grid (1,1) to avoid any tiling issues. That might be simpler and ensures the full array is processed in one block. The user says "Compute add for full-shape inputs". Tiling is okay, but using full block is also okay. However, some Pallas implementations might have limitations on block size (e.g., must be <= some limit). 256x256 float32 is 256KB, which is okay for GPU shared memory? Actually 256*256*4 = 262,144 bytes = 256 KB. That might be okay for some GPUs but could exceed shared memory if other things are used. To be safe, 128x128 is 64KB, which is very safe.

But the user didn't specify tiling. Either works. Let's stick with 128x128.

Wait, the user says "authentic normal-lowering JAX Pallas kernel". Normal lowering usually means it compiles to GPU/TPU kernels. Using tiling is more authentic.

Let's check if there are any syntax issues with pl.pallas_call. The signature is:

pl.pallas_call(
    kernel,
    out_shape=...,
    in_specs=...,
    out_specs=...,
    grid=...,
    interpret=False,  # default is False? Actually default might be False.
)(*inputs)

We don't need to pass interpret=False explicitly, but we must not pass interpret=True.

One more detail: The output semantics are defined independently by operation name "add". So just a + b.

Let's finalize.

Wait, I should check if `jax.ShapeDtypeStruct` is correct. Yes, `jax.ShapeDtypeStruct(shape, dtype)`.

Also, we might need to import `jax` and `jax.numpy as jnp`.

The module should define `workload`.

Let's write the final answer with just the code.

One more check: The user says "Compute add for full-shape inputs [[256, 256], [256, 256]] with output semantics defined independently by the operation name." This might mean that the operation is "add" and the output is just the sum.

I will provide the code.

Potential issue: In some JAX versions, `pl.BlockSpec` requires `index_map` to be a callable that takes the grid indices and returns a tuple of ints. Our `index_map` does that.

Let's write it cleanly.

Actually, to be absolutely safe with the API, let's use a lambda directly inside BlockSpec:

pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128))

This is exactly `pl.BlockSpec(block_shape, index_map)`.

So:

in_specs=[
    pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
    pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
],
out_specs=pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),

This is perfect.

Let's assemble.

One final thought: Should we include `if __name__ == "__main__":`? No, just the module.

I'll return the code block.<|end_message|><|message_model|><|content_text|>import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    a, b = inputs
    def add_kernel(a_ref, b_ref, o_ref):
        o_ref[...] = a_ref[...] + b_ref[...]
    block_shape = (128, 128)
    grid = (2, 2)
    out = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        in_specs=[
            pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128)),
            pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128)),
        ],
        out_specs=pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128)),
        grid=grid,
    )(a, b)
    return out<|end_message|><|content_model_end_sampling|>