<|message_model|><|content_thinking|>The user wants an authentic normal-lowering JAX Pallas kernel for rmsnorm-128x128 operation. Need to compute RMSNorm for full-shape inputs [[128, 128]] with output semantics defined independently by operation name. The callable must accept inputs with shapes [[128, 128]] and dtypes ['float32']. Must match independent semantic oracle at full declared shapes. No interpret mode or plain-JAX fallback. Return exactly one self-contained Python module defining workload(*inputs). Use current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second. Include every required import and complete kernel body.

We need to implement RMSNorm: For input x of shape [128, 128], typically RMSNorm computes sqrt(mean(x^2, axis=-1, keepdims=True) + eps) and then x / rms. But the operation name is "rmsnorm-128x128". The output semantics are defined independently by operation name. We need to infer what the oracle expects. Usually for a kernel benchmark, rmsnorm might compute the RMS normalization along the last dimension (or maybe along the first?). Let's think.

The input shape is [128, 128]. The operation is "rmsnorm-128x128". Often in ML, RMSNorm is applied to the last dimension (feature dimension). So for each row of 128 elements, compute RMS and divide. That yields output shape [128, 128]. The kernel should be a Pallas kernel that operates on blocks.

We need to use pl.pallas_call with a kernel that uses pl.load, pl.store, etc. Since it's normal-lowering, we need to define a grid and block specs.

Given full shape [128, 128], we can process the whole thing in one block if block_shape is (128, 128). But maybe they want tiled? The prompt says "Compute rmsnorm for full-shape inputs [[128, 128]]". It doesn't specify tiling, but we can use a single block of (128, 128) to keep it simple and correct. However, Pallas kernels often require block shapes that divide the array dimensions. 128 divides 128, so (128, 128) is fine.

But we need to be careful: The kernel must accept inputs with shapes [[128, 128]] and dtypes float32. It must match the independent semantic oracle. We need to define the semantic correctly.

Let's assume RMSNorm along the last axis (axis=-1). So for each row i: rms = sqrt(mean(x[i,:]^2) + eps). Then out[i,:] = x[i,:] / rms.

We need to choose eps. Usually 1e-6. The oracle likely uses a specific eps. Since it's not specified, maybe the operation is just "rmsnorm" without eps? But RMSNorm always has eps. Let's assume eps = 1e-6.

But maybe the oracle expects normalization along axis=0? Let's think about typical naming. "rmsnorm-128x128" could be a benchmark from a suite like "pallas-benchmarks" or "jax-pallas-kernels". In some benchmarks, they define operations like "rmsnorm" with input shape [M, N] and compute along the last dimension. Let's assume axis=-1.

We need to write a Pallas kernel that loads a block of input, computes RMS for each row (if block covers full width), and writes output.

If we use block_shape (128, 128) and grid (1, 1), the kernel gets the full array. Then inside the kernel, we can compute using JAX operations on the block (since it's a scalar block). But the prompt says "authentic normal-lowering JAX Pallas kernel". It might be okay to use JAX operations inside the kernel on the loaded block, as long as it's not a plain-JAX fallback (i.e., we must use pl.pallas_call). The kernel body can use jnp operations.

However, sometimes they want a kernel that uses pl.load and pl.store with manual loops? But using jnp on the block is standard for Pallas.

Let's design:

Imports:
import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl

Define kernel:
def rmsnorm_kernel(x_ref, o_ref):
    # Load full block
    x = pl.load(x_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
    # Compute RMS along last axis
    # x shape (128, 128)
    mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + 1e-6)
    out = x / rms
    pl.store(o_ref, (pl.dslice(0, 128), pl.dslice(0, 128)), out)

Wait, pl.load with dslice might need to specify indices. Actually pl.load(ref, idx) where idx is a tuple of slices or integers. For a block reference, we can load the whole block with pl.load(x_ref, (slice(None), slice(None)))? But inside kernel, the reference is already a block. Actually in Pallas, the kernel receives references that are already the block. So we can just do x = x_ref[...]? Let's check Pallas API.

In Pallas, inside the kernel, you can access the reference directly: x_ref[...] loads the whole block? Actually the reference object supports indexing: x_ref[:, :] returns a JAX array representing the block. Or you can use pl.load(x_ref, (pl.dslice(0, block_shape[0]), pl.dslice(0, block_shape[1]))). But if the block is the full array, maybe simpler: x = x_ref[:, :].

Wait, the Pallas kernel signature is def kernel(x_ref, o_ref, *args):. The references are of type pl.MemoryRef. You can index them: x = x_ref[:, :]. That loads the block.

Let's check current API. In jax.experimental.pallas, MemoryRef supports __getitem__ with slices. So x = x_ref[:, :] works.

Then compute:
x = x_ref[:, :]
mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
rms = jnp.sqrt(mean_sq + 1e-6)
o_ref[:, :] = x / rms

But we need to be careful: The output semantics might be defined as "rmsnorm" which could be x / sqrt(mean(x^2) + eps). Yes.

Now define workload(*inputs):
def workload(*inputs):
    x = inputs[0]
    # Define block specs
    # Since input is [128, 128], block_shape = (128, 128)
    # index_map: lambda i, j: (i, j) but grid is (1,1)
    # Actually for a single block, grid = (1, 1)
    # BlockSpec(block_shape, index_map)
    # index_map takes grid indices and returns start indices.
    # For grid (1,1), index_map = lambda i, j: (i * 128, j * 128) or just lambda i, j: (0, 0) if block is full.
    # But to be general, let's use lambda i, j: (i * 128, j * 128) with grid (1,1).
    # Actually if block_shape is (128, 128) and array is (128, 128), index_map should return (0,0) for (0,0).
    # Let's use lambda i, j: (i * 128, j * 128).
    # But grid must be (1, 1).
    # Let's define:
    block_shape = (128, 128)
    grid = (1, 1)
    # For input
    in_specs = [pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))]
    # For output
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    # Call
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x)

Wait, the API for pl.pallas_call: pl.pallas_call(kernel, out_shape=..., in_specs=..., out_specs=..., grid=...)(*inputs). Actually the arguments might be slightly different. Let's check.

In recent JAX versions, pl.pallas_call is used as:
pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(shape, dtype),
    in_specs=[pl.BlockSpec(...)],
    out_specs=pl.BlockSpec(...),
    grid=...,
)(*inputs)

But sometimes out_specs is a list if multiple outputs. Here one output.

Also, the kernel function signature: def kernel(x_ref, o_ref):.

We need to make sure we use pl.BlockSpec(block_shape, index_map) exactly with block_shape first.

Let's refine.

Potential issue: The kernel might need to handle the case where the block is not the full array, but since we declare grid (1,1) and block_shape (128,128), it's fine.

But is it "authentic normal-lowering"? It might require using pl.program_id to compute indices if tiled. But with single block, it's okay.

However, maybe the benchmark expects tiling into smaller blocks, e.g., (32, 32) or (128, 32) etc. But the prompt says "Compute rmsnorm for full-shape inputs [[128, 128]]". It doesn't specify tiling. A single block is simplest and correct.

But let's think: If we use a single block, the kernel is essentially just a JAX function wrapped in pallas_call. That is acceptable as long as it's not "interpret mode" or "plain-JAX fallback". The user explicitly says "Do not use interpret mode or a plain-JAX fallback." Using pl.pallas_call with a kernel that uses jnp operations is the standard way.

But maybe they want the kernel to manually compute RMS using loops over the block? Not necessary.

Let's check the exact semantics of RMSNorm. In some contexts, RMSNorm is defined as:
x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps)
with eps = 1e-6.

In others, it's:
x * rsqrt(mean(x^2, axis=-1, keepdims=True) + eps)

Same thing.

We need to match the oracle. Since we don't have the oracle, we must assume standard. Let's include eps = 1e-6.

But what if the oracle uses eps = 1e-5? Or no eps? The operation name "rmsnorm" strongly implies eps is used. Usually 1e-6 is standard in LLaMA etc.

Let's consider if the normalization is along axis=0 (columns). If input is [128, 128], maybe they treat first dim as batch and second as features, so axis=-1. That's most common.

Let's write the module.

One more thing: The user asks "Return exactly one self-contained Python module defining workload(*inputs)." So the module should define workload. It can also define helper functions.

We must include every required import.

Let's draft:

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def rmsnorm_kernel(x_ref, o_ref):
    # Load the block
    x = x_ref[:, :]
    # Compute RMS along last axis
    mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + 1e-6)
    o_ref[:, :] = x / rms

def workload(*inputs):
    x = inputs[0]
    assert x.shape == (128, 128)
    assert x.dtype == jnp.float32
    block_shape = (128, 128)
    grid = (1, 1)
    in_specs = [
        pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    ]
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x)

Wait, is the lambda correct? The index_map takes grid indices (i, j) and returns the start index in the array. For grid (1,1), i and j are 0. So (0,0). If we multiply by 128, it's (0,0). Good.

But what if the user passes inputs that are not exactly 128x128? The prompt says "must accept inputs with shapes [[128, 128]]". So we can assume that.

However, we need to be careful: In Pallas, the kernel is compiled for a specific block shape and grid. If we hardcode 128, it's okay.

But is there any issue with using jnp.mean inside the kernel? No, that's fine.

Wait, the prompt says "authentic normal-lowering JAX Pallas kernel". Could "normal-lowering" refer to a specific lowering mode? In JAX, Pallas kernels can be lowered to GPU/TPU. "Normal-lowering" might mean using the standard lowering path, not interpret mode. So using pl.pallas_call is correct.

One potential issue: The kernel must not use any Python control flow that depends on values, but using jnp.mean is fine.

Let's double-check the BlockSpec API: pl.BlockSpec(block_shape, index_map). Yes.

In some versions, index_map is a callable that takes *grid_indices. For 2D grid, it takes (i, j). So lambda i, j: (i * 128, j * 128) is correct.

But we need to make sure that the block_shape is a tuple of ints.

Also, out_specs: If we have one output, out_specs can be a single BlockSpec or a list? The docs say out_specs can be a sequence of BlockSpec or a single BlockSpec if one output. To be safe, we can pass a list: out_specs=[pl.BlockSpec(...)]? Actually for pallas_call, out_specs is typically a sequence if multiple outputs, but for one output it can be a single BlockSpec or a list. Let's check.

In jax.experimental.pallas, pallas_call signature:
def pallas_call(
    f,
    out_shape,
    in_specs=None,
    out_specs=None,
    grid=None,
    ...
):
If out_specs is None, it defaults to full array. If we pass a single BlockSpec, it might be interpreted as specs for a single output. But sometimes it expects a sequence. To be safe, let's pass out_specs=[pl.BlockSpec(...)]? Wait, if out_shape is a single ShapeDtypeStruct, then out_specs should be a single BlockSpec or a list of one? I think it can be a single BlockSpec. But let's look at examples.

Example from docs:
pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
    out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
    grid=(1,),
)(x)

So out_specs can be a single BlockSpec.

Thus our code is okay.

But wait: The index_map for input and output should be the same.

Let's refine the kernel to be robust: Use pl.load and pl.store? Actually indexing is simpler.

But is indexing x_ref[:, :] allowed? Yes, MemoryRef supports slicing.

However, some versions require using pl.load(x_ref, (slice(0, 128), slice(0, 128)))? Actually pl.load takes a reference and an index tuple. The index tuple can contain slices or integers. If we pass (slice(None), slice(None)), it loads the whole block. But indexing directly might be more straightforward.

Let's check: In Pallas, you can do:
x = pl.load(x_ref, (pl.dslice(0, 128), pl.dslice(0, 128)))
But pl.dslice is for dynamic slices? Actually pl.dslice(start, size) is used for dynamic indexing. For static full block, we can just do x_ref[:, :].

Wait, I recall that inside a Pallas kernel, you access the memory reference by indexing: x_ref[:, :] returns a JAX array. This is the standard way.

Let's confirm with a known example:
def add_kernel(x_ref, y_ref, o_ref):
  o_ref[:, :] = x_ref[:, :] + y_ref[:, :]

Yes, that's common.

So we can use that.

Now, one subtlety: The user asks for "full-shape inputs [[128, 128]]". The double brackets might indicate a list of inputs, each of shape [128, 128]. So workload(*inputs) receives one or more inputs. We just take inputs[0].

Let's consider if the operation might require two inputs: e.g., x and weight? But the prompt says "inputs with shapes [[128, 128]]". That suggests a single input of shape [128, 128]. The double brackets are just notation for a list of shapes.

Thus workload takes *inputs, uses inputs[0].

Now, is there any need for a second input like scale? RMSNorm sometimes has a weight parameter (gamma) of shape [128]. But the prompt says inputs shapes [[128, 128]] (plural? maybe just one). It says "accept inputs with shapes [[128, 128]] and dtypes ['float32']". That could mean the input is a single array of shape (128, 128). If there were weights, they'd mention them.

So we assume single input.

Let's write the final module.

Wait, the user says "Compute rmsnorm for full-shape inputs [[128, 128]] with output semantics defined independently by the operation name." This might mean that the operation is named "rmsnorm-128x128" and the semantics are defined by that name, which is standard RMSNorm.

Let's include eps = 1e-6.

One more check: Should we use jnp.sqrt or jax.lax.rsqrt? jnp.sqrt is fine.

Potential issue: The kernel might need to be jitted? No, pallas_call handles compilation.

Let's produce the final answer: Just the Python module.

Make sure to include "import jax" and "import jax.numpy as jnp" and "import jax.experimental.pallas as pl".

Let's write it cleanly.

One more thought: The user explicitly says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." So we must use that order.

Also "Include every required import and a complete kernel body. Do not return an incomplete kernel."

So the module should be self-contained.

Let's write:

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def _rmsnorm_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    sq = jnp.square(x)
    mean_sq = jnp.mean(sq, axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + 1e-6)
    o_ref[:, :] = x / rms

def workload(*inputs):
    x = inputs[0]
    block_shape = (128, 128)
    grid = (1, 1)
    in_specs = [
        pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    ]
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    return pl.pallas_call(
        _rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x)

Wait, is the lambda correct for index_map? The index_map receives the grid indices. For a 2D grid (1,1), it receives (0,0). The lambda returns (0,0). Good.

But what if the grid is (1,1) and block_shape is (128,128), the index_map should return the start index for each block. Since there's only one block, (0,0) is correct.

However, some Pallas versions require index_map to be a function that takes *args corresponding to grid dimensions. So lambda i, j: ... is fine.

One more thing: Should we specify input specs as a list? Yes, in_specs expects a sequence.

Let's check if out_specs needs to be a list when out_shape is a single struct. In the docs, out_specs can be a BlockSpec or a sequence of BlockSpecs. If it's a single BlockSpec, it's fine.

But to be safe, maybe pass out_specs=[pl.BlockSpec(...)]? Actually if out_shape is a single ShapeDtypeStruct, passing a list of one BlockSpec might also work. But passing a single BlockSpec is more direct.

Let's check an example from JAX source:
pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
    out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
    grid=(1,),
)(x)

Yes, out_specs is a single BlockSpec.

So our code is correct.

Wait, but the lambda for 1D grid takes one argument. For 2D grid, it takes two. Our grid is (1,1), so lambda i, j: ... is correct.

One final check: The user says "Return only the Python module, with no prose." So we must output just the code block.

Let's do it.

But I want to be absolutely sure about the RMSNorm semantics. Could it be that the oracle expects normalization over axis=0? Let's think about the naming "rmsnorm-128x128". In some benchmark suites (like "mlperf" or "tpu-benchmarks"), they have operations like "rmsnorm" with input [batch, seq_len, hidden] or [batch, hidden]. Here it's 2D [128, 128]. Usually for a 2D input, RMSNorm is applied to the last dimension (hidden). So axis=-1.

If it were applied to axis=0, it would be less common. Let's stick with axis=-1.

What about eps? If the oracle uses eps=1e-6, we're good. If it uses eps=1e-5, we might be off. But 1e-6 is the most standard. Some implementations use 1e-6, some use 1e-5. In LLaMA, eps=1e-6. In PyTorch's RMSNorm (if it existed), maybe 1e-6. Let's use 1e-6.

Could the oracle expect no eps? That would be "rms" not "rmsnorm". RMSNorm always includes eps for numerical stability.

Thus final code is as above.

Wait, one more subtlety: The prompt says "authentic normal-lowering JAX Pallas kernel". Could "normal-lowering" mean that the kernel should be lowered using the "normal" path, which might require using pl.program_id or pl.num_programs? Not necessarily. It just means don't use interpret mode.

But to be more "authentic", maybe we should compute the RMS using a reduction inside the kernel with pl.dot or something? No, that's overcomplicating.

Actually, for a tiled kernel, you'd load a block of rows and compute partial sums, then reduce. But with a single block, it's fine.

However, maybe the benchmark expects the kernel to work for any block size that divides 128, but the workload is called with full shape. The prompt says "Compute rmsnorm for full-shape inputs [[128, 128]]". It might be okay.

Let's consider if we should use a tiled approach to be safe. For example, block_shape = (32, 128) and grid = (4, 1). Then each block is a set of rows. The kernel computes RMS for each row within the block. That is also easy.

But if we use block_shape = (32, 128), the index_map is lambda i, j: (i*32, j*128). The kernel loads x_ref[:, :] which is (32, 128). Then computes mean over axis=-1, which is correct for each row. This is more "authentic" because it shows tiling.

But the prompt doesn't require tiling. It just requires correct output for full shape.

However, using a single block might be considered "not a real kernel" by some, but it's still a valid Pallas kernel.

To be safe and more authentic, let's use a tiled block that covers the full width but splits rows: block_shape = (32, 128), grid = (4, 1). Or even block_shape = (64, 64), grid = (2, 2). But then computing RMS along axis=-1 requires that the block contains the full width for each row. If we split columns, we'd need to compute partial sums and then combine, which is more complex.

So to keep it simple and correct without cross-block communication, we should keep the full width in each block: block_shape = (32, 128) or (128, 128). If we use (32, 128), the kernel is still simple.

But the simplest is (128, 128). Let's stick with that.

Wait, is there any issue with using jnp.mean inside a Pallas kernel on TPU? It should work because Pallas kernels are JAX functions that are traced and compiled. Using jnp.mean is fine.

One final check: The user says "The callable must accept inputs with shapes [[128, 128]] and dtypes ['float32']." So we should maybe assert or just rely on pallas_call. We don't need asserts, but they don't hurt.

Let's produce the final module.

Actually, I want to make sure about the import of pl. It's "jax.experimental.pallas". In newer JAX versions, it might be "jax.experimental.pallas" or "jax.pallas"? The prompt says "Use the current JAX API exactly: pl.BlockSpec". So importing as pl from jax.experimental.pallas is correct.

Let's write the code.

One more thing: Should we define the kernel with a specific name? It doesn't matter.

I'll write:

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def _kernel(x_ref, o_ref):
    x = x_ref[:, :]
    mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    o_ref[:, :] = x / jnp.sqrt(mean_sq + 1e-6)

def workload(*inputs):
    x = inputs[0]
    bs = (128, 128)
    grid = (1, 1)
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(bs, lambda i, j: (i * 128, j * 128))],
        out_specs=pl.BlockSpec(bs, lambda i, j: (i * 128, j * 128)),
        grid=grid,
    )(x)

This looks perfect.

Wait, I need to ensure that the lambda captures nothing incorrectly. It's fine.

One final thought: The user might expect the kernel to handle the case where input is passed as a list of arrays? No, workload(*inputs) unpacks arguments.

I'll output exactly this.<|end_message|><|message_model|><|content_text|>import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def _rmsnorm_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + 1e-6)
    o_ref[:, :] = x / rms

def workload(*inputs):
    x = inputs[0]
    block_shape = (128, 128)
    grid = (1, 1)
    in_specs = [
        pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    ]
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 128, j * 128))
    return pl.pallas_call(
        _rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )(x)<|end_message|><|content_model_end_sampling|>