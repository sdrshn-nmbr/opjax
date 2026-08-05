<|message_model|><|content_thinking|>We need to implement an authentic normal-lowering JAX Pallas kernel for rmsnorm-128x128 operation. Compute rmsnorm for full-shape inputs [[128, 128]] with output semantics defined independently by the operation name. The callable must accept inputs with shapes [[128, 128]] and dtypes ['float32']. It must match the independent semantic oracle at the full declared shapes. Do not use interpret mode or a plain-JAX fallback. Return exactly one self-contained Python module defining workload(*inputs). Use current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is first argument and index_map is second. Include every required import and a complete kernel body. Do not return incomplete kernel. Return only Python module, with no prose.

We need to write a Pallas kernel that computes RMSNorm over the last dimension? Usually rmsnorm: given input x of shape [M, N], compute rms = sqrt(mean(x^2, axis=-1, keepdims=True)) + eps, then output = x / rms. For 128x128, we can do a kernel that processes blocks.

We need to define workload(*inputs). It should accept inputs (probably a tuple of arrays). We need to use jax.experimental.pallas as pl.

We need to use pl.pallas_call with a kernel function, grid, in_specs, out_specs.

Since input is 128x128 float32, we can process the whole thing in one block? But they ask for authentic normal-lowering JAX Pallas kernel. Could use a grid of (1, 1) with block_shape (128, 128). But maybe they want tiling? The problem says "Compute rmsnorm for full-shape inputs [[128, 128]]". It might be okay to have block_shape (128, 128) and process entire array in one block. However, to be safe, maybe use a grid that splits along the first dimension? But RMSNorm requires reduction over the second dimension (or last dimension). If we split along first dimension, each block can compute local sum of squares over its row segment, but need global reduction across the second dimension? Actually if we split along first dimension (rows), each block contains full columns for a subset of rows. Then reduction over columns is local to each block (since each block has all columns for its rows). That works: block_shape (32, 128) with grid (4, 1). Then each block computes rms for its rows independently.

But the problem says "full-shape inputs [[128, 128]]". It doesn't specify tiling. But using a grid with block_shape (128, 128) is simplest and fully correct.

However, they ask for "authentic normal-lowering JAX Pallas kernel". Usually Pallas kernels are written with a grid and block specs. We can do grid = (1, 1) and block_shape = (128, 128). That is valid.

But maybe they want to see use of pl.BlockSpec with index_map. Let's include that.

We need to define the kernel body: def kernel(x_ref, o_ref): load x, compute rms, write o.

In Pallas, references are pl.MemoryRef. We can do x = x_ref[...]; but for block_shape (128,128), x_ref shape is (128,128). We can compute:

x = x_ref[:, :]
mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
rms = jnp.sqrt(mean_sq + 1e-6)  # eps maybe 1e-6? The oracle might use a specific eps. We need to match independent semantic oracle. Usually RMSNorm uses eps=1e-6. But we need to be careful. The problem says "output semantics defined independently by the operation name". It might be standard rmsnorm with eps=1e-6 or maybe 1e-5. We need to infer from typical definitions.

In many contexts, RMSNorm is defined as: x / sqrt(mean(x^2) + eps). The eps is often 1e-6. But sometimes 1e-5. Since they say "match the independent semantic oracle at the full declared shapes", we need to guess the exact eps. Usually in JAX or PyTorch, LayerNorm uses eps=1e-5. RMSNorm in some implementations uses eps=1e-6. But the problem might not care about eps if oracle uses same? Actually it must match exactly. We need to choose a standard value. Let's use eps=1e-6? Or maybe 1e-5? Let's think.

The operation is "rmsnorm-128x128". In the context of Pallas benchmarks (like from the "pallas" repo or "jax" benchmarks), there is a workload called "rmsnorm". I recall a benchmark suite "jax-pallas-benchmarks" or "pallas-kernels" that includes rmsnorm. The typical eps might be 1e-6. But I'm not entirely sure.

Let's search memory: In the "pallas" examples, there is a "rmsnorm" kernel used in LLaMA. The formula is: rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6). Actually in LLaMA, the RMSNorm uses eps=1e-6. Yes, the original LLaMA uses eps=1e-6. So likely eps=1e-6.

Thus compute: rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6). Then out = x / rms.

We need to write the module with workload(*inputs). It should return the result.

We need to import jax, jax.numpy as jnp, jax.experimental.pallas as pl.

We need to define the kernel and call pl.pallas_call.

Let's design:

def workload(*inputs):
    x = inputs[0]
    # Define kernel
    def rmsnorm_kernel(x_ref, o_ref):
        x = x_ref[:, :]
        # compute over last axis
        mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + 1e-6)
        o_ref[:, :] = x / rms
    # Block specs
    block_shape = (128, 128)
    # Since input is exactly 128x128, index_map is identity
    in_specs = pl.BlockSpec(block_shape, lambda i, j: (i, j))
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i, j))
    # Grid
    grid = (1, 1)
    # Call
    y = pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[in_specs],
        out_specs=out_specs,
        grid=grid,
    )(x)
    return y

Wait: pl.pallas_call signature: pl.pallas_call(kernel, out_shape=..., grid=..., in_specs=..., out_specs=...)(*inputs). Actually the modern API: pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(...), grid=..., in_specs=..., out_specs=...)(*inputs). Yes.

But we need to make sure block_shape is first argument to BlockSpec. So pl.BlockSpec((128, 128), lambda i, j: (i, j)). Good.

We need to include every required import.

Potential issue: The kernel uses jnp.mean which is fine inside Pallas? Yes, Pallas supports many JAX primitives, but some reductions might need to be done carefully. However, jnp.mean over axis=-1 with keepdims=True inside a kernel that operates on a block of shape (128,128) is okay because it's a local reduction over the block's dimensions. Since the block covers the full second dimension, it's correct.

But if we want to be more robust, we could split along first dimension: grid=(4,1), block_shape=(32,128). Then index_map: lambda i, j: (i*32, j). That also works and is more "authentic". Let's do that to show tiling. But we must ensure that the reduction is over the full second dimension, which is contained in each block (since block_shape second dim is 128). So each block computes rms for its 32 rows independently. That is correct.

Let's choose grid = (4, 1), block_shape = (32, 128). Then index_map = lambda i, j: (i * 32, j). This is a nice tiling.

But the problem says "Compute rmsnorm for full-shape inputs [[128, 128]]". It doesn't require tiling, but it's okay.

However, we must be careful: The kernel receives a block of shape (32, 128). The reduction over axis=-1 is over 128 elements, correct.

Let's write it with grid=(4,1) and block_shape=(32,128). That demonstrates BlockSpec usage.

But is there any issue with using lambda that captures nothing? It's fine.

Let's write the module.

One more thing: The callable must accept inputs with shapes [[128, 128]] and dtypes ['float32']. It must match the independent semantic oracle. We assume eps=1e-6.

We should also consider that the oracle might compute rmsnorm as x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6). We will do exactly that.

We need to return exactly one self-contained Python module defining workload(*inputs). No prose.

So final output should be just the code block.

Let's craft the code:

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        x = x_ref[:, :]
        mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + 1e-6)
        o_ref[:, :] = x / rms
    block_shape = (32, 128)
    in_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 32, j))
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 32, j))
    grid = (4, 1)
    y = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[in_specs],
        out_specs=out_specs,
        grid=grid,
    )(x)
    return y

Wait: The lambda for index_map takes grid indices. If grid is (4,1), indices are (i, j) where i in [0,3], j in [0,0]. So (i*32, j) is correct.

But we need to make sure that block_shape is first argument to BlockSpec. Yes: pl.BlockSpec(block_shape, lambda ...).

One subtlety: In Pallas, the index_map function signature is index_map(*grid_indices). So lambda i, j: (i*32, j) is correct.

We should also import jax.experimental.pallas as pl.

Let's check if pl.pallas_call requires `interpret=False`? By default it uses the normal lowering (not interpret). Actually in recent JAX versions, `pl.pallas_call` has an `interpret` parameter defaulting to False? Let's check. In jax.experimental.pallas, `pallas_call` has `interpret: bool = False`. So default is normal lowering. Good.

We should not set interpret=True.

We need to make sure we don't use any plain-JAX fallback.

The code looks complete.

But wait: The problem says "Use the current JAX API exactly: pl.BlockSpec(block_shape, index_map); block_shape is the first argument and index_map is the second." We have that.

Let's refine to be safe: Use block_shape = (128, 128) and grid = (1, 1) to avoid any tiling issues with reduction. Actually with grid=(4,1) and block_shape=(32,128), the reduction is local to each block because the second dimension is fully covered. That is correct. But is there any issue with the output shape? No.

However, some Pallas kernels require that the block_shape divides the array dimensions evenly. 32 divides 128, 128 divides 128. Good.

Let's stick with (32,128) to show tiling.

But I want to be absolutely sure about eps. If the oracle uses eps=1e-5, our result will be slightly off. Which is more likely? In LLaMA, eps=1e-6. The problem mentions "rmsnorm-128x128". In many benchmark suites (like "mlperf" or "pallas-benchmarks"), the RMSNorm uses eps=1e-6. Let's assume 1e-6.

Alternatively, we could avoid hardcoding eps by reading it from input? No, the operation is fixed.

Let's consider if the oracle might not include eps at all? Some definitions of RMSNorm are just x / sqrt(mean(x^2)). But usually there is an epsilon for numerical stability. The problem says "Compute rmsnorm". Usually includes eps. But if we include eps and oracle doesn't, it will be wrong. If we exclude eps and oracle includes, also wrong.

How to decide? Let's think about standard definitions.

In PyTorch's `nn.RMSNorm` (added recently), the default eps is 1e-6. In the original LLaMA code, eps=1e-6. In the "Transformer" paper, LayerNorm uses eps=1e-5. RMSNorm is a variant of LayerNorm, and many implementations use 1e-6.

Given the operation name "rmsnorm", it's likely the LLaMA-style RMSNorm with eps=1e-6.

But to be safe, could we compute both and choose? No.

Let's check if there's a way to infer from "normal-lowering JAX Pallas kernel". The benchmark might be from "jax-pallas" repo. I recall a file "rmsnorm.py" in some repo. Let me think: In the "pallas" repository, there is an example "rmsnorm" that uses eps=1e-6. Actually I think the example is:

def rmsnorm(x, weight, eps=1e-6):
    ...

Yes.

So we'll use 1e-6.

One more detail: The kernel should use `jnp.sqrt` and `jnp.mean`. These are fine.

Let's write the final module.

Wait: The problem says "Return exactly one self-contained Python module defining workload(*inputs)." So we should define `workload` at module level.

We should also include `if __name__ == "__main__":`? Not needed.

Let's produce the code.

Double-check imports: `import jax`, `import jax.numpy as jnp`, `from jax.experimental import pallas as pl`.

The kernel uses `x_ref[:, :]`. In Pallas, `x_ref` is a `pl.MemoryRef`. Indexing with `[:, :]` returns a JAX array of the block shape. Good.

Assignment to `o_ref[:, :]` writes back.

One more thing: The `pl.pallas_call` might need `compiler_params`? Not required.

Let's write it.

Actually, I realize that `pl.pallas_call` might require `grid` to be a tuple of ints, and `in_specs` and `out_specs` to be lists or tuples. We have that.

Let's finalize.

Wait: Should we use `jax.ShapeDtypeStruct(x.shape, x.dtype)` or `jax.ShapeDtypeStruct(x.shape, jnp.float32)`? Since x is float32, using x.dtype is fine.

Let's write the final answer.<|end_message|><|message_model|><|content_text|>import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(*inputs):
    x = inputs[0]
    def kernel(x_ref, o_ref):
        x = x_ref[:, :]
        mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_sq + 1e-6)
        o_ref[:, :] = x / rms
    block_shape = (32, 128)
    in_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 32, j))
    out_specs = pl.BlockSpec(block_shape, lambda i, j: (i * 32, j))
    grid = (4, 1)
    y = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[in_specs],
        out_specs=out_specs,
        grid=grid,
    )(x)
    return y<|end_message|><|content_model_end_sampling|>