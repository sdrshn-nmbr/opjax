module @jit__plain_jax_control attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x128xbf16>) -> (tensor<128x128xbf16> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<128x128xbf16>
    %1 = stablehlo.add %arg0, %0 : tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
