module @jit__interpreted_pallas_control attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x128xbf16>) -> (tensor<128x128xbf16> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0x7FC0> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<128x128xbf16>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.multiply %c, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %2:6 = stablehlo.while(%iterArg = %1, %iterArg_1 = %c_0, %iterArg_2 = %c_0, %iterArg_3 = %c_0, %iterArg_4 = %arg0, %iterArg_5 = %0) : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<128x128xbf16>, tensor<128x128xbf16>
     cond {
      %3 = stablehlo.convert %iterArg : tensor<i32>
      %4 = stablehlo.compare  LT, %iterArg_1, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      %c_6 = stablehlo.constant dense<128> : tensor<i32>
      %3 = stablehlo.multiply %c_6, %iterArg_2 : tensor<i32>
      %4 = stablehlo.multiply %c_6, %iterArg_3 : tensor<i32>
      %5 = stablehlo.multiply %c_6, %iterArg_2 : tensor<i32>
      %6 = stablehlo.multiply %c_6, %iterArg_3 : tensor<i32>
      %c_7 = stablehlo.constant dense<0> : tensor<i32>
      %7 = stablehlo.compare  LT, %3, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %8 = stablehlo.add %3, %c_6 : tensor<i32>
      %9 = stablehlo.select %7, %8, %3 : tensor<i1>, tensor<i32>
      %10 = stablehlo.compare  LT, %4, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %11 = stablehlo.add %4, %c_6 : tensor<i32>
      %12 = stablehlo.select %10, %11, %4 : tensor<i1>, tensor<i32>
      %13 = stablehlo.dynamic_slice %iterArg_4, %9, %12, sizes = [128, 128] : (tensor<128x128xbf16>, tensor<i32>, tensor<i32>) -> tensor<128x128xbf16>
      %14 = stablehlo.compare  LT, %5, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %15 = stablehlo.add %5, %c_6 : tensor<i32>
      %16 = stablehlo.select %14, %15, %5 : tensor<i1>, tensor<i32>
      %17 = stablehlo.compare  LT, %6, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %18 = stablehlo.add %6, %c_6 : tensor<i32>
      %19 = stablehlo.select %17, %18, %6 : tensor<i1>, tensor<i32>
      %20 = stablehlo.dynamic_slice %iterArg_5, %16, %19, sizes = [128, 128] : (tensor<128x128xbf16>, tensor<i32>, tensor<i32>) -> tensor<128x128xbf16>
      %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
      %21 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<128x128xbf16>
      %22 = stablehlo.add %13, %21 : tensor<128x128xbf16>
      %23 = stablehlo.compare  LT, %3, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %24 = stablehlo.add %3, %c_6 : tensor<i32>
      %25 = stablehlo.select %23, %24, %3 : tensor<i1>, tensor<i32>
      %26 = stablehlo.compare  LT, %4, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.add %4, %c_6 : tensor<i32>
      %28 = stablehlo.select %26, %27, %4 : tensor<i1>, tensor<i32>
      %29 = stablehlo.dynamic_update_slice %iterArg_4, %13, %25, %28 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<i32>, tensor<i32>) -> tensor<128x128xbf16>
      %30 = stablehlo.compare  LT, %5, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %31 = stablehlo.add %5, %c_6 : tensor<i32>
      %32 = stablehlo.select %30, %31, %5 : tensor<i1>, tensor<i32>
      %33 = stablehlo.compare  LT, %6, %c_7,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %34 = stablehlo.add %6, %c_6 : tensor<i32>
      %35 = stablehlo.select %33, %34, %6 : tensor<i1>, tensor<i32>
      %36 = stablehlo.dynamic_update_slice %iterArg_5, %22, %32, %35 : (tensor<128x128xbf16>, tensor<128x128xbf16>, tensor<i32>, tensor<i32>) -> tensor<128x128xbf16>
      %c_9 = stablehlo.constant dense<1> : tensor<i32>
      %37 = stablehlo.add %iterArg_1, %c_9 : tensor<i32>
      %38 = stablehlo.add %iterArg_3, %c_9 : tensor<i32>
      %c_10 = stablehlo.constant dense<true> : tensor<i1>
      %39 = func.call @_where(%c_10, %38, %iterArg_3) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %40 = stablehlo.compare  EQ, %39, %c_9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_11 = stablehlo.constant dense<0> : tensor<i32>
      %41 = func.call @_where_0(%40, %c_11, %39) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %42 = stablehlo.add %iterArg_2, %c_9 : tensor<i32>
      %43 = func.call @_where(%40, %42, %iterArg_2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %44 = stablehlo.compare  EQ, %43, %c_9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %45 = func.call @_where_0(%44, %c_11, %43) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      stablehlo.return %iterArg, %37, %45, %41, %29, %36 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<128x128xbf16>, tensor<128x128xbf16>
    }
    return %2#5 : tensor<128x128xbf16>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_0(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<i1>, tensor<i32>
    return %1 : tensor<i32>
  }
}
