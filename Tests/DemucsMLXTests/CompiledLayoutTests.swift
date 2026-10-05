import MLX
import Testing

// Exercise the backend's independently cached variants with the same graph.
extension NativeParityTests {
  @Test(arguments: 1...10, [DType.float32, DType.float16])
  func compiledLayoutVariants(rank: Int, dtype: DType) {
    let shape = Array(repeating: 3, count: rank)
    let count = shape.reduce(1, *)
    let row = (arange(count).asType(dtype) / 1024).reshaped(shape)
    let column = row.transposed(axes: Array((0..<rank).reversed()))
    let operation: (MLXArray, MLXArray) -> MLXArray = { ($0 + $1) * 0.5 }
    let compiled = compile(shapeless: true, operation)
    let broadcastShape = (0..<rank).map { $0.isMultiple(of: 2) ? 1 : 3 }
    let broadcast = ones(broadcastShape, dtype: dtype)
    let reversed = column[.stride(by: -1)]
    // Return to the initial layout after requesting other libraries: a cache
    // key that omits the selected variant can otherwise reuse the wrong code.
    for (x, y) in [
      (row, row), (column, row), (reversed, row),
      (column, broadcast), (row, row),
    ] {
      let expected = operation(x, y)
      let actual = compiled(x, y)
      #expect(arrayEqual(expected, actual).item(Bool.self))
    }
  }

  @Test(arguments: [1, 17, 255, 1025, 16385])
  func compiledContiguousWorkSizes(count: Int) {
    let operation: (MLXArray) -> MLXArray = { ($0 + 0.25) * 0.5 }
    let compiled = compile(shapeless: true, operation)
    for size in [count, count + 1, count] {
      let x = arange(size).asType(.float32) / 1024
      #expect(arrayEqual(operation(x), compiled(x)).item(Bool.self))
    }
  }
}
