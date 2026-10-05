import Foundation
import MLX
import Testing

@testable import DemucsMLX

private final class RewriteOperation: @unchecked Sendable {
  let call: Op
  init(_ call: @escaping Op) { self.call = call }
}
@Suite(.serialized)
struct RewriteGateTests {
  @Test func incompatibleBiasUsesOriginalRewrite() throws {
    _ = MetalTestResources.configure
    let store = WeightStore([
      "probe.conv_tr.conv.weight": zeros([4, 4, 1, 8]),
      "probe.conv_tr.conv.bias": zeros([4]),
      "probe.rewrite.conv.weight": zeros([16, 3, 3, 8]),
      "probe.rewrite.conv.bias": zeros([16], dtype: .float16),
    ])
    let config = ModelConfig(values: [
      "channels": .number(8), "norm_starts": .number(4), "dconv_mode": .number(0),
      "sources": .array([.string("stem")]),
    ])
    let layer = try HybridLayer(
      store: store, prefix: "probe", config: config, stage: 0, frequency: true,
      empty: false, last: true, kernel: 4, stride: 4, padding: 0, decode: true, legacy: false)
    #expect(layer.gatedRewrite == nil && layer.rewrite != nil)
    let x = ones([1, 8, 2, 3])
    #expect(layer.decode(x, skip: zeros(x.shape), length: 3).0.dtype == .float32)
  }
  @Test(arguments: [false, true]) func rewriteBiasPlacement(strided: Bool) throws {
    _ = MetalTestResources.configure
    let weights = sin(arange(12 * 3 * 3 * 6).asType(.float32) * 0.07).reshaped(12, 3, 3, 6) * 0.05
    let bias = linspace(-0.3, 0.4, count: 12)
    let store = WeightStore(["test.weight": weights, "test.bias": bias])
    let convolution = try store.conv("test", padding: 1, frequency: true, widthPadding: 1)
    let baseline = RewriteOperation { glu(convolution($0)) }
    let candidate = RewriteOperation(try store.gatedFrequencyRewrite("test", padding: 1))
    var input = cos(arange(2 * 6 * 17 * 19).asType(.float32) * 0.13).reshaped(2, 6, 17, 19)
    if strided { input = input[0..., 0..., .stride(by: -2), .stride(by: 2)] }
    #expect(arrayEqual(baseline.call(input), candidate.call(input)).item(Bool.self))
    let original = compile { [baseline] in [baseline.call($0[0])] }
    let actual = compile { [candidate] in [candidate.call($0[0])] }
    #expect(arrayEqual(original([input])[0], actual([input])[0]).item(Bool.self))
  }
}
