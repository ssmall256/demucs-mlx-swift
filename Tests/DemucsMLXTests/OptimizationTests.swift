import DemucsAudio
import Foundation
import MLX
import Testing

@testable import DemucsMLX

extension NativeParityTests {
  @Test(arguments: [false, true])
  func pointwiseConvolutionAgreement(_ frequency: Bool) throws {
    let input = sin(MLXArray(0..<(2 * 6 * 99)).asType(.float32) * 0.07)
      .reshaped(frequency ? [2, 6, 9, 11] : [2, 6, 99])
    let weight = cos(MLXArray(0..<(96 * 6)).asType(.float32) * 0.03)
      .reshaped(frequency ? [96, 1, 1, 6] : [96, 1, 6])
    let bias = sin(MLXArray(0..<96).asType(.float32))
    let store = WeightStore(["projection.weight": weight, "projection.bias": bias])
    let actual = try store.conv("projection", frequency: frequency)(input)
    let expected =
      frequency
      ? (conv2d(input.transposed(0, 2, 3, 1), weight) + bias).transposed(0, 3, 1, 2)
      : (conv1d(input.transposed(0, 2, 1), weight) + bias).transposed(0, 2, 1)
    #expect(actual.shape == expected.shape)
    #expect(max(abs(actual - expected)).item(Float.self) < 2e-5)
  }

  @Test(arguments: [false, true])
  func pcmBufferOwnsStridedSamples(_ half: Bool) throws {
    let strided = sin(MLXArray(0..<62).asType(.float32) * 0.17).reshaped(31, 2).T
      .asType(half ? .float16 : .float32)
    let expected = strided.asType(.float32).asArray(Float.self)
    let buffer = try DemucsAudio.pcmBuffer(from: strided, sampleRate: 44100)
    let restored = try DemucsAudio.tensor(from: buffer, sampleRate: 44100)
    #expect(restored.shape == [2, 31])
    #expect(restored.asArray(Float.self) == expected)
  }
}
