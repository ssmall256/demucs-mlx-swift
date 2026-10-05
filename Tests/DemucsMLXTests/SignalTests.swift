import DemucsAudio
import Foundation
import MLX
import Testing

@testable import DemucsMLX

// Serialized with model tests: MLX compilation and resource setup are process-wide.
extension NativeParityTests {
  @Test func spectralRoundTripAndOverlapEdges() {
    let signal = sin(MLXArray(0..<12031).asType(.float32) * 0.071).reshaped(1, -1)
    let spectral = SpectralTransform(nfft: 4096)
    let output = spectral.istft(spectral.stft(signal), length: signal.dim(-1))
    #expect(max(abs(signal - output)).item(Float.self) < 2e-6)
    let frames = stacked([MLXArray([Float(1), 2, 3, 4]), MLXArray([Float(5), 6, 7, 8])]).reshaped(
      1, 2, 4)
    let out = OverlapAdd.apply(frames, window: MLXArray([Float(1), 2, 2, 1]), stride: 3, length: 7)
    #expect(out.asArray(Float.self) == [1, 2, 3, 4.5, 6, 7, 8])
  }
  @Test func nativeAudioRoundTripAndResampling() throws {
    let signal = stacked([sin(MLXArray(0..<4800).asType(.float32) * 0.071), zeros([4800])])
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".wav")
    defer { try? FileManager.default.removeItem(at: file) }
    try DemucsAudio.save(signal, sampleRate: 48000, to: file, float32: true)
    let same = try DemucsAudio.load(file, sampleRate: 48000)
    #expect(same.shape == signal.shape)
    #expect(max(abs(same - signal)).item(Float.self) < 1e-7)
    let resampled = try DemucsAudio.load(file, sampleRate: 44100)
    #expect(abs(resampled.dim(-1) - 4410) <= 1)
    #expect(all(isFinite(resampled)).item(Bool.self))
  }
}
