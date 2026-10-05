import Foundation
import MLX
import Testing

@testable import DemucsMLX

extension NativeParityTests {
  @Test(arguments: [1, 17, 2048, 2049, 4096, 4097, 12031])
  func fusedFrameAgreement(_ length: Int) {
    let spectral = SpectralTransform(nfft: 4096)
    let backing = sin(MLXArray(0..<(4 * length)).asType(.float32) * 0.071).reshaped(2, -1)
    let x = backing[0..., .stride(by: 2)]
    let expected = spectral.originalFrames(x)
    for choice in [SpectralKernelChoice(width: 64), .init(width: 256, tileFrames: 4)] {
      let actual = spectral.windowedFrames(x, choice: choice)
      eval(actual, expected)
      #expect(actual.shape == expected.shape)
      #expect(arrayEqual(actual, expected).item(Bool.self))
    }
  }
  @Test func largeTiledFramesAndOddFFT() {
    for (nfft, rows, length) in [(4096, 8, 800003), (4095, 2, 12031)] {
      let spectral = SpectralTransform(nfft: nfft)
      let x = sin(MLXArray(0..<(rows * length)).asType(.float32) * 0.019).reshaped(rows, length)
      let expected = spectral.originalFrames(x)
      for choice in [
        SpectralKernelChoice(width: 128, tileFrames: 2), .init(width: 256, tileFrames: 4),
      ] {
        #expect(arrayEqual(spectral.windowedFrames(x, choice: choice), expected).item(Bool.self))
      }
      if rows == 8 { #expect(expected.nbytes >= 100_000_000) }
    }
  }
  @Test func spectralCacheRejectsInvalidFiles() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    try Data(repeating: 0, count: 262_145).write(to: file)
    #expect(SpectralKernelCache(url: file).choice(for: "x") == .init())
    try Data("broken".utf8).write(to: file)
    let cache = SpectralKernelCache(url: file)
    #expect(cache.choice(for: "x") == .init())
    cache.store(.init(width: 128), for: "x")
    #expect(SpectralKernelCache(url: file).choice(for: "x").width == 128)
    cache.store(.init(width: 7), for: "x")
    #expect(cache.choice(for: "x").width == 128)
    let wrongIdentity: [String: Any] = [
      "identity": "old-kernel", "choices": ["x": ["width": 512, "tileFrames": 1]],
    ]
    try JSONSerialization.data(withJSONObject: wrongIdentity).write(to: file)
    #expect(SpectralKernelCache(url: file).choice(for: "x") == .init())
    for i in 0..<300 { cache.store(.init(width: 64), for: String(i)) }
    let json = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
    #expect((json["choices"] as! [String: Any]).count <= 256)
  }
}

extension Separator {
  fileprivate func spectralCompilationCounts() -> (Int, Int) {
    (
      models.values.reduce(0) { $0 + $1.compiled.count },
      models.values.reduce(0) { $0 + $1.seen.count }
    )
  }
}

extension NativeParityTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil))
  func explicitTuningInvalidatesGraphsAndPreservesStems() async throws {
    var options = SeparationOptions()
    options.batchSize = 1
    options.shifts = 0
    options.seed = 481
    let separator = try Separator(
      cacheDirectory: URL(
        fileURLWithPath:
          ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!), options: options)
    let input = AudioTensor(sin(MLXArray(0..<12000).asType(.float32) * 0.01).reshaped(2, -1))
    let before = try await separator.separate(input)
    _ = try await separator.separate(input)
    #expect(await separator.spectralCompilationCounts().0 > 0)
    let report = try await separator.tuneSpectralKernels()
    #expect(report.measurements.count >= 8)
    #expect(report.measurements.filter(\.selected).count == 2)
    let counts = await separator.spectralCompilationCounts()
    #expect(counts.0 == 0 && counts.1 == 0)
    let after = try await separator.separate(input)
    #expect(arrayEqual(before.audio, after.audio).item(Bool.self))
  }
}
