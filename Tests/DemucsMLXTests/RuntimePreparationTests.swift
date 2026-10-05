import Foundation
import MLX
import Testing

@testable import DemucsMLX

struct RuntimePreparationTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test func parallelSampleCopyMatchesContiguousReadback() throws {
    // A row of a larger array has a nonzero storage offset and crosses the
    // multi-threaded copy threshold.
    let count = 3 * 2 * 400_003
    let values = (0..<count).map { Float($0 % 9973) * 0.5 - Float($0 % 7) }
    let backing = MLXArray(values, [3, 2, count / 6])
    let view = backing[1]
    let result = SeparationResult(
      audio: view, sources: ["a", "b"], sampleRate: 44100,
      statistics: InferenceStatistics(
        elapsedSeconds: 1, audioSeconds: 1, batchSize: 1))
    let expected = view.asArray(Float.self)
    let copied = result.samples()
    #expect(copied.count == expected.count)
    #expect(copied.elementsEqual(expected) { $0.bitPattern == $1.bitPattern })
    var prepared = result
    prepared.prepared = PreparedSamples(count: view.size)
    let first = prepared.samples()
    let second = prepared.samples()
    #expect(first.elementsEqual(expected) { $0.bitPattern == $1.bitPattern })
    #expect(second.elementsEqual(expected) { $0.bitPattern == $1.bitPattern })
  }
  @Test func assetVerificationNeedsNoSession() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    #expect(throws: DemucsError.self) {
      try Separator.verifyAssets(model: .htdemucs, cacheDirectory: directory)
    }
  }
}
