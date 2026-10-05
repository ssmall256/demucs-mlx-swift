import Foundation
import MLX
import Testing

@testable import DemucsMLX

extension NativeParityTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_FULL_FIXTURES"] != nil))
  func batchedAndBatchOneTailAgreement() async throws {
    let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_FULL_FIXTURES"]!)
    let cache = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!)
    let input = AudioTensor(
      try #require(loadArrays(url: root.appendingPathComponent("audio_30.safetensors"))["audio"]))
    var options = SeparationOptions()
    options.seed = 481
    options.batchSize = 1
    let single = try Separator(cacheDirectory: cache, options: options)
    let first = try await single.separate(input)
    options.batchSize = 8
    let batch = try Separator(cacheDirectory: cache, options: options)
    let second = try await batch.separate(input)
    #expect(first.audio.shape == [4, 2, 30 * 44100])
    let delta = first.audio - second.audio
    let agreement =
      (10 * log10(sum(first.audio * first.audio) / maximum(sum(delta * delta), 1e-30))).item(
        Float.self)
    print("> Batch-one tail agreement: **\(agreement) dB**")
    #expect(agreement > 60)
  }
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil))
  func memoryFailureAndInputValidation() async throws {
    let cache = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!)
    var options = SeparationOptions()
    options.memoryBudgetBytes = 1
    let separator = try Separator(cacheDirectory: cache, options: options)
    do {
      _ = try await separator.separate(AudioTensor(zeros([2, 1000])))
      Issue.record("Expected an explicit memory budget failure")
    } catch DemucsError.memoryBudget {}
    do {
      _ = try await separator.separate(AudioTensor(zeros([1, 2, 100])))
      Issue.record("Expected invalid tensor rank rejection")
    } catch DemucsError.invalidInput {}
    let cancelled = Task { try await separator.separate(AudioTensor(zeros([2, 1000]))) }
    cancelled.cancel()
    do {
      _ = try await cancelled.value
      Issue.record("Expected cancellation")
    } catch is CancellationError {}
  }
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil))
  func fineTunedStemSelection() async throws {
    let cache = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!)
    var options = SeparationOptions()
    options.stem = "vocals"
    let separator = try Separator(model: .htdemucsFT, cacheDirectory: cache, options: options)
    #expect(separator.sources == ["vocals"])
    let selection = await separator.selection
    #expect(selection == [3])
  }
}
