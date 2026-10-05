import AVFoundation
import CryptoKit
import Foundation
import MLX
import Testing

@testable import DemucsAudio
@testable import DemucsMLX

private func temporaryDirectory() throws -> URL {
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  return url
}
private func digest(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

struct ModelHubTests {
  /// A local directory standing in for the hub, holding one tiny model.
  private func publish(weights: Data, config: Data) throws -> (URL, ModelHub.PublishedModel) {
    let hub = try temporaryDirectory()
    try weights.write(to: hub.appendingPathComponent("htdemucs.safetensors"))
    try config.write(to: hub.appendingPathComponent("htdemucs_config.json"))
    return (
      hub,
      .init(
        weights: .init(size: Int64(weights.count), sha256: digest(weights)),
        config: .init(size: Int64(config.count), sha256: digest(config)))
    )
  }
  private let weights = Data(repeating: 0x77, count: 5000)
  private let config = Data(#"{"k": 1}"#.utf8)

  @Test func everyRegistryModelIsPublished() {
    #expect(Set(ModelHub.published.keys) == Set(DemucsModel.allCases))
    for files in ModelHub.published.values {
      #expect(files.weights.size > 0 && files.weights.sha256.count == 64)
      #expect(files.config.size > 0 && files.config.sha256.count == 64)
    }
    #expect(ModelHub.downloadSize(of: .htdemucs) == 168_005_865 + 4215)
  }
  @Test func downloadWritesVerifiedFiles() async throws {
    let (hub, files) = try publish(weights: weights, config: config)
    let cache = try temporaryDirectory().appendingPathComponent("cache")
    #expect(!ModelHub.isCached(.htdemucs, in: cache))
    try await ModelHub.download(.htdemucs, to: cache, from: hub, expecting: files, progress: nil)
    #expect(ModelHub.isCached(.htdemucs, in: cache))
    #expect(try Data(contentsOf: cache.appendingPathComponent("htdemucs.safetensors")) == weights)
    #expect(try Data(contentsOf: cache.appendingPathComponent("htdemucs_config.json")) == config)
    #expect(try FileManager.default.contentsOfDirectory(atPath: cache.path).count == 2)
  }
  @Test(arguments: ["wrong-bytes", "truncated", "oversized", "wrong-config", "missing"])
  func badDownloadLeavesCacheUntouched(_ fault: String) async throws {
    let (hub, files) = try publish(weights: weights, config: config)
    let served = hub.appendingPathComponent("htdemucs.safetensors")
    switch fault {
    case "wrong-bytes": try Data(repeating: 0x78, count: 5000).write(to: served)
    case "truncated": try Data(repeating: 0x77, count: 4999).write(to: served)
    case "oversized": try Data(repeating: 0x77, count: 9000).write(to: served)
    case "wrong-config":
      try Data(#"{"k": 2}"#.utf8).write(to: hub.appendingPathComponent("htdemucs_config.json"))
    default: try FileManager.default.removeItem(at: served)
    }
    let cache = try temporaryDirectory()
    let existing = cache.appendingPathComponent("htdemucs.safetensors")
    try Data("existing".utf8).write(to: existing)
    await #expect(throws: DemucsError.self) {
      try await ModelHub.download(.htdemucs, to: cache, from: hub, expecting: files, progress: nil)
    }
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: cache.path) == [
        "htdemucs.safetensors"
      ])
    #expect(try Data(contentsOf: existing) == Data("existing".utf8))
  }
  @Test func neverPolicyDoesNotTouchTheNetwork() async throws {
    let cache = try temporaryDirectory()
    await #expect(throws: DemucsError.self) {
      _ = try await Separator.load(model: .htdemucs, cacheDirectory: cache, download: .never)
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: cache.path).isEmpty)
  }
}

struct ResultAndFormatTests {
  init() {
    _ = MetalTestResources.configure
  }
  private func result() -> SeparationResult {
    // Four stems, two channels, eight samples; stem i holds the constant i + 1.
    let audio = stacked((1...4).map { full([2, 8], values: Float($0)) })
    return SeparationResult(
      audio: audio, sources: ["drums", "bass", "other", "vocals"], sampleRate: 44100,
      statistics: InferenceStatistics(elapsedSeconds: 1, audioSeconds: 1, batchSize: 1))
  }
  @Test func stemsDictionaryMatchesSources() throws {
    let result = result()
    #expect(Set(result.stems.keys) == Set(result.sources))
    #expect(result.stems["bass"]!.asArray(Float.self) == Array(repeating: 2, count: 16))
  }
  @Test func twoStemsSumsTheRest() throws {
    let reduced = try result().twoStems("vocals")
    #expect(reduced.sources == ["vocals", "no_vocals"])
    #expect(reduced.audio.shape == [2, 2, 8])
    #expect(reduced.stems["vocals"]!.asArray(Float.self) == Array(repeating: 4, count: 16))
    #expect(reduced.stems["no_vocals"]!.asArray(Float.self) == Array(repeating: 6, count: 16))
    #expect(throws: DemucsError.self) { try result().twoStems("guitar") }
  }
  @Test(arguments: [
    AudioFileFormat.wav(.pcm16), .wav(.pcm24), .wav(.float32), .flac(), .flac(bitDepth: 24),
    .alac(), .alac(bitDepth: 24), .aac(),
  ])
  func savesEveryFormat(_ format: AudioFileFormat) throws {
    let frames = 44100
    let phase = MLXArray(0..<frames).asType(.float32) * Float(2 * Double.pi * 440 / 44100)
    let audio = stacked([sin(phase) * 0.25, cos(phase) * 0.25])
    let url = try temporaryDirectory().appendingPathComponent("tone." + format.fileExtension)
    try DemucsAudio.save(audio, sampleRate: 44100, to: url, format: format)
    let file = try AVAudioFile(forReading: url)
    #expect(file.fileFormat.sampleRate == 44100)
    #expect(file.fileFormat.channelCount == 2)
    let decoded = try DemucsAudio.load(url, sampleRate: 44100)
    #expect(decoded.dim(0) == 2)
    if case .aac = format {
      // AAC adds encoder delay and padding; only check it is about the right length.
      #expect(abs(decoded.dim(1) - frames) < 4096)
    } else {
      #expect(decoded.dim(1) == frames)
      let error = abs(decoded - audio).max().item(Float.self)
      #expect(error < (format == .wav(.float32) ? 1e-7 : 1e-4))
    }
  }
  @Test func rejectsInvalidFormatOptions() {
    #expect(throws: DemucsError.self) { try AudioFileFormat.flac(bitDepth: 20).validate() }
    #expect(throws: DemucsError.self) { try AudioFileFormat.aac(bitRate: 1).validate() }
  }
  @Test func exportNamesFilesByStemAndFormat() throws {
    let directory = try temporaryDirectory()
    try DemucsAudio.export(try result().twoStems("drums"), to: directory, format: .flac())
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == [
        "drums.flac", "no_drums.flac",
      ])
  }
}
