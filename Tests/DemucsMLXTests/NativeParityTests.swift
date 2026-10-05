import Foundation
import MLX
import Testing

@testable import DemucsMLX

@Suite(
  .serialized, .enabled(if: ProcessInfo.processInfo.environment["DEMUCS_FIXTURE_DIRECTORY"] != nil))
struct NativeParityTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test(arguments: ["transformer", "hybrid", "wiener", "multifreq", "time"])
  func upstreamForward(_ name: String) throws {
    let root = URL(
      fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_FIXTURE_DIRECTORY"]!
    ).appendingPathComponent(name)
    let registry: DemucsModel =
      name == "transformer" ? .htdemucs : (name == "time" ? .mdx : .hdemucsMMI)
    let manifest = try CacheManifest.read(model: registry, directory: root, validateRegistry: false)
    let weights = try loadArrays(url: manifest.file)
    let model = try NativeModel(
      config: manifest.configurations[0], architecture: manifest.classes[0], arrays: weights,
      attention: .fp16)
    let fixture = try loadArrays(url: root.appendingPathComponent("forward.safetensors"))
    let input = try #require(fixture["audio"])
    let reference = try #require(fixture["torch"])
    let mlx = try #require(fixture["mlx"])
    let out = model.call(input[.newAxis])[0]
    eval(out)
    #expect(out.shape == reference.shape)
    let floor: Float = name == "transformer" ? 70 : (name == "time" ? 90 : 70)
    for i in 0..<4 {
      let upstreamSNR = snr(reference[i], out[i])
      let python = snr(mlx[i], out[i])
      print("| \(name) / \(i) | upstream **\(upstreamSNR) dB** | Python **\(python) dB** |")
      #expect(upstreamSNR > floor)
      // At >100 dB, backend reduction order dominates the difference.
      #expect(upstreamSNR > min(100, snr(reference[i], mlx[i])) - 1)
      #expect(python > 85)
    }
  }
  @Test func compiledMatchesEager() throws {
    let root = URL(
      fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_FIXTURE_DIRECTORY"]!
    ).appendingPathComponent("transformer")
    let manifest = try CacheManifest.read(
      model: .htdemucs, directory: root, validateRegistry: false)
    let model = try NativeModel(
      config: manifest.configurations[0], architecture: manifest.classes[0],
      arrays: loadArrays(url: manifest.file), attention: .fp16)
    let fixture = try loadArrays(url: root.appendingPathComponent("forward.safetensors"))
    let input = try #require(fixture["audio"])[.newAxis]
    let eager = model.forward(input, policy: .automatic)
    eval(eager)
    let compiled = model.forward(input, policy: .automatic)
    eval(compiled)
    #expect(snr(eager, compiled) > 85)
  }
  @Test(arguments: ["transformer", "hybrid"])
  func firstCompiledPreservesFirstAndCachedOutputs(_ name: String) throws {
    let root = URL(
      fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_FIXTURE_DIRECTORY"]!
    ).appendingPathComponent(name)
    let manifest = try CacheManifest.read(
      model: name == "transformer" ? .htdemucs : .hdemucsMMI, directory: root,
      validateRegistry: false)
    let weights = try loadArrays(url: manifest.file)
    func makeModel() throws -> NativeModel {
      try NativeModel(
        config: manifest.configurations[0], architecture: manifest.classes[0], arrays: weights,
        attention: .fp16)
    }
    let baseline = try makeModel()
    let candidate = try makeModel()
    let fixture = try loadArrays(url: root.appendingPathComponent("forward.safetensors"))
    let input = try #require(fixture["audio"])[.newAxis]
    for boundary in 0..<2 {
      let reference = baseline.forward(input, policy: .automatic)
      eval(reference)
      let actual = candidate.forward(input, policy: .firstCompiled)
      eval(actual)
      print("| first-compiled / \(name) / \(boundary) | **\(snr(reference, actual)) dB** |")
      #expect(reference.shape == actual.shape)
      #expect(sum((reference .!= actual).asType(.int32)).item(Int.self) == 0)
      #expect(candidate.arithmetic.divisor == nil)
      #expect(baseline.arithmetic.divisor == nil)
    }
    #expect(candidate.compiled.count == 1)
    #expect(baseline.compiled.count == 1)
  }
  @Test(
    .enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil),
    arguments: DemucsModel.allCases)
  func nativePCMValidationAndTransition(model: DemucsModel) async throws {
    let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!)
    var options = SeparationOptions()
    options.compilation = .firstCompiled
    options.batchSize = 2
    options.seed = 481
    let baseline = try Separator(model: model, cacheDirectory: root, options: options)
    options.cacheValidation = .verifiedIdentity
    let candidate = try Separator(model: model, cacheDirectory: root, options: options)
    let badInputs: [[Float]] = [[0, .nan, 0, 0], [0, .infinity, 0, 0], [0, -.infinity, 0, 0]]
    for samples in badInputs {
      await #expect(throws: DemucsError.self) {
        try await candidate.separate(samples: samples, channels: 2)
      }
    }
    await #expect(throws: DemucsError.self) {
      try await candidate.separate(samples: [0, 0, 0], channels: 2)
    }
    let samples = Array(repeating: Float(0.01), count: 2 * 4410)
    for _ in 0..<2 {
      let reference = try await baseline.separate(AudioTensor(MLXArray(samples, [2, 4410])))
      let actual = try await candidate.separate(samples: samples, channels: 2)
      #expect(reference.samples() == actual.samples())
    }
    var invalid = samples
    invalid[0] = .nan
    await #expect(throws: DemucsError.self) {
      try await candidate.separate(samples: invalid, channels: 2)
    }
  }
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil))
  func registryFirstCompiledTransition() throws {
    let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!)
    let manifest = try CacheManifest.read(model: .htdemucs, directory: root)
    let weights = try loadModelArrays(manifest, index: 0)
    func makeModel() throws -> NativeModel {
      try NativeModel(
        config: manifest.configurations[0], architecture: manifest.classes[0], arrays: weights,
        attention: .fp16)
    }
    let baseline = try makeModel()
    let candidate = try makeModel()
    let input = sin(MLXArray(0..<88200).asType(.float32) * 0.017).reshaped(1, 2, 44100)
    eval(input)
    for _ in 0..<2 {
      let reference = baseline.forward(input, policy: .automatic)
      eval(reference)
      let actual = candidate.forward(input, policy: .firstCompiled)
      eval(actual)
      #expect(sum((reference .!= actual).asType(.int32)).item(Int.self) == 0)
      #expect(candidate.arithmetic.divisor == nil)
    }
    #expect(candidate.compiled.count == 1)
  }
  private func snr(_ reference: MLXArray, _ estimate: MLXArray) -> Float {
    let error = reference - estimate
    return (10 * log10(sum(reference * reference) / maximum(sum(error * error), 1e-30))).item(
      Float.self)
  }
}
