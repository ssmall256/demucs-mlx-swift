import ArgumentParser
import DemucsAudio
@_spi(Profiling) import DemucsMLX
import Foundation
import MLX

extension DemucsModel: ExpressibleByArgument {}
extension AttentionPrecision: ExpressibleByArgument {}
extension CompilationPolicy: ExpressibleByArgument {}
extension CacheValidationPolicy: ExpressibleByArgument {}

@main struct DemucsCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "demucs-mlx-swift", abstract: "Native MLX music source separation",
    version: "0.1.0",
    subcommands: [Separate.self, Benchmark.self, Tensor.self, Tune.self, VerifyAssets.self],
    defaultSubcommand: Separate.self)
}

struct VerifyAssets: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "verify-assets",
    abstract: "Hash and validate model assets, preparing a receipt without GPU initialization")
  @Option(help: "Official registry model") var model: DemucsModel = .htdemucs
  @Option(help: "Directory containing versioned safetensors and JSON")
  var cache: String = Separator.defaultCacheDirectory.path
  func run() throws {
    defer { PerformanceTrace.flush() }
    try Separator.verifyAssets(model: model, cacheDirectory: URL(fileURLWithPath: cache))
    print(
      "✅ **Verified \(model.rawValue) assets.** Writable caches now contain a verified-digest receipt."
    )
  }
}
struct CommonOptions: ParsableArguments {
  @Option(name: [.short, .customLong("model")], help: "Official registry model") var model:
    DemucsModel = .htdemucs
  @Option(help: "Model directory; models are downloaded here when missing") var cache: String =
    Separator.defaultCacheDirectory.path
  @Flag(help: "Never download a missing model") var noDownload = false
  @Option(help: "Number of random shifts; zero disables shift averaging") var shifts: Int = 1
  @Option var overlap: Float = 0.25
  @Option var seed: UInt64?
  @Option var batchSize: Int?
  @Option var segment: Double?
  @Option var attention: AttentionPrecision = .fp16
  @Option(help: "Graph policy: automatic, first-compiled or eager")
  var compilation: CompilationPolicy = .automatic
  @Option(help: "Asset verification: always-hash or an unchanged verified-identity receipt")
  var cacheValidation: CacheValidationPolicy = .alwaysHash
  @Option var stem: String?
  @Option(help: "Explicit inference memory budget in MiB") var memoryBudget: Int?
  func separator() throws -> Separator {
    var options = SeparationOptions()
    options.shifts = shifts
    options.overlap = overlap
    options.seed = seed
    options.batchSize = batchSize
    options.segmentSeconds = segment
    options.attention = attention
    options.compilation = compilation
    options.cacheValidation = cacheValidation
    options.stem = stem
    if let memoryBudget {
      guard memoryBudget > 0, memoryBudget < Int.max / 1_048_576 else {
        throw ValidationError("Invalid memory budget")
      }
      options.memoryBudgetBytes = memoryBudget * 1_048_576
    }
    return try Separator(
      model: model, cacheDirectory: URL(fileURLWithPath: cache), options: options)
  }
  /// Fetch the model into the cache directory if it is not there yet.
  func prepareModel() async throws {
    let directory = URL(fileURLWithPath: cache)
    guard !noDownload, ModelHub.downloadsEnabled, !ModelHub.isCached(model, in: directory) else {
      return
    }
    let megabytes = ModelHub.downloadSize(of: model) / 1_048_576
    FileHandle.standardError.write(
      Data(
        "Downloading \(model.rawValue) (\(megabytes) MB) from \(ModelHub.baseURL.absoluteString) ...\n"
          .utf8))
    try await ModelHub.download(model, to: directory)
  }
}
struct Separate: AsyncParsableCommand {
  static let configuration = CommandConfiguration(abstract: "Separate a track into WAV stems")
  @OptionGroup var common: CommonOptions
  @Argument var input: [String] = []
  @Option(name: .shortAndLong) var output: String = "separated"
  @Flag(help: "Write 32-bit float WAV") var float32 = false
  @Option(help: "Stem format: wav, wav24, wav-float32, flac, flac24, alac, alac24 or aac")
  var format: String?
  @Option(help: "AAC bit rate in kbit/s") var aacBitrate: Int = 256
  @Option(help: "Write only this stem and the sum of the others (for example vocals)")
  var twoStems: String?
  @Flag var listModels = false
  @Option(help: "Bounded CPU stem writers (default: 2 on macOS)") var writeWorkers: Int?
  @Option(help: "Tracks decoded ahead of inference: 0 or 1") var prefetch: Int?
  @Option(help: "Additional resident input/output budget in MiB") var ioMemoryBudget: Int?
  @Option(help: "Write per-file stage timings as JSON") var json: String?
  @Option(help: "Audio decoder: automatic, apple or native-pcm (PCM WAV at native rate)")
  var audioDecoder: String = "automatic"
  func run() async throws {
    defer { PerformanceTrace.flush() }
    if listModels {
      print(DemucsModel.allCases.map(\.rawValue).joined(separator: "\n"))
      return
    }
    guard !input.isEmpty else { throw ValidationError("Provide audio files or --list-models") }
    let started = ContinuousClock.now
    try await common.prepareModel()
    let separator = try common.separator()
    var options = AudioBatchOptions()
    guard let decoder = AudioDecoder(rawValue: audioDecoder) else {
      throw ValidationError("Audio decoder must be automatic, apple or native-pcm")
    }
    options.decoder = decoder
    options.float32 = float32
    if let format {
      switch format.lowercased() {
      case "wav", "wav16": options.format = .wav(.pcm16)
      case "wav24": options.format = .wav(.pcm24)
      case "wav-float32", "float32": options.format = .wav(.float32)
      case "flac": options.format = .flac()
      case "flac24": options.format = .flac(bitDepth: 24)
      case "alac": options.format = .alac()
      case "alac24": options.format = .alac(bitDepth: 24)
      case "aac", "m4a": options.format = .aac(bitRate: aacBitrate * 1000)
      default:
        throw ValidationError(
          "Format must be wav, wav24, wav-float32, flac, flac24, alac, alac24 or aac")
      }
    }
    options.twoStems = twoStems
    if let writeWorkers { options.writerWorkers = writeWorkers }
    if let prefetch { options.prefetchDepth = prefetch }
    if let ioMemoryBudget {
      guard ioMemoryBudget >= 0, ioMemoryBudget < Int.max / 1_048_576 else {
        throw ValidationError("Invalid I/O memory budget")
      }
      options.overlapMemoryBudgetBytes = ioMemoryBudget * 1_048_576
    }
    let reports = try await DemucsAudio.separateAndExport(
      input.map { URL(fileURLWithPath: $0) }, using: separator,
      to: URL(fileURLWithPath: output).appendingPathComponent(common.model.rawValue),
      options: options)
    for report in reports {
      print("✅ \(report.sources.count) stems → \(report.directory.path)")
      print(
        String(
          format:
            "**Decode:** %.3f s · **Inference:** %.3f s · **Export:** %.3f s · **Inference RTFx:** %.1f×",
          report.decodeSeconds, report.statistics.elapsedSeconds, report.exportSeconds,
          report.statistics.rtfx))
    }
    let duration = started.duration(to: .now)
    let elapsed =
      Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    let seconds = reports.reduce(0) { $0 + $1.statistics.audioSeconds }
    print(
      String(
        format: "**Total:** %.3f s · **File-to-stems RTFx:** %.1f×", elapsed,
        seconds / max(elapsed, 1e-12)))
    if let json {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(reports).write(to: URL(fileURLWithPath: json), options: .atomic)
    }
  }
}
// CPU-only fixture reading keeps MLX initialization inside cold-runtime timing.
private func readCPUAudioFixture(_ url: URL) throws -> (samples: [Float], shape: [Int]) {
  let data = try Data(contentsOf: url)
  guard data.count >= 8 else { throw ValidationError("Truncated audio fixture") }
  let headerSize = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
  guard headerSize > 0, headerSize <= 16_777_216, headerSize <= data.count - 8 else {
    throw ValidationError("Invalid audio fixture header")
  }
  let base = 8 + Int(headerSize)
  guard let header = try JSONSerialization.jsonObject(with: data[8..<base]) as? [String: Any],
    let audio = header["audio"] as? [String: Any], audio["dtype"] as? String == "F32",
    let shape = audio["shape"] as? [Int], shape.count == 2, shape[0] > 0, shape[1] > 1,
    let offsets = audio["data_offsets"] as? [Int], offsets.count == 2,
    offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= data.count - base
  else { throw ValidationError("Expected Float32 [channels,samples] audio fixture") }
  let count = shape[0].multipliedReportingOverflow(by: shape[1])
  let size = count.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
  guard !count.overflow, !size.overflow, size.partialValue == offsets[1] - offsets[0] else {
    throw ValidationError("Audio fixture shape and payload disagree")
  }
  let samples = [Float](unsafeUninitializedCapacity: count.partialValue) { target, initialized in
    data.withUnsafeBytes { bytes in
      UnsafeMutableRawPointer(target.baseAddress!).copyMemory(
        from: bytes.baseAddress!.advanced(by: base + offsets[0]), byteCount: size.partialValue)
    }
    initialized = count.partialValue
  }
  return (samples, shape)
}

struct Benchmark: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Benchmark materialized native inference")
  @OptionGroup var common: CommonOptions
  @Option var seconds: Int = 30
  @Option var warmup: Int = 2
  @Option var iterations: Int = 5
  @Flag(help: "Include CPU input upload and stem readback in the measured call") var cpuRoundtrip =
    false
  @Flag(
    help:
      "Include session construction and first MLX initialization; requires CPU roundtrip, fixture and zero warmups"
  )
  var coldRuntime = false
  @Option(help: "Seconds of cooldown before each measured call") var cooldown: Double = 0
  @Flag(help: "Wait for nominal thermal state before each measured call") var thermalGate = false
  @Flag(help: "Warm once, then accept run lines on stdin and emit JSON measurements")
  var interactive = false
  @Option(help: "Optional safetensors fixture containing audio") var fixture: String?
  @Option(help: "Write machine-readable measurements") var json: String?
  @Option(help: "Save the final materialized stems for parity checking") var outputFixture: String?
  @Option(help: "Save the first measured stems after timing, for cold-call parity checking")
  var firstOutputFixture: String?
  func run() async throws {
    defer { PerformanceTrace.flush() }
    guard seconds > 0, seconds <= 600, warmup >= 0, iterations > 0,
      cooldown.isFinite, (0...60).contains(cooldown)
    else {
      throw ValidationError("Invalid benchmark duration or iteration count")
    }
    guard !coldRuntime || (cpuRoundtrip && fixture != nil && warmup == 0) else {
      throw ValidationError("Cold runtime requires --cpu-roundtrip, --fixture and --warmup 0")
    }
    try await common.prepareModel()
    var separator: Separator?
    var initializationSeconds = 0.0
    func session() throws -> Separator {
      if let separator { return separator }
      let started = ContinuousClock.now
      let loaded = try common.separator()
      let duration = started.duration(to: .now)
      initializationSeconds =
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
      separator = loaded
      return loaded
    }
    let input: MLXArray?
    let inputShape: [Int]
    let cpuInput: [Float]?
    if coldRuntime {
      let fixture = try readCPUAudioFixture(URL(fileURLWithPath: fixture!))
      input = nil
      inputShape = fixture.shape
      cpuInput = fixture.samples
    } else {
      let separator = try session()
      let audio: MLXArray
      if let fixture {
        guard let value = try loadArrays(url: URL(fileURLWithPath: fixture))["audio"] else {
          throw ValidationError("Fixture has no audio tensor")
        }
        audio = value
      } else {
        let t =
          MLXArray(0..<(seconds * separator.sampleRate)).asType(.float32)
          / Float(separator.sampleRate)
        let tones = (sin(t * (2 * Float.pi * 220)) + 0.5 * sin(t * (2 * Float.pi * 440))) * 0.05
        audio = stacked([tones, tones])
      }
      eval(audio)
      input = audio
      inputShape = audio.shape
      cpuInput = cpuRoundtrip ? audio.asArray(Float.self) : nil
    }
    func infer() async throws -> SeparationResult {
      let separator = try session()
      let result: SeparationResult
      if let cpuInput {
        result = try await separator.separate(samples: cpuInput, channels: inputShape[0])
      } else {
        result = try await separator.separate(AudioTensor(input!))
      }
      if cpuRoundtrip {
        let outputTrace = PerformanceTrace.begin("cpu.output.materialize")
        _ = result.samples()
        PerformanceTrace.end(outputTrace, gpuCompleted: true)
      }
      return result
    }
    func cool() async throws {
      let deadline = ContinuousClock.now.advanced(by: .seconds(300))
      repeat {
        if thermalGate {
          while ProcessInfo.processInfo.thermalState != .nominal {
            guard ContinuousClock.now < deadline else {
              throw ValidationError("Thermal gate timed out before a measurement")
            }
            try await Task.sleep(for: .seconds(1))
          }
        }
        if cooldown > 0 { try await Task.sleep(for: .seconds(cooldown)) }
        if !thermalGate || ProcessInfo.processInfo.thermalState == .nominal { return }
        guard ContinuousClock.now < deadline else {
          throw ValidationError("Thermal gate timed out before a measurement")
        }
      } while true
    }
    var coldInference: Double?
    for i in 0..<warmup {
      let started = ContinuousClock.now
      _ = try await infer()
      if i == 0 {
        let duration = started.duration(to: .now)
        coldInference =
          Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
      }
    }
    func emit(_ record: [String: Any]) throws {
      var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
      data.append(10)
      try FileHandle.standardOutput.write(contentsOf: data)
    }
    if interactive {
      try emit([
        "event": "ready",
        "audio_seconds": Double(inputShape[1]) / Double(separator?.sampleRate ?? 44100),
      ])
    }
    var timings: [Double] = []
    var thermalStates: [[Int]] = []
    var effectiveBatchSize = 0
    print("## 🚀 Native Demucs benchmark\n\n| Run | Time | RTFx |\n|---|---:|---:|")
    for i in 0..<iterations {
      if interactive {
        guard readLine() == "run" else {
          throw ValidationError("Interactive benchmark expects a run line for each measurement")
        }
      }
      try await cool()
      let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
      let callTrace = PerformanceTrace.begin("benchmark.complete-call")
      let started = ContinuousClock.now
      let result = try await infer()
      let duration = started.duration(to: .now)
      PerformanceTrace.end(callTrace, gpuCompleted: true)
      let elapsed =
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
      timings.append(elapsed)
      effectiveBatchSize = result.statistics.batchSize
      thermalStates.append([thermalBefore, ProcessInfo.processInfo.thermalState.rawValue])
      if i == 0, let firstOutputFixture {
        try save(arrays: ["stems": result.audio], url: URL(fileURLWithPath: firstOutputFixture))
      }
      if i == iterations - 1, let outputFixture {
        try save(arrays: ["stems": result.audio], url: URL(fileURLWithPath: outputFixture))
      }
      if interactive {
        try emit([
          "event": "measurement", "iteration": i, "seconds": elapsed,
          "rtfx": result.statistics.audioSeconds / elapsed,
          "thermal_states": thermalStates.last!, "effective_batch_size": effectiveBatchSize,
        ])
      }
      print(
        String(
          format: "| %d | %.4f s | **%.1f×** |", i + 1, elapsed,
          result.statistics.audioSeconds / elapsed))
    }
    let sorted = timings.sorted()
    let median = (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    let duration = Double(inputShape[1]) / Double(separator!.sampleRate)
    print(String(format: "\n> ✅ **Median: %.4f s · %.1f× RTFx**", median, duration / median))
    if let json {
      let data = try JSONSerialization.data(
        withJSONObject: [
          "model": common.model.rawValue, "times": timings,
          "cpu_roundtrip": cpuRoundtrip, "cold_runtime": coldRuntime, "cooldown_seconds": cooldown,
          "thermal_states": thermalStates, "thermal_gate": thermalGate,
          "effective_batch_size": effectiveBatchSize,
          "median": median, "rtfx": duration / median, "audio_seconds": duration,
          "peak_memory_bytes": Memory.peakMemory,
          "initialization_seconds": initializationSeconds,
          "initialization_in_first_timer": coldRuntime,
          "cold_inference_seconds": coldInference.map { $0 as Any } ?? NSNull(),
        ], options: [.prettyPrinted, .sortedKeys])
      try data.write(to: URL(fileURLWithPath: json), options: .atomic)
    }
  }
}
struct Tensor: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Separate an audio safetensors fixture without file I/O conversion")
  @OptionGroup var common: CommonOptions
  @Argument var input: String
  @Argument var output: String
  @Flag(help: "Use the native CPU PCM SDK entry point") var cpuInput = false
  func run() async throws {
    defer { PerformanceTrace.flush() }
    guard let audio = try loadArrays(url: URL(fileURLWithPath: input))["audio"] else {
      throw ValidationError("Input must contain an audio tensor")
    }
    try await common.prepareModel()
    let separator = try common.separator()
    let result =
      cpuInput
      ? try await separator.separate(samples: audio.asArray(Float.self), channels: audio.dim(0))
      : try await separator.separate(audio)
    try save(arrays: ["stems": result.audio], url: URL(fileURLWithPath: output))
    print(
      String(
        format: "✅ **%.4f s · %.1f× RTFx**", result.statistics.elapsedSeconds,
        result.statistics.rtfx))
  }
}

struct Tune: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Explicitly tune spectral kernels for this device")
  @OptionGroup var common: CommonOptions
  @Option(help: "Save the complete tuning report as JSON") var json: String?
  func run() async throws {
    defer { PerformanceTrace.flush() }
    try await common.prepareModel()
    let separator = try common.separator()
    let report = try await separator.tuneSpectralKernels()
    print(
      "## Spectral kernel tuning\n\n| Kernel | Shape | Width | Tile | Milliseconds |\n|---|---|---:|---:|---:|"
    )
    for row in report.measurements where row.selected {
      print(
        "| \(row.kernel) | `\(row.shape)` | \(row.threadWidth) | \(row.tileFrames) | **\(row.seconds * 1000)** |"
      )
    }
    print("> ✅ Cached choices: `\(report.cacheURL.path)`")
    if let json {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(report).write(to: URL(fileURLWithPath: json), options: .atomic)
    }
  }
}
