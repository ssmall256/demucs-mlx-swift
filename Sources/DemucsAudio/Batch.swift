import AVFoundation
@_spi(Profiling) import DemucsMLX
import Foundation
import MLX

public struct AudioBatchOptions: Sendable {
  /// Additional decoded tracks kept ahead of inference: zero or one.
  public var prefetchDepth: Int
  public var writerWorkers: Int
  /// Additional resident inputs/results beyond the active inference track.
  public var overlapMemoryBudgetBytes: Int
  /// Stem file format. `float32` is a shorthand for the two fast WAV encodings.
  public var format: AudioFileFormat = .wav(.pcm16)
  public var float32: Bool {
    get { format == .wav(.float32) }
    set { format = .wav(newValue ? .float32 : .pcm16) }
  }
  /// Write only this stem and the sum of the others (`<stem>` and `no_<stem>`).
  public var twoStems: String? = nil
  public var decoder: AudioDecoder = .automatic
  /// Hard platform ceiling; callers may request a smaller overlap allowance.
  public static var maximumOverlapMemoryBudgetBytes: Int {
    #if os(iOS)
      Int(min(64 * 1_048_576, ProcessInfo.processInfo.physicalMemory / 8))
    #else
      Int(min(512 * 1_048_576, ProcessInfo.processInfo.physicalMemory / 8))
    #endif
  }
  public init() {
    overlapMemoryBudgetBytes = Self.maximumOverlapMemoryBudgetBytes
    #if os(iOS)
      prefetchDepth = 0
      writerWorkers = 1
    #else
      prefetchDepth = 1
      writerWorkers = 2
    #endif
  }
}
public struct AudioFileReport: Codable, Sendable {
  public let input: URL
  public let directory: URL
  public let sources: [String]
  public let statistics: InferenceStatistics
  public let decodeSeconds: Double
  public let exportSeconds: Double
  public let elapsedSeconds: Double
  public let peakOverlapBytes: Int
}
public struct AudioBatchProgress: Sendable {
  public enum Stage: String, Sendable { case decoding, separating, exporting, completed }
  public let index: Int
  public let count: Int
  public let stage: Stage
  public let separation: SeparationProgress?
}

/// Registers detached work for cancellation and bounds extra resident tracks.
final class AudioBatchControl: @unchecked Sendable {
  private let lock = NSLock()
  private var failure: Error?
  private var cancellations: [UUID: @Sendable () -> Void] = [:]
  private var reserved = 0
  private var peak = 0
  let limit: Int
  init(limit: Int) { self.limit = limit }
  func reserve(_ bytes: Int) -> Bool {
    lock.withLock {
      guard bytes >= 0, bytes <= limit - reserved, failure == nil else { return false }
      reserved += bytes
      peak = max(peak, reserved)
      return true
    }
  }
  func release(_ bytes: Int) {
    lock.withLock {
      reserved -= bytes
      assert(reserved >= 0)
    }
  }
  var peakBytes: Int { lock.withLock { peak } }
  func register(_ cancellation: @escaping @Sendable () -> Void) -> UUID {
    let key = UUID()
    let failed = lock.withLock {
      cancellations[key] = cancellation
      return failure != nil
    }
    if failed { cancellation() }
    return key
  }
  func unregister(_ key: UUID) { _ = lock.withLock { cancellations.removeValue(forKey: key) } }
  func fail(_ error: Error) {
    let actions = lock.withLock {
      if failure == nil { failure = error }
      return Array(cancellations.values)
    }
    for action in actions { action() }
  }
  func check() throws {
    if let error = lock.withLock({ failure }) { throw error }
    try Task.checkCancellation()
  }
}

struct AudioMemoryEstimate {
  let decodedBytes: Int
  let frames: Int
  func outputBytes(channels: Int, sources: Int) -> Int? {
    guard channels > 0, sources > 0 else { return nil }
    var bytes = frames
    for factor in [channels, sources, 4] {
      let product = bytes.multipliedReportingOverflow(by: factor)
      guard !product.overflow else { return nil }
      bytes = product.partialValue
    }
    return bytes
  }
}

struct DecodedTrack: Sendable {
  let input: AudioTensor
  let seconds: Double
  let started: ContinuousClock.Instant
}
struct PendingDecode {
  let task: Task<DecodedTrack, Error>
  let registration: UUID
  let bytes: Int
}
struct PendingExport {
  let task: Task<AudioFileReport, Error>
  let registration: UUID
}
struct PreparedStem: Sendable {
  let samples: PreparedAudio
  let destination: URL
}

extension DemucsAudio {
  static func elapsed(_ start: ContinuousClock.Instant) -> Double {
    let d = start.duration(to: .now)
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
  }
  static func destinations(_ inputs: [URL], root: URL) throws -> [URL] {
    var used = Set<String>()
    return try inputs.map { input in
      guard input.isFileURL, root.isFileURL else {
        throw DemucsError.invalidInput("Batch inputs and output root must be file URLs")
      }
      let directory = root.appendingPathComponent(input.deletingPathExtension().lastPathComponent)
      let key = directory.standardizedFileURL.resolvingSymlinksInPath().path.lowercased()
      guard used.insert(key).inserted else {
        throw DemucsError.invalidInput(
          "Batch inputs have colliding output directories: \(directory.path)")
      }
      return directory
    }
  }
  static func estimateAudioMemory(_ url: URL, sampleRate: Int) throws -> AudioMemoryEstimate {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    guard file.length > 1, format.sampleRate.isFinite, format.sampleRate > 0 else {
      throw DemucsError.invalidInput("Invalid audio file metadata")
    }
    let frames = ceil(Double(file.length) * Double(sampleRate) / format.sampleRate) + 64
    let estimate =
      (frames * 4 + Double(getpagesize())) * Double(format.channelCount)
      + Double(65536 * 4) * Double(format.channelCount)
    guard estimate.isFinite, estimate < Double(Int.max) else {
      throw DemucsError.invalidInput("Audio estimate exceeds native capacity")
    }
    return AudioMemoryEstimate(decodedBytes: Int(estimate), frames: Int(frames))
  }
  static func prepare(
    _ result: SeparationResult, directory: URL, format: AudioFileFormat = .wav(.pcm16)
  ) throws -> [PreparedStem] {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    guard Set(result.sources.map { $0.lowercased() }).count == result.sources.count else {
      throw DemucsError.invalidCache("Colliding stem filenames")
    }
    return try result.sources.enumerated().map { index, name in
      guard !name.contains("/"), !name.contains("\\"), name != ".", name != "..", !name.isEmpty
      else {
        throw DemucsError.invalidCache("Invalid stem filename")
      }
      return PreparedStem(
        samples: try PreparedAudio(result.audio[index]),
        destination: directory.appendingPathComponent(name + "." + format.fileExtension))
    }
  }
  static func write(
    _ stems: [PreparedStem], sampleRate: Int, workers: Int, format: AudioFileFormat,
    control: AudioBatchControl
  ) async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
      var next = 0
      func add(_ stem: PreparedStem) {
        group.addTask {
          do {
            try control.check()
            try stem.samples.save(sampleRate: sampleRate, to: stem.destination, format: format)
          } catch {
            control.fail(error)
            throw error
          }
        }
      }
      while next < min(workers, stems.count) {
        add(stems[next])
        next += 1
      }
      while try await group.next() != nil {
        try control.check()
        if next < stems.count {
          add(stems[next])
          next += 1
        }
      }
    }
  }
  /// Separate files with one serialized model session and bounded decode/export overlap.
  /// Reports are returned in input order; callbacks may arrive from worker executors.
  /// Already completed files are preserved if another file fails or the caller cancels.
  public static func separateAndExport(
    _ inputs: [URL], using separator: Separator, to root: URL,
    options: AudioBatchOptions = .init(),
    progress: (@Sendable (AudioBatchProgress) -> Void)? = nil
  ) async throws -> [AudioFileReport] {
    guard (0...1).contains(options.prefetchDepth), (1...16).contains(options.writerWorkers),
      options.overlapMemoryBudgetBytes >= 0
    else {
      throw DemucsError.invalidOptions(
        "Prefetch must be 0...1, writers 1...16, and overlap budget nonnegative")
    }
    try options.format.validate()
    if let name = options.twoStems {
      guard separator.sources.count > 1, separator.sources.contains(name) else {
        throw DemucsError.invalidOptions(
          "Two-stem output needs one of: \(separator.sources.joined(separator: ", "))")
      }
    }
    let directories = try destinations(inputs, root: root)
    let control = AudioBatchControl(
      limit: min(
        options.overlapMemoryBudgetBytes, AudioBatchOptions.maximumOverlapMemoryBudgetBytes))
    return try await withTaskCancellationHandler {
      var decode: PendingDecode?
      var export: PendingExport?
      var reports: [AudioFileReport] = []
      func startDecode(_ index: Int, bytes: Int) -> PendingDecode {
        let task = Task.detached(priority: .userInitiated) {
          do {
            try control.check()
            progress?(
              AudioBatchProgress(
                index: index, count: inputs.count, stage: .decoding, separation: nil))
            let started = ContinuousClock.now
            let input = AudioTensor(
              try load(
                inputs[index], sampleRate: separator.sampleRate,
                decoder: options.decoder))
            try control.check()
            return DecodedTrack(input: input, seconds: elapsed(started), started: started)
          } catch {
            control.fail(error)
            throw error
          }
        }
        return PendingDecode(
          task: task, registration: control.register { task.cancel() }, bytes: bytes)
      }
      func finishExport() async throws {
        if let pending = export {
          do {
            let report = try await pending.task.value
            control.unregister(pending.registration)
            export = nil
            reports.append(report)
          } catch {
            control.unregister(pending.registration)
            export = nil
            throw error
          }
        }
      }
      do {
        for index in inputs.indices {
          try control.check()
          let estimate = try estimateAudioMemory(inputs[index], sampleRate: separator.sampleRate)
          let estimatedOutputBytes = estimate.outputBytes(
            channels: separator.audioChannels, sources: separator.sources.count)
          let serial =
            estimate.decodedBytes > control.limit
            || (estimatedOutputBytes.map { $0 > control.limit } ?? true)
          if serial { try await finishExport() }
          let pending = decode ?? startDecode(index, bytes: 0)
          decode = pending
          let decoded = try await pending.task.value
          control.unregister(pending.registration)
          if pending.bytes > 0 { control.release(pending.bytes) }
          decode = nil
          // At most one prefetched input, plus one completed result being exported.
          if !serial && options.prefetchDepth > 0 && index + 1 < inputs.count {
            let bytes = try estimateAudioMemory(
              inputs[index + 1], sampleRate: separator.sampleRate
            ).decodedBytes
            if control.reserve(bytes) { decode = startDecode(index + 1, bytes: bytes) }
          }
          progress?(
            AudioBatchProgress(
              index: index, count: inputs.count, stage: .separating, separation: nil))
          let inference = Task {
            try await separator.separate(decoded.input) { update in
              progress?(
                AudioBatchProgress(
                  index: index, count: inputs.count, stage: .separating, separation: update))
            }
          }
          let registration = control.register { inference.cancel() }
          let result: SeparationResult
          do {
            let separated = try await inference.value
            result = try options.twoStems.map { try separated.twoStems($0) } ?? separated
          } catch {
            control.unregister(registration)
            throw error
          }
          control.unregister(registration)
          try control.check()
          try await finishExport()
          let stems = try prepare(
            result, directory: directories[index], format: options.format)
          let outputBytes = result.audio.nbytes
          let overlap = !serial && index + 1 < inputs.count && control.reserve(outputBytes)
          let statistics = result.statistics
          let sources = result.sources
          let destination = directories[index]
          let inputURL = inputs[index]
          let decodeSeconds = decoded.seconds
          let decodedStarted = decoded.started
          let writer = Task.detached(priority: .userInitiated) {
            defer { if overlap { control.release(outputBytes) } }
            do {
              try control.check()
              progress?(
                AudioBatchProgress(
                  index: index, count: inputs.count, stage: .exporting, separation: nil))
              let start = ContinuousClock.now
              let trace = PerformanceTrace.begin("audio.export")
              defer { PerformanceTrace.end(trace) }
              try await write(
                stems, sampleRate: separator.sampleRate, workers: options.writerWorkers,
                format: options.format, control: control)
              let seconds = elapsed(start)
              progress?(
                AudioBatchProgress(
                  index: index, count: inputs.count, stage: .completed, separation: nil))
              return AudioFileReport(
                input: inputURL, directory: destination, sources: sources,
                statistics: statistics, decodeSeconds: decodeSeconds, exportSeconds: seconds,
                elapsedSeconds: elapsed(decodedStarted), peakOverlapBytes: control.peakBytes)
            } catch {
              control.fail(error)
              throw error
            }
          }
          export = PendingExport(task: writer, registration: control.register { writer.cancel() })
          if !overlap { try await finishExport() }
        }
        try await finishExport()
        return reports
      } catch {
        control.fail(error)
        if let pending = decode {
          _ = await pending.task.result
          control.unregister(pending.registration)
          if pending.bytes > 0 { control.release(pending.bytes) }
        }
        if let pending = export {
          _ = await pending.task.result
          control.unregister(pending.registration)
        }
        try control.check()
        throw error
      }
    } onCancel: {
      control.fail(CancellationError())
    }
  }
}
