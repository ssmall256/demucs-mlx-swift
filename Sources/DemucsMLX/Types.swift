import Foundation
import MLX

/// Models in the official demucs-mlx registry. Quantized names refer to the
/// original DiffQ checkpoints, exported as ordinary floating-point MLX weights.
public enum DemucsModel: String, CaseIterable, Codable, Sendable {
  case htdemucs
  case htdemucsFT = "htdemucs_ft"
  case htdemucs6s = "htdemucs_6s"
  case hdemucsMMI = "hdemucs_mmi"
  case mdx
  case mdxExtra = "mdx_extra"
  case mdxQ = "mdx_q"
  case mdxExtraQ = "mdx_extra_q"
}
public enum AttentionPrecision: String, Codable, Sendable { case fp16, fp32 }
public enum CompilationPolicy: String, Codable, Sendable {
  /// Eager first use, followed by cached compiled forwards.
  case automatic
  /// Compile eligible HTDemucs first forwards while preserving eager arithmetic.
  /// Other architectures and unsupported configurations use the automatic path.
  /// Precompiled Metal libraries additionally avoid runtime shader compilation.
  case firstCompiled = "first-compiled"
  /// Keep every forward eager.
  case eager
}
public enum CacheValidationPolicy: String, Codable, Sendable {
  /// Read and hash every asset on each session construction.
  case alwaysHash = "always-hash"
  /// Reuse a verified digest only while the complete file identity matches.
  /// Missing, changed or malformed receipts fall back to full hashing.
  case verifiedIdentity = "verified-identity"
}
public struct SeparationOptions: Sendable {
  public var shifts: Int = 1
  public var overlap: Float = 0.25
  public var seed: UInt64? = nil
  /// Nil uses the model's default segment length, including on iOS.
  public var segmentSeconds: Double? = nil
  /// Nil selects a hardware-aware batch size.
  public var batchSize: Int? = nil
  public var attention: AttentionPrecision = .fp16
  public var compilation: CompilationPolicy = .automatic
  public var cacheValidation: CacheValidationPolicy = .alwaysHash
  /// Fine-tuned model selection skips unrelated ensemble members.
  public var stem: String? = nil
  public var memoryBudgetBytes: Int? = nil
  public init() {}
}
public enum DemucsError: Error, LocalizedError, Sendable {
  case invalidInput(String)
  case invalidOptions(String)
  case invalidCache(String)
  case missingModel(String)
  case unsupported(String)
  case memoryBudget(required: Int, available: Int)
  case downloadFailed(String)
  public var errorDescription: String? {
    switch self {
    case .invalidInput(let s), .invalidOptions(let s), .invalidCache(let s), .missingModel(let s),
      .unsupported(let s), .downloadFailed(let s):
      return s
    case .memoryBudget(let r, let a):
      return
        "Estimated inference memory \(r / 1_048_576) MiB exceeds budget \(a / 1_048_576) MiB. Reduce batchSize, choose a smaller model, or raise memoryBudgetBytes if the device has room. Shorter segmentSeconds reduces legacy-model memory; HTDemucs still pads to its training length."
    }
  }
}
public struct SeparationProgress: Sendable {
  public let completedChunks: Int
  public let totalChunks: Int
  public let modelIndex: Int
  public let modelCount: Int
  public let shiftIndex: Int
  public let shiftCount: Int
  public var fractionCompleted: Double {
    let chunks = Double(completedChunks) / Double(max(1, totalChunks))
    let model = (Double(shiftIndex) + chunks) / Double(max(1, shiftCount))
    return (Double(modelIndex) + model) / Double(max(1, modelCount))
  }
}
public struct InferenceStatistics: Sendable, Codable {
  public let elapsedSeconds: Double
  public let audioSeconds: Double
  public let batchSize: Int
  public var rtfx: Double { audioSeconds / max(elapsedSeconds, 1e-12) }
}
/// Arrays are evaluated before returning. Callers must not mutate shared arrays
/// concurrently. Conversion to CPU PCM is explicit.
public struct SeparationResult: @unchecked Sendable {
  public let audio: MLXArray
  public let sources: [String]
  public let sampleRate: Int
  public let statistics: InferenceStatistics
  /// Destination prepared by the CPU PCM entry point; used by the first `samples()`.
  var prepared: PreparedSamples? = nil
  /// CPU PCM in [source, channel, sample] order. Materialization is explicit;
  /// large outputs are copied by several threads.
  public func samples() -> [Float] {
    let array = audio.dtype == .float32 ? audio : audio.asType(.float32)
    let count = array.size
    guard count >= 1 << 20 else { return array.asArray(Float.self) }
    let backing = array.asData(access: .noCopyIfContiguous)
    func copy(into target: UnsafeMutableRawPointer) {
      // Workers write disjoint byte ranges of storage that outlives the call.
      nonisolated(unsafe) let target = target
      backing.data.withUnsafeBytes { source in
        nonisolated(unsafe) let source = source
        let bytes = count * MemoryLayout<Float>.stride
        let chunk = 8 << 20
        DispatchQueue.concurrentPerform(iterations: (bytes + chunk - 1) / chunk) { i in
          let offset = i * chunk
          (target + offset).copyMemory(
            from: source.baseAddress! + offset, byteCount: min(chunk, bytes - offset))
        }
      }
    }
    if prepared?.count == count, var output = prepared?.take() {
      output.withUnsafeMutableBytes { copy(into: $0.baseAddress!) }
      return output
    }
    return [Float](unsafeUninitializedCapacity: count) { destination, initialized in
      copy(into: UnsafeMutableRawPointer(destination.baseAddress!))
      initialized = count
    }
  }
  public func stem(_ name: String) throws -> MLXArray {
    guard let i = sources.firstIndex(of: name) else {
      throw DemucsError.invalidInput("Unknown stem: \(name)")
    }
    return audio[i]
  }
  /// Each stem as a `[channels, samples]` array, keyed by source name.
  public var stems: [String: MLXArray] {
    Dictionary(uniqueKeysWithValues: sources.enumerated().map { ($1, audio[$0]) })
  }
  /// Reduce to two stems: `name` and everything else, summed, as `no_<name>`.
  ///
  /// For example `twoStems("vocals")` gives `vocals` and `no_vocals`. The
  /// result needs every stem of the model, so it cannot follow `options.stem`.
  public func twoStems(_ name: String) throws -> SeparationResult {
    guard let index = sources.firstIndex(of: name) else {
      throw DemucsError.invalidInput(
        "Unknown stem: \(name). Available: \(sources.joined(separator: ", "))")
    }
    guard sources.count > 1 else {
      throw DemucsError.invalidOptions("Two-stem output needs all of the model's stems")
    }
    let others = sources.indices.filter { $0 != index }.map { audio[$0] }
    let rest = others.dropFirst().reduce(others[0]) { $0 + $1 }
    let combined = stacked([audio[index], rest])
    eval(combined)
    return SeparationResult(
      audio: combined, sources: [name, "no_" + name], sampleRate: sampleRate,
      statistics: statistics)
  }
}

/// A CPU output array whose pages are faulted in on worker threads while
/// inference runs, so the final copy does not pay first-touch costs. Taken once.
final class PreparedSamples: @unchecked Sendable {
  let count: Int
  private let lock = NSLock()
  private let group = DispatchGroup()
  private var storage: [Float]?
  init(count: Int) {
    self.count = count
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async { [self] in
      let stride = Int(getpagesize()) / MemoryLayout<Float>.stride
      // Every element is overwritten before the array is returned to callers.
      let array = [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
        let chunk = 1 << 21
        nonisolated(unsafe) let base = buffer.baseAddress!
        DispatchQueue.concurrentPerform(iterations: (count + chunk - 1) / chunk) { i in
          var j = i * chunk
          let end = min(count, j + chunk)
          while j < end {
            base[j] = 0
            j += stride
          }
        }
        initialized = count
      }
      lock.withLock { storage = array }
      group.leave()
    }
  }
  func take() -> [Float]? {
    group.wait()
    return lock.withLock {
      defer { storage = nil }
      return storage
    }
  }
}

/// Explicit read-only borrowing of an MLX array across concurrency domains.
/// Keep the array and all aliases immutable until separation returns. This
/// wrapper permits reusing one evaluated input across repeated requests.
public struct AudioTensor: @unchecked Sendable {
  public let array: MLXArray
  public init(_ array: MLXArray) { self.array = array }
}
