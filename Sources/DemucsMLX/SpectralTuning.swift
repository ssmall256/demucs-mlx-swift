import Foundation
import MLX

public struct SpectralKernelMeasurement: Codable, Sendable {
  public let kernel: String
  public let shape: [Int]
  public let threadWidth: Int
  public let tileFrames: Int
  public let seconds: Double
  public let selected: Bool
}
public struct SpectralTuningReport: Codable, Sendable {
  public let cacheURL: URL
  public let deviceIdentity: String
  public let measurements: [SpectralKernelMeasurement]
}

enum SpectralTuner {
  static func seconds(_ body: () -> MLXArray) throws -> Double {
    for _ in 0..<2 {
      try Task.checkCancellation()
      eval(body())
    }
    var times: [Double] = []
    for _ in 0..<6 {
      try Task.checkCancellation()
      Stream.gpu.synchronize()
      let start = ContinuousClock.now
      eval(body())
      Stream.gpu.synchronize()
      let duration = start.duration(to: .now)
      times.append(
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18)
    }
    times.sort()
    return (times[2] + times[3]) / 2
  }
  static func tune(
    _ spectral: SpectralTransform, rows: Int, length: Int, outputRows: Int, outputLength: Int
  ) throws -> [SpectralKernelMeasurement] {
    let signal = sin(MLXArray(0..<(rows * length)).asType(.float32) * 0.003).reshaped(rows, length)
    let reference = spectral.originalFrames(signal)
    eval(signal, reference)
    var choices = [64, 128, 256, 512].filter { $0 <= spectral.nfft }.map {
      SpectralKernelChoice(width: $0)
    }
    if reference.nbytes >= 100_000_000 {
      for width in [128, 256] {
        for frames in [2, 4]
        where FrameExtraction.supportsTile(
          nfft: spectral.nfft, hop: spectral.hop, width: width, frames: frames)
        {
          choices.append(.init(width: width, tileFrames: frames))
        }
      }
    }
    var candidates: [(SpectralKernelChoice, Double)] = []
    for choice in choices {
      try Task.checkCancellation()
      let actual = spectral.windowedFrames(signal, choice: choice)
      guard arrayEqual(actual, reference).item(Bool.self) else {
        throw DemucsError.unsupported("Spectral tuning candidate changed extracted frames")
      }
      let time = try seconds { spectral.windowedFrames(signal, choice: choice) }
      candidates.append((choice, time))
    }
    let best = candidates.min { $0.1 < $1.1 }!.0
    let cache = SpectralKernelCache.shared
    let key = cache.key(
      kind: "frames", nfft: spectral.nfft, hop: spectral.hop, rows: rows, length: length)
    cache.store(best, for: key)
    var report = candidates.map {
      SpectralKernelMeasurement(
        kernel: "frames", shape: reference.shape, threadWidth: $0.0.width,
        tileFrames: $0.0.tileFrames, seconds: $0.1, selected: $0.0 == best)
    }
    // Match inverse FFT output layout. Materialize once, outside trials, so OLA
    // measurements do not include an implicit broadcast-to-contiguous copy.
    let frames = contiguous(
      broadcast(reference[0..<1], to: [outputRows, reference.dim(1), spectral.nfft]))
    let expected = OverlapAdd.apply(
      frames, window: spectral.window, stride: spectral.hop, length: outputLength,
      offset: spectral.nfft / 2, squareWindow: true)
    eval(frames, expected)
    var ola: [(SpectralKernelChoice, Double)] = []
    for width in [64, 128, 256, 512] {
      let choice = SpectralKernelChoice(width: width)
      let body = {
        OverlapAdd.apply(
          frames, window: spectral.window, stride: spectral.hop, length: outputLength,
          offset: spectral.nfft / 2, squareWindow: true, threadWidth: width)
      }
      guard arrayEqual(body(), expected).item(Bool.self) else {
        throw DemucsError.unsupported("Spectral tuning candidate changed overlap-add arithmetic")
      }
      ola.append((choice, try seconds(body)))
    }
    let selected = ola.min { $0.1 < $1.1 }!.0
    cache.store(
      selected,
      for: cache.key(
        kind: "ola", nfft: spectral.nfft, hop: spectral.hop, rows: outputRows, length: outputLength)
    )
    report += ola.map {
      SpectralKernelMeasurement(
        kernel: "ola", shape: frames.shape, threadWidth: $0.0.width, tileFrames: 1,
        seconds: $0.1, selected: $0.0 == selected)
    }
    return report
  }
}

extension Separator {
  /// Explicit device tuning. Run before separating tracks; no tuning occurs on first inference.
  /// Choices are persisted best-effort and existing compiled forwards are invalidated.
  public func tuneSpectralKernels() async throws -> SpectralTuningReport {
    guard FrameExtraction.device != nil else {
      throw DemucsError.unsupported("Spectral kernel tuning requires Metal")
    }
    defer {
      for model in models.values {
        model.compiled.removeAll()
        model.seen.removeAll()
      }
    }
    var measured = Set<String>()
    var report: [SpectralKernelMeasurement] = []
    for index in selection {
      try Task.checkCancellation()
      let c = manifest.configurations[index]
      guard manifest.classes[index] != "DemucsMLX" else { continue }
      let nfft = c.int("nfft", 4096)
      let hop = nfft / 4
      let segment = options.segmentSeconds ?? c.double("segment", 7.8)
      let length = Int(
        (manifest.classes[index] == "HTDemucsMLX" ? c.double("segment", 7.8) : segment)
          * Double(sampleRate))
      let chunks = (length + hop - 1) / hop
      let hybrid = manifest.classes[index] == "HTDemucsMLX" || c.bool("hybrid", true)
      let old = c.bool("hybrid_old", false)
      let inputLength = hybrid ? chunks * hop + 3 * hop : length
      let outputLength = hybrid ? chunks * hop + (old ? 0 : 3 * hop) : length
      let maximumBatch = options.batchSize ?? HardwarePolicy.batchSize
      let budget = options.memoryBudgetBytes ?? Self.defaultMemoryBudget()
      // Reference + candidate frames coexist, as do reference + candidate OLA outputs.
      // Include signal, windowed/padded scratch and source-sized reconstruction buffers.
      let frames = (inputLength - nfft % 2) / hop + 1
      let bytesPerBatch =
        audioChannels * 4
        * (max(3, c.sources.count + 1) * frames * nfft + 2 * inputLength
          + 2 * c.sources.count * outputLength)
      let batches = options.batchSize.map { [$0] } ?? Array(1...maximumBatch)
      for batch in batches {
        let required = batch * bytesPerBatch
        guard required <= budget / 2 else {
          if options.batchSize != nil || batch == 1 {
            throw DemucsError.memoryBudget(required: required, available: budget / 2)
          }
          continue
        }
        let key = "\(nfft):\(batch):\(inputLength):\(outputLength):\(c.sources.count)"
        if measured.insert(key).inserted {
          report += try SpectralTuner.tune(
            SpectralTransform(nfft: nfft), rows: batch * audioChannels, length: inputLength,
            outputRows: batch * audioChannels * c.sources.count, outputLength: outputLength)
        }
      }
    }
    return SpectralTuningReport(
      cacheURL: SpectralKernelCache.shared.url, deviceIdentity: SpectralKernelCache.shared.identity,
      measurements: report)
  }
}
