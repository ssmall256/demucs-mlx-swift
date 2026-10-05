import Foundation
import MLX
import Metal

/// Reusable inference session. Requests run serially on this actor, off the main
/// actor. Transfer ownership of input arrays, or pass `MLXArray(input)` when the
/// original array must remain available. Do not mutate either alias in flight.
public actor Separator {
  public nonisolated let sources: [String]
  public nonisolated let sampleRate: Int
  public nonisolated let audioChannels: Int
  public nonisolated let options: SeparationOptions
  let manifest: CacheManifest
  let selection: [Int]
  var models: [Int: NativeModel] = [:]
  var windows: [Int: MLXArray] = [:]
  /// First-member weights read in the background from construction onward.
  var prefetched: PendingModelArrays?
  /// Validate and hash assets away from the caller's executor. This is the
  /// preferred entry point for UI applications; inference remains actor-owned.
  ///
  /// With no `cacheDirectory` the model is looked up in `defaultCacheDirectory`.
  /// If it is not there and `download` allows it, its files are fetched from
  /// Hugging Face and checked against the digests built into this package
  /// before use (see `ModelHub`).
  public static func load(
    model: DemucsModel = .htdemucs, cacheDirectory: URL? = nil,
    options: SeparationOptions = .init(), download: ModelDownloadPolicy = .ifMissing,
    downloadProgress: (@Sendable (ModelDownloadProgress) -> Void)? = nil
  ) async throws -> Separator {
    try Task.checkCancellation()
    let cacheDirectory = cacheDirectory ?? defaultCacheDirectory
    if download == .ifMissing, ModelHub.downloadsEnabled,
      !ModelHub.isCached(model, in: cacheDirectory)
    {
      try await ModelHub.download(model, to: cacheDirectory, progress: downloadProgress)
    }
    let session = try await Task.detached(priority: .userInitiated) {
      try Separator(model: model, cacheDirectory: cacheDirectory, options: options)
    }.value
    try Task.checkCancellation()
    return session
  }
  /// Validate assets and write a verified-digest receipt where the cache is
  /// writable. CPU only: no session, weights or GPU runtime are prepared.
  public static func verifyAssets(model: DemucsModel = .htdemucs, cacheDirectory: URL) throws {
    _ = try CacheManifest.read(model: model, directory: cacheDirectory, validation: .alwaysHash)
  }
  /// Construction validates assets, then prepares the session in the
  /// background: MLX's Metal runtime initializes and the first model's weights
  /// are read. Neither builds or evaluates a graph; inference joins them.
  public init(
    model: DemucsModel = .htdemucs, cacheDirectory: URL, options: SeparationOptions = .init()
  ) throws {
    try Self.validate(options)
    RuntimePreparation.begin()
    let manifest = try CacheManifest.read(
      model: model, directory: cacheDirectory, validation: options.cacheValidation)
    self.manifest = manifest
    self.options = options
    sampleRate = manifest.configurations[0].int("samplerate", 44100)
    audioChannels = manifest.configurations[0].int("audio_channels", 2)
    if let stem = options.stem {
      guard model == .htdemucsFT, let i = manifest.configurations[0].sources.firstIndex(of: stem)
      else {
        throw DemucsError.invalidOptions(
          "Single-model stem selection requires htdemucs_ft and a valid source name")
      }
      let active = manifest.weights.indices.filter { manifest.weights[$0][i] > 0 }
      guard active.count == 1 else {
        throw DemucsError.invalidCache("Selected stem has multiple contributing models")
      }
      selection = active
      sources = [stem]
    } else {
      selection = Array(manifest.configurations.indices)
      sources = manifest.configurations[0].sources
    }
    prefetched = PendingModelArrays(manifest, index: selection[0])
  }
  public nonisolated func separate(
    _ input: sending MLXArray, progress: (@Sendable (SeparationProgress) -> Void)? = nil
  ) async throws -> SeparationResult {
    try await separate(AudioTensor(input), progress: progress)
  }
  public static func validate(_ options: SeparationOptions) throws {
    guard (0...64).contains(options.shifts), options.overlap.isFinite, options.overlap >= 0,
      options.overlap < 1
    else { throw DemucsError.invalidOptions("shifts must be 0...64 and overlap must be in [0,1)") }
    if let b = options.batchSize, !(1...128).contains(b) {
      throw DemucsError.invalidOptions("batchSize must be 1...128")
    }
    if let s = options.segmentSeconds, !s.isFinite || !(0.01...600).contains(s) {
      throw DemucsError.invalidOptions("segmentSeconds must be finite and in 0.01...600")
    }
    if let m = options.memoryBudgetBytes, m <= 0 {
      throw DemucsError.invalidOptions("memoryBudgetBytes must be positive")
    }
  }
  /// Recommended macOS cache location. iOS callers supply a bundle or sandbox URL.
  public nonisolated static var defaultCacheDirectory: URL {
    #if os(iOS)
      FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("demucs-mlx")
    #else
      FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/demucs-mlx")
    #endif
  }
  private func load(_ index: Int, pending: PendingModelArrays? = nil) throws -> NativeModel {
    if let existing = models[index] { return existing }
    #if os(iOS)
      models.removeAll()
      Memory.clearCache()
    #else
      if manifest.tensorBytes > (options.memoryBudgetBytes ?? Self.defaultMemoryBudget()) / 2 {
        models.removeAll()
        Memory.clearCache()
      }
    #endif
    let trace = PerformanceTrace.begin("model.load-and-build")
    let subset = try pending?.take() ?? loadModelArrays(manifest, index: index)
    defer { PerformanceTrace.end(trace) }
    let model = try NativeModel(
      config: manifest.configurations[index], architecture: manifest.classes[index], arrays: subset,
      attention: options.attention)
    #if os(iOS)
      models.removeAll()
    #endif
    models[index] = model
    return model
  }
  /// Separate channels-first CPU PCM at `sampleRate`. Copies the input into
  /// owned MLX storage; callers need no MLX types for input or `result.samples()`.
  public func separate(
    samples: [Float], channels: Int,
    progress: (@Sendable (SeparationProgress) -> Void)? = nil
  ) async throws -> SeparationResult {
    try Task.checkCancellation()
    guard channels > 0, samples.count % channels == 0, samples.count / channels > 1 else {
      throw DemucsError.invalidInput("Expected channels-first PCM with at least two samples")
    }
    let inputTrace = PerformanceTrace.begin("cpu.input.materialize")
    let tensor = AudioTensor(MLXArray(samples, [channels, samples.count / channels]))
    PerformanceTrace.end(inputTrace)
    // CPU callers read stems back; fault their destination in while inference runs.
    let count = sources.count * audioChannels * (samples.count / channels)
    let prepared = count >= 1 << 20 ? PreparedSamples(count: count) : nil
    var result = try await separate(tensor, progress: progress)
    result.prepared = prepared
    return result
  }
  public func separate(
    _ tensor: AudioTensor, progress: (@Sendable (SeparationProgress) -> Void)? = nil
  ) async throws -> SeparationResult {
    try Task.checkCancellation()
    return try performSeparation(tensor, progress: progress)
  }
  // No suspension points in the graph/cache phase: requests remain serialized.
  private func performSeparation(
    _ tensor: AudioTensor, progress: (@Sendable (SeparationProgress) -> Void)?
  ) throws -> SeparationResult {
    let trace = PerformanceTrace.begin("inference.whole-call")
    var completed = false
    defer { PerformanceTrace.end(trace, gpuCompleted: completed) }
    let input = tensor.array
    try Task.checkCancellation()
    guard input.ndim == 2, input.dim(0) > 0, input.dim(1) > 1,
      input.dtype == .float32 || input.dtype == .float16
    else {
      throw DemucsError.invalidInput(
        "Expected a floating-point [channels,samples] tensor with at least two samples")
    }
    var audio = input.asType(.float32)
    if audio.dim(0) != audioChannels {
      if audioChannels == 1 {
        audio = mean(audio, axis: 0, keepDims: true)
      } else if audio.dim(0) == 1 {
        audio = broadcast(audio, to: [audioChannels, audio.dim(1)])
      } else if audio.dim(0) > audioChannels {
        audio = audio[0..<audioChannels]
      } else {
        throw DemucsError.invalidInput("Unsupported input channel layout")
      }
    }
    let firstModel = selection.first!
    let pending =
      models.isEmpty ? (prefetched ?? PendingModelArrays(manifest, index: firstModel)) : nil
    prefetched = nil
    // Also join on invalid input/cancellation; no preparation escapes this call.
    defer { pending?.wait() }
    let validationTrace = PerformanceTrace.begin("input.finite-check")
    let finite = all(isFinite(audio)).item(Bool.self)
    PerformanceTrace.end(validationTrace, gpuCompleted: true)
    guard finite else {
      throw DemucsError.invalidInput("Audio contains non-finite samples")
    }
    let target = options.batchSize ?? Self.automaticBatchSize()
    let started = ContinuousClock.now
    var rng = ShiftRandom(seed: options.seed ?? UInt64.random(in: 0...UInt64.max))
    var output: MLXArray?
    var totals = Array(repeating: Float(0), count: sources.count)
    var effectiveBatch = target
    for index in manifest.configurations.indices {
      if !selection.contains(index) {
        for _ in 0..<options.shifts { _ = rng.offset(maximum: sampleRate / 2) }
        continue
      }
      try Task.checkCancellation()
      let model = try load(index, pending: index == firstModel ? pending : nil)
      let segment = options.segmentSeconds ?? model.segment
      guard model.architecture != "HTDemucsMLX" || segment <= model.segment else {
        throw DemucsError.invalidOptions("HTDemucs segment exceeds its training length")
      }
      let length = audio.dim(1)
      let shiftCount = max(1, options.shifts)
      let maxShift = sampleRate / 2
      var sumOfShifts: MLXArray?
      for shiftIndex in 0..<shiftCount {
        let offset = options.shifts > 0 ? rng.offset(maximum: maxShift) : 0
        let backing =
          options.shifts > 0 ? padLast(audio[.newAxis], maxShift, maxShift) : audio[.newAxis]
        let shiftLength = options.shifts > 0 ? length + maxShift - offset : length
        let estimate = try split(
          model, backing: backing, start: offset, length: shiftLength, segment: segment,
          batchTarget: target, index: selection.firstIndex(of: index)!, shiftIndex: shiftIndex,
          shiftCount: shiftCount, progress: progress)
        effectiveBatch = estimate.1
        let trimmed =
          options.shifts > 0 ? estimate.0[.ellipsis, (maxShift - offset)...] : estimate.0
        sumOfShifts = sumOfShifts.map { $0 + trimmed } ?? trimmed
        eval(sumOfShifts!)
      }
      var result = sumOfShifts! / Float(shiftCount)
      let weights: [Float]
      if let stem = options.stem, let i = model.sources.firstIndex(of: stem) {
        result = result[0..., i..<(i + 1), 0..., 0...]
        weights = [manifest.weights[index][i]]
      } else {
        weights = manifest.weights[index]
      }
      result = result * MLXArray(weights).reshaped(1, -1, 1, 1)
      for i in totals.indices { totals[i] += weights[i] }
      output = output.map { $0 + result } ?? result
      eval(output!)
    }
    let result = (output! / MLXArray(totals).reshaped(1, -1, 1, 1))[0]
    eval(result)
    completed = true
    let duration = started.duration(to: .now)
    let elapsed =
      Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    return SeparationResult(
      audio: result, sources: sources, sampleRate: sampleRate,
      statistics: InferenceStatistics(
        elapsedSeconds: elapsed, audioSeconds: Double(audio.dim(1)) / Double(sampleRate),
        batchSize: effectiveBatch))
  }
  private func split(
    _ model: NativeModel, backing: MLXArray, start: Int, length: Int, segment: Double,
    batchTarget: Int, index: Int, shiftIndex: Int, shiftCount: Int,
    progress: (@Sendable (SeparationProgress) -> Void)?
  ) throws -> (MLXArray, Int) {
    let segmentLength = Int(segment * Double(sampleRate))
    let stride = max(1, Int((1 - options.overlap) * Float(segmentLength)))
    let offsets = Array(Swift.stride(from: 0, to: length, by: stride))
    var batchSize =
      options.batchSize == nil
      ? (offsets.count + (offsets.count + batchTarget - 1) / batchTarget - 1)
        / ((offsets.count + batchTarget - 1) / batchTarget) : batchTarget
    let budget = options.memoryBudgetBytes ?? Self.defaultMemoryBudget()
    let outputBytes = sources.count * audioChannels * length * 4
    func estimate(_ b: Int) -> Int {
      models.values.reduce(0) { $0 + $1.parameterBytes } + outputBytes + b
        * Int(1_300_000_000 * Double(model.validLength(segmentLength)) / 343980)
    }
    while options.batchSize == nil && batchSize > 1 && estimate(batchSize) > budget {
      batchSize -= 1
    }
    guard estimate(batchSize) <= budget else {
      throw DemucsError.memoryBudget(required: estimate(batchSize), available: budget)
    }
    let window: MLXArray
    if let cached = windows[segmentLength] {
      window = cached
    } else {
      let windowTrace = PerformanceTrace.begin("inference.window.prepare")
      let windowGraphTrace = PerformanceTrace.begin("inference.window.swift-graph")
      let weights = concatenated([
        arange(1, segmentLength / 2 + 1, dtype: .int32),
        arange(segmentLength - segmentLength / 2, 0, step: -1, dtype: .int32),
      ]).asType(.float32)
      window = weights / max(weights)
      PerformanceTrace.end(windowGraphTrace)
      let windowEvalTrace = PerformanceTrace.begin("inference.window.eval")
      eval(window)
      PerformanceTrace.end(windowEvalTrace, gpuCompleted: true)
      PerformanceTrace.end(windowTrace, gpuCompleted: true)
      if windows.count >= 8 { windows.removeAll() }
      windows[segmentLength] = window
    }
    let out = zeros([1, model.sources.count, audioChannels, length])
    var pending: [MLXArray] = []
    var first = 0
    var done = 0
    var next = 0
    func prepare() throws -> (group: [Int], mix: MLXArray)? {
      guard next < offsets.count else { return nil }
      try Task.checkCancellation()
      func padded(_ length: Int) -> Int {
        model.architecture == "HTDemucsMLX" ? segmentLength : model.validLength(length)
      }
      let firstLength = min(segmentLength, length - offsets[next])
      let paddedLength = padded(firstLength)
      var group: [Int] = []
      while next < offsets.count && group.count < batchSize
        && padded(min(segmentLength, length - offsets[next])) == paddedLength
      {
        group.append(next)
        next += 1
      }
      let inputs = group.map { chunk -> MLXArray in
        let chunkLength = min(segmentLength, length - offsets[chunk])
        let delta = paddedLength - chunkLength
        let begin = start + offsets[chunk] - delta / 2
        let end = begin + paddedLength
        return padLast(
          backing[.ellipsis, max(0, begin)..<min(backing.dim(-1), end)], max(0, -begin),
          max(0, end - backing.dim(-1)))
      }
      let mix = concatenated(inputs, axis: 0)
      return (group, mix)
    }
    var ready = try prepare()
    while let current = ready {
      try Task.checkCancellation()
      let group = current.group
      let mix = current.mix
      let completed = group.last! + 1
      ready = try prepare()
      let enqueueTrace = PerformanceTrace.begin("inference.batch.enqueue")
      let prediction = model.forward(mix, policy: options.compilation)
      PerformanceTrace.end(enqueueTrace)
      for (j, chunk) in group.enumerated() {
        let chunkLength = min(segmentLength, length - offsets[chunk])
        let frame = padLast(trim(prediction[j], chunkLength), 0, segmentLength - chunkLength)
        pending.append(frame)
      }
      let end = completed == offsets.count ? length : min(length, completed * stride)
      let frames = stacked(pending, axis: 2)[.newAxis]
      let span = OverlapAdd.apply(
        frames, window: window, stride: stride, length: end - done, offset: done - first * stride)
      out[.ellipsis, done..<end] = span
      let submitTrace = PerformanceTrace.begin("inference.async-submit")
      asyncEval(out)
      PerformanceTrace.end(submitTrace)
      done = end
      let keep = max(first, Int(floor(Double(end - segmentLength) / Double(stride))) + 1)
      let drop = min(pending.count, max(0, keep - first))
      pending.removeFirst(drop)
      first += drop
      progress?(
        SeparationProgress(
          completedChunks: completed, totalChunks: offsets.count, modelIndex: index,
          modelCount: selection.count, shiftIndex: shiftIndex, shiftCount: shiftCount))
    }
    let joinTrace = PerformanceTrace.begin("inference.final-join")
    eval(out)
    PerformanceTrace.end(joinTrace, gpuCompleted: true)
    return (out, batchSize)
  }
  nonisolated static func automaticBatchSize() -> Int {
    HardwarePolicy.batchSize
  }
  nonisolated static func defaultMemoryBudget() -> Int { cachedMemoryBudget }
  // Creating a Metal device object costs milliseconds per query; the budget is constant.
  private nonisolated static let cachedMemoryBudget: Int = {
    #if os(iOS)
      return Int(min(ProcessInfo.processInfo.physicalMemory / 3, UInt64(2_500_000_000)))
    #else
      return Int(
        MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? ProcessInfo.processInfo
          .physicalMemory / 2)
    #endif
  }()
}
