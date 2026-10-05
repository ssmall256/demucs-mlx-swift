import AVFoundation
import CDemucsAudioIO
import Darwin
import Foundation
import MLX
import Testing

@testable import DemucsAudio
@testable import DemucsMLX

private final class ReleaseCounter: @unchecked Sendable {
  let lock = NSLock()
  private var value = 0
  var count: Int { lock.withLock { value } }
  func increment() { lock.withLock { value += 1 } }
}

extension NativeParityTests {
  @Test func ownedPCMStorageIsSharedAndReleased() throws {
    let count = ReleaseCounter()
    do {
      var tensor: MLXArray?
      var original: UnsafeMutableRawPointer?
      do {
        let storage = try OwnedPCMStorage(channels: 2, capacity: 6003) { count.increment() }
        original = storage.pointer
        let format = try #require(
          AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100, channels: 2, interleaved: false))
        let buffer = try storage.buffer(format: format)
        for c in 0..<2 {
          for i in 0..<6003 { buffer.floatChannelData![c][i] = Float(i + c) / 10000 }
        }
        tensor = try storage.tensor(frames: 6003)
        // Access/evaluation is permitted only after the writer has sealed storage.
        let bytes = tensor!.asData(access: .noCopy)
        bytes.data.withUnsafeBytes { #expect($0.baseAddress == UnsafeRawPointer(original)) }
        #expect(bytes.strides[0] == storage.planeStride)
        #expect(bytes.strides[1] == 1)
      }
      #expect(count.count == 0)
      let gpu = try #require(tensor) * 2
      #expect(gpu[1, 6002].item(Float.self) == Float(6003) / 10000 * 2)
      tensor = nil
      // Evaluated GPU graphs can release input storage; no raw pointer escapes its owner.
      eval(gpu)
      _ = original
    }
    Stream.gpu.synchronize()
    #expect(count.count == 1)
  }
  @Test func pcmSnapshotDoesNotAliasCaller() throws {
    let format = try #require(
      AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 44100, channels: 1, interleaved: false))
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 20))
    buffer.frameLength = 20
    for i in 0..<20 { buffer.floatChannelData![0][i] = Float(i) }
    let tensor = try DemucsAudio.tensor(from: buffer, sampleRate: 44100)
    buffer.floatChannelData![0][5] = -100
    #expect(tensor[0, 5].item(Float.self) == 5)
  }
  @Test(arguments: [false, true])
  func stridedWAVPayloadAgreement(_ half: Bool) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let input = sin(MLXArray(0..<80006).asType(.float32) * 0.003).reshaped(2, -1)
    let tiny = MLXArray([Float(0.125), 0.25, 0.5, -0.75]).reshaped(1, -1)
    let backing = half ? input.asType(.float16) : input
    let small = half ? tiny.asType(.float16) : tiny
    let views = [
      backing, backing[0..., .stride(by: 2)], backing[0..., .stride(by: -1)],
      broadcast(backing[0..<1], to: backing.shape),
      small[0..., .stride(by: 2)], small[0..., .stride(by: -1)],
      broadcast(small[0..., 0..<1], to: [2, 2]),
    ]
    for (i, view) in views.enumerated() {
      let audio = view
      for floating in [true, false] {
        let path = directory.appendingPathComponent("new-\(i)-\(floating).wav")
        let old = directory.appendingPathComponent("old-\(i)-\(floating).wav")
        try DemucsAudio.save(audio, sampleRate: 44100, to: path, float32: floating)
        try DemucsAudio.saveLegacy(audio, sampleRate: 44100, to: old, float32: floating)
        let actual = try DemucsAudio.load(path, sampleRate: 44100)
        let expected = try DemucsAudio.load(old, sampleRate: 44100)
        #expect(arrayEqual(actual, expected).item(Bool.self))
      }
    }
    let values: [Float] = [
      -2, -1, -0.5, -1 / 32768, 0, 1 / 32768,
      0.5, 0.99999, 1, 2, 0.5 / 32768, 1.5 / 32768, 2.5 / 32768,
    ]
    let boundary = MLXArray(values).reshaped(1, -1)
    for sample in [boundary, broadcast(boundary, to: [2, boundary.dim(1)])] {
      let new = directory.appendingPathComponent("bounds.wav")
      let old = directory.appendingPathComponent("old.wav")
      try DemucsAudio.save(sample, sampleRate: 44100, to: new)
      try DemucsAudio.saveLegacy(sample, sampleRate: 44100, to: old, float32: false)
      #expect(
        arrayEqual(
          try DemucsAudio.load(new, sampleRate: 44100), try DemucsAudio.load(old, sampleRate: 44100)
        ).item(Bool.self))
    }
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: directory.path).allSatisfy {
        !$0.contains("partial")
      })
  }
  @Test func ownedDecodeMatchesLegacyInBothDirections() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".wav")
    defer { try? FileManager.default.removeItem(at: file) }
    let input = sin(MLXArray(0..<240062).asType(.float32) * 0.002).reshaped(2, -1)
    try DemucsAudio.save(input, sampleRate: 48000, to: file, float32: true)
    for rate in [48000, 44100, 96000] {
      let actual = try DemucsAudio.load(file, sampleRate: rate)
      let expected = try DemucsAudio.loadLegacy(file, sampleRate: rate)
      #expect(abs(actual.dim(1) - expected.dim(1)) <= 1)
      let n = min(actual.dim(1), expected.dim(1))
      let a = actual[0..., 0..<n]
      let e = expected[0..., 0..<n]
      let snr = (10 * log10(sum(e * e) / maximum(sum((a - e) * (a - e)), 1e-30))).item(Float.self)
      #expect(snr > 100)
    }
  }
  @Test func cancelledWriterLeavesDestinationIntact() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("audio.wav")
    try Data("old".utf8).write(to: path)
    let prepared = try PreparedAudio(zeros([2, 10000]))
    let control = AudioBatchControl(limit: 0)
    let task = Task {
      // A deterministic gate, not a timing-dependent cancellation race.
      control.fail(CancellationError())
      try control.check()
      try prepared.save(sampleRate: 44100, to: path, float32: false)
    }
    _ = await task.result
    #expect(try Data(contentsOf: path) == Data("old".utf8))
    #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["audio.wav"])
    // Fail publication after a complete temporary WAV has actually been written.
    let blocked = directory.appendingPathComponent("blocked.wav")
    try FileManager.default.createDirectory(at: blocked, withIntermediateDirectories: false)
    #expect(throws: (any Error).self) {
      try prepared.save(sampleRate: 44100, to: blocked, float32: false)
    }
    #expect(
      try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        == ["audio.wav", "blocked.wav"])
    #expect(try Data(contentsOf: path) == Data("old".utf8))
    #expect(throws: (any Error).self) {
      try prepared.save(
        sampleRate: 44100, to: directory.appendingPathComponent("missing/a.wav"), float32: false)
    }
  }
}

@Suite struct AudioBatchPolicyTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test func boundedReservationAndFailure() throws {
    let control = AudioBatchControl(limit: 100)
    #expect(control.reserve(60))
    #expect(!control.reserve(50))
    control.release(60)
    #expect(control.reserve(100))
    #expect(control.peakBytes == 100)
    control.release(100)
    control.fail(CancellationError())
    #expect(!control.reserve(1))
    #expect(throws: CancellationError.self) { try control.check() }
  }
  @Test func nativeWriterCancelsBetweenChunksAndRejectsOversize() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    var callbacks = 0
    let samples = [Float](repeating: 0.25, count: 40000)
    let code = samples.withUnsafeBytes { data in
      withUnsafeMutablePointer(to: &callbacks) { counter in
        demucs_write_wav(
          file.path, data.baseAddress, 0, 40000, 1, 1, 0, 44100, 0,
          { context in
            let counter = context!.assumingMemoryBound(to: Int.self)
            counter.pointee += 1
            return counter.pointee >= 3 ? 1 : 0
          }, counter)
      }
    }
    #expect(code == ECANCELED)
    #expect(callbacks == 3)
    #expect(try Data(contentsOf: file).count == 44 + 16384 * 2)
    let huge = samples.withUnsafeBytes { data in
      demucs_write_wav(file.path, data.baseAddress, 0, Int64.max, 2, 1, 0, 44100, 0, nil, nil)
    }
    #expect(huge == EFBIG)
    #expect(try Data(contentsOf: file).count == 44 + 16384 * 2)
  }
  @Test func estimatesUseModelOutputChannelsWithoutDoubleCounting() throws {
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".wav")
    defer { try? FileManager.default.removeItem(at: path) }
    let samples = [Float](repeating: 0.25, count: 12000)
    let code = samples.withUnsafeBytes { bytes in
      demucs_write_wav(path.path, bytes.baseAddress, 0, 6000, 2, 2, 1, 44100, 1, nil, nil)
    }
    #expect(code == 0)
    let estimate = try DemucsAudio.estimateAudioMemory(path, sampleRate: 44100)
    let output = try #require(estimate.outputBytes(channels: 2, sources: 6))
    let control = AudioBatchControl(limit: 1_048_576)
    #expect(control.reserve(estimate.decodedBytes))
    #expect(control.reserve(output))
    #expect(control.peakBytes <= 1_048_576)
    #expect(
      AudioMemoryEstimate(decodedBytes: 0, frames: Int.max)
        .outputBytes(channels: 2, sources: 6) == nil)
  }
  @Test func collidingPathsRejectedBeforeWork() throws {
    let root = URL(fileURLWithPath: "/tmp/output")
    #expect(throws: DemucsError.self) {
      try DemucsAudio.destinations(
        [
          URL(fileURLWithPath: "/tmp/a/song.wav"),
          URL(fileURLWithPath: "/tmp/b/song.m4a"),
        ], root: root)
    }
    #expect(try DemucsAudio.destinations([], root: root).isEmpty)
  }
}

private final class BatchCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var action: (@Sendable () -> Void)?
  private var requested = false
  func install(_ action: @escaping @Sendable () -> Void) {
    let cancel = lock.withLock {
      self.action = action
      return requested
    }
    if cancel { action() }
  }
  func cancel() {
    let action = lock.withLock {
      requested = true
      return self.action
    }
    action?()
  }
}

extension NativeParityTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"] != nil))
  func audioBatchOrderingBackpressureAndCleanup() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var inputs: [URL] = []
    for i in 0..<3 {
      let url = root.appendingPathComponent("track\(i).wav")
      try DemucsAudio.save(
        sin(MLXArray(0..<6000).asType(.float32) * Float(0.01 + Double(i) * 0.001)).reshaped(1, -1),
        sampleRate: 44100, to: url, float32: true)
      inputs.append(url)
    }
    var separation = SeparationOptions()
    separation.seed = 481
    separation.shifts = 0
    separation.compilation = .eager
    separation.batchSize = 1
    let separator = try Separator(
      cacheDirectory: URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["DEMUCS_MODEL_DIRECTORY"]!),
      options: separation)
    var options = AudioBatchOptions()
    options.overlapMemoryBudgetBytes = 0
    let reports = try await DemucsAudio.separateAndExport(
      inputs, using: separator, to: root.appendingPathComponent("serial"), options: options)
    #expect(reports.map(\.input) == inputs)
    #expect(reports.allSatisfy { $0.peakOverlapBytes == 0 && $0.statistics.audioSeconds > 0 })
    options.overlapMemoryBudgetBytes = 1_048_576
    options.prefetchDepth = 1
    options.writerWorkers = 2
    let concurrent = try await DemucsAudio.separateAndExport(
      inputs, using: separator, to: root.appendingPathComponent("overlapped"), options: options)
    #expect(concurrent.map(\.input) == inputs)
    #expect(concurrent.contains { $0.peakOverlapBytes > 0 })
    #expect(concurrent.allSatisfy { $0.peakOverlapBytes <= options.overlapMemoryBudgetBytes })
    for (a, b) in zip(reports, concurrent) {
      for stem in a.sources {
        #expect(
          try Data(contentsOf: a.directory.appendingPathComponent(stem + ".wav"))
            == Data(contentsOf: b.directory.appendingPathComponent(stem + ".wav")))
      }
    }
    let cancelled = BatchCancellation()
    let cancelRoot = root.appendingPathComponent("cancelled")
    let task = Task { @Sendable [inputs, options] in
      try await DemucsAudio.separateAndExport(
        inputs, using: separator, to: cancelRoot, options: options
      ) { update in
        if update.stage == .decoding { cancelled.cancel() }
      }
    }
    cancelled.install { task.cancel() }
    let outcome = await task.result
    if case .success = outcome { Issue.record("Cancelled batch succeeded") }
    let failureRoot = root.appendingPathComponent("failure")
    try FileManager.default.createDirectory(at: failureRoot, withIntermediateDirectories: true)
    try Data("blocked".utf8).write(to: failureRoot.appendingPathComponent("track1"))
    do {
      _ = try await DemucsAudio.separateAndExport(
        inputs, using: separator, to: failureRoot, options: options)
      Issue.record("Invalid output directory succeeded")
    } catch {
      #expect(
        FileManager.default.fileExists(
          atPath: failureRoot.appendingPathComponent("track0/vocals.wav").path))
    }
    #expect(partialFiles(root).isEmpty)
  }
}

private func partialFiles(_ root: URL) -> [URL] {
  guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
    return []
  }
  return files.allObjects.compactMap { $0 as? URL }.filter {
    $0.lastPathComponent.contains(".partial.")
  }
}
