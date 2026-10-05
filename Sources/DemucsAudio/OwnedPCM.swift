@preconcurrency import AVFoundation
import CDemucsAudioIO
import DemucsMLX
import Foundation
import MLX

/// Internal decoder-owned storage. Never exposed while it is writable.
/// AVFoundation and MLX independently retain this allocation, without a retain cycle.
final class OwnedPCMStorage: @unchecked Sendable {
  let pointer: UnsafeMutableRawPointer
  let channels: Int
  let capacity: Int
  let planeStride: Int
  let floatCount: Int
  private let onRelease: (@Sendable () -> Void)?
  init(channels: Int, capacity: Int, onRelease: (@Sendable () -> Void)? = nil) throws {
    guard channels > 0, channels <= Int(UInt16.max), capacity > 1,
      capacity < Int(UInt32.max) / 4
    else { throw DemucsError.invalidInput("Audio dimensions exceed native PCM capacity") }
    let page = Int(getpagesize())
    let planeBytes = ((capacity * 4 + page - 1) / page) * page
    let (bytes, overflow) = planeBytes.multipliedReportingOverflow(by: channels)
    guard !overflow, bytes / 4 <= Int(Int32.max) else {
      throw DemucsError.invalidInput("Audio allocation exceeds MLX dimensions")
    }
    var allocation: UnsafeMutableRawPointer?
    guard posix_memalign(&allocation, page, bytes) == 0, let allocation else {
      throw DemucsError.invalidInput("Cannot allocate \(bytes) bytes for decoded audio")
    }
    pointer = allocation
    self.channels = channels
    self.capacity = capacity
    planeStride = planeBytes / 4
    floatCount = bytes / 4
    self.onRelease = onRelease
    memset(pointer, 0, bytes)
  }
  deinit {
    free(pointer)
    onRelease?()
  }

  func buffer(format: AVAudioFormat, offset: Int = 0, frames: Int? = nil) throws -> AVAudioPCMBuffer
  {
    let count = frames ?? capacity
    guard offset >= 0, count > 0, count <= capacity - offset,
      Int(format.channelCount) == channels, format.commonFormat == .pcmFormatFloat32,
      !format.isInterleaved
    else { throw DemucsError.invalidInput("Invalid owned PCM view") }
    let list = AudioBufferList.allocate(maximumBuffers: channels)
    for c in 0..<channels {
      list[c] = AudioBuffer(
        mNumberChannels: 1, mDataByteSize: UInt32(count * 4),
        mData: pointer.advanced(by: (c * planeStride + offset) * 4))
    }
    guard
      let buffer = AVAudioPCMBuffer(
        pcmFormat: format, bufferListNoCopy: list.unsafeMutablePointer,
        deallocator: { [self] _ in
          list.unsafeMutablePointer.deallocate()
          withExtendedLifetime(self) {}
        })
    else {
      list.unsafeMutablePointer.deallocate()
      throw DemucsError.invalidInput("Cannot create an owned PCM view")
    }
    buffer.frameLength = 0
    return buffer
  }
  /// Seal only after decoding finishes. The returned view owns this storage via MLX.
  func tensor(frames: Int) throws -> MLXArray {
    guard frames > 1, frames <= capacity else {
      throw DemucsError.invalidInput("Decoded audio must contain at least two frames")
    }
    let allocation = self
    let base = MLXArray(rawPointer: pointer, [floatCount], dtype: .float32) {
      withExtendedLifetime(allocation) {}
    }
    return asStrided(base, [channels, frames], strides: [planeStride, 1])
  }
}

extension DemucsAudio {
  static func loadOwned(_ url: URL, sampleRate: Int, preferNativeWAV: Bool = false) throws
    -> MLXArray
  {
    guard sampleRate > 0, sampleRate <= Int(Int32.max) else {
      throw DemucsError.invalidInput("Sample rate must be a positive integer")
    }
    try Task.checkCancellation()
    if preferNativeWAV, url.isFileURL, url.pathExtension.lowercased() == "wav",
      let audio = try loadNativeWAV(url, sampleRate: sampleRate)
    {
      return audio
    }
    return try loadAppleOwned(url, sampleRate: sampleRate)
  }
  static func loadAppleOwned(_ url: URL, sampleRate: Int) throws -> MLXArray {
    try Task.checkCancellation()
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    let sourceFormat = file.processingFormat
    guard file.length > 1, file.length < Int64(UInt32.max) / 4,
      sourceFormat.sampleRate.isFinite, sourceFormat.sampleRate > 0
    else { throw DemucsError.invalidInput("Audio file is empty or exceeds native PCM capacity") }
    let channels = Int(sourceFormat.channelCount)
    let chunk = 65536
    if sourceFormat.sampleRate == Double(sampleRate) {
      let storage = try OwnedPCMStorage(channels: channels, capacity: Int(file.length))
      var offset = 0
      while offset < storage.capacity {
        try Task.checkCancellation()
        let count = min(chunk, storage.capacity - offset)
        let buffer = try storage.buffer(format: sourceFormat, offset: offset, frames: count)
        do { try file.read(into: buffer, frameCount: UInt32(count)) } catch {
          let error = error as NSError
          if error.domain != NSOSStatusErrorDomain || error.code != -39 { throw error }
        }
        let read = Int(buffer.frameLength)
        if read == 0 { break }
        offset += read
      }
      return try storage.tensor(frames: offset)
    }
    guard
      let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
        channels: sourceFormat.channelCount, interleaved: false),
      let converter = AVAudioConverter(from: sourceFormat, to: target),
      let scratch = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: UInt32(chunk))
    else { throw DemucsError.invalidInput("Unsupported audio sample-rate conversion") }
    converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
    let estimate = ceil(Double(file.length) * Double(sampleRate) / sourceFormat.sampleRate) + 64
    guard estimate.isFinite, estimate < Double(UInt32.max / 4) else {
      throw DemucsError.invalidInput("Resampled audio exceeds native PCM capacity")
    }
    let storage = try OwnedPCMStorage(channels: channels, capacity: Int(estimate))
    var offset = 0
    nonisolated(unsafe) var inputError: Error?
    nonisolated(unsafe) var ended = false
    while true {
      try Task.checkCancellation()
      guard offset < storage.capacity else {
        throw DemucsError.invalidInput("Decoded audio exceeds the file's reported length")
      }
      let buffer = try storage.buffer(
        format: target, offset: offset, frames: min(chunk, storage.capacity - offset))
      var error: NSError?
      let status = converter.convert(to: buffer, error: &error) { requested, state in
        do {
          try Task.checkCancellation()
          if ended || file.framePosition >= file.length {
            state.pointee = .endOfStream
            return nil
          }
          scratch.frameLength = 0
          let remaining = UInt32(min(Int64(chunk), file.length - file.framePosition))
          do { try file.read(into: scratch, frameCount: min(remaining, max(1, requested))) } catch {
            let error = error as NSError
            if error.domain != NSOSStatusErrorDomain || error.code != -39 { throw error }
            ended = true
          }
          if scratch.frameLength == 0 {
            state.pointee = .endOfStream
            return nil
          }
          state.pointee = .haveData
          return scratch
        } catch {
          inputError = error
          state.pointee = .noDataNow
          return nil
        }
      }
      if let inputError { throw inputError }
      if let error { throw error }
      guard status != .error else { throw DemucsError.invalidInput("Audio conversion failed") }
      offset += Int(buffer.frameLength)
      if status == .endOfStream { break }
      guard buffer.frameLength > 0 else {
        throw DemucsError.invalidInput("Audio converter made no progress")
      }
    }
    return try storage.tensor(frames: offset)
  }

  static func loadNativeWAV(_ url: URL, sampleRate: Int) throws
    -> MLXArray?
  {
    guard sampleRate > 0, sampleRate <= Int(Int32.max) else {
      throw DemucsError.invalidInput("Sample rate must be a positive integer")
    }
    var reader: OpaquePointer?
    var info = demucs_pcm_wav_info()
    let status = demucs_open_pcm_wav(url.path, Int32(sampleRate), &reader, &info)
    guard status != 0 else { return nil }
    guard status > 0, let reader else {
      throw POSIXError(POSIXErrorCode(rawValue: -status) ?? .EIO)
    }
    defer { demucs_close_pcm_wav(reader) }
    let storage = try OwnedPCMStorage(channels: Int(info.channels), capacity: Int(info.frames))
    let error = demucs_read_pcm_wav(
      reader, storage.pointer.assumingMemoryBound(to: Float.self), Int64(storage.capacity),
      storage.planeStride, { _ in Task.isCancelled ? 1 : 0 }, nil)
    if error == ECANCELED { throw CancellationError() }
    guard error == 0 else { throw POSIXError(POSIXErrorCode(rawValue: error) ?? .EIO) }
    return try storage.tensor(frames: Int(info.frames))
  }
}
