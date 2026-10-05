import AVFoundation
import CDemucsAudioIO
import Cmlx
import DemucsMLX
import Foundation
import MLX

/// Prepared on one MLX executor, then borrowed immutably by CPU-only writer tasks.
/// The owner and its aliases must not be mutated during export.
final class PreparedAudio: @unchecked Sendable {
  let owner: MLXArray
  let pointer: UnsafeRawPointer
  let shape: [Int]
  let strides: [Int]
  let dtype: DType
  init(_ audio: MLXArray) throws {
    guard audio.ndim == 2, audio.dim(0) > 0, audio.dim(0) <= 16383,
      audio.dim(1) > 0, audio.dim(1) < Int(UInt32.max)
    else { throw DemucsError.invalidInput("Invalid audio shape \(audio.shape)") }
    owner = audio.dtype == .float32 || audio.dtype == .float16 ? audio : audio.asType(.float32)
    eval(owner)
    // Snapshot only after evaluation. Foundation Data can inline/copy tiny
    // no-copy buffers, which would invalidate the original strided layout.
    guard let data = mlx_array_data_uint8(owner.ctx),
      let nativeStrides = mlx_array_strides(owner.ctx)
    else {
      throw DemucsError.invalidInput("Cannot borrow evaluated audio storage")
    }
    pointer = UnsafeRawPointer(data)
    shape = owner.shape
    strides = (0..<owner.ndim).map { Int(nativeStrides[$0]) }
    dtype = owner.dtype
  }
  func save(sampleRate: Int, to url: URL, float32: Bool) throws {
    try save(sampleRate: sampleRate, to: url, format: .wav(float32 ? .float32 : .pcm16))
  }
  func save(sampleRate: Int, to url: URL, format: AudioFileFormat) throws {
    try format.validate()
    try withExtendedLifetime(owner) {
      try savePrepared(sampleRate: sampleRate, to: url, format: format)
    }
  }
  private func savePrepared(sampleRate: Int, to url: URL, format: AudioFileFormat) throws {
    guard sampleRate > 0, sampleRate <= Int(Int32.max), url.isFileURL else {
      throw DemucsError.invalidInput("Invalid audio destination or sample rate")
    }
    try Task.checkCancellation()
    let temporary = url.deletingLastPathComponent().appendingPathComponent(
      ".\(url.deletingPathExtension().lastPathComponent).\(UUID().uuidString).partial.\(url.pathExtension)"
    )
    defer { try? FileManager.default.removeItem(at: temporary) }
    let error: Int32
    let float32 = format.isFloat32WAV
    if format.usesNativeWAVWriter, url.pathExtension.lowercased() == "wav" {
      error = demucs_write_wav(
        temporary.path, pointer, dtype == .float32 ? 0 : 1,
        Int64(shape[1]), Int32(shape[0]), strides[1], strides[0],
        Int32(sampleRate), float32 ? 1 : 0, { _ in Task.isCancelled ? 1 : 0 }, nil)
    } else {
      error = EFBIG
    }
    if error == EFBIG {
      try saveWithAVFoundation(sampleRate: sampleRate, to: temporary, format: format)
    } else if error == ECANCELED {
      throw CancellationError()
    } else if error != 0 {
      throw POSIXError(POSIXErrorCode(rawValue: error) ?? .EIO)
    }
    try Task.checkCancellation()
    guard rename(temporary.path, url.path) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
  }
  private func saveWithAVFoundation(sampleRate: Int, to url: URL, format: AudioFileFormat) throws {
    let settings = format.settings(sampleRate: sampleRate, channels: shape[0])
    let file = try AVAudioFile(
      forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let count = max(1, 16384 / shape[0])
    guard
      let buffer = AVAudioPCMBuffer(
        pcmFormat: file.processingFormat, frameCapacity: UInt32(count))
    else { throw DemucsError.invalidInput("Cannot create bounded export buffer") }
    let base = pointer
    for start in stride(from: 0, to: shape[1], by: count) {
      try Task.checkCancellation()
      let n = min(count, shape[1] - start)
      for c in 0..<shape[0] {
        for i in 0..<n {
          let offset = (start + i) * strides[1] + c * strides[0]
          let value: Float
          if dtype == .float32 {
            value = base.assumingMemoryBound(to: Float.self).advanced(by: offset).pointee
          } else {
            value = Float(base.assumingMemoryBound(to: Float16.self).advanced(by: offset).pointee)
          }
          buffer.floatChannelData![c][i] = value
        }
      }
      buffer.frameLength = UInt32(n)
      try file.write(from: buffer)
    }
  }
}
