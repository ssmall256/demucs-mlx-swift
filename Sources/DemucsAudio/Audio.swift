import AVFoundation
@_spi(Profiling) import DemucsMLX
import Foundation
import MLX

public enum AudioDecoder: String, Sendable, CaseIterable {
  /// Current accepted platform defaults.
  case automatic
  case avFoundation = "apple"
  /// Prefer direct native-rate PCM WAV decoding; other inputs use Apple.
  case nativePCM = "native-pcm"
}

/// Native Apple audio decoding and sample-rate conversion. Internal Demucs
/// factor-two resampling uses the model's Julius filter, independently of I/O.
public enum DemucsAudio {
  public static func load(_ url: URL, sampleRate: Int) throws -> MLXArray {
    try load(url, sampleRate: sampleRate, decoder: .automatic)
  }
  public static func load(_ url: URL, sampleRate: Int, decoder: AudioDecoder) throws -> MLXArray {
    let trace = PerformanceTrace.begin("audio.decode")
    defer { PerformanceTrace.end(trace) }
    if decoder == .automatic,
      ProcessInfo.processInfo.environment["DEMUCS_AUDIO_DECODE"] == "baseline"
    {
      return try loadLegacy(url, sampleRate: sampleRate)
    }
    return try loadOwned(
      url, sampleRate: sampleRate,
      preferNativeWAV: decoder == .nativePCM
        || (decoder == .automatic
          && ProcessInfo.processInfo.environment["DEMUCS_WAV_DECODE"] == "native")
    )
  }
  static func loadLegacy(_ url: URL, sampleRate: Int) throws -> MLXArray {
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    guard file.length > 1, file.length < Int64(UInt32.max),
      let buffer = AVAudioPCMBuffer(
        pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
    else {
      throw DemucsError.invalidInput("Audio file is empty or too large for an in-memory PCM buffer")
    }
    try file.read(into: buffer)
    return try tensor(from: buffer, sampleRate: sampleRate)
  }
  public static func tensor(from input: AVAudioPCMBuffer, sampleRate: Int) throws -> MLXArray {
    guard input.frameLength > 1, input.format.commonFormat == .pcmFormatFloat32,
      !input.format.isInterleaved
    else { throw DemucsError.invalidInput("Expected non-interleaved Float32 PCM") }
    let buffer: AVAudioPCMBuffer
    if input.format.sampleRate == Double(sampleRate) {
      buffer = input
    } else {
      guard
        let format = AVAudioFormat(
          commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
          channels: input.format.channelCount, interleaved: false),
        let converter = AVAudioConverter(from: input.format, to: format)
      else { throw DemucsError.invalidInput("Unsupported audio sample-rate conversion") }
      // Match mlx-audio-io's default macOS "best" AudioToolbox resampler.
      converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
      let count =
        ceil(Double(input.frameLength) * Double(sampleRate) / input.format.sampleRate) + 64
      guard count < Double(UInt32.max),
        let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count))
      else { throw DemucsError.invalidInput("Resampled audio exceeds PCM capacity") }
      var provided = false
      var error: NSError?
      let status = converter.convert(to: output, error: &error) { _, status in
        if provided {
          status.pointee = .endOfStream
          return nil
        }
        provided = true
        status.pointee = .haveData
        return input
      }
      guard status != .error, error == nil else {
        throw error ?? DemucsError.invalidInput("Audio conversion failed") as NSError
      }
      buffer = output
    }
    guard let data = buffer.floatChannelData else {
      throw DemucsError.invalidInput("PCM channel data is unavailable")
    }
    let frames = Int(buffer.frameLength)
    let channels = (0..<Int(buffer.format.channelCount)).map {
      MLXArray(UnsafeBufferPointer(start: data[$0], count: frames))
    }
    return stacked(channels)
  }
  public static func pcmBuffer(from audio: MLXArray, sampleRate: Int) throws -> AVAudioPCMBuffer {
    guard audio.ndim == 2, audio.dim(0) > 0, audio.dim(1) > 0, audio.dim(1) < Int(UInt32.max),
      let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
        channels: AVAudioChannelCount(audio.dim(0)), interleaved: false),
      let buffer = AVAudioPCMBuffer(
        pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.dim(1))),
      let pointers = buffer.floatChannelData
    else {
      throw DemucsError.invalidInput("Cannot create PCM buffer for tensor shape \(audio.shape)")
    }
    buffer.frameLength = buffer.frameCapacity
    for channel in 0..<audio.dim(0) {
      let samples = audio[channel].asType(.float32)
      // Borrow contiguous MLX storage only while its owner is alive. Strided
      // channels use MLX's contiguous-copy fallback; PCM owns the final copy.
      withExtendedLifetime(samples) {
        let bytes = samples.asData(access: .noCopyIfContiguous).data
        bytes.withUnsafeBytes { source in
          pointers[channel].update(
            from: source.bindMemory(to: Float.self).baseAddress!, count: audio.dim(1))
        }
      }
    }
    return buffer
  }
  public static func save(_ audio: MLXArray, sampleRate: Int, to url: URL, float32: Bool = false)
    throws
  {
    if ProcessInfo.processInfo.environment["DEMUCS_AUDIO_WRITE"] == "baseline" {
      try saveLegacy(audio, sampleRate: sampleRate, to: url, float32: float32)
      return
    }
    try PreparedAudio(audio).save(sampleRate: sampleRate, to: url, float32: float32)
  }
  /// Write `[channels, samples]` audio in any supported format. The container
  /// is chosen by the URL's extension, which should be `format.fileExtension`.
  public static func save(
    _ audio: MLXArray, sampleRate: Int, to url: URL, format: AudioFileFormat
  ) throws {
    try PreparedAudio(audio).save(sampleRate: sampleRate, to: url, format: format)
  }
  static func saveLegacy(_ audio: MLXArray, sampleRate: Int, to url: URL, float32: Bool) throws {
    let pcm = try pcmBuffer(from: audio, sampleRate: sampleRate)
    let settings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate,
      AVNumberOfChannelsKey: audio.dim(0), AVLinearPCMBitDepthKey: float32 ? 32 : 16,
      AVLinearPCMIsFloatKey: float32, AVLinearPCMIsBigEndianKey: false,
      AVLinearPCMIsNonInterleaved: false,
    ]
    let file = try AVAudioFile(
      forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    try file.write(from: pcm)
  }
  public static func separate(
    _ url: URL, using separator: Separator,
    progress: (@Sendable (SeparationProgress) -> Void)? = nil
  ) async throws -> SeparationResult {
    try Task.checkCancellation()
    let decode = Task.detached(priority: .userInitiated) {
      AudioTensor(try load(url, sampleRate: separator.sampleRate))
    }
    let input = try await withTaskCancellationHandler {
      try await decode.value
    } onCancel: {
      decode.cancel()
    }
    try Task.checkCancellation()
    return try await separator.separate(input, progress: progress)
  }
  public static func export(_ result: SeparationResult, to directory: URL, float32: Bool = false)
    throws
  {
    try export(result, to: directory, format: .wav(float32 ? .float32 : .pcm16))
  }
  /// Write one file per stem into `directory`, named `<stem>.<extension>`.
  public static func export(
    _ result: SeparationResult, to directory: URL, format: AudioFileFormat
  ) throws {
    try format.validate()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    for (index, name) in result.sources.enumerated() {
      // Source names from user-provided metadata cannot escape the directory.
      guard !name.contains("/"), !name.contains("\\"), name != ".", name != ".." else {
        throw DemucsError.invalidCache("Invalid stem filename")
      }
      try save(
        result.audio[index], sampleRate: result.sampleRate,
        to: directory.appendingPathComponent(name + "." + format.fileExtension), format: format)
    }
  }
}
