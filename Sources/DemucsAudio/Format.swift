import AVFoundation
import DemucsMLX
import Foundation

/// Output file format for stems.
public enum AudioFileFormat: Sendable, Hashable {
  public enum WAVEncoding: Sendable, Hashable { case pcm16, pcm24, float32 }
  /// Uncompressed WAV. 16-bit and float32 use the built-in fast writer.
  case wav(WAVEncoding = .pcm16)
  /// FLAC, lossless. `bitDepth` is 16 or 24.
  case flac(bitDepth: Int = 16)
  /// Apple Lossless in an `.m4a` container. `bitDepth` is 16 or 24.
  case alac(bitDepth: Int = 16)
  /// AAC in an `.m4a` container, at `bitRate` bits per second.
  case aac(bitRate: Int = 256_000)

  public var fileExtension: String {
    switch self {
    case .wav: "wav"
    case .flac: "flac"
    case .alac, .aac: "m4a"
    }
  }

  /// True for the two encodings the native WAV writer produces directly.
  var usesNativeWAVWriter: Bool { self == .wav(.pcm16) || self == .wav(.float32) }
  var isFloat32WAV: Bool { self == .wav(.float32) }

  func validate() throws {
    switch self {
    case .wav: return
    case .flac(let depth), .alac(let depth):
      guard depth == 16 || depth == 24 else {
        throw DemucsError.invalidOptions("Lossless bit depth must be 16 or 24")
      }
    case .aac(let rate):
      guard (32_000...512_000).contains(rate) else {
        throw DemucsError.invalidOptions("AAC bit rate must be between 32000 and 512000")
      }
    }
  }

  func settings(sampleRate: Int, channels: Int) -> [String: Any] {
    var settings: [String: Any] = [AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: channels]
    switch self {
    case .wav(let encoding):
      settings[AVFormatIDKey] = kAudioFormatLinearPCM
      settings[AVLinearPCMBitDepthKey] = encoding == .pcm16 ? 16 : encoding == .pcm24 ? 24 : 32
      settings[AVLinearPCMIsFloatKey] = encoding == .float32
      settings[AVLinearPCMIsBigEndianKey] = false
      settings[AVLinearPCMIsNonInterleaved] = false
    case .flac(let depth):
      settings[AVFormatIDKey] = kAudioFormatFLAC
      settings[AVLinearPCMBitDepthKey] = depth
    case .alac(let depth):
      settings[AVFormatIDKey] = kAudioFormatAppleLossless
      settings[AVEncoderBitDepthHintKey] = depth
    case .aac(let rate):
      settings[AVFormatIDKey] = kAudioFormatMPEG4AAC
      settings[AVEncoderBitRateKey] = rate
    }
    return settings
  }
}
