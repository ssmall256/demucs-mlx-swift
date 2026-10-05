import AVFoundation
import CDemucsAudioIO
import Foundation
import MLX
import Testing

@testable import DemucsAudio

private func le(_ value: UInt32, _ bytes: Int) -> Data {
  Data((0..<bytes).map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) })
}
private func chunk(_ name: String, _ payload: Data) -> Data {
  Data(name.utf8) + le(UInt32(payload.count), 4) + payload
    + (payload.count.isMultiple(of: 2) ? Data() : Data([0]))
}
private func fixture(
  bits: Int, channels: Int, floating: Bool = false,
  extensible: Bool = false, frames: Int = 70001
) -> (Data, [Float]) {
  var samples = Data()
  var expected = [Float](repeating: 0, count: frames * channels)
  for i in 0..<frames {
    for c in 0..<channels {
      let edges: [UInt32] = [
        0, .max, 0x8000_0000, 0x7fff_ffff, 0x0080_0000,
        0x00ff_ffff, 0x0000_8000, 0x0000_7fff, 0xff,
      ]
      let raw =
        i < edges.count
        ? edges[(i + c) % edges.count]
        : UInt32(truncatingIfNeeded: i &* 7919 &+ c &* 3911)
      let value: Float
      if floating {
        value = Float(Int(raw % 2001) - 1000) / 1024
        samples += le(value.bitPattern, 4)
      } else {
        samples += le(raw, bits / 8)
        switch bits {
        case 8: value = Float(Int(UInt8(truncatingIfNeeded: raw)) - 128) / 128
        case 16: value = Float(Int16(truncatingIfNeeded: raw)) / 32768
        case 24: value = Float(Int32(bitPattern: raw << 8) >> 8) / 8_388_608
        default: value = Float(Int32(bitPattern: raw)) / 2_147_483_648
        }
      }
      expected[c * frames + i] = value
    }
  }
  var fmt =
    le(extensible ? 0xfffe : (floating ? 3 : 1), 2)
    + le(UInt32(channels), 2) + le(44100, 4)
    + le(UInt32(44100 * channels * bits / 8), 4)
    + le(UInt32(channels * bits / 8), 2) + le(UInt32(bits), 2)
  if extensible {
    fmt +=
      le(22, 2) + le(UInt32(bits), 2) + le(0, 4)
      + le(floating ? 3 : 1, 4)
      + Data([0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71])
  }
  let body =
    Data("WAVE".utf8) + chunk("JUNK", Data([1, 2, 3]))
    + chunk("fmt ", fmt) + chunk("data", samples)
  return (Data("RIFF".utf8) + le(UInt32(body.count), 4) + body, expected)
}
private func withFile<T>(_ data: Data, _ body: (URL) throws -> T) throws -> T {
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(
    UUID().uuidString + ".wav")
  try data.write(to: url)
  defer { try? FileManager.default.removeItem(at: url) }
  return try body(url)
}

@Suite struct PCMParserTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test(arguments: [8, 16, 24, 32], [1, 2, 3])
  func boundedPlanarConversion(bits: Int, channels: Int) throws {
    let (data, expected) = fixture(bits: bits, channels: channels)
    try withFile(data) { url in
      var reader: OpaquePointer?
      var info = demucs_pcm_wav_info()
      #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 1)
      let handle = try #require(reader)
      defer { demucs_close_pcm_wav(handle) }
      let stride = Int(info.frames) + 7
      var output = [Float](repeating: -99, count: stride * channels)
      let error = output.withUnsafeMutableBufferPointer {
        demucs_read_pcm_wav(handle, $0.baseAddress, info.frames, stride, nil, nil)
      }
      #expect(error == 0)
      for c in 0..<channels {
        #expect(
          Array(output[c * stride..<c * stride + Int(info.frames)])
            == Array(expected[c * Int(info.frames)..<(c + 1) * Int(info.frames)]))
        #expect(output[c * stride + Int(info.frames)] == -99)
      }
    }
  }
  @Test func rejectsUnsupportedOrMalformedAndCancels() throws {
    let (valid, _) = fixture(bits: 16, channels: 2)
    var badAlign = valid
    // RIFF + odd JUNK (12 bytes) + fmt header + block align offset.
    badAlign[44] = 1
    var hugeChunk = valid
    hugeChunk.replaceSubrange(16..<20, with: le(UInt32.max, 4))
    for data in [
      Data(valid.prefix(20)), Data(valid.dropLast()), badAlign, hugeChunk,
      Data("not wave".utf8), fixture(bits: 16, channels: 1, frames: 1).0,
    ] {
      try withFile(data) { url in
        var reader: OpaquePointer?
        var info = demucs_pcm_wav_info()
        #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 0)
        #expect(reader == nil)
      }
    }
    try withFile(valid) { url in
      var reader: OpaquePointer?
      var info = demucs_pcm_wav_info()
      #expect(demucs_open_pcm_wav(url.path, 48000, &reader, &info) == 0)
      #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 1)
      let handle = try #require(reader)
      defer { demucs_close_pcm_wav(handle) }
      var output = [Float](repeating: -99, count: Int(info.frames) * 2)
      var callbacks = 0
      let error = withUnsafeMutablePointer(to: &callbacks) { context in
        output.withUnsafeMutableBufferPointer {
          demucs_read_pcm_wav(
            handle, $0.baseAddress, info.frames, Int(info.frames),
            { p in
              let counter = p!.assumingMemoryBound(to: Int.self)
              counter.pointee += 1
              return counter.pointee == 2 ? 1 : 0
            }, context)
        }
      }
      #expect(error == ECANCELED)
      #expect(callbacks == 2)
      #expect(output[Int(info.frames) - 1] == -99)
    }
  }
  @Test func floatBitsAndExtensibleValidation() throws {
    var (data, _) = fixture(bits: 32, channels: 1, floating: true, frames: 9)
    let bits: [UInt32] = [
      0, 0x8000_0000, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000,
      0x3f80_0000, 0xbf80_0000, 1, 0x007f_ffff,
    ]
    data.replaceSubrange(56..<92, with: bits.reduce(Data()) { $0 + le($1, 4) })
    try withFile(data) { url in
      var reader: OpaquePointer?
      var info = demucs_pcm_wav_info()
      #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 1)
      let handle = try #require(reader)
      defer { demucs_close_pcm_wav(handle) }
      var output = [Float](repeating: 0, count: 9)
      #expect(
        output.withUnsafeMutableBufferPointer {
          demucs_read_pcm_wav(handle, $0.baseAddress, 9, 9, nil, nil)
        } == 0)
      #expect(output.map(\.bitPattern) == bits)
    }
    let (extended, _) = fixture(bits: 24, channels: 2, extensible: true)
    var wrongGUID = extended
    wrongGUID[60] = 1
    var partialBits = extended
    partialBits[50] = 20
    var rifx = extended
    rifx.replaceSubrange(0..<4, with: Data("RIFX".utf8))
    for data in [wrongGUID, partialBits, rifx] {
      try withFile(data) { url in
        var reader: OpaquePointer?
        var info = demucs_pcm_wav_info()
        #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 0)
        #expect(reader == nil)
      }
    }
  }
  @Test func changedFileAndOutputCapacityAreChecked() throws {
    let (data, _) = fixture(bits: 16, channels: 2)
    try withFile(data) { url in
      var reader: OpaquePointer?
      var info = demucs_pcm_wav_info()
      #expect(demucs_open_pcm_wav(url.path, 44100, &reader, &info) == 1)
      let handle = try #require(reader)
      defer { demucs_close_pcm_wav(handle) }
      var output = [Float](repeating: -99, count: Int(info.frames) * 2)
      #expect(
        output.withUnsafeMutableBufferPointer {
          demucs_read_pcm_wav(
            handle, $0.baseAddress, info.frames - 1,
            Int(info.frames), nil, nil)
        } == EINVAL)
      #expect(
        output.withUnsafeMutableBufferPointer {
          demucs_read_pcm_wav(
            handle, $0.baseAddress, info.frames,
            Int.max, nil, nil)
        } == EINVAL)
      let file = try FileHandle(forWritingTo: url)
      try file.truncate(atOffset: 60)
      try file.close()
      #expect(
        output.withUnsafeMutableBufferPointer {
          demucs_read_pcm_wav(
            handle, $0.baseAddress, info.frames,
            Int(info.frames), nil, nil)
        } == EIO)
    }
  }
}

extension NativeParityTests {
  @Test(arguments: [8, 16, 24, 32], [1, 2, 3])
  func nativePCMMatchesApple(bits: Int, channels: Int) throws {
    for extended in [false, true] {
      let (data, values) = fixture(bits: bits, channels: channels, extensible: extended)
      try withFile(data) { url in
        let native = try #require(try DemucsAudio.loadNativeWAV(url, sampleRate: 44100))
        let apple = try DemucsAudio.loadAppleOwned(url, sampleRate: 44100)
        #expect(arrayEqual(native, apple).item(Bool.self))
        #expect(native.asArray(Float.self) == values)
        #expect(try DemucsAudio.loadNativeWAV(url, sampleRate: 48000) == nil)
        #expect(
          arrayEqual(
            native,
            try DemucsAudio.load(url, sampleRate: 44100, decoder: .nativePCM)
          ).item(Bool.self))
        if bits == 16, channels == 2, !extended {
          #expect(
            arrayEqual(
              try DemucsAudio.load(url, sampleRate: 48000, decoder: .nativePCM),
              try DemucsAudio.load(url, sampleRate: 48000, decoder: .avFoundation)
            ).item(Bool.self))
        }
      }
    }
  }
  @Test func nativeFloatPCMMatchesApple() throws {
    for channels in [1, 2, 3] {
      for extended in [false, true] {
        let (data, values) = fixture(
          bits: 32, channels: channels, floating: true,
          extensible: extended)
        try withFile(data) { url in
          let native = try #require(try DemucsAudio.loadNativeWAV(url, sampleRate: 44100))
          let apple = try DemucsAudio.loadAppleOwned(url, sampleRate: 44100)
          #expect(arrayEqual(native, apple).item(Bool.self))
          #expect(native.asArray(Float.self) == values)
        }
      }
    }
  }
}
