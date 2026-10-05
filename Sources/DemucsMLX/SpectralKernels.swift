import Cmlx
import Foundation
import MLX
import Metal

// Adapted from mlx-spectro's fused_frame_extract/tiled_frame_extract.
// Copyright (c) 2026 ssmall256. MIT licensed; see LICENSE.
struct FrameExtraction {
  static let simple = MLXFast.metalKernel(
    name: "demucs_fused_frame_extract", inputNames: ["signal", "win", "params"],
    outputNames: ["out"],
    source: """
      int length = params[0], frames = params[1];
      int bin = int(thread_position_in_grid.x);
      int frame = int(thread_position_in_grid.y);
      ulong row = thread_position_in_grid.z;
      if (bin >= NFFT || frame >= frames) return;
      int pos = frame * HOP + bin - PAD;
      int index = pos < 0 ? -pos : (pos >= length ? 2 * length - 2 - pos : pos);
      out[(row * ulong(frames) + ulong(frame)) * NFFT + bin] =
          signal[row * ulong(length) + index] * win[bin];
      """)
  static let tiled = MLXFast.metalKernel(
    name: "demucs_tiled_frame_extract", inputNames: ["signal", "win", "params"],
    outputNames: ["out"],
    source: """
      threadgroup float shared_signal[CHUNK];
      int length = params[0], frames = params[1];
      int lx = int(thread_position_in_threadgroup.x);
      int ly = int(thread_position_in_threadgroup.y);
      int first = int(threadgroup_position_in_grid.x) * TILE;
      ulong row = threadgroup_position_in_grid.z;
      int count = min(TILE, frames - first);
      int needed = (count - 1) * HOP + NFFT;
      for (int j = ly * WIDTH + lx; j < needed; j += WIDTH * TILE) {
          int pos = first * HOP + j - PAD;
          int index = pos < 0 ? -pos : (pos >= length ? 2 * length - 2 - pos : pos);
          shared_signal[j] = signal[row * ulong(length) + index];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ly < count) {
          for (int bin = lx; bin < NFFT; bin += WIDTH) {
              out[(row * ulong(frames) + ulong(first + ly)) * NFFT + bin] =
                  shared_signal[ly * HOP + bin] * win[bin];
          }
      }
      """)
  static let device = MTLCreateSystemDefaultDevice()
  static func supportsTile(nfft: Int, hop: Int, width: Int, frames: Int) -> Bool {
    guard let device else { return false }
    return frames >= 2 && width >= 32 && width * frames <= device.maxThreadsPerThreadgroup.width
      && ((frames - 1) * hop + nfft) * 4 <= min(32768, device.maxThreadgroupMemoryLength)
  }
  static func apply(
    _ signal: MLXArray, window: MLXArray, nfft: Int, hop: Int,
    choice: SpectralKernelChoice
  ) -> MLXArray {
    let rows = signal.dim(0)
    let length = signal.dim(1)
    let frames = (length - nfft % 2) / hop + 1
    let params = MLXArray([Int32(length), Int32(frames)])
    let shape = [rows, frames, nfft]
    let width = min(choice.width, nfft)
    if choice.tileFrames > 1
      && supportsTile(
        nfft: nfft, hop: hop, width: width, frames: choice.tileFrames)
    {
      let tiles = (frames + choice.tileFrames - 1) / choice.tileFrames
      return tiled(
        [signal, window, params],
        template: [
          ("NFFT", nfft), ("HOP", hop), ("PAD", nfft / 2), ("WIDTH", width),
          ("TILE", choice.tileFrames), ("CHUNK", (choice.tileFrames - 1) * hop + nfft),
        ], grid: (tiles * width, choice.tileFrames, rows),
        threadGroup: (width, choice.tileFrames, 1), outputShapes: [shape], outputDTypes: [.float32])[
          0]
    }
    return simple(
      [signal, window, params], template: [("NFFT", nfft), ("HOP", hop), ("PAD", nfft / 2)],
      grid: (nfft, frames, rows), threadGroup: (width, 1, 1), outputShapes: [shape],
      outputDTypes: [.float32])[0]
  }
}

struct SpectralKernelChoice: Codable, Equatable, Sendable {
  var width: Int = 256
  var tileFrames: Int = 1
  var valid: Bool { [64, 128, 256, 512].contains(width) && [1, 2, 4].contains(tileFrames) }
}

/// Explicit tuning never runs from graph construction or ordinary separation.
final class SpectralKernelCache: @unchecked Sendable {
  static let shared = SpectralKernelCache()
  private let lock = NSLock()
  private var choices: [String: SpectralKernelChoice] = [:]
  let url: URL
  let identity: String
  init(url: URL? = nil) {
    self.url =
      url ?? ProcessInfo.processInfo.environment["DEMUCS_SPECTRAL_CACHE"].map {
        URL(fileURLWithPath: $0)
      }
      ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("demucs-mlx-swift/spectral-kernels-v1.json")
    let device = FrameExtraction.device
    var version = mlx_string_new()
    defer { mlx_string_free(version) }
    mlx_version(&version)
    let runtime = String(cString: mlx_string_data(version))
    identity =
      "v4:mlx\(runtime):\(device?.registryID ?? 0):\(device?.name ?? "cpu"):\(ProcessInfo.processInfo.operatingSystemVersionString)"
    if let data = Self.readBounded(self.url),
      let file = try? JSONDecoder().decode(File.self, from: data), file.identity == identity,
      file.choices.count <= 256
    {
      choices = file.choices.filter { $0.value.valid }
    }
  }
  private static func readBounded(_ url: URL) -> Data? {
    guard let file = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? file.close() }
    guard let data = try? file.read(upToCount: 262_145), data.count <= 262_144 else {
      return nil
    }
    return data
  }
  private struct File: Codable {
    let identity: String
    let choices: [String: SpectralKernelChoice]
  }
  func key(kind: String, nfft: Int, hop: Int, rows: Int, length: Int) -> String {
    "\(kind):f32:\(nfft):\(hop):\(rows):\(length)"
  }
  func choice(for key: String, fallback: SpectralKernelChoice = .init()) -> SpectralKernelChoice {
    lock.withLock { choices[key] ?? fallback }
  }
  func store(_ choice: SpectralKernelChoice, for key: String) {
    guard choice.valid, key.utf8.count <= 256 else { return }
    lock.withLock {
      if choices.count >= 256 && choices[key] == nil { choices.removeAll() }
      choices[key] = choice
      if let data = try? JSONEncoder().encode(File(identity: identity, choices: choices)) {
        try? FileManager.default.createDirectory(
          at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
      }
    }
  }
}
