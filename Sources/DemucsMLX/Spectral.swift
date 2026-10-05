import Foundation
import MLX

/// Deterministic gather overlap-add: each sample sums frames in ascending order.
/// No atomics, host copies, or output-sized scatter intermediates.
struct OverlapAdd {
  static let kernel = MLXFast.metalKernel(
    name: "demucs_overlap_add", inputNames: ["frames", "window", "params"], outputNames: ["out"],
    source: """
      ulong rows = params[0], length = params[1];
      int frame_count = params[2], offset = params[3];
      ulong row = thread_position_in_grid.y;
      uint sample = thread_position_in_grid.x;
      if (sample >= length || row >= rows) return;
      ulong i = row * length + sample;
      uint t = sample + offset;
      int first = max(0, (int(t) - FRAME_LENGTH + STRIDE) / STRIDE);
      int last = min(frame_count - 1, int(t) / STRIDE);
      float value = 0.0f;
      float weight = 0.0f;
      for (int f = first; f <= last; ++f) {
          uint j = t - f * STRIDE;
          float w = window[j];
          value += frames[(row * frame_count + f) * FRAME_LENGTH + j] * w;
          weight += SQUARE_WINDOW ? w * w : w;
      }
      out[i] = weight > 0.0f ? value / weight : 0.0f;
      """)
  static func apply(
    _ frames: MLXArray, window: MLXArray, stride: Int, length: Int, offset: Int = 0,
    squareWindow: Bool = false, threadWidth: Int = 256
  ) -> MLXArray {
    let rows = frames.shape.dropLast(2).reduce(1, *)
    let n = frames.dim(-2)
    let width = frames.dim(-1)
    return kernel(
      [frames, window, MLXArray([Int64(rows), Int64(length), Int64(n), Int64(offset)])],
      template: [
        ("FRAME_LENGTH", width), ("STRIDE", stride), ("SQUARE_WINDOW", squareWindow),
      ], grid: (length, rows, 1), threadGroup: (threadWidth, 1, 1),
      outputShapes: [Array(frames.shape.dropLast(2)) + [length]], outputDTypes: [.float32])[0]
  }
}
struct SpectralTransform {
  let nfft: Int
  let hop: Int
  let window: MLXArray
  init(nfft: Int) {
    self.nfft = nfft
    hop = nfft / 4
    window = 0.5 - 0.5 * cos(MLXArray(0..<nfft).asType(.float32) * (2 * Float.pi / Float(nfft)))
  }
  func windowedFrames(_ x: MLXArray, choice: SpectralKernelChoice? = nil) -> MLXArray {
    let rows = x.shape.dropLast().reduce(1, *)
    let length = x.dim(-1)
    let useFused = ProcessInfo.processInfo.environment["DEMUCS_SPECTRAL_MODE"] != "baseline"
    if x.dtype == .float32 && length > nfft / 2 && length < Int(Int32.max) / 2
      && FrameExtraction.device != nil && (useFused || choice != nil)
    {
      let n = (length - nfft % 2) / hop + 1
      var fallback = SpectralKernelChoice()
      if rows * n * nfft * 4 >= 100_000_000
        && FrameExtraction.supportsTile(nfft: nfft, hop: hop, width: 256, frames: 4)
      {
        fallback.tileFrames = 4
      }
      let key = SpectralKernelCache.shared.key(
        kind: "frames", nfft: nfft, hop: hop, rows: rows, length: length)
      return FrameExtraction.apply(
        x.reshaped(rows, length), window: window, nfft: nfft, hop: hop,
        choice: choice ?? SpectralKernelCache.shared.choice(for: key, fallback: fallback))
    }
    return originalFrames(x)
  }
  func originalFrames(_ x: MLXArray) -> MLXArray {
    let rows = x.shape.dropLast().reduce(1, *)
    let length = x.dim(-1)
    let y = padLast(x.reshaped(rows, length), nfft / 2, nfft / 2, reflect: true)
    let n = (y.dim(-1) - nfft) / hop + 1
    return asStrided(y, [rows, n, nfft], strides: [y.dim(-1), hop, 1]) * window
  }
  func stft(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    let frames = windowedFrames(x)
    let n = frames.dim(1)
    let z = rfft(frames, axis: -1).transposed(0, 2, 1)
    return z.reshaped(Array(shape.dropLast()) + [nfft / 2 + 1, n])
  }
  func istft(_ z: MLXArray, length: Int) -> MLXArray {
    let shape = z.shape
    let rows = shape.dropLast(2).reduce(1, *)
    let frames = shape.last!
    let waveform = irfft(
      z.reshaped(rows, nfft / 2 + 1, frames).transposed(0, 2, 1), n: nfft, axis: -1)
    let key = SpectralKernelCache.shared.key(
      kind: "ola", nfft: nfft, hop: hop, rows: rows, length: length)
    let result = OverlapAdd.apply(
      waveform, window: window, stride: hop, length: length, offset: nfft / 2, squareWindow: true,
      threadWidth: SpectralKernelCache.shared.choice(for: key).width)
    return result.reshaped(Array(shape.dropLast(2)) + [length])
  }
}
struct JuliusResampler {
  let old: Int, new: Int, width: Int
  let kernel: MLXArray
  init(old: Int, new: Int) {
    self.old = old
    self.new = new
    let rate = Float(min(old, new)) * 0.945
    let zeros: Float = 24
    width = Int(ceil(zeros * Float(old) / rate))
    let indices = MLXArray((-width)..<(width + old)).asType(.float32)
    var phases: [MLXArray] = []
    for i in 0..<new {
      let t =
        clip((indices / Float(old) - Float(i) / Float(new)) * rate, min: -zeros, max: zeros)
        * Float.pi
      let safe = which(t .== 0, MLXArray(Float(1)), t)
      let sinc = which(t .== 0, MLXArray(Float(1)), sin(t) / safe)
      let w = cos(t / zeros / 2)
      let k = sinc * w * w
      phases.append(k / sum(k))
    }
    kernel = stacked(phases)[.ellipsis, .newAxis]
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let length = x.dim(-1)
    let flat = x.reshaped(-1, length).asType(.float32)
    let paddedX = concatenated(
      [
        broadcast(flat[0..., 0..<1], to: [flat.dim(0), width]), flat,
        broadcast(flat[0..., (length - 1)..<length], to: [flat.dim(0), width + old]),
      ], axis: -1)
    let phases = conv1d(paddedX[.ellipsis, .newAxis], kernel, stride: old)
    let y = phases.reshaped(Array(x.shape.dropLast()) + [-1])
    return y[.ellipsis, 0..<(new * length / old)].asType(x.dtype)
  }
}
