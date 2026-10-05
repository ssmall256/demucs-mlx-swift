import Foundation
import MLX
import MLXNN

struct HybridLayer {
  let frequency: Bool, empty: Bool, last: Bool
  let stride: Int, padding: Int, channels: Int
  let convolution: Op, norm1: Op, norm2: Op, dconv: Op?, rewrite: Op?
  let gatedRewrite: Op?
  let arithmetic: CompilationArithmetic
  init(
    store w: WeightStore, prefix p: String, config c: ModelConfig, stage: Int, frequency: Bool,
    empty: Bool, last: Bool, kernel: Int, stride: Int, padding: Int, decode: Bool, legacy: Bool,
    multi: Bool = false
  ) throws {
    arithmetic = w.arithmetic
    self.frequency = frequency
    self.empty = empty
    self.last = last
    self.stride = stride
    self.padding = padding
    let normEnabled = stage >= c.int("norm_starts", 4)
    let convKey = p + (decode ? ".conv_tr.conv" : ".conv.conv")
    let tensor = try w.get(convKey + ".weight", rank: frequency ? 4 : 3)
    let waveform = p.hasPrefix("t")
    let base = waveform ? c.int("channels_time", c.int("channels", 48)) : c.int("channels", 48)
    let outputChannels = Int(Double(base) * pow(c.double("growth", 2), Double(stage)))
    let previous =
      stage == 0
      ? c.int("audio_channels", 2) * (waveform ? 1 : (c.bool("cac", true) ? 2 : 1))
      : Int(Double(base) * pow(c.double("growth", 2), Double(stage - 1)))
    let expectedInput = decode ? outputChannels : previous
    let expectedOutput =
      decode ? (stage == 0 ? previous * c.sources.count : previous) : outputChannels
    guard tensor.dim(0) == expectedOutput, tensor.dim(-1) == expectedInput, tensor.dim(1) == kernel
    else {
      throw DemucsError.invalidCache("Convolution dimensions disagree with constructor: \(convKey)")
    }
    channels = tensor.dim(-1)
    convolution = try w.conv(
      convKey, stride: stride, padding: decode ? 0 : padding, frequency: frequency,
      transposed: decode)
    norm1 =
      !empty && normEnabled ? try w.norm(p + ".norm1", groups: c.int("norm_groups", 4)) : identity
    norm2 =
      normEnabled && (decode || !empty)
      ? try w.norm(p + ".norm2", groups: c.int("norm_groups", 4)) : identity
    if !empty && c.bool("rewrite", true) {
      let context = c.int(decode ? "context" : "context_enc", decode ? 1 : 0)
      rewrite = try w.conv(
        p + ".rewrite.conv", padding: frequency && multi && decode ? 0 : context,
        frequency: frequency, widthPadding: context)
      let rewriteWeight = try w.get(p + ".rewrite.conv.weight")
      let rewriteBias = try w.get(p + ".rewrite.conv.bias", rank: 1)
      gatedRewrite =
        decode && !legacy && !multi && frequency && !normEnabled && context == 1
          && rewriteWeight.dtype == .float32 && rewriteBias.dtype == .float32
          && rewriteWeight.dim(1) == 3 && rewriteWeight.dim(2) == 3
        ? try w.gatedFrequencyRewrite(p + ".rewrite.conv", padding: context) : nil
      let rewritten = decode ? expectedInput : expectedOutput
      guard rewriteWeight.dim(0) == 2 * rewritten, rewriteWeight.dim(-1) == rewritten else {
        throw DemucsError.invalidCache("Rewrite dimensions disagree with convolution: \(p)")
      }
    } else {
      rewrite = nil
      gatedRewrite = nil
    }
    if !empty && c.int("dconv_mode", legacy ? 1 : 3) & (decode ? 2 : 1) != 0 {
      dconv = try w.dconv(p + ".dconv", config: c, stage: stage, legacy: legacy)
    } else {
      dconv = nil
    }
  }
  private func residual(_ y: MLXArray) -> MLXArray {
    guard let dconv else { return y }
    if !frequency { return dconv(y) }
    let b = y.dim(0)
    let c = y.dim(1)
    let f = y.dim(2)
    let t = y.dim(3)
    return dconv(y.transposed(0, 2, 1, 3).reshaped(-1, c, t)).reshaped(b, f, c, t).transposed(
      0, 2, 1, 3)
  }
  func encode(_ input: MLXArray, inject: MLXArray? = nil) -> MLXArray {
    var x = input
    if !frequency {
      if x.ndim == 4 { x = x.reshaped(x.dim(0), -1, x.dim(-1)) }
      let remainder = x.dim(-1) % stride
      if remainder != 0 { x = padLast(x, 0, stride - remainder) }
    }
    var y = convolution(x)
    if empty { return y }
    if let inject {
      y = y + (inject.ndim == 3 && y.ndim == 4 ? inject[0..., 0..., .newAxis, 0...] : inject)
    }
    y = residual(arithmetic.activation(norm1(y)))
    return rewrite.map { glu(norm2($0(y))) } ?? y
  }
  func decode(_ input: MLXArray, skip: MLXArray?, length: Int) -> (MLXArray, MLXArray) {
    var x = input
    if frequency && x.ndim == 3 { x = x.reshaped(x.dim(0), channels, -1, x.dim(-1)) }
    var y = x
    if !empty {
      x = x + skip!
      y = gatedRewrite?(x) ?? rewrite.map { glu(norm1($0(x))) } ?? x
      y = residual(y)
    }
    var z = norm2(convolution(y))
    if frequency {
      if padding > 0 { z = z[0..., 0..., padding..<(z.dim(2) - padding), 0...] }
    } else {
      z = z[.ellipsis, padding..<(padding + length)]
    }
    return (last ? z : arithmetic.activation(z), y)
  }
}
struct HybridStage {
  let layers: [HybridLayer]
  let ratios: [Double]
  let decode: Bool
  let biases: [MLXArray]
  let kernel: Int
  init(
    store w: WeightStore, prefix p: String, config c: ModelConfig, stage: Int, frequency: Bool,
    empty: Bool, last: Bool, kernel: Int, stride: Int, padding: Int, decode: Bool, legacy: Bool
  ) throws {
    self.decode = decode
    self.kernel = kernel
    ratios = stage < c.int("multi_freqs_depth", 3) ? c.floats("multi_freqs") : []
    var list: [HybridLayer] = []
    var bs: [MLXArray] = []
    for index in 0..<(ratios.isEmpty ? 1 : ratios.count + 1) {
      let path = ratios.isEmpty ? p : p + ".layers.\(index)"
      list.append(
        try HybridLayer(
          store: w, prefix: path, config: c, stage: stage, frequency: frequency, empty: empty,
          last: ratios.isEmpty ? last : true, kernel: kernel, stride: stride,
          padding: ratios.isEmpty ? padding : 0, decode: decode, legacy: legacy,
          multi: !ratios.isEmpty))
      if decode && !ratios.isEmpty { bs.append(try w.get(path + ".conv_tr.conv.bias")) }
    }
    layers = list
    biases = bs
    finalLast = last
  }
  let finalLast: Bool
  func encode(_ x: MLXArray, inject: MLXArray?) -> MLXArray {
    if ratios.isEmpty { return layers[0].encode(x, inject: inject) }
    var start = 0
    var outputs: [MLXArray] = []
    let f = x.dim(2)
    let pad = kernel / 4
    for (i, r) in (ratios + [1]).enumerated() {
      var limit = f
      if r != 1 {
        let le = Int((Double(f) * r).rounded(.toNearestOrEven)) - start + (start == 0 ? pad : 0)
        let frames = Int(
          (Double(le - kernel) / Double(layers[i].stride) + 1).rounded(.toNearestOrEven))
        limit = start + (frames - 1) * layers[i].stride + kernel - (start == 0 ? pad : 0)
      }
      var y = x[0..., 0..., start..<limit, 0...]
      y = padded(y, widths: [0, 0, [start == 0 ? pad : 0, r == 1 ? pad : 0], 0])
      outputs.append(layers[i].encode(y))
      start = limit - kernel + layers[i].stride
    }
    return concatenated(outputs, axis: 2)
  }
  func decode(_ x: MLXArray, skip: MLXArray, length: Int) -> (MLXArray, MLXArray) {
    if ratios.isEmpty { return layers[0].decode(x, skip: skip, length: length) }
    var start = 0
    var outputs: [MLXArray] = []
    let f = x.dim(2)
    for (i, r) in (ratios + [1]).enumerated() {
      let limit = r == 1 ? f : Int((Double(f) * r).rounded(.toNearestOrEven))
      let stride = layers[i].stride
      var y = layers[i].decode(
        x[0..., 0..., start..<limit, 0...], skip: skip[0..., 0..., start..<limit, 0...],
        length: length
      ).0
      if !outputs.isEmpty {
        let previous = outputs.removeLast()
        let end = previous.dim(2)
        let overlap =
          previous[0..., 0..., (end - stride)..<end, 0...] + y[0..., 0..., 0..<stride, 0...]
          - biases[i].reshaped(1, -1, 1, 1)
        outputs.append(
          concatenated([previous[0..., 0..., 0..<(end - stride), 0...], overlap], axis: 2))
        y = y[0..., 0..., stride..., 0...]
      }
      if r == 1 { y = y[0..., 0..., 0..<(y.dim(2) - stride / 2), 0...] }
      if start == 0 { y = y[0..., 0..., (stride / 2)..., 0...] }
      outputs.append(y)
      start = limit
    }
    let y = concatenated(outputs, axis: 2)
    return (finalLast ? y : gelu(y), y)
  }
}
