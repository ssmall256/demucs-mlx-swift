import Foundation
import MLX
import MLXNN

typealias Op = (MLXArray) -> MLXArray
var identity: Op { { $0 } }
// Keep eager arithmetic identical to Python MLX. MLXNN.gelu compiles itself,
// which changes rounding before FP16 attention even in an eager model pass.
func gelu(_ x: MLXArray) -> MLXArray {
  x * (1 + erf(x / sqrt(Float(2)))) / 2
}
func gelu(_ x: MLXArray, divisor: MLXArray) -> MLXArray {
  x * (1 + erf(x / divisor)) / 2
}
/// Only populated while tracing a separator-owned first forward. Passing the
/// divisor as a graph input avoids MLX 0.32's seven-digit scalar serialization.
final class CompilationArithmetic {
  let rootTwo = MLXArray(sqrt(Float(2)))
  var divisor: MLXArray?
  func activation(_ x: MLXArray) -> MLXArray {
    if let divisor { return gelu(x, divisor: divisor) }
    return gelu(x)
  }
}
/// Keep division separate from Power, whose fused Metal expression changes
/// positional embedding rounding. Depends preserves storage without a copy.
func positionalIndexDivide(_ x: MLXArray, _ denominator: MLXArray) -> MLXArray {
  let operands = depends(inputs: [x, broadcast(denominator, to: x.shape)], dependencies: [x])
  return depends(input: operands[0] / operands[1], dependencies: [operands[0]])
}
func trim(_ x: MLXArray, _ length: Int) -> MLXArray {
  let start = (x.dim(-1) - length) / 2
  return x[.ellipsis, start..<(start + length)]
}
func padLast(_ x: MLXArray, _ left: Int, _ right: Int, reflect: Bool = false) -> MLXArray {
  var x = x
  var left = left
  var right = right
  if reflect {
    let extra = max(0, max(left, right) - x.dim(-1) + 1)
    if extra > 0 {
      let r = min(right, extra)
      let l = extra - r
      x = padLast(x, l, r)
      left -= l
      right -= r
    }
    var pieces: [MLXArray] = []
    if left > 0 { pieces.append(x[.ellipsis, 1..<(left + 1)][.ellipsis, .stride(by: -1)]) }
    pieces.append(x)
    if right > 0 {
      pieces.append(
        x[.ellipsis, (x.dim(-1) - right - 1)..<(x.dim(-1) - 1)][.ellipsis, .stride(by: -1)])
    }
    return concatenated(pieces, axis: -1)
  }
  var widths = Array(repeating: IntOrPair(0), count: x.ndim)
  widths[x.ndim - 1] = IntOrPair((left, right))
  return padded(x, widths: widths)
}
func glu(_ x: MLXArray, axis: Int = 1) -> MLXArray {
  let halves = split(x, parts: 2, axis: axis)
  return halves[0] * sigmoid(halves[1])
}
/// Constructed on one inference executor; immutable operators hold owned arrays.
final class WeightStore {
  let arrays: [String: MLXArray]
  var used = Set<String>()
  let arithmetic = CompilationArithmetic()
  init(_ arrays: [String: MLXArray]) { self.arrays = arrays }
  func has(_ key: String) -> Bool { arrays[key] != nil }
  func verify(_ key: String, shape: [Int]) throws {
    guard try get(key).shape == shape else {
      throw DemucsError.invalidCache("Unexpected tensor shape: \(key); expected \(shape)")
    }
  }
  func get(_ key: String, rank: Int? = nil) throws -> MLXArray {
    guard let a = arrays[key], rank == nil || a.ndim == rank else {
      throw DemucsError.invalidCache("Missing or invalid tensor: \(key)")
    }
    used.insert(key)
    return a
  }
  func conv(
    _ p: String, stride: Int = 1, padding: Int = 0, dilation: Int = 1, frequency: Bool = false,
    widthPadding: Int? = nil, transposed: Bool = false
  ) throws -> Op {
    let w = try get(p + ".weight", rank: frequency ? 4 : 3)
    let b = try get(p + ".bias", rank: 1)
    guard b.size == w.dim(0) else {
      throw DemucsError.invalidCache("Convolution bias shape mismatch: \(p)")
    }
    if !transposed && stride == 1 && padding == 0 && (widthPadding ?? 0) == 0
      && w.shape.dropFirst().dropLast().allSatisfy({ $0 == 1 })
    {
      // Kernel-one convolution is a channel projection. Route it directly
      // through GEMM with fused bias, retaining FP32 weights and arithmetic.
      let matrix = w.reshaped(w.dim(0), w.dim(-1)).T
      return { x in
        if frequency {
          return addMM(b, x.transposed(0, 2, 3, 1), matrix).transposed(0, 3, 1, 2)
        }
        return addMM(b, x.transposed(0, 2, 1), matrix).transposed(0, 2, 1)
      }
    }
    if transposed && stride == 4 && padding == 0 && w.dim(1) == 8 && (!frequency || w.dim(2) == 1) {
      let phases = (0..<4).map { i in stacked([w[0..., i + 4], w[0..., i]], axis: 1) }
      let weights = concatenated(phases, axis: 0)
      return { x in
        if frequency {
          let a = x.transposed(0, 2, 3, 1)
          let paddedX = padded(a, widths: [0, [1, 1], 0, 0])
          let y = conv2d(paddedX, weights).reshaped(x.dim(0), x.dim(2) + 1, x.dim(3), 4, w.dim(0))
            .transposed(0, 1, 3, 2, 4)
          return (y.reshaped(x.dim(0), 4 * (x.dim(2) + 1), x.dim(3), w.dim(0)) + b).transposed(
            0, 3, 1, 2)
        }
        let a = padded(x.transposed(0, 2, 1), widths: [0, [1, 1], 0])
        let y = conv1d(a, weights).reshaped(x.dim(0), 4 * (x.dim(2) + 1), w.dim(0))
        return (y + b).transposed(0, 2, 1)
      }
    }
    return { x in
      if frequency {
        let a = x.transposed(0, 2, 3, 1)
        let y: MLXArray
        if transposed {
          y = convTransposed2d(a, w, stride: [stride, 1], padding: [padding, widthPadding ?? 0])
        } else {
          y = conv2d(
            a, w, stride: [stride, 1], padding: [padding, widthPadding ?? 0],
            dilation: [dilation, 1])
        }
        return (y + b).transposed(0, 3, 1, 2)
      }
      let a = x.transposed(0, 2, 1)
      let y =
        transposed
        ? convTransposed1d(a, w, stride: stride, padding: padding, dilation: dilation)
        : conv1d(a, w, stride: stride, padding: padding, dilation: dilation)
      return (y + b).transposed(0, 2, 1)
    }
  }
  /// Standard HTDemucs decoder rewrite with identity normalization. The
  /// convolution is unchanged; split before bias to fuse the bias and gate.
  func gatedFrequencyRewrite(_ p: String, padding: Int) throws -> Op {
    let w = try get(p + ".weight", rank: 4)
    let b = try get(p + ".bias", rank: 1)
    guard w.dtype == .float32, b.dtype == .float32, w.dim(1) == 3, w.dim(2) == 3,
      w.dim(0) % 2 == 0, b.size == w.dim(0)
    else { throw DemucsError.invalidCache("Invalid gated rewrite: \(p)") }
    let bias = split(b, parts: 2)
    return { x in
      let y = conv2d(x.transposed(0, 2, 3, 1), w, padding: [padding, padding])
      let halves = split(y, parts: 2, axis: -1)
      return ((halves[0] + bias[0]) * sigmoid(halves[1] + bias[1])).transposed(0, 3, 1, 2)
    }
  }
  func norm(_ p: String, groups: Int = 1, channelLast: Bool = false) throws -> Op {
    guard has(p + ".weight") || has(p + ".gn.weight") else {
      throw DemucsError.invalidCache("Missing normalization tensors: \(p)")
    }
    let key = has(p + ".weight") ? p : p + ".gn"
    let w = try get(key + ".weight", rank: 1)
    let b = try get(key + ".bias", rank: 1)
    guard w.shape == b.shape, w.size % groups == 0 else {
      throw DemucsError.invalidCache("Normalization dimensions mismatch: \(p)")
    }
    if channelLast && key == p { return { MLXFast.layerNorm($0, weight: w, bias: b, eps: 1e-5) } }
    return { x in
      let y = channelLast ? x.transposed(0, 2, 1) : x
      let n = MLXFast.layerNorm(y.reshaped(y.dim(0), groups, -1), eps: 1e-5).reshaped(y.shape)
      let shape = [1, w.size] + Array(repeating: 1, count: y.ndim - 2)
      let out = n * w.reshaped(shape) + b.reshaped(shape)
      return channelLast ? out.transposed(0, 2, 1) : out
    }
  }
  func linear(_ p: String) throws -> Op {
    let w = try get(p + ".weight", rank: 2)
    let b = try get(p + ".bias", rank: 1)
    guard b.size == w.dim(0) else { throw DemucsError.invalidCache("Linear shape mismatch: \(p)") }
    return { addMM(b, $0, w.T) }
  }
  func scale(_ p: String, channelLast: Bool = false) throws -> Op {
    guard has(p + ".scale") else { return identity }
    let w = try get(p + ".scale", rank: 1)
    return channelLast ? { $0 * w } : { $0 * w[0..., .newAxis] }
  }
  func dconv(_ p: String, config c: ModelConfig, stage: Int = 0, legacy: Bool = false) throws -> Op
  {
    let depth = abs(c.int("dconv_depth", 2))
    let useLSTM = legacy && stage >= c.int("dconv_lstm", 4)
    let useAttention = legacy && stage >= c.int("dconv_attn", 4)
    // Standard DConv remains channels-last across its entire residual chain.
    if !useLSTM && !useAttention {
      var native: [Op] = []
      for i in 0..<depth {
        let q = p + ".layers.\(i).layers"
        let w1 = try get(q + ".0.conv.weight", rank: 3)
        let b1 = try get(q + ".0.conv.bias", rank: 1)
        let w2 = try get(q + ".3.conv.weight", rank: 3)
        let b2 = try get(q + ".3.conv.bias", rank: 1)
        let nw1 = try get(q + ".1.weight", rank: 1)
        let nb1 = try get(q + ".1.bias", rank: 1)
        let nw2 = try get(q + ".4.weight", rank: 1)
        let nb2 = try get(q + ".4.bias", rank: 1)
        let scale = try get(q + ".6.scale", rank: 1)
        guard w1.dim(1) == 3, w2.dim(1) == 1, w1.dim(0) == w2.dim(2), w2.dim(0) == 2 * w1.dim(2),
          b1.size == w1.dim(0), nw1.shape == b1.shape, nb1.shape == b1.shape,
          b2.size == w2.dim(0), nw2.shape == b2.shape, nb2.shape == b2.shape,
          scale.size == w1.dim(2)
        else {
          throw DemucsError.invalidCache("Invalid residual convolution dimensions: \(q)")
        }
        let dilation = c.int("dconv_depth", 2) > 0 ? 1 << i : 1
        let projection = w2.reshaped(w2.dim(0), w2.dim(-1)).T
        let gateWeight = split(nw2, parts: 2)
        let gateBias = split(nb2, parts: 2)
        native.append { x in
          let h = conv1d(x, w1, padding: dilation, dilation: dilation) + b1
          let hn =
            MLXFast.layerNorm(h.reshaped(h.dim(0), 1, -1), eps: 1e-5).reshaped(h.shape) * nw1 + nb1
          let o = addMM(b2, self.arithmetic.activation(hn), projection)
          let normalized =
            MLXFast.layerNorm(o.reshaped(o.dim(0), 1, -1), eps: 1e-5).reshaped(o.shape)
          // Split the normalized view before affine arithmetic. A slice after
          // affine forces the compiler to materialize both large gate planes;
          // this order lets affine, GLU, scale and residual addition fuse.
          let halves = split(normalized, parts: 2, axis: -1)
          let left = halves[0] * gateWeight[0] + gateBias[0]
          let right = halves[1] * gateWeight[1] + gateBias[1]
          return x + left * sigmoid(right) * scale
        }
      }
      return { input in
        native.reduce(input.transposed(0, 2, 1)) { x, f in f(x) }.transposed(0, 2, 1)
      }
    }
    var blocks: [Op] = []
    for i in 0..<depth {
      let q = p + ".layers.\(i).layers"
      let dilation = c.int("dconv_depth", 2) > 0 ? 1 << i : 1
      let c1 = try conv(q + ".0.conv", padding: dilation, dilation: dilation)
      let n1 = try norm(q + ".1")
      var offset = 3
      var extra: [Op] = []
      if useLSTM {
        extra.append(try blstm(q + ".\(offset)", layers: 2, maxSteps: 200, skip: true))
        offset += 1
      }
      if useAttention {
        extra.append(try localState(q + ".\(offset)"))
        offset += 1
      }
      let c2 = try conv(q + ".\(offset).conv")
      let n2 = try norm(q + ".\(offset+1)")
      let s = try scale(q + ".\(offset+3)")
      let middle = extra
      blocks.append { x in
        var y = self.arithmetic.activation(n1(c1(x)))
        for operation in middle { y = operation(y) }
        return x + s(glu(n2(c2(y))))
      }
    }
    return { x in blocks.reduce(x) { y, f in f(y) } }
  }
  func localState(_ p: String) throws -> Op {
    let content = try conv(p + ".content.conv")
    let query = try conv(p + ".query.conv")
    let key = try conv(p + ".key.conv")
    let projection = try conv(p + ".proj.conv")
    let decay = try conv(p + ".query_decay.conv")
    return { x in
      let b = x.dim(0)
      let channels = x.dim(1)
      let t = x.dim(2)
      let heads = 4
      let index = MLXArray(0..<t).asType(x.dtype)
      let delta = index[.newAxis, 0...] - index[0..., .newAxis]
      let queries = query(x).reshaped(b, heads, -1, t)
      let keys = key(x).reshaped(b, heads, -1, t)
      var dots = matmul(keys.transposed(0, 1, 3, 2), queries) / sqrt(Float(channels / heads))
      let decayQ = sigmoid(decay(x).reshaped(b, heads, 4, t)) * 0.5
      let coefficient = sum(decayQ * MLXArray(1...4).asType(x.dtype).reshaped(1, 1, 4, 1), axis: 2)
      dots = dots - abs(delta).reshaped(1, 1, t, t) * coefficient.reshaped(b, heads, 1, t) * 0.5
      dots = which(eye(t, dtype: .bool), MLXArray(-100, dtype: dots.dtype), dots)
      let weights = softmax(dots, axis: 2)
      let result = matmul(content(x).reshaped(b, heads, -1, t), weights).reshaped(b, channels, t)
      return x + projection(result)
    }
  }
  func blstm(_ p: String, layers: Int, maxSteps: Int? = nil, skip: Bool = false) throws -> Op {
    var fw: [Op] = []
    var bw: [Op] = []
    for i in 0..<layers {
      fw.append(try lstm(p + ".forward_lstms.\(i)"))
      bw.append(try lstm(p + ".backward_lstms.\(i)"))
    }
    let projection = try linear(p + ".linear")
    return { input in
      let b = input.dim(0)
      let c = input.dim(1)
      let t = input.dim(2)
      var x = input.transposed(0, 2, 1)
      var frameCount = 0
      let width = maxSteps ?? t
      let stride = width / 2
      if t > width {
        frameCount = max(1, Int(ceil(Double(t - width) / Double(stride))) + 1)
        let flat = padLast(input, 0, (frameCount - 1) * stride + width - t)
        x = asStrided(
          flat, [b, c, frameCount, width], strides: [c * flat.dim(2), flat.dim(2), stride, 1]
        ).transposed(0, 2, 3, 1).reshaped(-1, width, c)
      }
      for i in 0..<layers {
        x = concatenated(
          [fw[i](x), bw[i](x[0..., .stride(by: -1), 0...])[0..., .stride(by: -1), 0...]], axis: -1)
      }
      x = projection(x).transposed(0, 2, 1)
      if frameCount > 0 {
        let frames = x.reshaped(b, frameCount, c, width)
        let limit = stride / 2
        let pieces = (0..<frameCount).map { k in
          frames[
            0..., k, 0..., (k == 0 ? 0 : limit)..<(k == frameCount - 1 ? width : width - limit)]
        }
        x = concatenated(pieces, axis: -1)[.ellipsis, 0..<t]
      }
      return skip ? x + input : x
    }
  }
  private func lstm(_ p: String) throws -> Op {
    let wx = try get(p + ".Wx", rank: 2)
    let wh = try get(p + ".Wh", rank: 2)
    let bias = try get(p + ".bias", rank: 1)
    guard wx.dim(0) == wh.dim(0), wh.dim(0) == 4 * wh.dim(1), bias.size == wh.dim(0) else {
      throw DemucsError.invalidCache("LSTM gate dimensions mismatch: \(p)")
    }
    return { input in
      let projected = addMM(bias, input, wx.T)
      var hidden: MLXArray?
      var cell: MLXArray?
      var outputs: [MLXArray] = []
      for t in 0..<input.dim(1) {
        var gates = projected[0..., t, 0...]
        if let h = hidden { gates = addMM(gates, h, wh.T) }
        let g = split(gates, parts: 4, axis: -1)
        let next = sigmoid(g[0]) * tanh(g[2])
        cell = cell.map { sigmoid(g[1]) * $0 + next } ?? next
        hidden = sigmoid(g[3]) * tanh(cell!)
        outputs.append(hidden!)
      }
      return stacked(outputs, axis: 1)
    }
  }
}
