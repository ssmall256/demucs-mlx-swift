import Foundation
import MLX
import MLXNN

struct Attention {
  let q: Op, out: Op
  let qkvWeight: MLXArray, qkvBias: MLXArray, kvWeight: MLXArray, kvBias: MLXArray
  let heads: Int, dtype: DType
  init(_ w: WeightStore, _ p: String, heads: Int, precision: AttentionPrecision) throws {
    self.heads = heads
    dtype = precision == .fp16 ? .float16 : .float32
    q = try w.linear(p + ".query_proj")
    out = try w.linear(p + ".out_proj")
    let qw = try w.get(p + ".query_proj.weight")
    let kw = try w.get(p + ".key_proj.weight")
    let vw = try w.get(p + ".value_proj.weight")
    let qb = try w.get(p + ".query_proj.bias")
    let kb = try w.get(p + ".key_proj.bias")
    let vb = try w.get(p + ".value_proj.bias")
    guard qw.shape == kw.shape, kw.shape == vw.shape, qw.dim(0) == qw.dim(1), qw.dim(0) % heads == 0
    else { throw DemucsError.invalidCache("Attention projection dimensions mismatch") }
    if w.has(p + ".qkv_proj.weight") {
      qkvWeight = try w.get(p + ".qkv_proj.weight", rank: 2)
      qkvBias = try w.get(p + ".qkv_proj.bias", rank: 1)
    } else {
      qkvWeight = concatenated([qw, kw, vw], axis: 0)
      qkvBias = concatenated([qb, kb, vb])
    }
    if w.has(p + ".kv_proj.weight") {
      kvWeight = try w.get(p + ".kv_proj.weight", rank: 2)
      kvBias = try w.get(p + ".kv_proj.bias", rank: 1)
    } else {
      kvWeight = concatenated([kw, vw], axis: 0)
      kvBias = concatenated([kb, vb])
    }
    guard qkvWeight.shape == [3 * qw.dim(0), qw.dim(1)], qkvBias.size == 3 * qw.dim(0),
      kvWeight.shape == [2 * qw.dim(0), qw.dim(1)], kvBias.size == 2 * qw.dim(0)
    else {
      throw DemucsError.invalidCache("Fused attention projection dimensions mismatch")
    }
  }
  func callAsFunction(_ queries: MLXArray, _ keys: MLXArray? = nil) -> MLXArray {
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    if let keys {
      q = self.q(queries)
      let kv = split(addMM(kvBias, keys, kvWeight.T), parts: 2, axis: -1)
      k = kv[0]
      v = kv[1]
    } else {
      let parts = split(addMM(qkvBias, queries, qkvWeight.T), parts: 3, axis: -1)
      q = parts[0]
      k = parts[1]
      v = parts[2]
    }
    func heads(_ x: MLXArray) -> MLXArray {
      x.reshaped(x.dim(0), x.dim(1), self.heads, -1).transposed(0, 2, 1, 3).asType(dtype)
    }
    let value = MLXFast.scaledDotProductAttention(
      queries: heads(q), keys: heads(k), values: heads(v),
      scale: 1 / sqrt(Float(q.dim(-1) / self.heads)), mask: nil)
    return out(value.asType(queries.dtype).transposed(0, 2, 1, 3).reshaped(queries.shape))
  }
}
struct TransformerLayer {
  let attention: Attention
  let arithmetic: CompilationArithmetic
  let n1: Op, n2: Op, n3: Op, nout: Op, g1: Op, g2: Op, l1: Op, l2: Op
  let normFirst: Bool, cross: Bool, geluActivation: Bool
  init(_ w: WeightStore, _ p: String, c: ModelConfig, cross: Bool, precision: AttentionPrecision)
    throws
  {
    arithmetic = w.arithmetic
    self.cross = cross
    normFirst = c.bool("t_norm_first", true)
    geluActivation = c.bool("t_gelu", true)
    attention = try Attention(
      w, p + (cross ? ".cross_attn" : ".attn"), heads: c.int("t_heads", 8), precision: precision)
    let groups = c.int("t_group_norm", 1)
    n1 = try w.norm(p + ".norm1", groups: groups, channelLast: true)
    n2 = try w.norm(p + ".norm2", groups: groups, channelLast: true)
    n3 = cross ? try w.norm(p + ".norm3", groups: groups, channelLast: true) : identity
    nout =
      normFirst && c.bool("t_norm_out", true)
      ? try w.norm(p + ".norm_out", groups: c.int("t_norm_out", 1), channelLast: true) : identity
    g1 = try w.scale(p + ".gamma_1", channelLast: true)
    g2 = try w.scale(p + ".gamma_2", channelLast: true)
    l1 = try w.linear(p + ".linear1")
    l2 = try w.linear(p + ".linear2")
  }
  func feed(_ x: MLXArray) -> MLXArray {
    l2(geluActivation ? arithmetic.activation(l1(x)) : maximum(l1(x), 0))
  }
  func callAsFunction(_ q: MLXArray, _ k: MLXArray? = nil) -> MLXArray {
    if normFirst {
      let a = cross ? attention(n1(q), n2(k!)) : attention(n1(q))
      let x = q + g1(a)
      return nout(x + g2(feed(cross ? n3(x) : n2(x))))
    }
    let x = n1(q + g1(attention(q, k)))
    return n2(x + g2(feed(x)))
  }
}
/// Owned by the separator executor; positional caches are bounded per model.
final class CrossTransformer {
  let c: ModelConfig
  let n: Op, nt: Op
  let layers: [TransformerLayer], timeLayers: [TransformerLayer]
  let scaled: MLXArray?
  var positions: [String: MLXArray] = [:]
  init(_ w: WeightStore, c: ModelConfig, precision: AttentionPrecision) throws {
    self.c = c
    n =
      c.bool("t_norm_in", true)
      ? try w.norm(
        "crosstransformer.norm_in", groups: c.int("t_norm_in_group", 1), channelLast: true)
      : identity
    nt =
      c.bool("t_norm_in", true)
      ? try w.norm(
        "crosstransformer.norm_in_t", groups: c.int("t_norm_in_group", 1), channelLast: true)
      : identity
    var a: [TransformerLayer] = []
    var b: [TransformerLayer] = []
    for i in 0..<c.int("t_layers", 5) {
      let cross = i % 2 != (c.bool("t_cross_first", false) ? 1 : 0)
      a.append(
        try TransformerLayer(
          w, "crosstransformer.layers.\(i)", c: c, cross: cross, precision: precision))
      b.append(
        try TransformerLayer(
          w, "crosstransformer.layers_t.\(i)", c: c, cross: cross, precision: precision))
    }
    layers = a
    timeLayers = b
    scaled =
      c.string("t_emb", "sin") == "scaled"
      ? try w.get("crosstransformer.position_embeddings.embedding.weight") : nil
    guard !c.bool("t_sparse_self_attn", false), !c.bool("t_sparse_cross_attn", false),
      c.int("t_sin_random_shift", 0) == 0
    else {
      throw DemucsError.unsupported(
        "Sparse attention and random positional shifts are not supported by the public registry")
    }
  }
  func embedding(_ channels: Int, _ f: Int, _ t: Int) -> MLXArray {
    let key = "\(channels):\(f):\(t)"
    if let cached = positions[key] { return cached }
    let half = channels / 2
    let div = exp(
      MLXArray(stride(from: Float(0), to: Float(half), by: 2))
        * Float(-log(c.double("t_max_period", 10000)) / Double(half)))
    let pw = MLXArray(0..<t).asType(.float32)[0..., .newAxis] * div
    let ph = MLXArray(0..<f).asType(.float32)[0..., .newAxis] * div
    let sw = broadcast(sin(pw).T.reshaped(-1, 1, t), to: [channels / 4, f, t])
    let cw = broadcast(cos(pw).T.reshaped(-1, 1, t), to: [channels / 4, f, t])
    let sh = broadcast(sin(ph).T.reshaped(-1, f, 1), to: [channels / 4, f, t])
    let ch = broadcast(cos(ph).T.reshaped(-1, f, 1), to: [channels / 4, f, t])
    let pe = concatenated(
      [
        stacked([sw, cw], axis: 1).reshaped(half, f, t),
        stacked([sh, ch], axis: 1).reshaped(half, f, t),
      ], axis: 0)[.newAxis]
    if positions.count >= 8 { positions.removeAll() }
    positions[key] = pe
    return pe
  }
  func callAsFunction(_ input: MLXArray, _ time: MLXArray, preservePositionArithmetic: Bool = false)
    -> (MLXArray, MLXArray)
  {
    let b = input.dim(0)
    let channels = input.dim(1)
    let f = input.dim(2)
    let t = input.dim(3)
    let tt = time.dim(2)
    let weight = Float(c.double("t_weight_pos_embed", 1))
    var x =
      n(input.transposed(0, 3, 2, 1).reshaped(b, t * f, channels)) + weight
      * embedding(channels, f, t).transposed(0, 3, 2, 1).reshaped(1, t * f, channels)
    var pos = MLXArray(0..<tt).asType(.float32).reshaped(1, tt, 1)
    let pe: MLXArray
    if let scaled {
      pe = scaled[0..<tt][.newAxis] * 0.2
    } else {
      if c.string("t_emb", "sin") == "cape" && c.bool("t_cape_mean_normalize", true) {
        pos = pos - mean(pos, axis: 1, keepDims: true)
      }
      let indices = MLXArray(0..<(channels / 2)).asType(.float32).reshaped(1, 1, -1)
      let adim =
        preservePositionArithmetic
        ? positionalIndexDivide(indices, MLXArray(Float(channels / 2 - 1)))
        : indices / Float(channels / 2 - 1)
      let phase = pos / pow(MLXArray(Float(c.double("t_max_period", 10000))), adim)
      pe = concatenated([cos(phase), sin(phase)], axis: -1)
    }
    var xt = sideBranch { nt(time.transposed(0, 2, 1)) + weight * pe }
    for i in layers.indices {
      let oldX = x
      let oldT = xt
      if layers[i].cross {
        x = layers[i](oldX, oldT)
        xt = sideBranch { timeLayers[i](oldT, oldX) }
      } else {
        x = layers[i](oldX)
        xt = sideBranch { timeLayers[i](oldT) }
      }
    }
    return (x.reshaped(b, t, f, channels).transposed(0, 3, 2, 1), xt.transposed(0, 2, 1))
  }
}
