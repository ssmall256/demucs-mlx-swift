import Foundation
import MLX
import MLXNN

struct TimeDemucs {
  let c: ModelConfig
  let encoders: [Op], decoders: [Op], lstm: Op?
  let up: JuliusResampler, down: JuliusResampler
  init(_ w: WeightStore, c: ModelConfig) throws {
    self.c = c
    up = JuliusResampler(old: 1, new: 2)
    down = JuliusResampler(old: 2, new: 1)
    var encoders: [Op] = []
    var decoders: [Op] = []
    let depth = c.int("depth", 6)
    let stride = c.int("stride", 4)
    let act: Op = c.bool("glu_act", true) ? { glu($0) } : { maximum($0, 0) }
    let act2: Op = c.bool("gelu_act", true) ? { gelu($0) } : { maximum($0, 0) }
    for stage in 0..<depth {
      let outputChannels = Int(
        Double(c.int("channels", 64)) * pow(c.double("growth", 2), Double(stage)))
      let inputChannels =
        stage == 0
        ? c.int("audio_channels", 2)
        : Int(Double(c.int("channels", 64)) * pow(c.double("growth", 2), Double(stage - 1)))
      let scale = c.bool("glu_act", true) ? 2 : 1
      let normalized = stage >= c.int("norm_starts", 4)
      let groups = c.int("norm_groups", 4)
      let p = "encoder.\(stage).layers"
      try w.verify(
        p + ".0.conv.weight", shape: [outputChannels, c.int("kernel_size", 8), inputChannels])
      var encode: [Op] = [
        try w.conv(p + ".0.conv", stride: stride),
        normalized ? try w.norm(p + ".1", groups: groups) : identity, act2,
      ]
      var index = 3
      if c.int("dconv_mode", 1) & 1 != 0 {
        encode.append(try w.dconv(p + ".\(index)", config: c, stage: stage, legacy: true))
        index += 1
      }
      if c.bool("rewrite", true) {
        try w.verify(
          p + ".\(index).conv.weight", shape: [scale * outputChannels, 1, outputChannels])
        encode.append(try w.conv(p + ".\(index).conv"))
        encode.append(normalized ? try w.norm(p + ".\(index+1)", groups: groups) : identity)
        encode.append(act)
      }
      let encoded = encode
      encoders.append { x in encoded.reduce(x) { y, f in f(y) } }
      let q = "decoder.\(depth-1-stage).layers"
      var decode: [Op] = []
      index = 0
      if c.bool("rewrite", true) {
        try w.verify(
          q + ".0.conv.weight",
          shape: [scale * outputChannels, 2 * c.int("context", 1) + 1, outputChannels])
        decode.append(try w.conv(q + ".0.conv", padding: c.int("context", 1)))
        decode.append(normalized ? try w.norm(q + ".1", groups: groups) : identity)
        decode.append(act)
        index = 3
      }
      if c.int("dconv_mode", 1) & 2 != 0 {
        decode.append(try w.dconv(q + ".\(index)", config: c, stage: stage, legacy: true))
        index += 1
      }
      try w.verify(
        q + ".\(index).conv.weight",
        shape: [
          stage == 0 ? c.sources.count * c.int("audio_channels", 2) : inputChannels,
          c.int("kernel_size", 8), outputChannels,
        ])
      decode.append(try w.conv(q + ".\(index).conv", stride: stride, transposed: true))
      index += 1
      if stage > 0 {
        decode.append(normalized ? try w.norm(q + ".\(index)", groups: groups) : identity)
        decode.append(act2)
      }
      let decoded = decode
      decoders.insert({ x in decoded.reduce(x) { y, f in f(y) } }, at: 0)
    }
    self.encoders = encoders
    self.decoders = decoders
    lstm = c.int("lstm_layers", 0) > 0 ? try w.blstm("lstm", layers: c.int("lstm_layers", 0)) : nil
  }
  func validLength(_ input: Int) -> Int {
    let resample = c.bool("resample", true)
    let kernel = c.int("kernel_size", 8)
    let stride = c.int("stride", 4)
    let depth = c.int("depth", 6)
    var length = resample ? input * 2 : input
    for _ in 0..<depth { length = max(1, Int(ceil(Double(length - kernel) / Double(stride))) + 1) }
    for _ in 0..<depth { length = (length - 1) * stride + kernel }
    return resample ? (length + 1) / 2 : length
  }
  func callAsFunction(_ input: MLXArray) -> MLXArray {
    var x = input
    let length = x.dim(-1)
    let normalize = c.bool("normalize", true)
    let mono = mean(x, axis: 1, keepDims: true)
    let m = normalize ? mean(mono, axis: -1, keepDims: true) : MLXArray(Float(0))
    let s = normalize ? std(mono, axis: -1, keepDims: true, ddof: 1) : MLXArray(Float(1))
    if normalize { x = (x - m) / (s + 1e-5) }
    let delta = validLength(length) - length
    x = padLast(x, delta / 2, delta - delta / 2)
    if c.bool("resample", true) { x = up(x) }
    var skips: [MLXArray] = []
    for encoder in encoders {
      x = encoder(x)
      skips.append(x)
    }
    if let lstm { x = lstm(x) }
    for decoder in decoders { x = decoder(x + trim(skips.removeLast(), x.dim(-1))) }
    if c.bool("resample", true) { x = down(x) }
    return trim(x * s + m, length).reshaped(
      input.dim(0), c.sources.count, c.int("audio_channels", 2), length)
  }
}
