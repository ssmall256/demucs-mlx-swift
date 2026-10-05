import Foundation
import MLX
import MLXNN

/// A model is exclusively used by its separator. Compilation captures immutable
/// weights; mutable caches are never accessed concurrently.
final class NativeModel: @unchecked Sendable {
  let config: ModelConfig
  let architecture: String
  let sources: [String], channels: Int, sampleRate: Int, segment: Double
  let spectral: SpectralTransform
  let encoder: [HybridStage], decoder: [HybridStage], timeEncoder: [HybridLayer],
    timeDecoder: [HybridLayer]
  let embedding: MLXArray?
  let transformer: CrossTransformer?
  let up: Op, down: Op, upTime: Op, downTime: Op
  let timeNetwork: TimeDemucs?
  var compiled: [String: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
  var seen = Set<String>()
  let parameterBytes: Int
  let arithmetic: CompilationArithmetic
  init(
    config c: ModelConfig, architecture: String, arrays: [String: MLXArray],
    attention: AttentionPrecision
  ) throws {
    config = c
    self.architecture = architecture
    sources = c.sources
    channels = c.int("audio_channels", 2)
    sampleRate = c.int("samplerate", 44100)
    segment = c.double("segment", 7.8)
    spectral = SpectralTransform(nfft: c.int("nfft", 4096))
    parameterBytes = arrays.values.reduce(0) { $0 + $1.nbytes }
    let w = WeightStore(arrays)
    arithmetic = w.arithmetic
    if architecture == "DemucsMLX" {
      timeNetwork = try TimeDemucs(w, c: c)
      encoder = []
      decoder = []
      timeEncoder = []
      timeDecoder = []
      embedding = nil
      transformer = nil
      up = identity
      down = identity
      upTime = identity
      downTime = identity
    } else {
      timeNetwork = nil
      let legacy = architecture == "HDemucsMLX"
      let depth = c.int("depth", legacy ? 6 : 4)
      let hybrid = !legacy || c.bool("hybrid", true)
      var e: [HybridStage] = []
      var d: [HybridStage] = []
      var te: [HybridLayer] = []
      var td: [HybridLayer] = []
      var freqs = c.int("nfft", 4096) / 2
      for i in 0..<depth {
        let frequency = freqs > 1
        let lastFreq = frequency && freqs <= c.int("kernel_size", 8)
        let kernel =
          frequency ? (lastFreq ? freqs : c.int("kernel_size", 8)) : c.int("time_stride", 2) * 2
        let stride = frequency ? c.int("stride", 4) : c.int("time_stride", 2)
        let padding = lastFreq ? 0 : kernel / 4
        e.append(
          try HybridStage(
            store: w, prefix: "encoder.\(i)", config: c, stage: i, frequency: frequency,
            empty: false, last: false, kernel: kernel, stride: stride, padding: padding,
            decode: false, legacy: legacy))
        d.insert(
          try HybridStage(
            store: w, prefix: "decoder.\(depth-1-i)", config: c, stage: i, frequency: frequency,
            empty: false, last: i == 0, kernel: kernel, stride: stride, padding: padding,
            decode: true, legacy: legacy), at: 0)
        if hybrid && frequency {
          te.append(
            try HybridLayer(
              store: w, prefix: "tencoder.\(i)", config: c, stage: i, frequency: false,
              empty: lastFreq, last: false, kernel: c.int("kernel_size", 8),
              stride: c.int("stride", 4), padding: c.int("kernel_size", 8) / 4, decode: false,
              legacy: legacy))
          // Frequency encoder stages are known before constructing time decoder names.
        }
        if frequency { freqs = lastFreq ? 1 : freqs / stride }
      }
      // Waveform decoders are stored in reverse encoder order.
      for j in te.indices {
        let stage = te.count - 1 - j
        let enc = te[stage]
        td.append(
          try HybridLayer(
            store: w, prefix: "tdecoder.\(j)", config: c, stage: stage, frequency: false,
            empty: enc.empty, last: stage == 0, kernel: c.int("kernel_size", 8),
            stride: c.int("stride", 4), padding: c.int("kernel_size", 8) / 4, decode: true,
            legacy: legacy))
      }
      encoder = e
      decoder = d
      timeEncoder = te
      timeDecoder = td
      embedding = w.has("freq_emb.embedding.weight") ? try w.get("freq_emb.embedding.weight") : nil
      transformer =
        !legacy && c.int("t_layers", 5) > 0
        ? try CrossTransformer(w, c: c, precision: attention) : nil
      if !legacy && c.int("bottom_channels", 0) > 0 {
        up = try w.conv("channel_upsampler.conv")
        down = try w.conv("channel_downsampler.conv")
        upTime = try w.conv("channel_upsampler_t.conv")
        downTime = try w.conv("channel_downsampler_t.conv")
      } else {
        up = identity
        down = identity
        upTime = identity
        downTime = identity
      }
      guard c.int("wiener_iters", 0) == 0 else {
        throw DemucsError.unsupported("Public registry models use zero Wiener EM iterations")
      }
    }
    let unused = Set(arrays.keys).subtracting(w.used)
    guard unused.isEmpty else {
      throw DemucsError.invalidCache(
        "Unexpected tensors: \(unused.sorted().prefix(5).joined(separator:", "))")
    }
  }
  func validLength(_ length: Int) -> Int {
    if architecture == "HTDemucsMLX" { return Int(segment * Double(sampleRate)) }
    return timeNetwork?.validLength(length) ?? length
  }
  func forward(_ x: MLXArray, policy: CompilationPolicy) -> MLXArray {
    guard policy != .eager else {
      let trace = PerformanceTrace.begin("model.eager.enqueue")
      defer { PerformanceTrace.end(trace) }
      return call(x)
    }
    let key = "\(x.shape):\(x.dtype)"
    if let function = compiled[key] {
      let trace = PerformanceTrace.begin("model.cached.enqueue")
      defer { PerformanceTrace.end(trace) }
      return function([x])[0]
    }
    if seen.insert(key).inserted {
      let trace = PerformanceTrace.begin(
        policy == .firstCompiled ? "model.first-compiled.enqueue" : "model.first-eager.enqueue")
      defer { PerformanceTrace.end(trace) }
      if architecture == "HTDemucsMLX", policy == .firstCompiled {
        // This path is verified for the released transformer geometry. Smaller
        // custom transformers can fuse differently (the fixture without layer
        // scaling reaches only 90 dB); they retain automatic execution.
        guard config.int("channels", 48) == 48, config.double("growth", 2) == 2,
          encoder.count == 4, config.int("bottom_channels", 0) == 512,
          config.int("t_layers", 5) == 5, config.bool("t_layer_scale", true),
          config.int("norm_starts", 4) >= encoder.count, spectral.nfft == 4096,
          config.bool("cac", true), config.int("context_enc", 0) == 0,
          encoder.allSatisfy({ $0.ratios.isEmpty && $0.layers[0].frequency })
        else { return call(x) }
        let length = max(x.dim(-1), validLength(x.dim(-1)))
        let frames = (length + spectral.hop - 1) / spectral.hop
        var frequency = spectral.nfft / 2
        for stage in encoder {
          let layer = stage.layers[0]
          frequency = (frequency + 2 * layer.padding - stage.kernel) / layer.stride + 1
        }
        let channels =
          config.int("bottom_channels", 0) > 0
          ? config.int("bottom_channels", 0)
          : Int(
            Double(config.int("channels", 48))
              * pow(config.double("growth", 2), Double(encoder.count - 1)))
        let constantsTrace = PerformanceTrace.begin("model.constants.prepare")
        let positions = transformer.map { $0.embedding(channels, frequency, frames) }
        eval([spectral.window, arithmetic.rootTwo] + (positions.map { [$0] } ?? []))
        PerformanceTrace.end(constantsTrace, gpuCompleted: true)
        let frontendTrace = PerformanceTrace.begin("model.frontend.graph")
        let frontend = firstForwardFrontend(x)
        PerformanceTrace.end(frontendTrace)
        // This function is deliberately used once. Later forwards keep their
        // accepted compiled arithmetic and cache keys; none capture the temporary
        // divisor or frontend. The tracing context belongs to this model only.
        let first = compile { [weak self] arrays in
          guard let self else { preconditionFailure("Compiled model used after its session ended") }
          let graphTrace = PerformanceTrace.begin("model.first.swift-graph")
          defer { PerformanceTrace.end(graphTrace) }
          self.arithmetic.divisor = arrays.last!
          defer { self.arithmetic.divisor = nil }
          return [
            self.call(
              arrays[0], preservePositionArithmetic: true, frontend: Array(arrays[1..<7]))
          ]
        }
        return first([x] + frontend + [arithmetic.rootTwo])[0]
      }
      return call(x)
    }
    let trace = PerformanceTrace.begin("model.compile.enqueue")
    defer { PerformanceTrace.end(trace) }
    let function = compile { [weak self] arrays in
      guard let self else { preconditionFailure("Compiled model used after its session ended") }
      let graphTrace = PerformanceTrace.begin("model.cached.swift-graph")
      defer { PerformanceTrace.end(graphTrace) }
      return [self.call(arrays[0])]
    }
    if compiled.count >= 8 {
      compiled.removeAll()
      seen.removeAll()
    }
    compiled[key] = function
    return function([x])[0]
  }
  /// Build the original eager normalization graph outside first-call tracing.
  /// This keeps MLX's rounded reciprocal literals away from FP16 attention.
  private func firstForwardFrontend(_ input: MLXArray) -> [MLXArray] {
    let length = max(input.dim(-1), validLength(input.dim(-1)))
    let mix = input.dim(-1) < length ? padLast(input, 0, length - input.dim(-1)) : input
    let wave = sideBranch { () -> [MLXArray] in
      let m = mean(mix, axes: [1, 2], keepDims: true)
      let s = std(mix, axes: [1, 2], keepDims: true, ddof: 1)
      return [m, s, (mix - m) / (s + 1e-5)]
    }
    let frames = (length + spectral.hop - 1) / spectral.hop
    let pad = spectral.hop / 2 * 3
    let signal = padLast(
      mix, pad, pad + frames * spectral.hop - length,
      reflect: !config.bool("hybrid_old", false))
    let z = spectral.stft(signal)[.ellipsis, 0..<(spectral.nfft / 2), 2..<(2 + frames)]
    let x = stacked([z.realPart(), z.imaginaryPart()], axis: 2)
      .reshaped(z.dim(0), channels * 2, z.dim(2), z.dim(3))
    let m = mean(x, axes: [1, 2, 3], keepDims: true)
    let s = std(x, axes: [1, 2, 3], keepDims: true, ddof: 1)
    return wave + [m, s, (x - m) / (s + 1e-5)]
  }
  func call(
    _ input: MLXArray, preservePositionArithmetic: Bool = false, frontend: [MLXArray]? = nil
  ) -> MLXArray {
    if let timeNetwork { return timeNetwork(input) }
    let transformerModel = architecture == "HTDemucsMLX"
    let hybrid = transformerModel || config.bool("hybrid", true)
    let old = config.bool("hybrid_old", false)
    let originalLength = input.dim(-1)
    let length = transformerModel ? Int(segment * Double(sampleRate)) : originalLength
    let mix = originalLength < length ? padLast(input, 0, length - originalLength) : input
    var saved: [MLXArray] = []
    var savedTime: [MLXArray] = []
    var lengths: [Int] = []
    var timeLengths: [Int] = []
    let independent = transformerModel && !timeEncoder.contains(where: { $0.empty })
    // Keep waveform reductions on its own stream. Putting these on the spectral
    // stream makes the first waveform convolution wait for the STFT and norms.
    func normalizeTime() -> (MLXArray, MLXArray, MLXArray) {
      if let frontend { return (frontend[0], frontend[1], frontend[2]) }
      let mt = mean(mix, axes: [1, 2], keepDims: true)
      let st = std(mix, axes: [1, 2], keepDims: true, ddof: 1)
      return (mt, st, (mix - mt) / (st + 1e-5))
    }
    let (mt, st, normalizedTime) = independent ? sideBranch(normalizeTime) : normalizeTime()
    var xt = normalizedTime
    if independent {
      sideBranch {
        for layer in timeEncoder {
          timeLengths.append(xt.dim(-1))
          xt = layer.encode(xt)
          savedTime.append(xt)
        }
      }
    }
    let hop = spectral.hop
    let frames = (length + hop - 1) / hop
    let pad = hop / 2 * 3
    let spectralInput = hybrid ? padLast(mix, pad, pad + frames * hop - length, reflect: !old) : mix
    var z = spectral.stft(spectralInput)[.ellipsis, 0..<(spectral.nfft / 2), 0...]
    if hybrid { z = z[.ellipsis, 2..<(2 + frames)] }
    let b = z.dim(0)
    let f = z.dim(2)
    let t = z.dim(3)
    let cac = config.bool("cac", true)
    var x =
      cac
      ? stacked([z.realPart(), z.imaginaryPart()], axis: 2).reshaped(b, channels * 2, f, t) : abs(z)
    let m = frontend?[3] ?? mean(x, axes: [1, 2, 3], keepDims: true)
    let s = frontend?[4] ?? std(x, axes: [1, 2, 3], keepDims: true, ddof: 1)
    x = frontend?[5] ?? (x - m) / (s + 1e-5)
    for i in encoder.indices {
      lengths.append(x.dim(-1))
      var inject: MLXArray?
      if hybrid && !independent && i < timeEncoder.count {
        timeLengths.append(xt.dim(-1))
        xt = timeEncoder[i].encode(xt)
        if timeEncoder[i].empty { inject = xt } else { savedTime.append(xt) }
      }
      x = encoder[i].encode(x, inject: inject)
      if i == 0, let embedding {
        x =
          x + Float(config.double("freq_emb", 0.2) * config.double("emb_scale", 10))
          * embedding.T.reshaped(1, embedding.dim(1), embedding.dim(0), 1)
      }
      saved.append(x)
    }
    if let transformer {
      let before = x.shape
      if config.int("bottom_channels", 0) > 0 {
        x = up(x.reshaped(b, before[1], -1)).reshaped(
          b, config.int("bottom_channels", 0), before[2], before[3])
        xt = sideBranch { upTime(xt) }
      }
      (x, xt) = transformer(x, xt, preservePositionArithmetic: preservePositionArithmetic)
      if config.int("bottom_channels", 0) > 0 {
        x = down(x.reshaped(b, x.dim(1), -1)).reshaped(before)
        xt = sideBranch { downTime(xt) }
      }
    } else if !transformerModel {
      x = zeros(like: x)
      xt = zeros(like: x)
    }
    let offset = encoder.count - timeDecoder.count
    if independent {
      sideBranch {
        for layer in timeDecoder {
          xt = layer.decode(xt, skip: savedTime.removeLast(), length: timeLengths.removeLast()).0
        }
      }
    }
    for i in decoder.indices {
      let pair = decoder[i].decode(x, skip: saved.removeLast(), length: lengths.removeLast())
      x = pair.0
      if hybrid && !independent && i >= offset {
        let layer = timeDecoder[i - offset]
        let le = timeLengths.removeLast()
        if layer.empty {
          xt = layer.decode(pair.1[0..., 0..., 0, 0...], skip: nil, length: le).0
        } else {
          xt = layer.decode(xt, skip: savedTime.removeLast(), length: le).0
        }
      }
    }
    x =
      x.reshaped(b, sources.count, -1, f, t) * s[0..., .newAxis, 0..., 0..., 0...]
      + m[0..., .newAxis, 0..., 0..., 0...]
    let separated: MLXArray
    if cac {
      let packed = x.reshaped(b, sources.count, channels, 2, f, t)
      separated =
        packed[0..., 0..., 0..., 0, 0..., 0...].asType(.complex64) + packed[
          0..., 0..., 0..., 1, 0..., 0...] * MLXArray(real: 0, imaginary: 1)
    } else {
      let phase = atan2(z.imaginaryPart(), z.realPart())[0..., .newAxis, 0..., 0..., 0...]
      separated = x.asType(.complex64) * exp(phase * MLXArray(real: 0, imaginary: 1))
    }
    var spec = separated
    spec = padded(spec, widths: [0, 0, 0, [0, 1], IntOrPair(hybrid ? (2, 2) : (0, 0))])
    let outputLength = hybrid ? frames * hop + (old ? 0 : 2 * pad) : length
    var output = spectral.istft(spec, length: outputLength)
    output = output[.ellipsis, (hybrid && !old ? pad : 0)..<((hybrid && !old ? pad : 0) + length)]
    if hybrid {
      let wave = sideBranch {
        xt.reshaped(b, sources.count, channels, length) * st[0..., .newAxis, 0..., 0...]
          + mt[0..., .newAxis, 0..., 0...]
      }
      output = output + wave
    }
    return output[.ellipsis, 0..<originalLength]
  }
}

/// Independent branches use a scoped MLX stream on larger desktop GPUs.
/// Cross-branch array dependencies remain explicit and are synchronized by MLX.
func sideBranch<R>(_ body: () -> R) -> R {
  if HardwarePolicy.dualStream { return Stream.withNewDefaultStream(body) }
  return body()
}
