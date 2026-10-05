import Foundation
import MLX

indirect enum JSONValue: Codable, Sendable, Equatable {
  case number(Double)
  case bool(Bool)
  case string(String)
  case array([JSONValue])
  case object([String: JSONValue])
  case null
  init(from decoder: Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() {
      self = .null
    } else if let x = try? c.decode(Bool.self) {
      self = .bool(x)
    } else if let x = try? c.decode(Double.self) {
      self = .number(x)
    } else if let x = try? c.decode(String.self) {
      self = .string(x)
    } else if let x = try? c.decode([JSONValue].self) {
      self = .array(x)
    } else {
      self = .object(try c.decode([String: JSONValue].self))
    }
  }
  var number: Double? {
    if case .number(let x) = self { return x }
    if case .object(let d) = self, d["__type__"]?.string == "fraction",
      let n = d["numerator"]?.number, let v = d["denominator"]?.number, v != 0
    {
      return n / v
    }
    return nil
  }
  var string: String? {
    if case .string(let x) = self { return x }
    return nil
  }
  var array: [JSONValue]? {
    if case .array(let x) = self { return x }
    return nil
  }
  var object: [String: JSONValue]? {
    if case .object(let x) = self { return x }
    return nil
  }
}
struct ModelConfig: Sendable {
  let values: [String: JSONValue]
  func int(_ k: String, _ fallback: Int) -> Int { Int(double(k, Double(fallback))) }
  func double(_ k: String, _ fallback: Double) -> Double { values[k]?.number ?? fallback }
  func bool(_ k: String, _ fallback: Bool) -> Bool {
    if case .bool(let x) = values[k] { return x }
    return fallback
  }
  func string(_ k: String, _ fallback: String) -> String { values[k]?.string ?? fallback }
  func floats(_ k: String) -> [Double] { values[k]?.array?.compactMap(\.number) ?? [] }
  var sources: [String] { values["sources"]?.array?.compactMap(\.string) ?? [] }
}
struct CacheManifest: Sendable {
  let file: URL
  let configurations: [ModelConfig]
  let classes: [String]
  let weights: [[Float]]
  let digest: String
  let tensorBytes: Int
  let isBag: Bool
  let fileIdentity: VerifiedFileIdentity
  static func read(
    model: DemucsModel, directory: URL, validateRegistry: Bool = true,
    validation: CacheValidationPolicy = .alwaysHash
  ) throws -> Self {
    let trace = PerformanceTrace.begin("cache.validation")
    defer { PerformanceTrace.end(trace) }
    let file = directory.appendingPathComponent(model.rawValue + ".safetensors")
    let configFile = directory.appendingPathComponent(model.rawValue + "_config.json")
    guard FileManager.default.fileExists(atPath: file.path),
      FileManager.default.fileExists(atPath: configFile.path)
    else {
      throw DemucsError.missingModel(
        "No \(model.rawValue) model in \(directory.path). Separator.load downloads it automatically; otherwise fetch it with ModelHub.download, from https://huggingface.co/\(ModelHub.repository), or convert it with tools/export_models.py."
      )
    }
    let metadata = try boundedJSON(configFile, maximum: 1_048_576)
    guard metadata["format_version"]?.number == 1, metadata["model_name"]?.string == model.rawValue,
      let digest = metadata["safetensors_sha256"]?.string, digest.count == 64,
      let countNumber = metadata["num_models"]?.number, countNumber >= 1, countNumber <= 16,
      countNumber.rounded() == countNumber,
      metadata["source_artifacts"]?.array?.count == Int(countNumber),
      ["BagOfModelsMLX", "HTDemucsMLX", "HDemucsMLX", "DemucsMLX"].contains(
        metadata["model_class"]?.string ?? "")
    else {
      throw DemucsError.invalidCache(
        "Invalid or unversioned model metadata: \(configFile.path). Regenerate with the restricted Python converter; legacy pickle caches are never loaded."
      )
    }
    let count = Int(countNumber)
    let classes =
      metadata["per_model_classes"]?.array?.compactMap(\.string)
      ?? Array(
        repeating: metadata["sub_model_class"]?.string ?? metadata["model_class"]?.string ?? "",
        count: count)
    if validateRegistry {
      let entry = RegistryEntry.entry(model)
      let artifacts = metadata["source_artifacts"]?.array ?? []
      guard classes == entry.classes, artifacts.count == entry.artifacts.count,
        zip(artifacts, entry.artifacts).allSatisfy({ actual, expected in
          actual.object?["signature"]?.string == expected.0
            && actual.object?["checksum"]?.string == expected.1
        })
      else {
        throw DemucsError.invalidCache(
          "Model artifacts or architectures disagree with the official registry")
      }
    }
    let configs: [[String: JSONValue]]
    if let list = metadata["per_model_kwargs"]?.array {
      configs = list.compactMap(\.object)
    } else {
      configs = Array(repeating: metadata["kwargs"]?.object ?? [:], count: count)
    }
    let arguments =
      metadata["per_model_args"]?.array?.map { $0.array ?? [] }
      ?? Array(repeating: metadata["args"]?.array ?? [], count: count)
    guard classes.count == count, configs.count == count, arguments.count == count,
      arguments.allSatisfy(\.isEmpty),
      classes.allSatisfy({ ["HTDemucsMLX", "HDemucsMLX", "DemucsMLX"].contains($0) })
    else {
      throw DemucsError.invalidCache("Unsupported model constructors or positional arguments")
    }
    let parsed = configs.map { ModelConfig(values: $0) }
    for c in parsed {
      guard c.values.values.allSatisfy(validConstructorValue) else {
        throw DemucsError.invalidCache("Constructor values exceed numeric or structural bounds")
      }
      guard (1...8).contains(c.sources.count), Set(c.sources).count == c.sources.count,
        (1...2).contains(c.int("audio_channels", 2)), (1...10).contains(c.int("depth", 4)),
        (1...2048).contains(c.int("channels", 48)),
        (1...192000).contains(c.int("samplerate", 44100)),
        c.double("segment", 7.8).isFinite, (0.01...600).contains(c.double("segment", 7.8)),
        (2...16).contains(c.int("stride", 4)), (2...64).contains(c.int("kernel_size", 8)),
        c.double("growth", 2).isFinite, (1...4).contains(c.double("growth", 2)),
        (-8...8).contains(c.int("dconv_depth", 2)), (1...128).contains(c.double("dconv_comp", 4)),
        (1...256).contains(c.int("norm_groups", 4)), (0...16).contains(c.int("t_layers", 5)),
        (1...64).contains(c.int("t_heads", 8)), (16...16384).contains(c.int("nfft", 4096)),
        c.int("nfft", 4096).nonzeroBitCount == 1
      else { throw DemucsError.invalidCache("Invalid model dimensions or constructor bounds") }
    }
    guard
      parsed.allSatisfy({
        $0.sources == parsed[0].sources
          && $0.int("audio_channels", 2) == parsed[0].int("audio_channels", 2)
          && $0.int("samplerate", 44100) == parsed[0].int("samplerate", 44100)
      })
    else { throw DemucsError.invalidCache("Ensemble members disagree on audio metadata") }
    let weights =
      metadata["weights"]?.array?.map { $0.array?.compactMap { $0.number.map(Float.init) } ?? [] }
      ?? Array(repeating: Array(repeating: Float(1), count: parsed[0].sources.count), count: count)
    guard weights.count == count,
      weights.allSatisfy({
        $0.count == parsed[0].sources.count && $0.allSatisfy { $0.isFinite && $0 >= 0 }
      }),
      (0..<parsed[0].sources.count).allSatisfy({ i in weights.reduce(Float(0)) { $0 + $1[i] } > 0 })
    else { throw DemucsError.invalidCache("Invalid ensemble weights") }
    let identity = try verifyModelFile(file, digest: digest, policy: validation)
    let bytes = Int(identity.size)
    let names = try validateSafetensors(file, fileBytes: bytes, identity: identity)
    let isBag = metadata["model_class"]?.string == "BagOfModelsMLX"
    if isBag {
      let prefixes = (0..<count).map { "model_\($0)." }
      guard names.allSatisfy({ key in prefixes.contains(where: { key.hasPrefix($0) }) }),
        prefixes.allSatisfy({ prefix in names.contains(where: { $0.hasPrefix(prefix) }) })
      else {
        throw DemucsError.invalidCache("Missing ensemble member or unexpected unassigned tensor")
      }
    } else {
      guard count == 1, !names.contains(where: { $0.hasPrefix("model_") }) else {
        throw DemucsError.invalidCache("Single-model cache has inconsistent tensor prefixes")
      }
    }
    return Self(
      file: file, configurations: parsed, classes: classes, weights: weights, digest: digest,
      tensorBytes: bytes, isBag: isBag, fileIdentity: identity)
  }
}
func boundedJSON(_ url: URL, maximum: Int) throws -> [String: JSONValue] {
  let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? maximum + 1
  guard size <= maximum else { throw DemucsError.invalidCache("Metadata exceeds size limit") }
  return try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
}
@discardableResult
func validateSafetensors(
  _ url: URL, fileBytes: Int, identity: VerifiedFileIdentity? = nil
) throws -> Set<String> {
  let f = try FileHandle(forReadingFrom: url)
  defer { try? f.close() }
  if let identity, try VerifiedFileIdentity.read(f) != identity {
    throw DemucsError.invalidCache("Model file changed before header validation")
  }
  guard let prefix = try f.read(upToCount: 8), prefix.count == 8 else {
    throw DemucsError.invalidCache("Truncated safetensors header")
  }
  let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
  guard length > 0, length <= 16_777_216, length <= fileBytes - 8,
    let data = try f.read(upToCount: Int(length)), data.count == length
  else { throw DemucsError.invalidCache("Invalid safetensors header length") }
  let header = try JSONDecoder().decode([String: JSONValue].self, from: data)
  var regions: [(Int, Int)] = []
  for (key, entry) in header where key != "__metadata__" {
    guard let d = entry.object, let dtype = d["dtype"]?.string,
      let itemSize = ["F32": 4, "F16": 2, "BF16": 2][dtype],
      let shape = d["shape"]?.array?.compactMap(\.number), !shape.isEmpty, shape.count <= 5,
      let offsets = d["data_offsets"]?.array?.compactMap(\.number), offsets.count == 2,
      shape.count == d["shape"]?.array?.count, d["data_offsets"]?.array?.count == 2,
      key.utf8.count <= 512,
      shape.allSatisfy({ $0 >= 1 && $0 <= 16_777_216 && $0.rounded() == $0 }),
      offsets.allSatisfy({ $0 >= 0 && $0 <= Double(fileBytes) && $0.rounded() == $0 })
    else { throw DemucsError.invalidCache("Invalid safetensors entry: \(key)") }
    let elements = shape.reduce(Double(1), *)
    guard elements * Double(itemSize) == offsets[1] - offsets[0],
      offsets[1] <= Double(fileBytes - 8 - Int(length))
    else { throw DemucsError.invalidCache("Invalid tensor byte range: \(key)") }
    regions.append((Int(offsets[0]), Int(offsets[1])))
  }
  let sorted = regions.sorted { $0.0 < $1.0 }
  guard !sorted.isEmpty, sorted[0].0 == 0 else {
    throw DemucsError.invalidCache("Empty or incomplete tensor data")
  }
  for i in 1..<sorted.count where sorted[i].0 != sorted[i - 1].1 {
    throw DemucsError.invalidCache("Overlapping or incomplete tensor data")
  }
  guard sorted.last!.1 == fileBytes - 8 - Int(length) else {
    throw DemucsError.invalidCache("Trailing tensor data")
  }
  if let identity, try VerifiedFileIdentity.read(f) != identity {
    throw DemucsError.invalidCache("Model file changed during header validation")
  }
  return Set(header.keys.filter { $0 != "__metadata__" })
}

private func validConstructorValue(_ value: JSONValue) -> Bool {
  switch value {
  case .number(let n): return n.isFinite && abs(n) <= 10_000_000
  case .object(let object):
    guard object["__type__"]?.string == "fraction", let n = value.number else { return false }
    return n.isFinite && abs(n) <= 10_000_000
  case .array(let values): return values.count <= 128 && values.allSatisfy(validConstructorValue)
  case .string(let s): return s.utf8.count <= 512
  case .bool, .null: return true
  }
}
