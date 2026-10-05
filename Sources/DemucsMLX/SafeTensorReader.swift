import Foundation
import MLX

/// Reads only the selected ensemble member. No other member is allocated or
/// evaluated, which keeps four-model bags usable on memory-constrained devices.
func loadModelArrays(_ manifest: CacheManifest, index: Int) throws -> [String: MLXArray] {
  let trace = PerformanceTrace.begin("model.array-loading")
  defer { PerformanceTrace.end(trace) }
  let file = try FileHandle(forReadingFrom: manifest.file)
  defer { try? file.close() }
  guard try VerifiedFileIdentity.read(file) == manifest.fileIdentity else {
    throw DemucsError.invalidCache("Model file changed since session construction")
  }
  guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
    throw DemucsError.invalidCache("Truncated safetensors header")
  }
  let headerSize = prefix.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
  guard headerSize > 0, headerSize <= 16_777_216,
    let data = try file.read(upToCount: headerSize), data.count == headerSize
  else {
    throw DemucsError.invalidCache("Invalid safetensors header")
  }
  let header = try JSONDecoder().decode([String: JSONValue].self, from: data)
  let bag = manifest.isBag
  let member = "model_\(index)."
  var tensors: [(name: String, shape: [Int], dtype: DType, offset: Int, count: Int)] = []
  for (key, entry) in header where key != "__metadata__" && (!bag || key.hasPrefix(member)) {
    guard let object = entry.object,
      let dimensions = object["shape"]?.array?.compactMap(\.number),
      let offsets = object["data_offsets"]?.array?.compactMap(\.number), offsets.count == 2,
      let dtype = object["dtype"]?.string.flatMap({
        ["F32": DType.float32, "F16": .float16, "BF16": .bfloat16][$0]
      })
    else {
      throw DemucsError.invalidCache("Malformed tensor: \(key)")
    }
    tensors.append(
      (
        bag ? String(key.dropFirst(member.count)) : key, dimensions.map(Int.init), dtype,
        8 + headerSize + Int(offsets[0]), Int(offsets[1] - offsets[0])
      ))
  }
  // Positional reads of byte-balanced groups proceed concurrently; each owned
  // leaf array is created by the thread that read it. No graph work occurs.
  let groups = min(4, tensors.count)
  var members = Array(repeating: [Int](), count: groups)
  var loads = Array(repeating: 0, count: groups)
  for i in tensors.indices.sorted(by: { tensors[$0].count > tensors[$1].count }) {
    let g = loads.indices.min(by: { loads[$0] < loads[$1] })!
    members[g].append(i)
    loads[g] += tensors[i].count
  }
  let descriptor = file.fileDescriptor
  let collected = LoadedTensors()
  let plan = members
  let entries = tensors
  DispatchQueue.concurrentPerform(iterations: groups) { g in
    for i in plan[g] {
      let t = entries[i]
      var bytes = Data(count: t.count)
      let complete = bytes.withUnsafeMutableBytes { buffer -> Bool in
        var done = 0
        while done < t.count {
          let n = pread(
            descriptor, buffer.baseAddress! + done, t.count - done, off_t(t.offset + done))
          guard n > 0 else { return false }
          done += n
        }
        return true
      }
      guard complete else {
        collected.fail(DemucsError.invalidCache("Truncated tensor: \(t.name)"))
        return
      }
      collected.add(t.name, MLXArray(bytes, t.shape, dtype: t.dtype))
    }
  }
  let arrays = try collected.result()
  guard try VerifiedFileIdentity.read(file) == manifest.fileIdentity else {
    throw DemucsError.invalidCache("Model file changed while loading weights")
  }
  guard !arrays.isEmpty else {
    throw DemucsError.invalidCache("No tensors for ensemble member \(index)")
  }
  return arrays
}

private final class LoadedTensors: @unchecked Sendable {
  private let lock = NSLock()
  private var arrays: [String: MLXArray] = [:]
  private var failure: Error?
  func add(_ name: String, _ array: MLXArray) { lock.withLock { arrays[name] = array } }
  func fail(_ error: Error) { lock.withLock { failure = failure ?? error } }
  func result() throws -> [String: MLXArray] {
    try lock.withLock {
      if let failure { throw failure }
      return arrays
    }
  }
}

/// Only owned leaf arrays cross threads; graph construction stays on the actor.
final class PendingModelArrays: @unchecked Sendable {
  private let group = DispatchGroup()
  private let lock = NSLock()
  private var outcome: Result<[String: MLXArray], Error>?
  private let manifest: CacheManifest
  private let index: Int
  init(_ manifest: CacheManifest, index: Int) {
    self.manifest = manifest
    self.index = index
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async { [self] in
      let value = Result { try loadModelArrays(self.manifest, index: self.index) }
      lock.withLock { outcome = value }
      group.leave()
    }
  }
  func wait() { group.wait() }
  func take() throws -> [String: MLXArray] {
    group.wait()
    let value = lock.withLock {
      let value = outcome!
      outcome = nil
      return value
    }
    return try value.get()
  }
}
