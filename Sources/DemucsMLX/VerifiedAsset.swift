import CryptoKit
import Darwin
import Foundation

/// Compatible with demucs-mlx's verified-digest receipt. Metadata changes
/// invalidate the receipt, including writes that preserve size and mtime.
struct VerifiedFileIdentity: Codable, Equatable, Sendable {
  let size: Int64
  let modificationNanoseconds: Int64
  let changeNanoseconds: Int64
  let inode: UInt64
  let device: UInt64
  enum CodingKeys: String, CodingKey {
    case size
    case modificationNanoseconds = "mtime_ns"
    case changeNanoseconds = "ctime_ns"
    case inode = "ino"
    case device = "dev"
  }
  static func read(_ file: FileHandle) throws -> Self {
    var info = stat()
    guard fstat(file.fileDescriptor, &info) == 0, info.st_size >= 0 else {
      throw DemucsError.invalidCache("Could not read model file identity")
    }
    func nanos(_ time: timespec) throws -> Int64 {
      let product = Int64(time.tv_sec).multipliedReportingOverflow(by: 1_000_000_000)
      let sum = product.partialValue.addingReportingOverflow(Int64(time.tv_nsec))
      guard !product.overflow, !sum.overflow else {
        throw DemucsError.invalidCache("Model file timestamp exceeds receipt range")
      }
      return sum.partialValue
    }
    return try Self(
      size: info.st_size, modificationNanoseconds: nanos(info.st_mtimespec),
      changeNanoseconds: nanos(info.st_ctimespec), inode: info.st_ino,
      device: UInt64(UInt32(bitPattern: info.st_dev)))
  }
  static func read(_ url: URL) throws -> Self {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    return try read(file)
  }
}
private struct VerifiedDigest: Codable {
  let identity: VerifiedFileIdentity
  let sha256: String
}

func verifyModelFile(
  _ url: URL, digest: String, policy: CacheValidationPolicy
) throws -> VerifiedFileIdentity {
  let file = try FileHandle(forReadingFrom: url)
  defer { try? file.close() }
  let identity = try VerifiedFileIdentity.read(file)
  let receiptURL = url.appendingPathExtension("verified.json")
  if policy == .verifiedIdentity,
    let receiptFile = try? FileHandle(forReadingFrom: receiptURL)
  {
    let receipt: VerifiedDigest? = {
      defer { try? receiptFile.close() }
      guard let data = try? receiptFile.read(upToCount: 4097), data.count <= 4096 else {
        return nil
      }
      return try? JSONDecoder().decode(VerifiedDigest.self, from: data)
    }()
    if receipt?.sha256 == digest, receipt?.identity == identity {
      guard try VerifiedFileIdentity.read(url) == identity else {
        throw DemucsError.invalidCache("Model file changed during validation")
      }
      return identity
    }
  }
  let trace = PerformanceTrace.begin("cache.sha256")
  defer { PerformanceTrace.end(trace) }
  var hash = SHA256()
  while let data = try file.read(upToCount: 1_048_576), !data.isEmpty {
    hash.update(data: data)
  }
  guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == digest else {
    throw DemucsError.invalidCache("Safetensors SHA-256 mismatch for \(url.lastPathComponent)")
  }
  guard try VerifiedFileIdentity.read(file) == identity,
    try VerifiedFileIdentity.read(url) == identity
  else { throw DemucsError.invalidCache("Model file changed during validation") }
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  if let data = try? encoder.encode(VerifiedDigest(identity: identity, sha256: digest)) {
    // Read-only caches retain full verification on subsequent loads.
    try? data.write(to: receiptURL, options: .atomic)
  }
  return identity
}
