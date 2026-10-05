import CryptoKit
import Darwin
import Foundation
import Testing

@testable import DemucsMLX

struct VerifiedAssetTests {
  init() {
    _ = MetalTestResources.configure
  }
  private func fixture(_ body: (URL, String) throws -> Void) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let file = directory.appendingPathComponent("model.safetensors")
    let bytes = Data("model-weights".utf8)
    try bytes.write(to: file)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    try body(file, digest)
  }
  @Test func malformedReceiptFallsBackToHash() throws {
    try fixture { file, digest in
      let initial = try verifyModelFile(file, digest: digest, policy: .verifiedIdentity)
      let receipt = file.appendingPathExtension("verified.json")
      #expect(FileManager.default.fileExists(atPath: receipt.path))
      #expect(try verifyModelFile(file, digest: digest, policy: .verifiedIdentity) == initial)
      try Data(repeating: 32, count: 4097).write(to: receipt)
      #expect(try verifyModelFile(file, digest: digest, policy: .verifiedIdentity) == initial)
      try Data("different-hash".utf8).write(to: receipt)
      #expect(throws: DemucsError.self) {
        try verifyModelFile(
          file, digest: String(repeating: "0", count: 64), policy: .verifiedIdentity)
      }
    }
  }
  @Test func detectsWriteWithPreservedSizeAndModificationTime() throws {
    try fixture { file, digest in
      let original = try verifyModelFile(file, digest: digest, policy: .alwaysHash)
      var before = stat()
      let statResult = file.withUnsafeFileSystemRepresentation { lstat($0!, &before) }
      #expect(statResult == 0)
      try Data("MODEL-WEIGHTS".utf8).write(to: file)
      let times = [before.st_atimespec, before.st_mtimespec]
      let result = file.withUnsafeFileSystemRepresentation { path in
        times.withUnsafeBufferPointer { utimensat(AT_FDCWD, path!, $0.baseAddress!, 0) }
      }
      #expect(result == 0)
      let changed = try VerifiedFileIdentity.read(file)
      #expect(changed.size == original.size)
      #expect(changed.modificationNanoseconds == original.modificationNanoseconds)
      #expect(changed.changeNanoseconds != original.changeNanoseconds)
      #expect(throws: DemucsError.self) {
        try verifyModelFile(file, digest: digest, policy: .verifiedIdentity)
      }
      #expect(throws: DemucsError.self) {
        try verifyModelFile(file, digest: digest, policy: .alwaysHash)
      }
    }
  }
  @Test func readOnlyDirectoryKeepsVerificationUsable() throws {
    try fixture { file, digest in
      let directory = file.deletingLastPathComponent()
      try FileManager.default.setAttributes(
        [.posixPermissions: 0o555], ofItemAtPath: directory.path)
      defer {
        try? FileManager.default.setAttributes(
          [.posixPermissions: 0o700], ofItemAtPath: directory.path)
      }
      #expect(try verifyModelFile(file, digest: digest, policy: .verifiedIdentity).size == 13)
      #expect(
        !FileManager.default.fileExists(atPath: file.appendingPathExtension("verified.json").path))
    }
  }
}
