import Foundation
import Testing

@testable import DemucsMLX

struct CacheTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test func missingCacheHasActionableError() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    #expect(throws: DemucsError.self) { try Separator(cacheDirectory: directory) }
  }
  @Test func rejectsUnversionedAndWrongRegistryMetadata() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data().write(to: directory.appendingPathComponent("htdemucs.safetensors"))
    try Data("{\"format_version\":0}".utf8).write(
      to: directory.appendingPathComponent("htdemucs_config.json"))
    #expect(throws: DemucsError.self) {
      try CacheManifest.read(model: .htdemucs, directory: directory)
    }
  }
  @Test func boundedMetadata() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    try Data(repeating: 32, count: 2048).write(to: file)
    #expect(throws: DemucsError.self) { try boundedJSON(file, maximum: 1024) }
  }
  @Test func progressIncludesEnsemblesAndShifts() {
    let progress = SeparationProgress(
      completedChunks: 5, totalChunks: 10, modelIndex: 1, modelCount: 4, shiftIndex: 1,
      shiftCount: 2)
    #expect(progress.fractionCompleted == 0.4375)
  }
}
