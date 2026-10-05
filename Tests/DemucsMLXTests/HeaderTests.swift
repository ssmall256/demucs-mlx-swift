import Foundation
import Testing

@testable import DemucsMLX

struct HeaderTests {
  init() {
    _ = MetalTestResources.configure
  }
  @Test(arguments: ["overlap", "malformed-shape", "unsupported-dtype"])
  func rejectsUnsafeTensorHeader(_ problem: String) throws {
    var header: [String: Any] = ["weight": ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]]]
    if problem == "overlap" {
      header["other"] = ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]]
    }
    if problem == "malformed-shape" {
      header["weight"] = ["dtype": "F32", "shape": [1, "invalid"], "data_offsets": [0, 4]]
    }
    if problem == "unsupported-dtype" {
      header["weight"] = ["dtype": "I32", "shape": [1], "data_offsets": [0, 4]]
    }
    let json = try JSONSerialization.data(withJSONObject: header)
    var length = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }
    bytes.append(json)
    bytes.append(Data(repeating: 0, count: 4))
    let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: path) }
    try bytes.write(to: path)
    #expect(throws: DemucsError.self) { try validateSafetensors(path, fileBytes: bytes.count) }
  }
}
