import Foundation

struct RegistryEntry {
  let artifacts: [(String, String)]
  let classes: [String]
  static func entry(_ model: DemucsModel) -> RegistryEntry {
    switch model {
    case .htdemucs: return .init(artifacts: [("955717e8", "8726e21a")], classes: ["HTDemucsMLX"])
    case .htdemucsFT:
      return .init(
        artifacts: [
          ("f7e0c4bc", "ba3fe64a"), ("d12395a8", "e57c48e6"), ("92cfc3b6", "ef3bcb9c"),
          ("04573f0d", "f3cf25b2"),
        ], classes: Array(repeating: "HTDemucsMLX", count: 4))
    case .htdemucs6s: return .init(artifacts: [("5c90dfd2", "34c22ccb")], classes: ["HTDemucsMLX"])
    case .hdemucsMMI: return .init(artifacts: [("75fc33f5", "1941ce65")], classes: ["HDemucsMLX"])
    case .mdx:
      return .init(
        artifacts: [
          ("0d19c1c6", "0f06f20e"), ("7ecf8ec1", "70f50cc9"), ("c511e2ab", "fe698775"),
          ("7d865c68", "3d5dd56b"),
        ], classes: ["DemucsMLX", "DemucsMLX", "HDemucsMLX", "HDemucsMLX"])
    case .mdxExtra:
      return .init(
        artifacts: [
          ("e51eebcc", "c1b80bdd"), ("a1d90b5c", "ae9d2452"), ("5d2d6c55", "db83574e"),
          ("cfa93e08", "61801ae1"),
        ], classes: Array(repeating: "HDemucsMLX", count: 4))
    case .mdxQ:
      return .init(
        artifacts: [
          ("6b9c2ca1", "3fd82607"), ("b72baf4e", "8778635e"), ("42e558d4", "196e0e1b"),
          ("305bc58f", "18378783"),
        ], classes: ["DemucsMLX", "DemucsMLX", "HDemucsMLX", "HDemucsMLX"])
    case .mdxExtraQ:
      return .init(
        artifacts: [
          ("83fc094f", "4a16d450"), ("464b36d7", "e5a9386e"), ("14fc6a69", "a89dd0ee"),
          ("7fd6ef75", "a905dd85"),
        ], classes: Array(repeating: "HDemucsMLX", count: 4))
    }
  }
}
