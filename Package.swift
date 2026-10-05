// swift-tools-version: 6.3
import PackageDescription

let package = Package(
  name: "demucs-mlx-swift",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [
    .library(name: "DemucsMLX", targets: ["DemucsMLX"]),
    .library(name: "DemucsAudio", targets: ["DemucsAudio"]),
    .executable(name: "demucs-mlx-swift", targets: ["DemucsCLI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
  ],
  targets: [
    .target(
      name: "DemucsMLX",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
      ]),
    .target(name: "CDemucsAudioIO", publicHeadersPath: "include"),
    .target(name: "DemucsAudio", dependencies: ["DemucsMLX", "CDemucsAudioIO"]),
    .executableTarget(
      name: "DemucsCLI",
      dependencies: [
        "DemucsMLX", "DemucsAudio",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]),
    .testTarget(
      name: "DemucsMLXTests", dependencies: ["DemucsMLX", "DemucsAudio", "CDemucsAudioIO"]),
  ],
  swiftLanguageModes: [.v6]
)
