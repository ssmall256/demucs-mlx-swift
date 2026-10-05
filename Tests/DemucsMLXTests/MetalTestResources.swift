import Foundation
import MLX

private final class MetalTestBundleMarker: NSObject {}

/// Swift Testing may load an XCTest image without registering its resources
/// in NSBundle.allBundles. Resolve the image's bundle before MLX starts.
enum MetalTestResources {
  static let configure: Void = {
    if let path = ProcessInfo.processInfo.environment["DEMUCS_METALLIB"] {
      GPU.metallib = URL(fileURLWithPath: path)
      return
    }
    guard GPU.metallib == nil else { return }
    let bundle = Bundle(for: MetalTestBundleMarker.self)
    let bundles = [bundle, Bundle.main] + Bundle.allBundles
    let roots = bundles.flatMap { [$0.bundleURL, $0.resourceURL].compactMap { $0 } }
    for root in roots {
      for suffix in [
        "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
        "mlx-swift_Cmlx.bundle/default.metallib", "default.metallib",
      ] {
        let url = root.appendingPathComponent(suffix)
        if FileManager.default.fileExists(atPath: url.path) {
          GPU.metallib = url
          return
        }
      }
    }
  }()
}
