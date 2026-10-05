import Foundation

#if os(macOS)
  import IOKit
#endif

struct HardwarePolicy {
  static let gpuCores: Int = {
    #if os(macOS)
      let service = IOServiceGetMatchingService(
        kIOMainPortDefault, IOServiceMatching("IOAccelerator"))
      guard service != 0 else { return 8 }
      defer { IOObjectRelease(service) }
      return
        (IORegistryEntryCreateCFProperty(
          service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue()
        as? NSNumber)?.intValue ?? 8
    #else
      return 6
    #endif
  }()
  static var dualStream: Bool {
    #if os(iOS)
      return false
    #else
      return gpuCores >= 16
    #endif
  }
  static var batchSize: Int {
    #if os(iOS)
      return 1
    #else
      let ram = ProcessInfo.processInfo.physicalMemory
      if gpuCores >= 38 && ram >= 64 * 1_073_741_824 { return 8 }
      return gpuCores >= 18 && ram >= 32 * 1_073_741_824 ? 3 : 2
    #endif
  }
}
