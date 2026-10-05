import Foundation
import os

/// Opt-in host spans. No evaluation or synchronization is added by tracing.
/// The CLI flushes once after work; normal inference records nothing.
@_spi(Profiling) public enum PerformanceTrace {
  public struct Token: @unchecked Sendable {
    fileprivate let name: String
    fileprivate let start: Double
    fileprivate let id: OSSignpostID
  }
  private final class State: @unchecked Sendable {
    let lock = NSLock()
    let log = OSLog(subsystem: "demucs-mlx-swift", category: "Performance")
    let path = ProcessInfo.processInfo.environment["DEMUCS_PHASE_TRACE"]
    var records: [[String: Any]] = []
  }
  private static let state = State()
  public static func begin(_ name: String) -> Token? {
    guard state.path != nil else { return nil }
    let token = Token(
      name: name, start: ProcessInfo.processInfo.systemUptime,
      id: OSSignpostID(log: state.log))
    os_signpost(
      .begin, log: state.log, name: "DemucsPhase", signpostID: token.id,
      "%{public}@", name as NSString)
    return token
  }
  public static func end(_ token: Token?, gpuCompleted: Bool = false) {
    guard let token else { return }
    let end = ProcessInfo.processInfo.systemUptime
    os_signpost(
      .end, log: state.log, name: "DemucsPhase", signpostID: token.id,
      "%{public}@", token.name as NSString)
    state.lock.withLock {
      state.records.append([
        "phase": token.name, "startUptime": token.start,
        "seconds": end - token.start, "gpuCompletedAtEnd": gpuCompleted,
      ])
    }
  }
  public static func flush() {
    guard let path = state.path else { return }
    let records = state.lock.withLock { state.records }
    do {
      let data = try JSONSerialization.data(
        withJSONObject: [
          "schemaVersion": 1,
          "pid": ProcessInfo.processInfo.processIdentifier,
          "boundary": "host spans; only marked existing joins imply GPU completion",
          "records": records,
        ], options: [.prettyPrinted, .sortedKeys])
      try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    } catch {
      // Diagnostics must never fail separation.
      FileHandle.standardError.write(Data("Phase trace could not be saved: \(error)\n".utf8))
    }
  }
}
