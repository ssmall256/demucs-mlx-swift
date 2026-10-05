import Foundation
import MLX

/// Process-wide runtime preparation, started once by session construction and
/// run off the caller's thread. It builds no graph and evaluates nothing: MLX's
/// allocator and Metal device initialize (a backend with a pipeline archive may
/// begin loading it), and the default memory budget is cached. Later MLX use
/// joins this initialization through MLX's own one-time initializers.
enum RuntimePreparation {
  private static let started: Void = {
    DispatchQueue.global(qos: .userInitiated).async {
      let trace = PerformanceTrace.begin("runtime.prepare")
      _ = Memory.activeMemory
      _ = Separator.defaultMemoryBudget()
      PerformanceTrace.end(trace)
    }
  }()
  static func begin() { started }
}
