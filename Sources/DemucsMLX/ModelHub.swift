import CryptoKit
import Foundation

/// When `Separator.load` may fetch a model that is not in the cache directory.
public enum ModelDownloadPolicy: Sendable {
  /// Download the model's two files if either is missing.
  case ifMissing
  /// Never use the network; a missing model throws `DemucsError.missingModel`.
  case never
}

public struct ModelDownloadProgress: Sendable {
  public let model: DemucsModel
  public let bytesReceived: Int64
  public let bytesExpected: Int64
  public var fractionCompleted: Double {
    bytesExpected > 0 ? min(1, Double(bytesReceived) / Double(bytesExpected)) : 0
  }
}

/// Published model files and their verified download.
///
/// Each model is `<model>.safetensors` plus `<model>_config.json`. Their sizes
/// and SHA-256 digests are fixed here, so a download is accepted only if it is
/// byte-identical to the files this release was built against. Nothing is
/// placed in the cache until both files have been verified.
public enum ModelHub {
  public static let repository = "ssmall256/demucs-mlx"

  struct PublishedFile: Sendable {
    let size: Int64
    let sha256: String
  }
  struct PublishedModel: Sendable {
    let weights: PublishedFile
    let config: PublishedFile
  }
  // Sizes and digests of the files at https://huggingface.co/ssmall256/demucs-mlx,
  // identical to those pinned by the demucs-mlx Python package.
  static let published: [DemucsModel: PublishedModel] = [
    .htdemucs: .init(
      weights: .init(
        size: 168_005_865,
        sha256: "339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297"),
      config: .init(
        size: 4215, sha256: "23657b19db14771aecf366ceedbf846c1386836c10043c97f4569acf05526b78")),
    .htdemucsFT: .init(
      weights: .init(
        size: 672_024_519,
        sha256: "53f03b1ad4b4d211025a35da65460ba61a17547adf9c0544cad0ebcc8d7bbabb"),
      config: .init(
        size: 10253, sha256: "6708535a698d4d4dd285a6f0a80d9d3c228fdd6514e53bdf1ddbc2ee9b0fa872")),
    .htdemucs6s: .init(
      weights: .init(
        size: 109_726_583,
        sha256: "d298f7f746bf53c21baad44fb08e88807ef47feb551dd22f1601a546c85b8e02"),
      config: .init(
        size: 4302, sha256: "d5e18c0209be583027d6eb5ec3013a02cd7ef55bc3096e644d8d3577d92e9bd0")),
    .hdemucsMMI: .init(
      weights: .init(
        size: 334_522_864,
        sha256: "39f359110433930c2a589131f84c03c26bbd209e89e10e6abbcb6062c131debc"),
      config: .init(
        size: 2400, sha256: "0a3a645fc281824d8b38077bf081afdae625eab7728b68f66e334e1751bfb938")),
    .mdx: .init(
      weights: .init(
        size: 1_381_657_640,
        sha256: "c95dab261c766fc50caadcd047aa2c125759b9b48efdac3f2aaab0fa35d8c41f"),
      config: .init(
        size: 4834, sha256: "ea7f81dce21e668e91c1ec36f8e11fa138d4828b6b622e0b4e9001afba7357a7")),
    .mdxQ: .init(
      weights: .init(
        size: 1_381_657_640,
        sha256: "d7f31edb6b37b5ee391d104e1f88cb70c56ca82e3f6f9e8f4f3cd2df6e9bddfc"),
      config: .init(
        size: 4836, sha256: "6f976a1c96128daa7bf4652399e49cfffd7bb68ba156965533f4d880699fa7a5")),
    .mdxExtra: .init(
      weights: .init(
        size: 1_338_062_104,
        sha256: "d1c969aa0a69417e767b23f97d10df944a4c9883febe853ab6d91ad0cb2276fe"),
      config: .init(
        size: 5602, sha256: "d925ba3dbad48ccc99d48e151e5d96a0fcc71ef57bd3862f4cb46d5a71c0db11")),
    .mdxExtraQ: .init(
      weights: .init(
        size: 1_338_062_104,
        sha256: "82310bf4d1f32b8044cba6c192af77ab8d12a0acdedd7bf841caa78a61bd5839"),
      config: .init(
        size: 5599, sha256: "a45609a3cefd6fd69fb5058046a4ed33716f11ec313237146372fbe39a583530")),
  ]

  /// Size in bytes of a model's weights, for showing before a download.
  public static func downloadSize(of model: DemucsModel) -> Int64 {
    guard let files = published[model] else { return 0 }
    return files.weights.size + files.config.size
  }

  /// Whether both of a model's files are present in `directory`. Presence only;
  /// `Separator` verifies their contents when it loads them.
  public static func isCached(_ model: DemucsModel, in directory: URL) -> Bool {
    let manager = FileManager.default
    return manager.fileExists(atPath: weightsURL(model, directory).path)
      && manager.fileExists(atPath: configURL(model, directory).path)
  }

  /// Where files are fetched from: `<base>/<filename>`. `DEMUCS_MLX_HUB_URL`
  /// replaces the whole base; `HF_ENDPOINT` selects a Hugging Face mirror.
  public static var baseURL: URL {
    let environment = ProcessInfo.processInfo.environment
    if let override = environment["DEMUCS_MLX_HUB_URL"], let url = URL(string: override),
      !override.isEmpty
    {
      return url
    }
    let endpoint = environment["HF_ENDPOINT"].flatMap { $0.isEmpty ? nil : $0 }
    let host = endpoint ?? "https://huggingface.co"
    return URL(string: host)!.appendingPathComponent(repository)
      .appendingPathComponent("resolve/main")
  }

  /// False when `DEMUCS_MLX_NO_DOWNLOAD` or `HF_HUB_OFFLINE` asks for offline use.
  public static var downloadsEnabled: Bool {
    let environment = ProcessInfo.processInfo.environment
    return !["DEMUCS_MLX_NO_DOWNLOAD", "HF_HUB_OFFLINE"].contains { name in
      ["1", "true", "yes", "on"].contains(environment[name]?.lowercased() ?? "")
    }
  }

  /// Fetch and verify one model into `directory`, replacing any existing copy.
  ///
  /// Throws `DemucsError.downloadFailed` if the network fails or either file
  /// does not match its published size and SHA-256. On failure the directory is
  /// left as it was.
  public static func download(
    _ model: DemucsModel, to directory: URL,
    progress: (@Sendable (ModelDownloadProgress) -> Void)? = nil
  ) async throws {
    guard let files = published[model] else {
      throw DemucsError.downloadFailed("No published weights for \(model.rawValue)")
    }
    try await download(model, to: directory, from: baseURL, expecting: files, progress: progress)
  }

  static func download(
    _ model: DemucsModel, to directory: URL, from base: URL, expecting files: PublishedModel,
    progress: (@Sendable (ModelDownloadProgress) -> Void)?
  ) async throws {
    let manager = FileManager.default
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    var staged: [URL] = []
    defer { for file in staged { try? manager.removeItem(at: file) } }
    let config = try await fetch(
      base.appendingPathComponent(configURL(model, directory).lastPathComponent),
      expecting: files.config, into: directory, label: "\(model.rawValue) config", report: nil)
    staged.append(config)
    let total = files.weights.size
    var report: (@Sendable (Int64) -> Void)?
    if let progress {
      report = { received in
        progress(ModelDownloadProgress(model: model, bytesReceived: received, bytesExpected: total))
      }
    }
    let weights = try await fetch(
      base.appendingPathComponent(weightsURL(model, directory).lastPathComponent),
      expecting: files.weights, into: directory, label: "\(model.rawValue) weights",
      report: report)
    staged.append(weights)
    // Weights first: an interruption between the two leaves an incomplete pair,
    // which the loader reports as missing, never a config without weights.
    try replace(weightsURL(model, directory), with: weights)
    try replace(configURL(model, directory), with: config)
    staged.removeAll()
  }

  static func weightsURL(_ model: DemucsModel, _ directory: URL) -> URL {
    directory.appendingPathComponent(model.rawValue + ".safetensors")
  }
  static func configURL(_ model: DemucsModel, _ directory: URL) -> URL {
    directory.appendingPathComponent(model.rawValue + "_config.json")
  }

  private static func replace(_ destination: URL, with source: URL) throws {
    guard rename(source.path, destination.path) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
  }

  private static func fetch(
    _ url: URL, expecting expected: PublishedFile, into directory: URL, label: String,
    report: (@Sendable (Int64) -> Void)?
  ) async throws -> URL {
    let destination = directory.appendingPathComponent(".download-\(UUID().uuidString).part")
    let transfer = Transfer(destination: destination, limit: expected.size, report: report)
    do {
      try await transfer.run(url)
      try Task.checkCancellation()
      let size =
        (try FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? NSNumber)?
        .int64Value
      guard size == expected.size else {
        throw DemucsError.downloadFailed(
          "\(label) is \(size ?? -1) bytes; the published file is \(expected.size)")
      }
      guard try sha256(of: destination) == expected.sha256 else {
        throw DemucsError.downloadFailed("\(label) does not match its published SHA-256")
      }
      return destination
    } catch {
      try? FileManager.default.removeItem(at: destination)
      if error is CancellationError || error is DemucsError { throw error }
      if (error as? URLError)?.code == .cancelled { throw CancellationError() }
      throw DemucsError.downloadFailed("Could not download \(label): \(error.localizedDescription)")
    }
  }

  private static func sha256(of file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hash = SHA256()
    while true {
      try Task.checkCancellation()
      guard let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty else { break }
      hash.update(data: chunk)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

/// One download task bridged to async/await, with progress and cancellation.
private final class Transfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
  private let destination: URL
  private let limit: Int64
  private let report: (@Sendable (Int64) -> Void)?
  private let lock = NSLock()
  private var continuation: CheckedContinuation<Void, Error>?
  private var failure: Error?
  private var task: URLSessionDownloadTask?

  init(destination: URL, limit: Int64, report: (@Sendable (Int64) -> Void)?) {
    self.destination = destination
    self.limit = limit
    self.report = report
  }

  func run(_ url: URL) async throws {
    let session = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: nil)
    defer { session.finishTasksAndInvalidate() }
    var request = URLRequest(url: url)
    request.setValue("demucs-mlx-swift", forHTTPHeaderField: "User-Agent")
    request.timeoutInterval = 30
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        let task = session.downloadTask(with: request)
        lock.withLock {
          self.continuation = continuation
          self.task = task
        }
        task.resume()
      }
    } onCancel: {
      lock.withLock { task }?.cancel()
    }
  }

  private func finish(_ error: Error?) {
    let pending = lock.withLock { () -> CheckedContinuation<Void, Error>? in
      defer { continuation = nil }
      return continuation
    }
    if let error { pending?.resume(throwing: error) } else { pending?.resume() }
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
  ) {
    if totalBytesWritten > limit {
      lock.withLock {
        failure = DemucsError.downloadFailed("Download is larger than the published file")
      }
      downloadTask.cancel()
      return
    }
    report?(totalBytesWritten)
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    // The temporary file is removed when this method returns, so move it now.
    if let response = downloadTask.response as? HTTPURLResponse,
      !(200..<300).contains(response.statusCode)
    {
      lock.withLock {
        failure = DemucsError.downloadFailed("Server returned HTTP \(response.statusCode)")
      }
      return
    }
    do {
      try FileManager.default.moveItem(at: location, to: destination)
    } catch {
      lock.withLock { failure = error }
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    finish(lock.withLock { failure } ?? error)
  }
}
