import DemucsAudio
import DemucsMLX
import Foundation
import MLX
import UIKit

@main final class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions options: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    UIApplication.shared.isIdleTimerDisabled = true
    return true
  }
  func application(
    _ application: UIApplication, configurationForConnecting session: UISceneSession,
    options: UIScene.ConnectionOptions
  ) -> UISceneConfiguration {
    let config = UISceneConfiguration(name: nil, sessionRole: session.role)
    config.delegateClass = HarnessScene.self
    return config
  }
}
final class HarnessScene: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?
  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession, options: UIScene.ConnectionOptions
  ) {
    guard let scene = scene as? UIWindowScene else { return }
    let window = UIWindow(windowScene: scene)
    window.rootViewController = HarnessController()
    window.makeKeyAndVisible()
    self.window = window
  }
}
final class HarnessController: UIViewController {
  let label = UILabel()
  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .systemBackground
    label.numberOfLines = 0
    label.text = "Running native Demucs device validation…"
    label.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(label)
    NSLayoutConstraint.activate([
      label.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
      label.trailingAnchor.constraint(
        equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
      label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
    ])
    Task {
      do {
        let message = try await DeviceValidation.run()
        label.text = message
        print(message)
        if ProcessInfo.processInfo.arguments.contains("--validate-and-exit") { exit(0) }
      } catch {
        let message = "FAILED: \(error)"
        label.text = message
        print(message)
        if ProcessInfo.processInfo.arguments.contains("--validate-and-exit") { exit(1) }
      }
    }
  }
}

actor DeviceValidation {
  private static func seconds(since start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now)
    return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
  }
  static func run() async throws -> String {
    guard let root = Bundle.main.url(forResource: "iOSAssets", withExtension: nil) else {
      throw DemucsError.missingModel("Stage device fixtures with script/device-assets")
    }
    var options = SeparationOptions()
    options.seed = 481
    options.batchSize = 1
    let loadStarted = ContinuousClock.now
    let separator = try await Separator.load(cacheDirectory: root, options: options)
    let loadSeconds = seconds(since: loadStarted)
    let fixture = try loadArrays(url: root.appendingPathComponent("audio_1.safetensors"))
    guard let audio = fixture["audio"] else { throw DemucsError.invalidInput("Missing fixture") }
    print("Device validation: cold separation")
    let coldStarted = ContinuousClock.now
    _ = try await separator.separate(AudioTensor(audio))
    let coldSeconds = seconds(since: coldStarted)
    print("Device validation: warm separation")
    let warmStarted = ContinuousClock.now
    let result = try await separator.separate(AudioTensor(audio))
    let warmSeconds = seconds(since: warmStarted)
    let reference = try loadArrays(
      url: root.appendingPathComponent("reference_1_gpu.safetensors"))[
        "stems"]!
    let error = result.audio - reference
    let snr = (10 * log10(sum(reference * reference) / maximum(sum(error * error), 1e-30))).item(
      Float.self)
    let perStem = (0..<4).map { source in
      (10
        * log10(
          sum(reference[source] * reference[source])
            / maximum(sum(error[source] * error[source]), 1e-30))).item(Float.self)
    }
    guard result.audio.shape == [4, 2, 44100], all(isFinite(result.audio)).item(Bool.self),
      snr > 60, perStem.allSatisfy({ $0 > 60 })
    else {
      throw DemucsError.invalidInput("Device parity failed: \(snr) dB")
    }
    let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    try DemucsAudio.export(result, to: directory.appendingPathComponent("stems"), float32: true)
    let decoded = try DemucsAudio.load(
      directory.appendingPathComponent("stems/vocals.wav"), sampleRate: 44100)
    guard decoded.shape == [2, 44100] else {
      throw DemucsError.invalidInput("Audio round-trip failed")
    }
    let report: [String: Any] = [
      "status": "passed", "snr_db": snr,
      "initialization_seconds": loadSeconds, "cold_seconds": coldSeconds,
      "warm_seconds": warmSeconds, "rtfx": result.statistics.audioSeconds / warmSeconds,
      "per_stem_snr_db": perStem,
      "shape": result.audio.shape, "peak_memory_bytes": Memory.peakMemory,
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: directory.appendingPathComponent("validation.json"))
    return
      "PASS: Native Demucs\n\(snr) dB parity\n\(result.statistics.audioSeconds / warmSeconds)× RTFx\n4 WAV stems written"
  }
}
