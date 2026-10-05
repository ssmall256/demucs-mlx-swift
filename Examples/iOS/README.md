# Physical-device validation harness

This UIKit developer harness consumes the root Swift package from an external Xcode project. It runs standard HTDemucs with default 7.8-second segments and batch one, compares evaluated stems against Python, writes Float32 WAV stems into Documents, and reloads vocals to check audio I/O. It writes `Documents/validation.json`, including cold/warm timing, shape, aggregate and per-stem SNR, and peak MLX allocation.

1. Export HTDemucs and generate full fixtures as described in the root README.
2. Run `./script/device-assets` from the repository root (optional first argument: cache directory).
3. Open `DemucsDeviceHarness.xcodeproj`; select your signing team and physical device. `project.yml` is the reproducible XcodeGen source.
4. Build in Release, then launch the device harness with Xcode's tools:

```sh
xcrun devicectl device process launch --device YOUR_UDID \
  --terminate-existing --console com.example.demucsmlx.deviceharness \
  --validate-and-exit
```

Every stem must exceed 60 dB against the Python fixture. Without `--validate-and-exit`, the status screen stays open. Keep the device foreground and unlocked during inference; the harness disables the idle timer while open.

Retrieve a report with:

```sh
xcrun devicectl device copy from --device YOUR_UDID \
  --domain-type appDataContainer \
  --domain-identifier com.example.demucsmlx.deviceharness \
  --source Documents/validation.json --destination validation.json
```

Assets are staged into `.build/iOSAssets` and are not committed. Set your own development team and bundle identifier. The generated project requires Xcode 27 and iOS 27.

Validation passed on an **iPhone 15 Pro, iOS 27.0**: four `[2,44100]` stems, all stems above 60 dB, successful WAV round trip, and approximately **1.6 GB** peak MLX allocation. A one-second fixture still executes a 7.8-second model segment. See [the validation report](../../docs/validation.md) for exact device measurements and desktop comparisons.
