# ``DemucsMLX``

Separate music into instrument stems using native MLX inference on Apple silicon.

## Overview

Create a reusable ``Separator`` with ``Separator/load(model:cacheDirectory:options:download:downloadProgress:)``, which downloads and verifies the model on first use (see ``ModelHub``), then pass floating-point audio at its sample rate. A session owns its weights and compiled graphs, serializes inference, and returns evaluated ``SeparationResult`` arrays.

```swift
var options = SeparationOptions()
options.seed = 481
let session = try await Separator.load(options: options)
let result = try await session.separate(AudioTensor(audio))
let vocals = try result.stem("vocals")
```

Inputs have shape `[channels, samples]`; outputs have shape `[sources, channels, samples]`. ``AudioTensor`` borrows immutable storage for reuse across requests. Passing a raw MLX array instead transfers ownership. Keep aliases immutable while a request runs.

For channels-first `[Float]` PCM at `session.sampleRate`, use `try await session.separate(samples: pcm, channels: 2)` and `result.samples()` for `[source, channel, sample]` CPU output. Input ownership, finite-value validation, and output materialization remain explicit work inside those calls. Tensor and CPU PCM callers use the same validation and inference paths.

Use the `DemucsAudio` library for AVFoundation files, sample-rate conversion, PCM buffers, and WAV export. Progress callbacks execute on the inference executor. Cancellation is observed between batches. iOS retains the model's default segment length, uses batch one, and loads ensemble members sequentially.

For file batches, `DemucsAudio.separateAndExport(_:using:to:options:progress:)` returns ordered file reports with bounded decode/export overlap. WAV writers borrow evaluated immutable samples and publish each file by rename. Caller-owned PCM conversion remains a snapshot.

Call `try await session.tuneSpectralKernels()` explicitly to benchmark and persist device-specific spectral choices. It invalidates the session's compiled forwards. Ordinary separation never runs tuning trials and uses safe defaults when the cache is unavailable.

Constructing a ``Separator`` validates assets, then starts MLX/Metal initialization and reads the first model's weights on background threads; no graph is built or evaluated until a separation call, which joins that work. `Separator.verifyAssets(model:cacheDirectory:)` validates assets without starting either. `separate(samples:channels:)` also prepares the CPU destination for `result.samples()` while inference runs.

Set `options.compilation = .firstCompiled` to compile eligible released HTDemucs first forwards while preserving their eager arithmetic. Other configurations retain automatic execution, and subsequent cached graphs keep their existing arithmetic. Precompiled Metal libraries additionally avoid runtime shader compilation; the ordinary dependency supports this policy with normal compilation.

Full weight hashing is the default on every session construction. For a controlled local asset cache, opt into `options.cacheValidation = .verifiedIdentity` (CLI: `--cache-validation verified-identity`). A successful full verification writes a Python-compatible `.verified.json` receipt beside the weights. Subsequent sessions reuse it only when digest, size, nanosecond modification/change times, inode and device match. Missing, malformed or changed receipts trigger full hashing; read-only caches remain usable. Registry and tensor-header validation always run. Metadata receipts do not detect corruption that leaves all metadata unchanged; use the default `.alwaysHash` when that guarantee is needed. Use `Separator.verifyAssets(model:cacheDirectory:)` or the CPU-only CLI `verify-assets --model htdemucs --cache assets` to prepare a receipt explicitly; both hash and validate without separation or GPU initialization. Prepare or verify deployment assets before measuring receipt-hit startup, and identify that requirement explicitly.

## Topics

### Inference

- ``Separator``
- ``AudioTensor``
- ``SeparationResult``
- ``ModelHub``
- ``ModelDownloadPolicy``
- ``ModelDownloadProgress``
- ``SeparationProgress``
- ``InferenceStatistics``
- ``SpectralTuningReport``
- ``SpectralKernelMeasurement``

### Configuration

- ``DemucsModel``
- ``SeparationOptions``
- ``AttentionPrecision``
- ``CompilationPolicy``
- ``CacheValidationPolicy``
- ``DemucsError``
