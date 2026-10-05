# Demucs MLX for Swift

Native music source separation for Apple silicon, with a Swift 6 actor API, Apple audio I/O, and a macOS command-line tool. The inference path uses MLX Swift directly. It is the Swift counterpart of the [`demucs-mlx`](https://github.com/ssmall256/demucs-mlx) Python package and shares its models; Python is needed only to convert weights yourself or generate reference fixtures.

Requires Swift 6.3 or later and Apple silicon. The deployment target is macOS 14 / iOS 17; development and testing are on macOS 27 and iOS 27 with Xcode 27. The package resolves **[MLX Swift 0.32.3](https://github.com/ml-explore/mlx-swift/releases/tag/0.32.3)** and **[ArgumentParser 1.8.2](https://github.com/apple/swift-argument-parser/releases/tag/1.8.2)**. Build inference executables in release mode.

## Use the SDK

Add this repository's URL to your package dependencies, or add a checkout as a local package in Xcode. Link `DemucsMLX` for tensors; add `DemucsAudio` for files and `AVAudioPCMBuffer` integration. The iOS example consumes these products as an external Xcode project.

```swift
import DemucsMLX
import DemucsAudio

let separator = try await Separator.load()   // htdemucs; downloaded and verified on first use
let result = try await DemucsAudio.separate(trackURL, using: separator)
try DemucsAudio.export(result, to: outputDirectory)   // drums.wav, bass.wav, other.wav, vocals.wav
```

Choose a model, options and an output format, or keep just one stem and the rest of the mix:

```swift
var options = SeparationOptions()
options.seed = 481
let separator = try await Separator.load(model: .htdemucsFT, options: options) { progress in
    print("Downloading: \(Int(progress.fractionCompleted * 100))%")
}
let result = try await DemucsAudio.separate(trackURL, using: separator)
let vocals = result.stems["vocals"]                    // [channels, samples] MLXArray
try DemucsAudio.export(try result.twoStems("vocals"), to: outputDirectory, format: .flac())
print(result.statistics.rtfx)
```

Stems can be written as WAV (16-bit, 24-bit or float32), FLAC, Apple Lossless or AAC. `Separator.load` is async, honours task cancellation, and reports download progress through the trailing closure.

For multiple files, the SDK keeps one separator, bounds decode/export overlap, and returns ordered file reports without retaining stem tensors:

```swift
var batch = AudioBatchOptions() // macOS: one prefetched track, two writers
batch.format = .flac()          // or .wav(.float32), .alac(), .aac(), ...
batch.twoStems = "vocals"       // optional: vocals + no_vocals only
let files = try await DemucsAudio.separateAndExport(
    [firstTrack, secondTrack], using: separator, to: outputRoot, options: batch
)
print(files[0].decodeSeconds, files[0].statistics.rtfx, files[0].exportSeconds)
```

`outputRoot/<track>/<stem>.<extension>` is the SDK convention. Duplicate track names are rejected before decoding. Completed files survive cancellation or a later failure; incomplete sibling temporary files are removed. I/O overlap is capped at 512 MiB on macOS or 64 MiB on iOS, and one-eighth of RAM. iOS defaults to zero prefetch and one writer. Tracks beyond the overlap allowance run serially. Callbacks can arrive from worker executors; dispatch UI updates to the main actor.

File decode uses internally owned, page-aligned planar Float32 storage imported into MLX without a copy, with best-quality resampling and bounded scratch. `tensor(from:)` still snapshots caller-owned mutable PCM. WAV export reads evaluated signed strides directly with bounded conversion/encoding, including sliced, reversed, broadcast and Float16 arrays.

For MLX callers, inputs are floating-point `[channels, samples]` arrays at `separator.sampleRate`. Results are evaluated `[sources, channels, samples]` arrays; `result.stem("vocals")` selects a stem. A raw array passed to `separate` transfers ownership under Swift's `sending` rules. To reuse an input, wrap it in `AudioTensor` and keep it and all aliases immutable until the request finishes.

```swift
let result = try await separator.separate(AudioTensor(audio)) { progress in
    print(progress.fractionCompleted)
}
let vocals = try result.stem("vocals")
```

Reuse a separator across tracks to retain weights and compiled graphs. Each separator serializes requests, checks cancellation between batches, and returns evaluated output. Progress callbacks execute on the inference executor; dispatch UI updates to the main actor. `Separator.load` hashes and validates assets away from the caller's executor.

## Models

`Separator.load` looks for the model in `Separator.defaultCacheDirectory` (`~/.cache/demucs-mlx` on macOS, the app's Caches directory on iOS). If it is not there, the two files for that model are downloaded from [ssmall256/demucs-mlx](https://huggingface.co/ssmall256/demucs-mlx) on Hugging Face, 160 MB for the default `htdemucs`. A download is accepted only if its size and SHA-256 match the values built into this package, and nothing is written to the cache until both files have passed.

- Pass `download: .never`, or set `DEMUCS_MLX_NO_DOWNLOAD=1` or `HF_HUB_OFFLINE=1`, to stay offline. `HF_ENDPOINT` selects a mirror.
- Pass `cacheDirectory:` to keep models elsewhere, for example inside an app bundle.
- `ModelHub.download(_:to:)` fetches a model ahead of time, and `ModelHub.isCached(_:in:)` checks for one.
- The cache is the same one the [`demucs-mlx`](https://github.com/ssmall256/demucs-mlx) Python package uses, so a model downloaded by either is shared.

Before building a model the loader checks registry identities, constructor bounds, SHA-256, tensor byte ranges, expected parameter names and layer shapes. It never reads Python pickle files.

To convert from Meta's original checkpoints yourself, run the exporter with a Python environment that has [`demucs-mlx[convert]`](https://github.com/ssmall256/demucs-mlx) installed; the result is byte-identical to the published files:

```sh
python tools/export_models.py --output "$HOME/.cache/demucs-mlx"
```

Original `mdx_q` / `mdx_extra_q` conversion also requires `diffq`; see [conversion notes](docs/conversion.md). Those names identify the original DiffQ checkpoints; the converted weights are float32.

## Command-line use

```sh
swift build -c release
./script/build_and_run.sh separate song.wav
./script/build_and_run.sh separate song.mp3 --format flac --two-stems vocals
./script/build_and_run.sh separate first.m4a second.wav --model htdemucs_ft --json stages.json
./script/build_and_run.sh separate --list-models
./script/build_and_run.sh benchmark --seconds 30 --seed 481
```

The build script builds and runs the release executable; with no arguments it prints help. `separate` downloads the model on first use (`--no-download` prevents that, `--cache` chooses the directory) and writes stems under `separated/<model>/<track>/`. `--format` selects `wav`, `wav24`, `wav-float32`, `flac`, `flac24`, `alac`, `alac24` or `aac`, and `--two-stems vocals` writes `vocals` and `no_vocals` only. `--prefetch 0 --write-workers 1 --io-memory-budget 0` selects serial I/O. Reports distinguish decode, inference and export, and the CLI reports total file-to-stems time. `tensor` reads an `audio` safetensors array and writes a `stems` array. `--help` describes every option.

| Model | Architecture | Sources | Ensemble |
|---|---|---:|---:|
| `htdemucs` | Hybrid transformer | 4 | 1 |
| `htdemucs_ft` | Hybrid transformer | 4 | 4 |
| `htdemucs_6s` | Hybrid transformer | 6 | 1 |
| `hdemucs_mmi` | Legacy hybrid | 4 | 1 |
| `mdx`, `mdx_q` | Time + hybrid | 4 | 4 |
| `mdx_extra`, `mdx_extra_q` | Legacy hybrid | 4 | 4 |

`--model htdemucs_ft --stem vocals` loads and runs only the member contributing to vocals. Options preserve Python's default shift count, overlap, segment lengths, and deterministic integer seed behavior. Mono inputs are broadcast to stereo; inputs with extra channels use the leading model channels. File I/O performs native sample-rate conversion.

## Performance and memory

The hot path uses FP32 weights/projections, FP16 attention, fused attention and normalization, channels-last residual chains, direct FP32 matrix products for kernel-one projections, phased decoder convolutions, cached positional embeddings, shape-bounded whole-forward compilation, and deterministic Metal overlap-add. Larger desktop GPUs use independent waveform/spectral streams. Batch selection considers GPU topology and RAM, then reduces batching to fit the configured memory estimate.

`--attention fp32` and `--compilation eager` are explicit diagnostic alternatives. The eager path preserves Python MLX arithmetic; compilation can change floating-point rounding. Numeric parity is tested per stem, not asserted to be bit-identical across hardware.

On iOS, the default batch is one and ensemble members load sequentially. Default model segments are retained. A memory estimate that cannot fit throws an actionable error instead of silently changing segments. `--segment` / `segmentSeconds` is explicit: shorter segments reduce legacy-model memory, while HTDemucs still pads to its training length. File and tensor APIs retain the full input/output track; very long tracks need an appropriate memory budget.

### Measured throughput

Apple M4 Max (128 GiB), macOS 27, MLX Swift 0.32.3, standard `htdemucs`, default options, October 5, 2026. RTFx is audio duration divided by wall time.

| Scenario | Result |
|---|---:|
| Warmed call, 120 s of audio, CPU samples in → CPU stems out (median of 6) | **1.07 s · 112× RTFx** |
| First call in a new process, same input | 1.30–1.61 s |
| Complete CLI run on a 181 s ALAC song: launch, load, decode, separate, write four WAVs (median of 5) | **2.89 s · 63× RTFx** |
| 60 back-to-back 120 s calls with no pause (two hours of audio) | 103 s · 70× aggregate |

The first ~20 back-to-back calls hold about 116×; after that the machine's power management reduces throughput, so sustained batch rates are lower than warmed single-call rates. Peak MLX allocation for these runs is about 4.7 GB with automatic batching. These are measurements on one machine, not guarantees; see [validation](docs/validation.md) for numeric parity against the Python reference, the matched Python comparison and iPhone results.

### Optional paths

- `try await separator.tuneSpectralKernels()` (CLI `tune`) benchmarks spectral extraction/overlap-add kernels for the device and persists the choices. Ordinary separation never runs trials.
- `--compilation first-compiled` compiles the first forward of eligible released HTDemucs models while preserving its eager arithmetic.
- `--cache-validation verified-identity` reuses a verified-asset receipt instead of rehashing weights on every launch. Full hashing is the default; `verify-assets` prepares a receipt without touching the GPU.
- `--audio-decoder native-pcm` selects the built-in native-rate PCM WAV reader. Other formats and resampling use Apple's decoders.
- `separator.separate(samples:channels:)` and `result.samples()` accept and return plain `[Float]` PCM.
- `DEMUCS_PHASE_TRACE=/absolute/path.json` records host phase timings.

See also [architecture](docs/architecture.md), [asset conversion](docs/conversion.md) and the [iPhone harness](Examples/iOS/README.md).

## Development

```sh
# CPU-only API/cache checks, then GPU checks if fixtures exist.
./script/test

# Produce reference fixtures (Python environment with demucs-mlx installed).
python tools/make_fixtures.py --output .build/fixtures
python tools/make_fixtures.py --full --output .build/full-fixtures
```

[tools/compare_benchmarks.py](tools/compare_benchmarks.py) compares matched, warmed Python and Swift inference. [tools/check_registry.py](tools/check_registry.py) checks every exported public model. [tools/check_cli.py](tools/check_cli.py) exercises decoding, resampling, inference, and WAV export end to end. Generated models and fixtures live outside version control.

MIT licensed. Adapted from [`demucs-mlx`](https://github.com/ssmall256/demucs-mlx) and Meta's [Demucs](https://github.com/adefossez/demucs); see [LICENSE](LICENSE).
