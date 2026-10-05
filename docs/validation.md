# Validation and performance

Validation performed October 2, 2026 with Xcode 27, Swift 6.4, MLX Swift 0.32.3 and Python MLX 0.32.3.

## Matched desktop comparison

The desktop is an Apple M4 Max with 128 GiB RAM. Both implementations load byte-identical HTDemucs weights (`339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297`). Each arm uses stereo 44.1 kHz input, 7.8-second model segments, seed 481, one shift, 0.25 overlap, a target batch of eight, FP32 weights/projections, FP16 attention and whole-forward compilation.

Each case has two warmups and three measured iterations per round, with two rounds ordered Python → Swift, then Swift → Python. The reported median includes all six timings; no slow observations are discarded. Timing surrounds the whole awaited separation call and ends after output is materialized. Loading, warmup, output fixture serialization and audio file I/O are excluded from warmed RTFx. The executable is pinned throughout the comparison. RTFx is audio duration divided by wall time; a larger number is faster. The target is Swift/Python RTFx ≥ 0.95.

| Audio | Device | Python RTFx | Swift RTFx | Swift / Python | Per-stem SNR |
|---|---|---:|---:|---:|---:|
| 30 s | GPU | 88.36× | **87.61×** | 0.992 | 84.0–104.8 dB |
| 60 s | GPU | 92.53× | **93.81×** | 1.014 | 84.7–108.4 dB |

Both cases meet the 95% target: Swift is within 1% of Python on 30-second input and 1.4% faster on 60-second input in this matched run. These figures are from the initial validation on October 2, 2026; both implementations have since become faster, and the README carries current throughput.

| Audio | Device | Asset validation | Cold inference | Peak MLX allocation |
|---|---|---:|---:|---:|
| 30 s | GPU | 0.070–0.082 s | 0.559–0.581 s | 5.07 GB |
| 60 s | GPU | 0.073–0.074 s | 0.985–1.001 s | 5.23 GB |

Background host activity and CPU load varied substantially; ratios are comparisons within a matched run, not a hardware ceiling or a claim of universal fastest performance.

## Numeric correctness

All eight public registry models passed using the same exported safetensors in Python and Swift, seed 481, one-second stereo input, default model segments, batch two and FP16 attention. Per-stem agreement must exceed 60 dB. This checks implementation agreement, not source-separation quality against musical ground truth. The six-source model's low-energy stem is the closest to the threshold.

| Model | Minimum–maximum per-stem SNR |
|---|---:|
| `htdemucs` | 65.05–89.33 dB |
| `htdemucs_ft` | 68.95–82.16 dB |
| `htdemucs_6s` | 60.45–89.31 dB |
| `hdemucs_mmi` | 98.73–129.95 dB |
| `mdx` | 117.90–129.96 dB |
| `mdx_extra` | 117.74–133.14 dB |
| `mdx_q` | 117.17–124.72 dB |
| `mdx_extra_q` | 116.73–139.05 dB |

Full 30/60-second benchmark outputs also receive per-stem comparisons against the Python reference.

The Swift Testing suite passes eight CPU test definitions and seven GPU test definitions, with parameterized cases. Small independently exported models exercise transformer attention, legacy hybrid/local attention/BLSTM, phase reconstruction, multiple frequency bands and pure time Demucs against both PyTorch and Python MLX. Direct PyTorch agreement spans 73.9–123.3 dB across these architecture fixtures. Additional checks cover compiled/eager agreement, STFT/iSTFT and overlap edges, native sample-rate conversion and WAV round trips, fine-tuned stem selection, input/memory errors, cancellation and seed behavior. Batch-one and batch-eight reconstruction agree at **100.8 dB** on a 30-second tail case.

The native CLI also passed an end-to-end 48 kHz stereo file test: native decoding and resampling, inference, and four 44.1 kHz stereo PCM16 WAV files of exactly 44,100 samples. Its measured cold process + cache validation + decoding + inference + export took **2.224 seconds** in that run. This is separate from warmed inference throughput.

## Physical iPhone

The harness ran on an **iPhone 15 Pro, iOS 27.0**, in Release mode with batch one and the default 7.8-second segment. The run passed shape/finite checks, **every stem exceeded 60 dB** against the matching Python fixture, and four Float32 WAV stems were written with vocals reloaded successfully.

| Device | Initialization | Cold inference | Warm inference | Warm RTFx | Per-stem SNR | Peak MLX allocation |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 0.120 s | 1.105 s | 0.685 s | 1.46× | 66.07–84.72 dB | 1.63 GB |

Each fixture is one second long, so the model still processes a padded 7.8-second segment. These RTFx figures are not long-track batched desktop throughput. Initialization measures manifest reading/hashing; cold inference includes weight loading and the first materialized prediction. Warm inference measures the whole awaited call. Peak memory is **MLX allocation**, not total process footprint. Only standard HTDemucs was exercised on the physical device; all registry architectures were exercised on macOS. Very long tracks retain their complete input/output, and the memory estimate is conservative rather than an OS jetsam guarantee.

## Reproduce

Build and export assets as described in the root README. Model caches, fixtures and separated audio are untracked. The Python tools need an environment with `demucs-mlx` installed; they run GPU inference, so avoid building or running other GPU work during a comparison.

```sh
./script/test
python tools/check_registry.py
python tools/compare_benchmarks.py --output .build/benchmarks/comparison.json
python tools/check_cli.py
```

`tools/compare_benchmarks.py` supports explicit durations, warmup/iteration/round counts, batch sizes and eager compilation. See [the iOS harness](../Examples/iOS/README.md) for device reproduction.
