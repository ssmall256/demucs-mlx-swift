# Changelog

All notable changes to this project are documented in this file. The release
workflow reads the section for the version being released.

## 0.1.0 - 2026-10-05

Initial release.

- `DemucsMLX`: native Demucs music source separation on MLX for all eight
  models (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `hdemucs_mmi`, `mdx`,
  `mdx_q`, `mdx_extra`, `mdx_extra_q`), behind an actor-based async API with
  progress reporting and task cancellation.
- Models are downloaded on first use from
  [ssmall256/demucs-mlx](https://huggingface.co/ssmall256/demucs-mlx) and
  verified against SHA-256 digests built into the package. The cache is shared
  with the [`demucs-mlx`](https://github.com/ssmall256/demucs-mlx) Python
  package.
- `DemucsAudio`: file decoding with sample-rate conversion, stem export as WAV
  (16-bit, 24-bit, float32), FLAC, Apple Lossless or AAC, two-stem output, and
  bounded-memory separation of multiple files.
- `demucs-mlx-swift` command-line tool.
- macOS 14 and iOS 17 deployment targets; an iOS validation harness is in
  `Examples/iOS`.
- About 112x realtime for `htdemucs` on an M4 Max once warm; see the README for
  how that was measured.
