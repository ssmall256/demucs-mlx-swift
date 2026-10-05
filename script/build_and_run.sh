#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
# Standard developer launcher; no private benchmark tooling is required.
swift build -c release --product demucs-mlx-swift
binary="$(swift build -c release --show-bin-path)/demucs-mlx-swift"
if [[ $# -eq 0 ]]; then
  exec "$binary" --help
fi
exec "$binary" "$@"
