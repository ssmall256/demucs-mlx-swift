#!/bin/bash
# Build the command-line tool and run it with the given arguments.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
./script/build build
binary="$root/.build/xcode/Build/Products/Release/demucs-mlx-swift"
if [[ $# -eq 0 ]]; then
  exec "$binary" --help
fi
exec "$binary" "$@"
