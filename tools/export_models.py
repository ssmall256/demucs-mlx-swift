#!/usr/bin/env python3
"""One-time safe asset export. Run with demucs-mlx's Python environment.

Existing versioned caches are copied byte-for-byte. Invalid/legacy caches are
regenerated into the destination through the restricted upstream converter.
No Python runtime is needed by the resulting Swift SDK.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path.home() / ".cache/demucs-mlx"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "htdemucs",
            "htdemucs_ft",
            "htdemucs_6s",
            "hdemucs_mmi",
            "mdx",
            "mdx_extra",
            "mdx_q",
            "mdx_extra_q",
        ],
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow fetching missing official Torch checkpoints during conversion",
    )
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Export destination must differ from the existing cache")
    from demucs_mlx.mlx_convert import (
        _load_safe_cache_config,
        _sha256_file,
        convert_htdemucs_weights,
    )
    from demucs_mlx.mlx_registry import MLX_MODEL_REGISTRY

    args.output.mkdir(parents=True, exist_ok=True)
    print("## 📦 Swift model export", flush=True)
    for name in args.models:
        if name not in MLX_MODEL_REGISTRY:
            parser.error(f"Unknown registry model: {name}")
        existing = args.output / f"{name}_config.json"
        existing_weights = args.output / f"{name}.safetensors"
        try:
            metadata = _load_safe_cache_config(existing, name)
            valid = _sha256_file(existing_weights) == metadata["safetensors_sha256"]
        except (OSError, ValueError):
            valid = False
        if valid:
            print(f"- ✅ `{name}` already exported with verified SHA-256", flush=True)
            continue
        config = args.source / f"{name}_config.json"
        weights = args.source / f"{name}.safetensors"
        try:
            data = _load_safe_cache_config(config, name)
            if _sha256_file(weights) != data["safetensors_sha256"]:
                raise ValueError("Digest mismatch")
        except (OSError, ValueError):
            print(f"- Regenerating `{name}` through restricted conversion", flush=True)
            if name in ("mdx_q", "mdx_extra_q"):
                from importlib.util import find_spec

                if find_spec("diffq") is None:
                    raise RuntimeError(
                        "Original quantized checkpoints require optional diffq. Install tools/export-requirements.txt in a separate conversion environment."
                    )
            if not args.allow_download:
                import torch

                def no_download(*_args, **_kwargs):
                    raise RuntimeError(
                        "Missing local Torch checkpoint. Populate the reference checkpoint cache, or explicitly pass --allow-download."
                    )

                torch.hub.download_url_to_file = no_download
            convert_htdemucs_weights(
                name, output_dir=str(args.output), verify=True, verbose=False
            )
        else:
            shutil.copy2(weights, args.output / weights.name)
            shutil.copy2(config, args.output / config.name)
            print(f"- ✅ `{name}` copied with verified SHA-256", flush=True)
    print(f"\n> ✅ Assets exported to `{args.output}`", flush=True)


if __name__ == "__main__":
    main()
