#!/usr/bin/env python3
"""Validate every public model against Python MLX using the same exported weights."""

import argparse
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import mlx.core as mx
from demucs_mlx.mlx_convert import load_mlx_model_from_safetensors
from demucs_mlx.mlx_registry import MLX_MODEL_REGISTRY
from demucs_mlx.apply_mlx import apply_model
from demucs_mlx.mlx_transformer import set_attention_dtype

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache", type=Path, default=Path(".build/models"))
p.add_argument(
    "--binary", type=Path, default=Path(".build/out/Products/Release/demucs-mlx-swift")
)
p.add_argument("--output", type=Path, default=Path(".build/registry-parity"))
p.add_argument("--models", nargs="+", default=list(MLX_MODEL_REGISTRY))
p.add_argument(
    "--baseline", type=Path, help="Pinned native baseline; require >100 dB before/after"
)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
audio = mx.load(".build/full-fixtures/audio_1.safetensors")["audio"]
mx.eval(audio)
mx.save_safetensors(str(a.output / "input.safetensors"), {"audio": audio})
records = []
print("## 🧪 Full registry parity", flush=True)
for name in a.models:
    model = load_mlx_model_from_safetensors(name, cache_dir=str(a.cache))
    for sub in getattr(model, "models", [model]):
        set_attention_dtype(sub, mx.float16)
    out = apply_model(
        model, audio[None], shifts=1, seed=481, overlap=0.25, batch_size=2
    )
    mx.eval(out)
    reference = np.array(out[0])
    del model, out
    gc.collect()
    mx.clear_cache()
    target = a.output / f"{name}.safetensors"
    subprocess.run(
        [
            str(a.binary.resolve()),
            "tensor",
            "--cache",
            str(a.cache.resolve()),
            "--model",
            name,
            "--seed",
            "481",
            "--batch-size",
            "2",
            str(a.output / "input.safetensors"),
            str(target),
        ],
        check=True,
    )
    native = np.array(mx.load(str(target))["stems"])
    snrs = []
    for ref, got in zip(reference, native):
        snrs.append(
            float(
                10
                * np.log10(
                    np.sum(ref.astype(np.float64) ** 2)
                    / max(np.sum((ref.astype(np.float64) - got) ** 2), 1e-30)
                )
            )
        )
    agreement = None
    if a.baseline:
        baseline_target = a.output / f"{name}-baseline.safetensors"
        subprocess.run(
            [
                str(a.baseline.resolve()),
                "tensor",
                "--cache",
                str(a.cache.resolve()),
                "--model",
                name,
                "--seed",
                "481",
                "--batch-size",
                "2",
                str(a.output / "input.safetensors"),
                str(baseline_target),
            ],
            check=True,
        )
        baseline = np.array(mx.load(str(baseline_target))["stems"])
        assert baseline.shape == native.shape
        agreement = {
            "bit_identical": bool(np.array_equal(baseline, native)),
            "snr_db": [
                float(
                    10
                    * np.log10(
                        np.sum(x.astype(np.float64) ** 2)
                        / max(
                            np.sum((x.astype(np.float64) - y.astype(np.float64)) ** 2),
                            1e-30,
                        )
                    )
                )
                for x, y in zip(baseline, native)
            ],
        }
        assert min(agreement["snr_db"]) > 100, agreement
        del baseline
    records.append(
        dict(
            model=name,
            shape=list(native.shape),
            native_baseline=agreement,
            snr_db=snrs,
            max_error=float(np.max(np.abs(reference - native))),
        )
    )
    print(
        f"- {'✅' if min(snrs) > 60 else '❌'} `{name}`: **{min(snrs):.2f}–{max(snrs):.2f} dB**",
        flush=True,
    )
    (a.output / "results.json").write_text(json.dumps(records, indent=2))
    if min(snrs) <= 60:
        raise AssertionError(f"{name} parity failed: {snrs}")
    del native, reference
    gc.collect()
    mx.clear_cache()
