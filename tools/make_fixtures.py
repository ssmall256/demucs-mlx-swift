#!/usr/bin/env python3
"""Produce native Swift fixtures from Python MLX and upstream Demucs.
Run with the demucs-mlx conversion environment.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx


def encode(value):
    if isinstance(value, Fraction):
        return {
            "__type__": "fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    raise TypeError(type(value))


def signal(seconds):
    rng = np.random.default_rng(481 + int(seconds))
    n = int(44100 * seconds)
    t = np.arange(n, dtype=np.float32) / 44100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    return (
        0.05 * tones[None, :] + rng.standard_normal((2, n), dtype=np.float32) * 0.01
    ).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.full:
        from demucs_mlx import Separator

        for seconds in (1, 30, 60):
            audio = mx.array(signal(seconds))
            mx.save_safetensors(
                str(args.output / f"audio_{seconds}.safetensors"), {"audio": audio}
            )
            for backend in ("gpu",):
                with Separator(seed=481, batch_size=8) as sep:
                    _, stems = sep.separate_tensor(audio, return_mx=True)
                    out = mx.stack(list(stems.values()))
                    mx.eval(out)
                    mx.save_safetensors(
                        str(args.output / f"reference_{seconds}_{backend}.safetensors"),
                        {"stems": out},
                    )
            print(f"- ✅ {seconds}s full-model references", flush=True)
        return
    import torch
    from demucs.htdemucs import HTDemucs
    from demucs.hdemucs import HDemucs
    from demucs.demucs import Demucs
    from demucs_mlx.mlx_convert import convert_single_model, _adapt_mlx_constructor

    sources = ["drums", "bass", "other", "vocals"]
    torch.manual_seed(0)
    cases = [
        (
            "transformer",
            "htdemucs",
            HTDemucs(
                sources,
                channels=8,
                depth=4,
                segment=1,
                bottom_channels=16,
                t_layers=4,
                t_heads=2,
                dconv_mode=3,
                t_layer_scale=False,
                dconv_init=1,
            ),
            1,
        ),
        (
            "hybrid",
            "hdemucs_mmi",
            HDemucs(
                sources,
                channels=8,
                depth=6,
                segment=4,
                cac=True,
                norm_starts=4,
                dconv_attn=4,
                dconv_lstm=4,
                dconv_init=1,
            ),
            2,
        ),
        (
            "wiener",
            "hdemucs_mmi",
            HDemucs(
                sources,
                channels=8,
                depth=6,
                segment=4,
                cac=False,
                hybrid_old=True,
                norm_starts=999,
            ),
            2,
        ),
        (
            "multifreq",
            "hdemucs_mmi",
            HDemucs(
                sources,
                channels=8,
                depth=6,
                segment=4,
                cac=True,
                multi_freqs=[0.1, 0.3],
                multi_freqs_depth=2,
                norm_starts=999,
            ),
            2,
        ),
        (
            "time",
            "mdx",
            Demucs(
                sources,
                channels=8,
                depth=6,
                segment=4,
                resample=True,
                dconv_attn=4,
                dconv_lstm=4,
                norm_starts=4,
                rewrite=False,
                dconv_init=1,
            ),
            2,
        ),
    ]
    print("## 🧪 Architecture fixtures", flush=True)
    for case, name, upstream, seconds in cases:
        folder = args.output / case
        folder.mkdir(exist_ok=True)
        upstream.eval()
        model = convert_single_model(upstream)
        model.eval()
        init_args, kwargs = _adapt_mlx_constructor(upstream, type(model))
        if init_args:
            kwargs["sources"] = init_args[0]
            init_args = []
        audio = mx.array(signal(seconds))
        # Direct-forward fixtures retain upstream valid lengths, not split padding.
        length = (
            model.valid_length(audio.shape[-1])
            if hasattr(model, "valid_length")
            else audio.shape[-1]
        )
        audio = mx.pad(audio, [(0, 0), (0, length - audio.shape[-1])])
        ref_mlx = model(audio[None])[0]
        with torch.no_grad():
            ref_torch = upstream(torch.from_numpy(np.array(audio))[None])[0].numpy()
        mx.eval(ref_mlx)
        file = folder / f"{name}.safetensors"
        mx.save_safetensors(
            str(file),
            dict(
                __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
                    model.parameters()
                )
            ),
        )
        metadata = dict(
            format_version=1,
            model_name=name,
            model_class=type(model).__name__,
            sub_model_class=None,
            num_models=1,
            args=init_args,
            kwargs=kwargs,
            per_model_args=[init_args],
            per_model_kwargs=[kwargs],
            per_model_classes=[type(model).__name__],
            weights=[[1] * 4],
            source_artifacts=[{"signature": "fixture", "checksum": "fixture"}],
            verification_passed=True,
            safetensors_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        )
        (folder / f"{name}_config.json").write_text(
            json.dumps(metadata, default=encode)
        )
        mx.save_safetensors(
            str(folder / "forward.safetensors"),
            {"audio": audio, "mlx": ref_mlx, "torch": mx.array(ref_torch)},
        )
        print(f"- ✅ `{case}` ({length} samples)", flush=True)


if __name__ == "__main__":
    main()
