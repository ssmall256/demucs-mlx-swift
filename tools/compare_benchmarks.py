#!/usr/bin/env python3
"""Matched, warmed HTDemucs comparison with alternating arm order.
Runs GPU inference in child processes; avoid other GPU work meanwhile.
"""

import argparse
import atexit
import gc
import hashlib
import json
import platform
import statistics
import shutil
import tempfile
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--binary",
        type=Path,
        default=Path(".build/out/Products/Release/demucs-mlx-swift"),
    )
    p.add_argument("--cache", type=Path, default=Path(".build/models"))
    p.add_argument("--fixtures", type=Path, default=Path(".build/full-fixtures"))
    p.add_argument(
        "--output", type=Path, default=Path(".build/benchmarks/comparison.json")
    )
    p.add_argument("--seconds", nargs="+", type=int, default=[30, 60])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--compilation", choices=["eager", "automatic"], default="automatic")
    a = p.parse_args()
    import mlx.core as mx
    import numpy as np
    from demucs_mlx import Separator
    from demucs_mlx.model_converter import get_mlx_cache_dir

    a.output.parent.mkdir(parents=True, exist_ok=True)
    if min(a.iterations, a.rounds) <= 0 or a.warmup < 0:
        p.error("Invalid iteration counts")

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    # Keep a byte-identical executable beside its resource bundle, so a separate
    # build cannot silently change an arm halfway through a comparison.
    with tempfile.NamedTemporaryFile(
        prefix=".demucs-benchmark-", dir=a.binary.resolve().parent, delete=False
    ) as pinned:
        pinned_binary = Path(pinned.name)
    shutil.copy2(a.binary, pinned_binary)
    atexit.register(lambda: pinned_binary.unlink(missing_ok=True))
    a.binary = pinned_binary
    binary_digest = digest(a.binary)
    weight_digest = digest(a.cache / "htdemucs.safetensors")
    if weight_digest != digest(get_mlx_cache_dir() / "htdemucs.safetensors"):
        raise ValueError("Python and Swift must load byte-identical weights")
    records = []
    metadata = dict(
        host=platform.node(),
        chip=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        ram_bytes=int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        swift=subprocess.check_output(["swift", "--version"], text=True).strip(),
        mlx=mx.__version__,
        weight_sha256=weight_digest,
        binary_sha256=binary_digest,
        batch=a.batch_size,
        seed=481,
        overlap=0.25,
        shifts=1,
        attention="fp16",
        warmup=a.warmup,
        rounds=a.rounds,
        compilation=a.compilation,
    )
    print("## 🚀 Matched Python / Swift comparison", flush=True)
    for seconds in a.seconds:
        fixture = a.fixtures / f"audio_{seconds}.safetensors"
        for backend in ["gpu"]:
            python_times = []
            swift_times = []
            rounds = []
            reference = None
            native = None
            for round_index in range(a.rounds):
                order = (
                    ["python", "swift"] if round_index % 2 == 0 else ["swift", "python"]
                )
                round_record = dict(order=order)
                for arm in order:
                    if arm == "python":
                        audio = mx.load(str(fixture))["audio"]
                        mx.eval(audio)
                        times = []
                        with Separator(
                            seed=481,
                            batch_size=a.batch_size,
                            attention_precision="fp16",
                            compile=a.compilation == "automatic",
                        ) as sep:
                            for i in range(a.warmup + a.iterations):
                                started = time.perf_counter()
                                _, stems = sep.separate_tensor(audio, return_mx=True)
                                mx.eval(*stems.values())
                                elapsed = time.perf_counter() - started
                                if i >= a.warmup:
                                    times.append(elapsed)
                                if i == a.warmup + a.iterations - 1:
                                    reference = np.stack(
                                        [np.array(x) for x in stems.values()]
                                    )
                                del stems
                        del sep, audio
                        gc.collect()
                        mx.clear_cache()
                        python_times.extend(times)
                        round_record["python_times"] = times
                    else:
                        if digest(a.binary) != binary_digest:
                            raise RuntimeError(
                                "Benchmark executable changed during comparison; rebuild before submitting"
                            )
                        target = (
                            a.output.parent
                            / f"swift_{seconds}_{backend}_{round_index}.json"
                        )
                        tensor = target.with_suffix(".safetensors")
                        subprocess.run(
                            [
                                str(a.binary.resolve()),
                                "benchmark",
                                "--cache",
                                str(a.cache.resolve()),
                                "--fixture",
                                str(fixture.resolve()),
                                "--seed",
                                "481",
                                "--batch-size",
                                str(a.batch_size),
                                "--compilation",
                                a.compilation,
                                "--warmup",
                                str(a.warmup),
                                "--iterations",
                                str(a.iterations),
                                "--json",
                                str(target.resolve()),
                                "--output-fixture",
                                str(tensor.resolve()),
                            ],
                            check=True,
                        )
                        swift = json.loads(target.read_text())
                        swift_times.extend(swift["times"])
                        round_record["swift"] = swift
                        native = np.array(mx.load(str(tensor))["stems"])
                        gc.collect()
                        mx.clear_cache()
                rounds.append(round_record)
            snr = [
                float(
                    10
                    * np.log10(
                        np.sum(ref.astype(np.float64) ** 2)
                        / max(np.sum((ref.astype(np.float64) - got) ** 2), 1e-30)
                    )
                )
                for ref, got in zip(reference, native)
            ]
            python = statistics.median(python_times)
            swift = statistics.median(swift_times)
            record = dict(
                seconds=seconds,
                backend=backend,
                python_times=python_times,
                swift_times=swift_times,
                python_median=python,
                swift_median=swift,
                python_rtfx=seconds / python,
                swift_rtfx=seconds / swift,
                ratio=python / swift,
                snr_db=snr,
                rounds=rounds,
            )
            records.append(record)
            print(
                f"\n> {'✅' if record['ratio'] >= 0.95 else '⚠️'} **{seconds}s {backend}: Python {seconds / python:.2f}× / Swift {seconds / swift:.2f}×; ratio {record['ratio']:.3f}; parity {min(snr):.1f}–{max(snr):.1f} dB**",
                flush=True,
            )
            a.output.write_text(json.dumps(dict(**metadata, records=records), indent=2))
            if min(snr) < 60:
                raise AssertionError(f"Parity failed: {snr}")
    print(
        "\n| Audio | Backend | Python RTFx | Swift RTFx | Swift / Python |\n|---|---|---:|---:|---:|"
    )
    for r in records:
        print(
            f"| {r['seconds']}s | {r['backend']} | {r['python_rtfx']:.2f}× | {r['swift_rtfx']:.2f}× | {r['ratio']:.3f} |"
        )


if __name__ == "__main__":
    main()
