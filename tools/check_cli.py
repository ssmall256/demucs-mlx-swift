#!/usr/bin/env python3
"""Exercise native decoding, resampling, inference and stem writing.
This invokes the GPU-backed native CLI.
"""

import argparse
import math
import struct
import subprocess
import time
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(".build/xcode/Build/Products/Release/demucs-mlx-swift"),
    )
    parser.add_argument("--cache", type=Path, default=Path(".build/models"))
    parser.add_argument("--output", type=Path, default=Path(".build/cli-validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    track = args.output / "tone-48k.wav"
    with wave.open(str(track), "wb") as audio:
        audio.setparams((2, 2, 48000, 48000, "NONE", "not compressed"))
        audio.writeframes(
            b"".join(
                struct.pack(
                    "<hh",
                    int(2000 * math.sin(2 * math.pi * 220 * i / 48000)),
                    int(1800 * math.sin(2 * math.pi * 330 * i / 48000)),
                )
                for i in range(48000)
            )
        )
    started = time.perf_counter()
    subprocess.run(
        [
            str(args.binary.resolve()),
            "separate",
            str(track.resolve()),
            "--cache",
            str(args.cache.resolve()),
            "--seed",
            "481",
            "--output",
            str(args.output.resolve()),
        ],
        check=True,
    )
    elapsed = time.perf_counter() - started
    folder = args.output / "htdemucs" / track.stem
    stems = sorted(folder.glob("*.wav"))
    assert {p.stem for p in stems} == {"drums", "bass", "other", "vocals"}
    for path in stems:
        with wave.open(str(path), "rb") as audio:
            assert (
                audio.getnchannels(),
                audio.getframerate(),
                audio.getnframes(),
                audio.getsampwidth(),
            ) == (2, 44100, 44100, 2), path
            assert len(audio.readframes(audio.getnframes())) == 44100 * 2 * 2, path
    print(
        f"\n## ✅ File API / CLI\n\n- **4 stereo WAV stems**, 44.1 kHz, 44,100 frames each\n- Native 48 → 44.1 kHz resampling\n- **Cold process + validation + decode + inference + export:** {elapsed:.3f} s"
    )


if __name__ == "__main__":
    main()
