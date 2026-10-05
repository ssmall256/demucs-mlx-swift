# Asset conversion

The Swift loader accepts version-one safetensors and JSON produced by `demucs-mlx`'s restricted converter. It verifies the registry's original checkpoint signatures/checksum identifiers and the complete safetensors SHA-256. The JSON binds the cached weights to their constructors and ensemble weighting. This detects corruption and inconsistent caches; it is not a signature authenticating a third party's assets.

`tools/export_models.py` copies valid source caches byte-for-byte and regenerates invalid or legacy caches into a separate destination. Its source defaults to `~/.cache/demucs-mlx`; use `--source ~/.cache/mlx-weights/demucs-mlx` for the newer shared cache. It leaves the Python project and source cache untouched. Conversion uses only the restricted checkpoint loader; there is no unrestricted pickle fallback.

The original quantized checkpoints require `diffq==0.2.4`. Install it into a separate conversion environment so benchmarking the reference project uses unchanged dependencies. One way to reuse the project's existing dependencies is to make a temporary venv, add its existing site-packages directory in a `.pth` file, and supply the source root with `PYTHONPATH`; then install diffq there. Alternatively, install `demucs-mlx` and `tools/export-requirements.txt` into a separate venv. Use that interpreter to run `tools/export_models.py`.

All exported parameters are floating-point arrays. The `_q` registry names preserve checkpoint identity, not a new Swift quantization format.

Native inference never downloads or converts assets. The one-time exporter also defaults to offline checkpoint use. Pre-populate the reference project's Torch checkpoint cache, or explicitly pass `--allow-download` to fetch missing official checkpoints during conversion.
