# whisper-wrapper

Local HTTP API server wrapping [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text. Built as a companion service so apps like [neo-scan](../neo-scan) can transcribe audio without bundling the Whisper runtime themselves.

On Apple Silicon Macs (M-series), the server automatically uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) instead, which targets the GPU and Neural Engine via Apple's Metal stack and is typically 3-5× faster than CPU-based inference.

## Quick start

```bash
# install (editable, for development)
pip install -e ".[dev]"

# on Apple Silicon, also install the mlx backend
pip install -e ".[mlx]"

# run the server (defaults to 127.0.0.1:8765)
whisper-wrapper serve --preload base

# smoke test
curl http://127.0.0.1:8765/health
```

Transcribe a WAV file:

```bash
curl -F audio=@tests/fixtures/hello.wav \
     -F model=base \
     http://127.0.0.1:8765/v1/transcribe
```

## Backend selection

The server picks its inference backend automatically at startup:

| Hardware | mlx-whisper installed? | Backend chosen |
|---|---|---|
| Apple Silicon (arm64 Mac) | yes | `mlx-whisper` — GPU + Neural Engine via Metal |
| Apple Silicon (arm64 Mac) | no | `faster-whisper` — CPU fallback |
| Everything else | — | `faster-whisper` |

Override the automatic choice with `--backend` or `WHISPER_WRAPPER_BACKEND`:

```bash
# force mlx-whisper even if detection would choose otherwise
whisper-wrapper serve --backend mlx-whisper

# force faster-whisper on an Apple Silicon Mac (e.g. for debugging)
whisper-wrapper serve --backend faster-whisper
```

The active backend is reported in the `/health` response (`backend`, `device`, `compute_type` fields).

### faster-whisper (default on non-Apple-Silicon)

Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for quantised CPU/CUDA inference.  
Default compute type is `int8` — ~4× smaller models, best CPU throughput, negligible accuracy impact for speech.  
Supports CUDA if an NVIDIA GPU is present. With `--device auto` (the default) the GPU is selected automatically when one is visible; force it with `--device cuda` or pin to `--device cpu`. For GPU inference pair it with `--compute-type float16`. A float16 compute type requested on a CPU device is automatically downgraded to `int8` (CTranslate2 cannot run float16 on CPU).

> **Concurrency:** transcription is serialized to one in-flight call at a time. The CTranslate2 model is shared across requests and is not thread-safe — calling it from multiple worker threads corrupts the process heap and eventually aborts with no traceback (observed on Windows after ~20–30h). Audio decode and VAD still run concurrently; only the model call is gated.

### mlx-whisper (Apple Silicon)

Uses Apple's [MLX](https://ml-explore.github.io/mlx/) framework. Models run on the GPU and Neural Engine via Metal using unified memory — no VRAM/RAM boundary.  
Models are downloaded from [mlx-community](https://huggingface.co/mlx-community) on HuggingFace and cached at `~/.cache/huggingface/hub`.  
Compute type is always `float16`.

## CLI

```
whisper-wrapper serve [options]
  --host 127.0.0.1            # bind address (use 0.0.0.0 to expose; requires --auth-token)
  --port 8765                 # bind port
  --model-cache-dir PATH      # where faster-whisper models are downloaded/cached
  --default-model base        # used when a request omits "model"
  --preload tiny,base         # models to load at startup
  --max-concurrent N          # retained for compatibility; currently a no-op — model inference is always serialized (see Concurrency note above)
  --compute-type int8         # int8 | int8_float16 | float16 | float32 (faster-whisper only)
  --device auto               # auto | cpu | cuda (faster-whisper only; auto picks CUDA when a GPU is present)
  --backend auto              # auto | faster-whisper | mlx-whisper
  --auth-token TOKEN          # required when host != 127.0.0.1/::1
  --log-level info            # debug | info | warning | error

whisper-wrapper download MODEL    # pre-fetch a faster-whisper model and exit
whisper-wrapper version           # print version info
```

Env vars override defaults but are overridden by CLI flags: `WHISPER_WRAPPER_HOST`, `WHISPER_WRAPPER_PORT`, `WHISPER_WRAPPER_MODEL_CACHE_DIR`, `WHISPER_WRAPPER_BACKEND`, etc.

## API

| Method | Path                         | Purpose |
| ------ | ---------------------------- | ------- |
| GET    | `/health`                    | Probe; returns version, loaded models, active backend, device |
| GET    | `/v1/models`                 | Supported sizes, loaded, cached-on-disk |
| POST   | `/v1/transcribe`             | Multipart: `audio`, `model`, `language`, `vad`, `word_timestamps`, `initial_prompt`, `temperature` |
| POST   | `/v1/models/{size}/preload`  | Trigger background download + load |

See `src/whisper_wrapper/schemas.py` for the full response shapes.

## Model storage

**faster-whisper** models download to a platform-appropriate user data directory:

- macOS: `~/Library/Application Support/whisper-wrapper/models`
- Linux: `~/.local/share/whisper-wrapper/models`
- Windows: `%LOCALAPPDATA%\whisper-wrapper\models`

Override with `--model-cache-dir`. Sizes: `tiny` ~75MB, `base` ~145MB, `small` ~460MB, `medium` ~1.5GB, `large-v3` ~3GB.

The following model name aliases are accepted in API requests and resolved to their canonical name:

| Alias | Resolves to |
|---|---|
| `large` | `large-v3` |

**mlx-whisper** models download via HuggingFace Hub to `~/.cache/huggingface/hub` (or `$HF_HOME/hub`). Model sizes are similar to above.

## Distribution

Releases are built per-OS via GitLab CI as standalone PyInstaller binaries. Models are **not** bundled — they download to the user cache on first use.

## Development

```bash
pip install -e ".[dev]"
pip install -e ".[mlx]"   # Apple Silicon only
pytest
ruff check src tests
```

Tests use the `tiny` model and download it to a temporary cache on first run (~75MB).
