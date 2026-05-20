# whisper-wrapper

Local HTTP API server wrapping [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text. Built as a companion service so apps like [neo-scan](../neo-scan) can transcribe audio without bundling the Whisper runtime themselves.

## Quick start

```bash
# install (editable, for development)
pip install -e ".[dev]"

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

## CLI

```
whisper-wrapper serve [options]
  --host 127.0.0.1            # bind address (use 0.0.0.0 to expose; requires --auth-token)
  --port 8765                 # bind port
  --model-cache-dir PATH      # where models are downloaded/cached
  --default-model base        # used when a request omits "model"
  --preload tiny,base         # models to load at startup
  --max-concurrent N          # max parallel transcriptions (default: min(cpu, 4))
  --compute-type int8         # int8 | int8_float16 | float16 | float32
  --device auto               # auto | cpu | cuda
  --auth-token TOKEN          # required when host != 127.0.0.1/::1
  --log-level info            # debug | info | warning | error

whisper-wrapper download MODEL    # pre-fetch a model and exit
whisper-wrapper version           # print version info
```

Env vars override defaults but are overridden by CLI flags: `WHISPER_WRAPPER_HOST`, `WHISPER_WRAPPER_PORT`, `WHISPER_WRAPPER_MODEL_CACHE_DIR`, etc.

## API

| Method | Path                         | Purpose |
| ------ | ---------------------------- | ------- |
| GET    | `/health`                    | Probe; returns version, loaded models, device |
| GET    | `/v1/models`                 | Supported sizes, loaded, cached-on-disk |
| POST   | `/v1/transcribe`             | Multipart: `audio`, `model`, `language`, `vad`, `word_timestamps`, `initial_prompt`, `temperature` |
| POST   | `/v1/models/{size}/preload`  | Trigger background download + load |

See `src/whisper_wrapper/schemas.py` for the full response shapes.

## Model storage

Models are downloaded on first use to a platform-appropriate user data directory:

- macOS: `~/Library/Application Support/whisper-wrapper/models`
- Linux: `~/.local/share/whisper-wrapper/models`
- Windows: `%LOCALAPPDATA%\whisper-wrapper\models`

Override with `--model-cache-dir`. Sizes: `tiny` ~75MB, `base` ~145MB, `small` ~460MB, `medium` ~1.5GB, `large-v3` ~3GB.

## Distribution

Releases are built per-OS via GitLab CI as standalone PyInstaller binaries. Models are **not** bundled — they download to the user cache on first use.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Tests use the `tiny` model and download it to a temporary cache on first run (~75MB).
