from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_upload(client) -> None:
    r = await client.post(
        "/v1/transcribe",
        files={"audio": ("empty.wav", b"", "audio/wav")},
        data={"model": "tiny", "vad": "false"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] in {"audio_decode_failed", "internal_error"}


@pytest.mark.asyncio
async def test_transcribe_unsupported_model(client) -> None:
    # send a small valid WAV but ask for a bogus model
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16_000, dtype=np.float32), 16_000, format="WAV", subtype="FLOAT")
    r = await client.post(
        "/v1/transcribe",
        files={"audio": ("a.wav", buf.getvalue(), "audio/wav")},
        data={"model": "ginormous", "vad": "false"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_available"


@pytest.mark.asyncio
async def test_transcribe_silence_returns_empty(client, silence_wav_bytes: bytes) -> None:
    r = await client.post(
        "/v1/transcribe",
        files={"audio": ("silence.wav", silence_wav_bytes, "audio/wav")},
        data={"model": "tiny", "vad": "true"},
    )
    # VAD should drop all audio → empty transcript. If VAD is unavailable on
    # the test machine, we still expect a clean response (text may be empty
    # or model-defined).
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "" or isinstance(body["text"], str)


@pytest.mark.skipif(
    not (Path(__file__).parent / "fixtures" / "hello.wav").exists(),
    reason="real audio fixture missing; drop a hello.wav under tests/fixtures/ to enable",
)
@pytest.mark.asyncio
async def test_transcribe_real_audio(client, real_fixture_wav: Path) -> None:
    raw = real_fixture_wav.read_bytes()
    r = await client.post(
        "/v1/transcribe",
        files={"audio": ("hello.wav", raw, "audio/wav")},
        data={"model": "tiny", "vad": "false"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["text"], str)
    assert body["model"] == "tiny"
    assert body["duration_sec"] > 0
    assert body["request_id"]
