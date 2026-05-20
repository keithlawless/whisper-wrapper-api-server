from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_ok(client) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "whisper-wrapper"
    assert body["backend"] == "faster-whisper"
    assert body["api_version"] == "v1"
    assert isinstance(body["loaded_models"], list)


@pytest.mark.asyncio
async def test_list_models(client) -> None:
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert "tiny" in body["supported"]
    assert "base" in body["supported"]
    assert isinstance(body["loaded"], list)
    assert isinstance(body["cached_on_disk"], list)


@pytest.mark.asyncio
async def test_request_id_header(client) -> None:
    r = await client.get("/health")
    assert r.headers.get("x-request-id")
