from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    api_version: str
    backend: str
    loaded_models: list[str]
    device: str
    compute_type: str


class ModelsResponse(BaseModel):
    supported: list[str]
    loaded: list[str]
    cached_on_disk: list[str]


class Segment(BaseModel):
    start: float
    end: float
    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class WordTiming(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str | None
    language_probability: float | None = None
    duration_sec: float
    model: str
    segments: list[Segment] = Field(default_factory=list)
    words: list[WordTiming] | None = None
    vad_applied: bool
    request_id: str


class PreloadResponse(BaseModel):
    status: str
    size: str
