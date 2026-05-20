from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whisper_wrapper.schemas import Segment, WordTiming


@dataclass
class TranscribeResult:
    text: str
    language: str | None
    language_probability: float | None
    segments: list[Segment] = field(default_factory=list)
    words: list[WordTiming] | None = None


class Transcriber:
    """Abstract base for a loaded Whisper model."""

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool,
        initial_prompt: str | None,
        temperature: float,
    ) -> TranscribeResult:
        raise NotImplementedError
