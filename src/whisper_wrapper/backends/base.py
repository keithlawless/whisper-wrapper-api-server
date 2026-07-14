from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whisper_wrapper.schemas import Segment, WordTiming

# Whisper's temperature-fallback ladder. A single fixed temperature disables
# fallback, which lets greedy decoding get trapped in a repetition loop
# ("Thank you. Thank you. …") that runs past the end of the audio and drops
# the real speech underneath it. Re-decoding a window at successively higher
# temperatures whenever the result looks like a hallucination — its text
# compresses too well or its average logprob is too low — is Whisper's built-in
# escape from that loop, and it only engages when temperature is a schedule.
FALLBACK_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4  # gzip ratio above this ⇒ re-decode hotter
LOGPROB_THRESHOLD = -1.0           # avg logprob below this ⇒ re-decode hotter


def temperature_schedule(temperature: float) -> tuple[float, ...]:
    """Expand a requested temperature into a fallback schedule.

    The default (0.0) yields the full fallback ladder so hallucination loops
    can be escaped. A caller that explicitly asks for a non-zero temperature
    gets it as a single-pass decode, for deterministic/experimental use.
    """
    if temperature > 0.0:
        return (temperature,)
    return FALLBACK_TEMPERATURES


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
        no_speech_threshold: float,
    ) -> TranscribeResult:
        raise NotImplementedError
