from __future__ import annotations

import numpy as np

from whisper_wrapper.backends.base import Transcriber, TranscribeResult
from whisper_wrapper.schemas import Segment, WordTiming


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model) -> None:
        self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool,
        initial_prompt: str | None,
        temperature: float,
    ) -> TranscribeResult:
        segments_gen, info = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            vad_filter=False,
        )
        segments_raw = list(segments_gen)

        segments: list[Segment] = []
        words: list[WordTiming] | None = [] if word_timestamps else None
        text_parts: list[str] = []

        for seg in segments_raw:
            text_parts.append(seg.text)
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text,
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                    avg_logprob=getattr(seg, "avg_logprob", None),
                )
            )
            if word_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    words.append(  # type: ignore[union-attr]
                        WordTiming(
                            start=float(w.start),
                            end=float(w.end),
                            word=w.word,
                            probability=getattr(w, "probability", None),
                        )
                    )

        return TranscribeResult(
            text="".join(text_parts).strip(),
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            segments=segments,
            words=words,
        )
