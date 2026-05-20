from __future__ import annotations

import numpy as np

from whisper_wrapper.backends.base import Transcriber, TranscribeResult
from whisper_wrapper.schemas import Segment, WordTiming

# Maps whisper-wrapper model size names to mlx-community HuggingFace repos.
MLX_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class MlxTranscriber(Transcriber):
    def __init__(self, repo: str) -> None:
        self._repo = repo

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool,
        initial_prompt: str | None,
        temperature: float,
    ) -> TranscribeResult:
        import mlx_whisper  # lazy import so non-Apple platforms load cleanly

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=False,
            no_speech_threshold=0.8,
            verbose=None,
        )

        segments: list[Segment] = []
        words: list[WordTiming] | None = [] if word_timestamps else None

        for seg in result.get("segments", []):
            segments.append(
                Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"],
                    no_speech_prob=seg.get("no_speech_prob"),
                    avg_logprob=seg.get("avg_logprob"),
                )
            )
            if word_timestamps and seg.get("words"):
                for w in seg["words"]:
                    words.append(  # type: ignore[union-attr]
                        WordTiming(
                            start=float(w["start"]),
                            end=float(w["end"]),
                            word=w["word"],
                            probability=w.get("probability"),
                        )
                    )

        return TranscribeResult(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            language_probability=None,  # mlx-whisper doesn't report this
            segments=segments,
            words=words,
        )
