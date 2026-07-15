"""Unit tests for _drop_segments_past — the phantom-segment guard that trims
decode-loop hallucinations whose timestamps run past the end of the audio."""
from __future__ import annotations

from whisper_wrapper.backends.base import TranscribeResult
from whisper_wrapper.routes.transcribe import _drop_segments_past
from whisper_wrapper.schemas import Segment, WordTiming


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def _result(segments, words=None) -> TranscribeResult:
    return TranscribeResult(
        text="".join(s.text for s in segments).strip(),
        language="en",
        language_probability=0.99,
        segments=list(segments),
        words=words,
    )


def test_drops_segments_past_cutoff_and_rebuilds_text():
    # Real 30s clip: two real segments, then a "Thank you." loop out to 56s.
    segs = [
        _seg(0.0, 5.0, " Navy 12, sneak up."),
        _seg(5.0, 28.0, " Cleared up. Very good."),
    ] + [_seg(float(t), float(t + 1), " Thank you.") for t in range(28, 56)]
    result, dropped = _drop_segments_past(_result(segs), cutoff_sec=31.0)
    assert dropped == 25                       # every segment starting >= 31s
    assert len(result.segments) == 5           # 2 real + 28..30 within cutoff
    assert result.text.endswith("Thank you.")  # rebuilt from survivors
    assert "Navy 12" in result.text


def test_no_drop_when_all_within_cutoff():
    segs = [_seg(0.0, 5.0, " one."), _seg(5.0, 9.0, " two.")]
    result, dropped = _drop_segments_past(_result(segs), cutoff_sec=31.0)
    assert dropped == 0
    assert result.text == "one. two."


def test_words_are_trimmed_to_cutoff():
    segs = [_seg(0.0, 5.0, " hi."), _seg(40.0, 41.0, " Thank you.")]
    words = [
        WordTiming(start=0.0, end=1.0, word="hi"),
        WordTiming(start=40.0, end=41.0, word="Thank you"),
    ]
    result, dropped = _drop_segments_past(_result(segs, words), cutoff_sec=31.0)
    assert dropped == 1
    assert [w.word for w in result.words] == ["hi"]
