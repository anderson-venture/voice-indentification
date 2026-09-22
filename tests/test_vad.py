"""VAD segmentation.

Two halves, tested separately:

* ``segments_from_probabilities`` -- hysteresis, duration filtering and
  padding, driven by hand-built probability sequences with no model involved,
  so boundaries can be asserted exactly;
* segmentation on the fixtures, where speech must be found in an utterance and
  **no** speech in room tone.
"""

from __future__ import annotations

import numpy as np
import pytest

from speakerid.audio.io import TARGET_SAMPLE_RATE, load_audio
from speakerid.audio.vad import (
    SILERO_CONTEXT_SAMPLES,
    SILERO_WINDOW_SAMPLES,
    Segment,
    SileroVad,
    concatenate_speech,
    speech_mask,
    total_speech_seconds,
)
from speakerid.errors import ModelError
from speakerid.models.base import RuntimeOptions

from .conftest import FakeVadSession, fake_session_factory

WINDOW_MS = SILERO_WINDOW_SAMPLES / 16.0  # 32 ms per window at 16 kHz


def build_vad(**overrides) -> SileroVad:
    settings = dict(
        threshold=0.5,
        neg_threshold=0.35,
        window_samples=SILERO_WINDOW_SAMPLES,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=30,
        max_speech_seconds=60.0,
    )
    settings.update(overrides)
    options = RuntimeOptions(2, 1, "sequential", "all", ["CPUExecutionProvider"], 1, False)
    return SileroVad(
        model_path="models/vad/silero_vad.onnx",
        options=options,
        session_factory=fake_session_factory,
        **settings,
    )


def probabilities(pattern: str) -> np.ndarray:
    """'.' is silence, 'S' is speech -- one character per 512-sample window."""
    return np.array([0.9 if char == "S" else 0.05 for char in pattern], dtype=np.float32)


# ---------------------------------------------------------------------------
# Segmentation logic
# ---------------------------------------------------------------------------


def test_window_size_is_enforced():
    with pytest.raises(ModelError, match="512-sample"):
        build_vad(window_samples=256)


def test_boundaries_land_where_expected():
    vad = build_vad(min_speech_ms=0, min_silence_ms=0, speech_pad_ms=0)
    # 5 silent windows, then 20 speech windows, then silence.
    pattern = "." * 5 + "S" * 20 + "." * 5
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    assert len(segments) == 1
    assert segments[0].start == 5 * SILERO_WINDOW_SAMPLES
    # Windows 5..24 are speech, so the segment closes at the start of window
    # 25 -- the first window that fell below the threshold.
    assert segments[0].end == 25 * SILERO_WINDOW_SAMPLES


def test_padding_extends_segments_on_both_sides():
    vad = build_vad(min_speech_ms=0, min_silence_ms=0, speech_pad_ms=64)  # 2 windows
    pattern = "." * 10 + "S" * 20 + "." * 10
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    pad = int(64 * TARGET_SAMPLE_RATE / 1000)
    assert segments[0].start == 10 * SILERO_WINDOW_SAMPLES - pad


def test_short_bursts_are_discarded():
    """A two-window blip is a door or a keyboard, not a word."""
    vad = build_vad(min_speech_ms=250, min_silence_ms=0, speech_pad_ms=0)
    pattern = "." * 5 + "SS" + "." * 20
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    assert segments == []


def test_hysteresis_bridges_a_single_quiet_frame():
    """A dip below the upper threshold must not split a word in half."""
    vad = build_vad(min_speech_ms=0, min_silence_ms=200, speech_pad_ms=0)
    pattern_scores = np.array(
        [0.05] * 3 + [0.9] * 10 + [0.4] + [0.9] * 10 + [0.05] * 5, dtype=np.float32
    )
    segments = vad.segments_from_probabilities(
        pattern_scores, total_samples=pattern_scores.size * SILERO_WINDOW_SAMPLES
    )
    assert len(segments) == 1, "0.4 is above neg_threshold and must not end the segment"


def test_long_silence_splits_into_two_segments():
    vad = build_vad(min_speech_ms=0, min_silence_ms=100, speech_pad_ms=0)
    pattern = "S" * 10 + "." * 20 + "S" * 10
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    assert len(segments) == 2


def test_max_speech_forces_a_cut():
    vad = build_vad(min_speech_ms=0, min_silence_ms=100, speech_pad_ms=0, max_speech_seconds=1.0)
    pattern = "S" * 200  # 6.4 s of continuous speech
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    assert len(segments) > 1
    assert all(segment.duration_seconds() <= 1.5 for segment in segments)


def test_overlapping_padded_segments_merge():
    vad = build_vad(min_speech_ms=0, min_silence_ms=32, speech_pad_ms=200)
    pattern = "S" * 10 + "." * 2 + "S" * 10
    segments = vad.segments_from_probabilities(
        probabilities(pattern), total_samples=len(pattern) * SILERO_WINDOW_SAMPLES
    )
    assert len(segments) == 1, "padding pushed the two segments together"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_speech_mask_and_concatenation_agree():
    samples = np.arange(1000, dtype=np.float32)
    segments = [Segment(100, 200), Segment(500, 600)]
    mask = speech_mask(segments, samples.size)
    assert mask.sum() == 200
    speech = concatenate_speech(samples, segments)
    assert speech.size == 200
    assert np.array_equal(speech, samples[mask])


def test_concatenate_with_no_segments_is_empty():
    samples = np.arange(100, dtype=np.float32)
    assert concatenate_speech(samples, []).size == 0


def test_segment_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Segment(500, 100)


# ---------------------------------------------------------------------------
# On fixtures
# ---------------------------------------------------------------------------


def test_speech_found_in_an_utterance(fixture_path):
    vad = build_vad()
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    segments = vad.segments(samples)
    assert segments, "no speech detected in a 9 s utterance"
    seconds = total_speech_seconds(segments)
    # The fixture is 9 s of bursts separated by pauses, so speech is a clear
    # majority of it but nowhere near all of it.
    assert 5.0 < seconds < 8.0, "detected {0:.2f}s".format(seconds)


def test_no_speech_in_room_tone(fixture_path):
    """The single most important VAD assertion in the suite.

    Room tone that reaches the embedder produces a confident garbage
    embedding, which is the most common source of false accepts in systems
    like this (CLAUDE.md §6).
    """
    vad = build_vad()
    samples = load_audio(fixture_path("room_tone.wav"), peak_dbfs=-3.0)
    segments = vad.segments(samples)
    assert total_speech_seconds(segments) < 0.5


def test_segments_are_ordered_and_disjoint(fixture_path):
    vad = build_vad()
    samples = load_audio(fixture_path("bob_01.wav"), peak_dbfs=-3.0)
    segments = vad.segments(samples)
    for earlier, later in zip(segments, segments[1:]):
        assert earlier.end <= later.start


def test_segments_stay_inside_the_buffer(fixture_path):
    vad = build_vad()
    samples = load_audio(fixture_path("alice_02.wav"), peak_dbfs=-3.0)
    for segment in vad.segments(samples):
        assert 0 <= segment.start < segment.end <= samples.size


# ---------------------------------------------------------------------------
# Silero v5 context window
# ---------------------------------------------------------------------------


def test_feed_carries_the_64_sample_context():
    """Each window must reach the model as 64 context + 512 new samples.

    This is the single most dangerous silent failure in the front end. The
    real Silero v5 ONNX does not reject a bare 512-sample window -- it simply
    returns ~0.0 for every frame, so the system finds no speech anywhere and
    rejects every request with `insufficient_speech`. It looks like a tuning
    problem and it is not.
    """
    captured = []

    class RecordingSession(FakeVadSession):
        def _compute(self, feeds):
            captured.append(np.asarray(feeds["input"]).shape)
            return super(RecordingSession, self)._compute(feeds)

    options = RuntimeOptions(2, 1, "sequential", "all", ["CPUExecutionProvider"], 1, False)
    vad = SileroVad(
        model_path="models/vad/silero_vad.onnx",
        options=options,
        threshold=0.5,
        neg_threshold=0.35,
        window_samples=SILERO_WINDOW_SAMPLES,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=30,
        max_speech_seconds=60.0,
        session_factory=lambda path, opts: RecordingSession(),
    )
    vad.speech_probabilities(np.zeros(SILERO_WINDOW_SAMPLES * 4, dtype=np.float32))

    expected = SILERO_WINDOW_SAMPLES + SILERO_CONTEXT_SAMPLES
    assert captured, "the VAD never ran"
    assert all(shape == (1, expected) for shape in captured), captured


def test_context_is_the_previous_windows_tail():
    """The context is the last 64 samples of the window before it."""
    seen = []

    class RecordingSession(FakeVadSession):
        def _compute(self, feeds):
            seen.append(np.asarray(feeds["input"], dtype=np.float32).reshape(-1).copy())
            return super(RecordingSession, self)._compute(feeds)

    options = RuntimeOptions(2, 1, "sequential", "all", ["CPUExecutionProvider"], 1, False)
    vad = SileroVad(
        model_path="models/vad/silero_vad.onnx",
        options=options,
        threshold=0.5,
        neg_threshold=0.35,
        window_samples=SILERO_WINDOW_SAMPLES,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=30,
        max_speech_seconds=60.0,
        session_factory=lambda path, opts: RecordingSession(),
    )
    del seen[:]  # drop the warmup call; we want the real windows
    samples = np.arange(SILERO_WINDOW_SAMPLES * 3, dtype=np.float32) / 1000.0
    vad.speech_probabilities(samples)

    context = SILERO_CONTEXT_SAMPLES
    # First window starts from silence, as the reference implementation does.
    assert np.all(seen[0][:context] == 0.0)
    assert np.array_equal(seen[0][context:], samples[:SILERO_WINDOW_SAMPLES])
    # Second window carries the tail of the first.
    assert np.array_equal(
        seen[1][:context], samples[SILERO_WINDOW_SAMPLES - context : SILERO_WINDOW_SAMPLES]
    )
    assert np.array_equal(
        seen[1][context:], samples[SILERO_WINDOW_SAMPLES : 2 * SILERO_WINDOW_SAMPLES]
    )
