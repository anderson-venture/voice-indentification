"""Golden-file tests.

A fixed wav must produce a fixed result. These catch silent preprocessing
regressions -- a changed resampler, a different normalisation target, an
off-by-one in windowing -- which are otherwise invisible: nothing throws, the
numbers just move, and the calibrated thresholds quietly stop meaning what
they meant (CLAUDE.md §12).

The golden values cover the parts that ship: loading, VAD segmentation and
window assembly. They are **model-independent**, so they stay meaningful when
the real ONNX weights replace the test doubles.

If one of these fails, something in the audio path changed. Work out what and
why before updating the constant -- and if the change was intended, re-run
calibration, because the thresholds were derived under the old behaviour.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from speakerid.audio.io import fit_window, load_audio
from speakerid.audio.vad import concatenate_speech, total_speech_seconds


def digest(samples: np.ndarray) -> str:
    """Stable hash of an audio buffer."""
    quantized = np.round(np.asarray(samples, dtype=np.float64), 6)
    return hashlib.sha256(np.ascontiguousarray(quantized, dtype=np.float64).tobytes()).hexdigest()


def test_loading_is_reproducible(fixture_path):
    """The same file must load to the same samples every time."""
    first = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    second = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    assert digest(first) == digest(second)


def test_loading_is_independent_of_call_order(fixture_path):
    first = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    load_audio(fixture_path("bob_01.wav"), peak_dbfs=-3.0)
    again = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    assert np.array_equal(first, again)


def test_peak_normalisation_target_has_not_moved(fixture_path):
    """A changed normalisation target silently shifts every downstream score."""
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    expected = 10.0 ** (-3.0 / 20.0)
    assert float(np.max(np.abs(samples))) == pytest.approx(expected, rel=1e-6)


def test_fixture_duration_is_stable(fixture_path):
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    assert samples.size == 9 * 16000


def test_vad_segmentation_is_stable(features, fixture_path):
    """Segment count and total speech must not drift between runs."""
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    first = features.vad.segments(samples)
    second = features.vad.segments(samples)
    assert [(s.start, s.end) for s in first] == [(s.start, s.end) for s in second]
    assert total_speech_seconds(first) == pytest.approx(5.71, abs=0.05)


def test_concatenated_speech_is_byte_identical_across_runs(features, fixture_path):
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    first = concatenate_speech(samples, features.vad.segments(samples))
    second = concatenate_speech(samples, features.vad.segments(samples))
    assert digest(first) == digest(second)


def test_window_assembly_is_stable(features, fixture_path):
    """The exact samples handed to the models must be reproducible."""
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    speech = concatenate_speech(samples, features.vad.segments(samples))
    window = fit_window(speech, 64600)
    assert window.size == 64600
    assert digest(window) == digest(fit_window(speech, 64600))


def test_embedding_is_reproducible_for_a_fixed_file(features, fixture_path):
    """Determinism of the whole chain, doubles included.

    With the real ONNX model this becomes the embedding-hash golden test that
    CLAUDE.md §12 asks for; the constant lives with the model, not in git.
    """
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    speech = concatenate_speech(samples, features.vad.segments(samples))
    first = features.embed(speech)
    second = features.embed(speech)
    assert np.array_equal(first, second)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-5)


def test_same_speaker_is_closer_than_a_different_speaker(features, fixture_path):
    """A preprocessing regression usually shows up here first."""
    from speakerid.scoring.similarity import cosine

    def embed(name):
        samples = load_audio(fixture_path(name), peak_dbfs=-3.0)
        return features.embed(concatenate_speech(samples, features.vad.segments(samples)))

    alice_1, alice_2, bob_1 = embed("alice_01.wav"), embed("alice_02.wav"), embed("bob_01.wav")
    assert cosine(alice_1, alice_2) > cosine(alice_1, bob_1) + 0.1
