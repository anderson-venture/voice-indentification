"""The 16 kHz mono float32 contract, windowing, and quality measurement."""

from __future__ import annotations

import numpy as np
import pytest

from speakerid.audio.io import (
    TARGET_SAMPLE_RATE,
    ensure_audio,
    estimate_snr_db,
    fit_window,
    load_audio,
    peak_normalize,
    resample,
    sliding_windows,
    to_mono,
)
from speakerid.errors import AudioError


def test_ensure_audio_rejects_wrong_sample_rate():
    with pytest.raises(AudioError, match="sample rate"):
        ensure_audio(np.zeros(16, dtype=np.float32), sample_rate=44100)


def test_ensure_audio_rejects_wrong_dtype():
    with pytest.raises(AudioError, match="float32"):
        ensure_audio(np.zeros(16, dtype=np.float64))


def test_ensure_audio_rejects_stereo():
    with pytest.raises(AudioError, match="mono"):
        ensure_audio(np.zeros((16, 2), dtype=np.float32))


def test_ensure_audio_rejects_nan():
    samples = np.zeros(16, dtype=np.float32)
    samples[3] = np.nan
    with pytest.raises(AudioError, match="NaN"):
        ensure_audio(samples)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [2.0, 3.0])


def test_resample_preserves_duration_and_tone():
    duration = 0.5
    source_rate = 44100
    time = np.arange(int(duration * source_rate)) / float(source_rate)
    tone = np.sin(2 * np.pi * 440.0 * time).astype(np.float32)

    resampled = resample(tone, source_rate, TARGET_SAMPLE_RATE)

    assert resampled.dtype == np.float32
    assert abs(resampled.size - int(duration * TARGET_SAMPLE_RATE)) <= 2
    # The dominant frequency must survive the rate change.
    spectrum = np.abs(np.fft.rfft(resampled))
    peak_hz = np.fft.rfftfreq(resampled.size, 1.0 / TARGET_SAMPLE_RATE)[int(np.argmax(spectrum))]
    assert abs(peak_hz - 440.0) < 5.0


def test_resample_is_identity_at_matching_rate():
    samples = np.random.RandomState(0).randn(1000).astype(np.float32)
    assert np.array_equal(resample(samples, 16000, 16000), samples)


def test_peak_normalize_hits_target():
    samples = (0.01 * np.random.RandomState(1).randn(4000)).astype(np.float32)
    normalized = peak_normalize(samples, -3.0)
    expected_peak = 10.0 ** (-3.0 / 20.0)
    assert float(np.max(np.abs(normalized))) == pytest.approx(expected_peak, rel=1e-5)


def test_peak_normalize_leaves_digital_silence_alone():
    """Scaling silence would only amplify a noise floor into false speech."""
    silence = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(peak_normalize(silence, -3.0), silence)


def test_fit_window_repeats_short_audio_rather_than_zero_padding():
    """CLAUDE.md §5: short clips are filled by repeating speech."""
    samples = np.linspace(0.1, 1.0, 100, dtype=np.float32)
    fitted = fit_window(samples, 250)

    assert fitted.size == 250
    assert not np.any(fitted == 0.0), "zero padding would look like a channel artefact"
    assert np.allclose(fitted[:100], samples)
    assert np.allclose(fitted[100:200], samples)


def test_fit_window_centre_crops_long_audio():
    samples = np.arange(1000, dtype=np.float32)
    fitted = fit_window(samples, 100)
    assert fitted.size == 100
    assert fitted[0] == pytest.approx(450.0)


def test_fit_window_exact_length_is_untouched():
    samples = np.arange(100, dtype=np.float32)
    assert np.array_equal(fit_window(samples, 100), samples)


def test_sliding_windows_covers_the_tail():
    samples = np.arange(1000, dtype=np.float32)
    windows = sliding_windows(samples, length=400, hop=200, max_windows=8)
    assert all(window.size == 400 for window in windows)
    # The final window must reach the end of the buffer.
    assert windows[-1][-1] == pytest.approx(999.0)


def test_sliding_windows_respects_the_cap():
    samples = np.arange(100000, dtype=np.float32)
    windows = sliding_windows(samples, length=1000, hop=500, max_windows=4)
    assert len(windows) == 4


def test_estimate_snr_db_separates_loud_speech_from_quiet_noise():
    rng = np.random.RandomState(0)
    samples = (0.01 * rng.randn(16000)).astype(np.float32)
    samples[4000:8000] += 1.0
    mask = np.zeros(16000, dtype=bool)
    mask[4000:8000] = True
    assert estimate_snr_db(samples, mask) > 20.0


def test_estimate_snr_db_with_no_speech_is_negative_infinity():
    samples = np.ones(100, dtype=np.float32)
    assert estimate_snr_db(samples, np.zeros(100, dtype=bool)) == float("-inf")


def test_load_audio_returns_the_contract(fixture_path):
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    ensure_audio(samples)
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert samples.size == pytest.approx(9 * TARGET_SAMPLE_RATE, abs=100)


def test_load_audio_enforces_the_length_limit(fixture_path):
    with pytest.raises(AudioError, match="longer than"):
        load_audio(fixture_path("alice_01.wav"), max_seconds=1.0)


def test_load_audio_missing_file():
    with pytest.raises(AudioError, match="not found"):
        load_audio("/nonexistent/file.wav")
