"""Audio loading and the 16 kHz mono float32 contract.

Every function in this package that touches audio takes and returns
``np.ndarray`` float32, mono, 16 kHz, and asserts it at the boundary
(CLAUDE.md §12). Sample-rate bugs are silent and lethal: nothing crashes, the
model just sees speech at the wrong speed and returns a confident wrong answer.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np

from ..errors import AudioError
from ..nptypes import BoolArray, FloatArray

LOGGER = logging.getLogger(__name__)

#: The only sample rate permitted across a module boundary.
TARGET_SAMPLE_RATE = 16000

#: Amplitude below which a buffer is treated as digital silence.
_SILENCE_FLOOR = 1e-9


def ensure_audio(
    samples: FloatArray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    name: str = "audio",
) -> FloatArray:
    """Assert the audio contract and return the array unchanged.

    Call this at every boundary. It is cheap and it is the only thing standing
    between a resampling mistake and a silent accuracy collapse.
    """
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioError(
            "{0}: sample rate must be {1} Hz, got {2}".format(name, TARGET_SAMPLE_RATE, sample_rate)
        )
    if not isinstance(samples, np.ndarray):
        raise AudioError("{0}: expected np.ndarray, got {1}".format(name, type(samples).__name__))
    if samples.ndim != 1:
        raise AudioError("{0}: expected mono 1-D array, got shape {1}".format(name, samples.shape))
    if samples.dtype != np.float32:
        raise AudioError("{0}: expected float32, got {1}".format(name, samples.dtype))
    if samples.size and not np.all(np.isfinite(samples)):
        raise AudioError("{0}: contains NaN or Inf".format(name))
    return samples


def to_mono(samples: FloatArray) -> FloatArray:
    """Downmix to mono by averaging channels."""
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        # soundfile returns (frames, channels).
        downmixed: FloatArray = samples.mean(axis=1)
        return downmixed
    raise AudioError("cannot downmix array with shape {0}".format(samples.shape))


def resample(
    samples: FloatArray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> FloatArray:
    """Resample with a polyphase filter.

    ``scipy.signal.resample_poly`` is deterministic, anti-aliased and has no
    hidden state, which matters for the golden-file test: the same wav must
    produce the same embedding hash on every machine.
    """
    if source_rate <= 0:
        raise AudioError("invalid source sample rate: {0}".format(source_rate))
    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)

    from scipy.signal import resample_poly  # imported lazily; scipy is heavy

    divisor = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate // divisor)
    down = int(source_rate // divisor)
    resampled = resample_poly(samples.astype(np.float64), up, down)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def peak_normalize(samples: FloatArray, target_dbfs: float) -> FloatArray:
    """Scale so the peak sits at ``target_dbfs``.

    Digital silence is returned untouched: scaling it would only amplify the
    noise floor into something the VAD might mistake for speech.
    """
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < _SILENCE_FLOOR:
        LOGGER.debug("peak normalisation skipped: buffer is silent")
        return samples
    target_amplitude = float(10.0 ** (target_dbfs / 20.0))
    return np.ascontiguousarray(samples * (target_amplitude / peak), dtype=np.float32)


def load_audio(
    path: str,
    target_rate: int = TARGET_SAMPLE_RATE,
    peak_dbfs: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> FloatArray:
    """Load any soundfile-readable file as 16 kHz mono float32.

    This is the only place in the package that reads audio from disk.
    """
    if not os.path.isfile(path):
        raise AudioError("audio file not found: {0}".format(path))

    import soundfile  # lazily imported so the DB-only code paths stay light

    try:
        data, source_rate = soundfile.read(path, dtype="float32", always_2d=False)
    except RuntimeError as exc:
        raise AudioError("could not decode {0}: {1}".format(path, exc)) from exc

    samples: FloatArray = np.ascontiguousarray(to_mono(np.asarray(data)), dtype=np.float32)

    if max_seconds is not None:
        limit = int(max_seconds * source_rate)
        if samples.size > limit:
            raise AudioError(
                "{0} is {1:.1f}s, longer than the configured limit of {2:.1f}s".format(
                    path, samples.size / float(source_rate), max_seconds
                )
            )

    samples = resample(samples, int(source_rate), target_rate)
    if peak_dbfs is not None:
        samples = peak_normalize(samples, peak_dbfs)
    return ensure_audio(samples, target_rate, name=os.path.basename(path))


def write_audio(path: str, samples: FloatArray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Write 16 kHz mono float32 to a 16-bit PCM wav.

    Used only by fixture generation and by explicitly-consented retention
    paths. Never call this on a live recognition request (CLAUDE.md §11).
    """
    ensure_audio(samples, sample_rate, name=path)
    import soundfile

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    soundfile.write(path, samples, sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Windowing -- how audio is fitted to a model's fixed input length
# ---------------------------------------------------------------------------


def fit_window(samples: FloatArray, length: int) -> FloatArray:
    """Fit ``samples`` to exactly ``length`` by repeating or centre-cropping.

    Short clips are filled by **repeating** the speech, never by zero-padding
    (CLAUDE.md §5): a model trained on continuous speech reads a block of
    zeros as a channel artefact and the embedding drifts toward it.
    """
    if length <= 0:
        raise AudioError("window length must be positive, got {0}".format(length))
    ensure_audio(samples, name="fit_window input")

    if samples.size == 0:
        raise AudioError("cannot build a window from an empty buffer")
    if samples.size == length:
        return samples
    if samples.size > length:
        start = (samples.size - length) // 2  # centre-crop
        return np.ascontiguousarray(samples[start : start + length], dtype=np.float32)

    repeats = int(math.ceil(length / float(samples.size)))
    tiled = np.tile(samples, repeats)
    return np.ascontiguousarray(tiled[:length], dtype=np.float32)


def sliding_windows(
    samples: FloatArray, length: int, hop: int, max_windows: int
) -> Tuple[FloatArray, ...]:
    """Split into overlapping fixed-length windows for mean-pooled embedding.

    Preferred over a single centre-crop when plenty of speech is available: it
    averages away a single unlucky window (a cough, a door) instead of betting
    the identity decision on it.
    """
    if hop <= 0:
        raise AudioError("window hop must be positive, got {0}".format(hop))
    ensure_audio(samples, name="sliding_windows input")

    if samples.size <= length:
        return (fit_window(samples, length),)

    windows: List[FloatArray] = []
    start = 0
    while start + length <= samples.size and len(windows) < max_windows:
        windows.append(np.ascontiguousarray(samples[start : start + length], dtype=np.float32))
        start += hop

    # Always include the tail, otherwise the last seconds of a turn are dropped.
    if windows and len(windows) < max_windows:
        tail = samples[samples.size - length :]
        if not np.array_equal(tail, windows[-1]):
            windows.append(np.ascontiguousarray(tail, dtype=np.float32))
    return tuple(windows)


# ---------------------------------------------------------------------------
# Quality measurement (enrolment gates, CLAUDE.md §7)
# ---------------------------------------------------------------------------


def estimate_snr_db(samples: FloatArray, speech_mask: BoolArray) -> float:
    """Crude speech-vs-noise power ratio, in dB.

    Uses the VAD's own segmentation to split speech from background, which is
    the cheapest estimate that is not simply wrong. It is good enough for an
    enrolment gate and is not used in any identity decision.
    """
    ensure_audio(samples, name="snr input")
    if speech_mask.shape != samples.shape:
        raise AudioError("speech mask shape does not match audio")
    if not speech_mask.any() or speech_mask.all():
        # No contrast available: report a value that fails an SNR gate rather
        # than one that passes it by accident.
        return float("-inf") if not speech_mask.any() else float("inf")

    speech_power = float(np.mean(np.square(samples[speech_mask], dtype=np.float64)))
    noise_power = float(np.mean(np.square(samples[~speech_mask], dtype=np.float64)))
    if noise_power < _SILENCE_FLOOR:
        return float("inf")
    if speech_power < _SILENCE_FLOOR:
        return float("-inf")
    return float(10.0 * math.log10(speech_power / noise_power))
