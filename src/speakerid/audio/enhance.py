"""Optional denoising -- DISABLED BY DEFAULT, and that default is load-bearing.

Aggressive denoising strips exactly the fine spectral detail that identifies a
speaker. It sounds better to a human ear and scores worse on the metric that
matters. Raw audio into a well-augmented model very often wins (CLAUDE.md §9.5).

Enabling this requires a measured improvement on the operator's own eval set,
recorded in ``docs/DEVIATIONS.md``. ``scripts/evaluate.py --compare-enhance``
runs the A/B that would justify it.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..errors import ConfigError
from ..nptypes import FloatArray
from .io import TARGET_SAMPLE_RATE, ensure_audio

LOGGER = logging.getLogger(__name__)


class Enhancer(object):
    """Dispatches to a denoising backend, or passes audio through untouched."""

    def __init__(self, enabled: bool, backend: str) -> None:
        self.enabled = enabled
        self.backend = backend
        if enabled and backend == "none":
            raise ConfigError("enhance.enabled is true but enhance.backend is 'none'")
        if enabled:
            LOGGER.warning(
                "audio enhancement is ENABLED (backend=%s). Denoising removes "
                "speaker-identifying detail; confirm a measured gain on the "
                "operator's own eval set is recorded in docs/DEVIATIONS.md",
                backend,
            )

    @classmethod
    def from_config(cls, config: Config) -> Enhancer:
        return cls(
            enabled=config.get_bool("enhance.enabled"),
            backend=config.get_str("enhance.backend"),
        )

    def __call__(self, samples: FloatArray) -> FloatArray:
        ensure_audio(samples, TARGET_SAMPLE_RATE, name="enhancer input")
        if not self.enabled:
            return samples
        if self.backend == "spectral_gate":
            enhanced = _spectral_gate(samples)
        else:
            raise ConfigError("unknown enhance.backend: {0}".format(self.backend))
        return ensure_audio(enhanced, TARGET_SAMPLE_RATE, name="enhancer output")


def _spectral_gate(
    samples: FloatArray,
    frame_length: int = 512,
    hop_length: int = 128,
    noise_percentile: float = 10.0,
    over_subtraction: float = 1.0,
) -> FloatArray:
    """Conservative spectral-subtraction gate.

    Deliberately mild. A stronger denoiser would sound cleaner and identify
    people worse. Provided so the A/B in §9.5 can actually be run, not because
    it is recommended.
    """
    from scipy.signal import istft, stft

    _, _, spectrum = stft(
        samples.astype(np.float64), nperseg=frame_length, noverlap=frame_length - hop_length
    )
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Per-bin noise floor from the quietest frames, which are noise-only in
    # any recording that contains pauses.
    noise_floor = np.percentile(magnitude, noise_percentile, axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - over_subtraction * noise_floor, 0.0)

    _, reconstructed = istft(
        cleaned * np.exp(1j * phase),
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
    )
    result: FloatArray = np.zeros(samples.size, dtype=np.float32)
    usable = min(samples.size, reconstructed.size)
    result[:usable] = reconstructed[:usable].astype(np.float32)
    return result
