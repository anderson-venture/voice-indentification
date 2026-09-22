"""Exception hierarchy.

Every failure mode the operator can hit has a named type, so the CLI and the
API can turn it into an actionable message instead of a traceback.
"""

from __future__ import annotations


class SpeakerIdError(Exception):
    """Base class for every error raised by this package."""


# -- configuration -----------------------------------------------------------


class ConfigError(SpeakerIdError):
    """Malformed, missing, or internally inconsistent configuration."""


class ThresholdsNotCalibratedError(ConfigError):
    """config/thresholds.yaml has not been filled in for this site.

    Raised loudly and early rather than falling back to a default, because a
    guessed threshold is an unquantified false-accept rate (CLAUDE.md §8).
    """


# -- audio -------------------------------------------------------------------


class AudioError(SpeakerIdError):
    """Audio could not be loaded, or violates the 16 kHz mono float32 contract."""


class InsufficientSpeechError(AudioError):
    """VAD found less usable speech than the configured minimum."""


# -- models ------------------------------------------------------------------


class ModelError(SpeakerIdError):
    """An ONNX model is missing, corrupt, or has an unexpected signature."""


class ManifestError(ModelError):
    """A model file's SHA-256 does not match models/MANIFEST.json."""


# -- database ----------------------------------------------------------------


class StoreError(SpeakerIdError):
    """Database-level failure."""


class ConsentError(StoreError):
    """Enrolment attempted without a recorded consent row (CLAUDE.md §11)."""


class CryptoError(StoreError):
    """Encryption key unavailable, or a ciphertext failed authentication."""


# -- enrolment / decisions ---------------------------------------------------


class EnrollmentError(SpeakerIdError):
    """An enrolment utterance failed a quality gate (CLAUDE.md §7)."""


class ChallengeError(SpeakerIdError):
    """A challenge-response token is unknown, expired, or already consumed."""
