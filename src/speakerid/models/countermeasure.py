"""Replay / spoof countermeasure (CLAUDE.md stage C).

AASIST-L, 85K parameters. It runs **before** the embedder: it is roughly 40x
cheaper and short-circuits the expensive path on rejection (CLAUDE.md §6).

Read this before trusting the score
-----------------------------------
AASIST-L's headline 0.99% EER is on ASVspoof 2019 LA, which is *synthetic*
speech -- TTS and voice conversion. The threat here is a **physical replay**:
a phone or Bluetooth speaker held up to the microphone. Countermeasures that
excel on lab-generated synthetic audio routinely fail on physical replay.

So this score is one signal, never a security guarantee. The defences that
actually carry the weight are system-level: challenge-response (see
``speakerid.challenge``), a microphone array, and a second factor.
``docs/FIRST_RUN.md`` records this model failing on a real corpus, and
CLAUDE.md §10 is the standing instruction to fine-tune it on physical replay
before it protects anything.

Interface contract::

    input:  float32 [1, 64600]  raw waveform, 16 kHz, ~4.04 s
    output: float32 [1, 2]      logits; softmax, index 1 = bona fide
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from ..audio.io import TARGET_SAMPLE_RATE, ensure_audio, fit_window
from ..config import Config
from ..errors import ModelError
from ..nptypes import FloatArray
from .base import OnnxModel, RuntimeOptions, SessionFactory

LOGGER = logging.getLogger(__name__)


def softmax(logits: FloatArray) -> FloatArray:
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalized: FloatArray = (exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)).astype(
        np.float32
    )
    return normalized


class Countermeasure(OnnxModel):
    """Wraps AASIST-L and returns a bona-fide probability in [0, 1]."""

    slot = "countermeasure"

    def __init__(
        self,
        model_path: str,
        options: RuntimeOptions,
        input_samples: int,
        bona_fide_index: int,
        models_dir: Optional[str] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        self.input_samples = input_samples
        self.bona_fide_index = bona_fide_index
        super(Countermeasure, self).__init__(
            model_path, options, models_dir=models_dir, session_factory=session_factory
        )

    @classmethod
    def from_config(
        cls, config: Config, session_factory: Optional[SessionFactory] = None
    ) -> Countermeasure:
        return cls(
            model_path=config.resolve_path("countermeasure.model_path"),
            options=RuntimeOptions.from_config(config),
            input_samples=config.get_int("countermeasure.input_samples"),
            bona_fide_index=config.get_int("countermeasure.bona_fide_index"),
            models_dir=config.resolve_relative("models"),
            session_factory=session_factory,
        )

    # -- OnnxModel hooks -----------------------------------------------------

    def _validate_signature(self) -> None:
        if len(self.input_names) != 1:
            raise ModelError(
                "countermeasure must take exactly one input; got {0}".format(self.input_names)
            )

    def _warmup_feeds(self) -> Dict[str, FloatArray]:
        feeds: Dict[str, FloatArray] = {
            self.input_names[0]: np.zeros((1, self.input_samples), dtype=np.float32)
        }
        return feeds

    # -- inference -----------------------------------------------------------

    def score(self, speech: FloatArray) -> float:
        """Return P(bona fide) in [0, 1]. Higher means more likely a live human.

        As with the embedder, pass VAD-selected speech: the countermeasure is
        looking for channel artefacts of a loudspeaker, and a buffer of room
        tone gives it nothing to find.
        """
        ensure_audio(speech, TARGET_SAMPLE_RATE, name="countermeasure input")
        if speech.size == 0:
            raise ModelError("refusing to score an empty buffer")

        window = fit_window(speech, self.input_samples)
        feeds: Dict[str, FloatArray] = {
            self.input_names[0]: window.reshape(1, -1).astype(np.float32, copy=False)
        }
        outputs = self._run(feeds)

        logits: FloatArray = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if logits.size != 2:
            raise ModelError("countermeasure returned {0} logits, expected 2".format(logits.size))
        if not 0 <= self.bona_fide_index < 2:
            raise ModelError(
                "countermeasure.bona_fide_index must be 0 or 1; got {0}".format(
                    self.bona_fide_index
                )
            )
        probabilities = softmax(logits)
        return float(probabilities[self.bona_fide_index])
