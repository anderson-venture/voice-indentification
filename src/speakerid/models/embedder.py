"""Speaker embedding (CLAUDE.md stage B).

The embedding model is **frozen**. It maps a waveform to a point in a 192-d
space where the same speaker lands close to themselves. Enrolling a person
stores vectors; it never trains anything. If you are ever tempted to add a
softmax layer sized to the employee count, re-read CLAUDE.md §1.

Interface contract, not to be deviated from::

    input:  float32 [1, N] or [1, 1, N]   raw waveform, 16 kHz mono,
                                          N = 64000..96000
    output: float32 [1, 192]              L2-normalized

Real exports differ on both points, so neither is assumed. The input layout is
read from the graph, and the output is re-normalized here: the ReDimNet-B2
export in use returns vectors of norm ~18, and trusting it to be unit-norm
would have inflated every cosine score.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

from ..audio.io import TARGET_SAMPLE_RATE, ensure_audio, fit_window, sliding_windows
from ..config import Config
from ..errors import ModelError
from ..nptypes import FloatArray
from .base import OnnxModel, RuntimeOptions, SessionFactory

LOGGER = logging.getLogger(__name__)


def l2_normalize(vector: FloatArray, epsilon: float = 1e-12) -> FloatArray:
    """Project onto the unit sphere.

    Cosine similarity assumes unit norm. Normalising defensively here means a
    re-exported model that forgot to normalise cannot quietly inflate scores.
    """
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < epsilon:
        raise ModelError("embedding has near-zero norm; the model saw no usable signal")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class SpeakerEmbedder(OnnxModel):
    """ReDimNet2-B2 (or any export honouring the interface contract)."""

    slot = "embedder"

    def __init__(
        self,
        model_path: str,
        options: RuntimeOptions,
        embedding_dim: int,
        window_samples: int,
        max_window_samples: int,
        multi_window_enabled: bool,
        multi_window_trigger_samples: int,
        multi_window_hop_samples: int,
        max_windows: int,
        models_dir: Optional[str] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.window_samples = window_samples
        self.max_window_samples = max_window_samples
        self.multi_window_enabled = multi_window_enabled
        self.multi_window_trigger_samples = multi_window_trigger_samples
        self.multi_window_hop_samples = multi_window_hop_samples
        self.max_windows = max_windows
        super(SpeakerEmbedder, self).__init__(
            model_path, options, models_dir=models_dir, session_factory=session_factory
        )

    @classmethod
    def from_config(
        cls, config: Config, session_factory: Optional[SessionFactory] = None
    ) -> SpeakerEmbedder:
        rate = config.sample_rate
        return cls(
            model_path=config.resolve_path("embedder.model_path"),
            options=RuntimeOptions.from_config(config),
            embedding_dim=config.get_int("embedder.embedding_dim"),
            window_samples=int(config.get_float("embedder.window_seconds") * rate),
            max_window_samples=int(config.get_float("embedder.max_window_seconds") * rate),
            multi_window_enabled=config.get_bool("embedder.multi_window.enabled"),
            multi_window_trigger_samples=int(
                config.get_float("embedder.multi_window.trigger_seconds") * rate
            ),
            multi_window_hop_samples=int(
                config.get_float("embedder.multi_window.hop_seconds") * rate
            ),
            max_windows=config.get_int("embedder.multi_window.max_windows"),
            models_dir=config.resolve_relative("models"),
            session_factory=session_factory,
        )

    # -- OnnxModel hooks -----------------------------------------------------

    def _validate_signature(self) -> None:
        if len(self.input_names) != 1:
            raise ModelError(
                "embedder must take exactly one input (the waveform); got {0}".format(
                    self.input_names
                )
            )
        rank = self.input_rank()
        if rank not in (2, 3):
            raise ModelError(
                "embedder input must be [batch, samples] or [batch, 1, samples]; "
                "this export declares rank {0}".format(rank)
            )

    def _shape_waveform(self, samples: FloatArray) -> FloatArray:
        """Lay the waveform out the way this particular export expects."""
        if self.input_rank() == 3:
            return samples.reshape(1, 1, -1)
        return samples.reshape(1, -1)

    def _warmup_feeds(self) -> Dict[str, FloatArray]:
        feeds: Dict[str, FloatArray] = {
            self.input_names[0]: self._shape_waveform(
                np.zeros(self.window_samples, dtype=np.float32)
            )
        }
        return feeds

    # -- inference -----------------------------------------------------------

    def embed_window(self, samples: FloatArray) -> FloatArray:
        """Embed exactly one window. Returns a unit-norm ``(dim,)`` vector."""
        ensure_audio(samples, TARGET_SAMPLE_RATE, name="embedder input")
        if samples.size != self.window_samples:
            samples = fit_window(samples, self.window_samples)

        feeds: Dict[str, FloatArray] = {
            self.input_names[0]: self._shape_waveform(samples.astype(np.float32, copy=False))
        }
        outputs = self._run(feeds)
        embedding: FloatArray = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if embedding.size != self.embedding_dim:
            raise ModelError(
                "embedder returned {0} dimensions, expected {1}; the model file "
                "does not match config/default.yaml".format(embedding.size, self.embedding_dim)
            )
        return l2_normalize(embedding)

    def embed(self, speech: FloatArray) -> FloatArray:
        """Embed a speech buffer, pooling across windows when there is enough.

        The caller must pass **speech only** -- VAD-selected and concatenated.
        Feeding silence, keyboard noise or HVAC hum to this model produces a
        confident-looking garbage embedding, which is the single most common
        source of false accepts in systems like this (CLAUDE.md §6).
        """
        ensure_audio(speech, TARGET_SAMPLE_RATE, name="embedder input")
        if speech.size == 0:
            raise ModelError("refusing to embed an empty buffer")

        use_multi = self.multi_window_enabled and speech.size >= self.multi_window_trigger_samples
        if not use_multi:
            # Below the trigger, centre-crop a single window (up to the model's
            # maximum) and embed it once.
            length = min(
                max(self.window_samples, min(speech.size, self.max_window_samples)),
                self.max_window_samples,
            )
            if speech.size < self.window_samples:
                length = self.window_samples
            return self.embed_window(fit_window(speech, length))

        windows = sliding_windows(
            speech,
            length=self.window_samples,
            hop=self.multi_window_hop_samples,
            max_windows=self.max_windows,
        )
        LOGGER.debug("mean-pooling embedding over %d windows", len(windows))
        return self.pool([self.embed_window(window) for window in windows])

    @staticmethod
    def pool(embeddings: Sequence[FloatArray]) -> FloatArray:
        """Mean-pool unit vectors and re-normalize.

        Averaging first and re-normalising afterwards keeps the result on the
        unit sphere, where cosine similarity is meaningful.
        """
        if not embeddings:
            raise ModelError("cannot pool an empty set of embeddings")
        stacked: FloatArray = np.stack(
            [np.asarray(item, dtype=np.float32) for item in embeddings], axis=0
        )
        return l2_normalize(stacked.mean(axis=0))
