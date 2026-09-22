"""Adaptive symmetric score normalization (AS-Norm).

The cheapest large accuracy win available in this system (CLAUDE.md §9.3):
one matrix multiply at inference, and it typically cuts error substantially.

The idea: a raw cosine of 0.61 means nothing on its own. What matters is how
unusual it is. AS-Norm answers that by scoring both the enrolment vector and
the probe against a **cohort** of a few hundred non-enrolled embeddings
recorded in the same room, and expressing the raw score as a z-score against
those two distributions::

    s_norm = 0.5 * ( (s - mu_enroll) / sigma_enroll
                   + (s - mu_probe)  / sigma_probe )

"Adaptive" means only the top-N most similar cohort members are used, so the
statistics come from the part of the space the speaker actually occupies.

The cohort must be recorded in the deployment environment. A cohort of clean
studio speech normalises against the wrong distribution and helps nothing.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from ..config import Config
from ..errors import ConfigError, SpeakerIdError
from ..nptypes import FloatArray
from .similarity import check_unit_norm, cosine_matrix

LOGGER = logging.getLogger(__name__)

#: Below this many cohort members the statistics are too noisy to trust.
MIN_USEFUL_COHORT = 20


class AsNorm(object):
    """Holds the cohort and the per-enrolment statistics derived from it."""

    def __init__(self, cohort: FloatArray, top_n: int, epsilon: float) -> None:
        cohort = np.atleast_2d(np.asarray(cohort, dtype=np.float32))
        if cohort.size == 0:
            cohort = np.zeros((0, 0), dtype=np.float32)
        else:
            check_unit_norm(cohort, name="cohort embeddings")
        self.cohort: FloatArray = np.ascontiguousarray(cohort, dtype=np.float32)
        self.top_n = top_n
        self.epsilon = epsilon

        if 0 < self.size < MIN_USEFUL_COHORT:
            LOGGER.warning(
                "cohort has only %d members; AS-Norm statistics will be unstable. "
                "CLAUDE.md §9.3 calls for 200-500 recorded in the deployment room.",
                self.size,
            )

    @property
    def size(self) -> int:
        return int(self.cohort.shape[0]) if self.cohort.ndim == 2 else 0

    @property
    def enabled(self) -> bool:
        return self.size > 0

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(cls, config: Config, cohort: Optional[FloatArray] = None) -> AsNorm:
        top_n = config.get_int("scoring.asnorm.top_n")
        epsilon = config.get_float("scoring.asnorm.epsilon")
        if not config.get_bool("scoring.asnorm.enabled"):
            LOGGER.warning(
                "AS-Norm is disabled. This is the cheapest accuracy win available; "
                "build a cohort with scripts/build_cohort.py before deploying."
            )
            return cls(np.zeros((0, 0), dtype=np.float32), top_n, epsilon)
        if cohort is None:
            cohort = load_cohort_file(config.resolve_path("scoring.asnorm.cohort_path"))
        return cls(cohort, top_n, epsilon)

    # -- statistics ----------------------------------------------------------

    def _stats(self, scores: FloatArray) -> Tuple[float, float]:
        """Mean and std of the top-N cohort scores."""
        if scores.size == 0:
            return 0.0, 1.0
        count = min(self.top_n, scores.size)
        # Only the top-N matter, so partition rather than sort the whole thing.
        selected = np.partition(scores, scores.size - count)[scores.size - count :]
        mean = float(np.mean(selected))
        std = float(np.std(selected))
        return mean, max(std, self.epsilon)

    def enrollment_stats(self, gallery_matrix: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Per-enrolment-vector cohort mean and std.

        Computed once at startup: it depends only on the gallery and the
        cohort, never on the probe, so it must not be in the request path.
        """
        gallery_matrix = np.atleast_2d(np.asarray(gallery_matrix, dtype=np.float32))
        count = gallery_matrix.shape[0]
        if not self.enabled or count == 0:
            return np.zeros(count, dtype=np.float32), np.ones(count, dtype=np.float32)

        means: FloatArray = np.zeros(count, dtype=np.float32)
        stds: FloatArray = np.ones(count, dtype=np.float32)
        for index in range(count):
            mean, std = self._stats(cosine_matrix(gallery_matrix[index], self.cohort))
            means[index] = mean
            stds[index] = std
        LOGGER.debug("computed AS-Norm enrolment statistics for %d vectors", count)
        return means, stds

    def probe_stats(self, probe: FloatArray) -> Tuple[float, float]:
        """Cohort mean and std for the probe. One matrix multiply per request."""
        if not self.enabled:
            return 0.0, 1.0
        return self._stats(cosine_matrix(probe, self.cohort))

    # -- normalization -------------------------------------------------------

    def normalize(
        self,
        raw_scores: FloatArray,
        probe: FloatArray,
        enrollment_means: FloatArray,
        enrollment_stds: FloatArray,
    ) -> FloatArray:
        """Apply symmetric normalization to a vector of raw cosine scores."""
        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        if not self.enabled:
            return raw_scores
        if raw_scores.shape != enrollment_means.shape:
            raise SpeakerIdError(
                "AS-Norm statistics do not match the score vector: {0} vs {1}. "
                "The gallery changed without recomputing enrolment statistics.".format(
                    raw_scores.shape, enrollment_means.shape
                )
            )
        probe_mean, probe_std = self.probe_stats(probe)
        enrolment_side = (raw_scores - enrollment_means) / np.maximum(enrollment_stds, self.epsilon)
        probe_side = (raw_scores - probe_mean) / max(probe_std, self.epsilon)
        return np.ascontiguousarray(0.5 * (enrolment_side + probe_side), dtype=np.float32)


# ---------------------------------------------------------------------------
# Cohort file IO
# ---------------------------------------------------------------------------


def save_cohort_file(path: str, cohort: FloatArray, model_id: str, room_id: str = "") -> None:
    """Persist a cohort matrix with the model it was built with.

    The model id is stored because a cohort is only valid for the embedding
    model that produced it; mixing them yields silently wrong statistics.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        cohort=np.ascontiguousarray(cohort, dtype=np.float32),
        model_id=np.array(model_id),
        room_id=np.array(room_id),
    )
    LOGGER.info("wrote cohort of %d embeddings to %s", cohort.shape[0], path)


def load_cohort_file(path: str, expected_model_id: Optional[str] = None) -> FloatArray:
    if not os.path.isfile(path):
        raise ConfigError(
            "AS-Norm is enabled but the cohort file {0} does not exist. Build one "
            "with scripts/build_cohort.py from 200-500 recordings made in the "
            "deployment environment, or set scoring.asnorm.enabled to false and "
            "accept the accuracy cost.".format(path)
        )
    with np.load(path, allow_pickle=False) as archive:
        cohort: FloatArray = np.asarray(archive["cohort"], dtype=np.float32)
        stored_model = str(archive["model_id"]) if "model_id" in archive else ""
    if expected_model_id and stored_model and stored_model != expected_model_id:
        raise ConfigError(
            "cohort at {0} was built with model {1}, but the configured embedder "
            "is {2}. Rebuild the cohort.".format(path, stored_model, expected_model_id)
        )
    return cohort
