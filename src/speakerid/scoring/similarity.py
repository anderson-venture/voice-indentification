"""Cosine similarity over unit-norm embeddings."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..errors import SpeakerIdError
from ..nptypes import FloatArray

#: Vectors are expected unit-norm; this is the tolerance before we complain.
_NORM_TOLERANCE = 1e-3


def check_unit_norm(matrix: FloatArray, name: str = "embeddings") -> None:
    """Verify rows are unit-norm, because cosine silently assumes it.

    A model re-export that stopped normalising would otherwise inflate every
    score by an arbitrary factor and move the operating point without any
    visible failure.
    """
    if matrix.size == 0:
        return
    norms = np.linalg.norm(np.atleast_2d(matrix), axis=1)
    if np.any(np.abs(norms - 1.0) > _NORM_TOLERANCE):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise SpeakerIdError(
            "{0} are not unit-norm (max deviation {1:.2e}); cosine similarity "
            "would be meaningless".format(name, worst)
        )


def cosine(left: FloatArray, right: FloatArray) -> float:
    """Cosine similarity between two 1-D vectors."""
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise SpeakerIdError(
            "cannot compare embeddings of shape {0} and {1}".format(left.shape, right.shape)
        )
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise SpeakerIdError("cannot take cosine of a zero vector")
    return float(np.dot(left, right) / denominator)


def cosine_matrix(probe: FloatArray, matrix: FloatArray) -> FloatArray:
    """Cosine of one probe against every row of ``matrix``. Returns ``(N,)``."""
    probe = np.asarray(probe, dtype=np.float32).reshape(-1)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    if matrix.shape[1] != probe.shape[0]:
        raise SpeakerIdError(
            "embedding dimension mismatch: probe {0}, gallery {1}".format(
                probe.shape[0], matrix.shape[1]
            )
        )
    probe_norm = float(np.linalg.norm(probe))
    if probe_norm == 0.0:
        raise SpeakerIdError("cannot score a zero-norm probe")
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0.0] = 1.0
    return np.ascontiguousarray((matrix @ probe) / (row_norms * probe_norm), dtype=np.float32)


def top_k(scores: FloatArray, k: int) -> Tuple[FloatArray, FloatArray]:
    """Return ``(indices, values)`` of the k highest scores, best first."""
    if scores.size == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    k = min(k, scores.size)
    # argpartition is O(n); we only sort the k we keep.
    partition = np.argpartition(-scores, k - 1)[:k]
    ordered = partition[np.argsort(-scores[partition], kind="stable")]
    return ordered.astype(np.int64), np.ascontiguousarray(scores[ordered], dtype=np.float32)


def aggregate_person_scores(
    per_embedding_scores: FloatArray, owners: List[str], method: str = "max"
) -> List[Tuple[str, float]]:
    """Collapse per-embedding scores to one score per person.

    ``max`` is the default and the documented choice (CLAUDE.md §7): a person
    is recognised if the probe resembles *any* of their enrolments, which is
    what makes multiple enrolments worth keeping in the first place.
    """
    if method not in ("max", "mean"):
        raise SpeakerIdError("unknown aggregation method: {0}".format(method))
    best: Dict[str, List[float]] = {}
    for score, owner in zip(per_embedding_scores, owners):
        value = float(score)
        if owner not in best:
            best[owner] = [value]
        else:
            best[owner].append(value)
    if method == "max":
        collapsed = [(owner, max(values)) for owner, values in best.items()]
    else:
        collapsed = [(owner, sum(values) / len(values)) for owner, values in best.items()]
    collapsed.sort(key=lambda item: item[1], reverse=True)
    return collapsed
