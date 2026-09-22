"""Verification and open-set identification metrics.

Shared by ``scripts/calibrate.py`` and ``scripts/evaluate.py`` so that the
threshold the operator calibrates and the number the report shows are computed
by the same code. Two implementations would drift, and the drift would be
invisible.

The metric that actually matters here is :func:`open_set_report`'s
``impostor_rejection_rate``. Top-1 accuracy over enrolled speakers is the
number people quote and the one that hides the failure: a system that
recognises every employee correctly *and* confidently admits every stranger
scores 100% top-1 (CLAUDE.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .errors import SpeakerIdError
from .nptypes import Float64Array


@dataclass(frozen=True)
class DetCurve:
    """False-accept and false-reject rates across every operating point."""

    thresholds: Float64Array
    far: Float64Array  # P(accept | impostor)
    frr: Float64Array  # P(reject | genuine)

    def at_threshold(self, threshold: float) -> Tuple[float, float]:
        index = int(np.searchsorted(self.thresholds, threshold, side="left"))
        index = min(max(index, 0), self.thresholds.size - 1)
        return float(self.far[index]), float(self.frr[index])


def _validate(
    scores: Sequence[float], labels: Sequence[int]
) -> Tuple[Float64Array, npt.NDArray[np.int_]]:
    score_array: Float64Array = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_array: npt.NDArray[np.int_] = np.asarray(labels).reshape(-1).astype(int)
    if score_array.size != label_array.size:
        raise SpeakerIdError("scores and labels have different lengths")
    if score_array.size == 0:
        raise SpeakerIdError("no trials supplied")
    unique = set(np.unique(label_array).tolist())
    if not unique.issubset({0, 1}):
        raise SpeakerIdError("labels must be 0 (impostor) or 1 (genuine)")
    if unique != {0, 1}:
        raise SpeakerIdError(
            "trials must contain both genuine and impostor pairs; got only "
            "{0}. A threshold cannot be set from one class.".format(unique)
        )
    return score_array, label_array


def det_curve(scores: Sequence[float], labels: Sequence[int]) -> DetCurve:
    """Compute FAR and FRR at every distinct score threshold.

    Convention: a trial is accepted when ``score >= threshold``.
    """
    score_array, label_array = _validate(scores, labels)
    order = np.argsort(score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]

    n_genuine = int(np.sum(sorted_labels == 1))
    n_impostor = int(np.sum(sorted_labels == 0))

    # Sweep the threshold upward. At threshold t, everything below t is
    # rejected: the genuine trials below t are false rejects, and the impostor
    # trials at or above t are false accepts.
    genuine_below = np.cumsum(sorted_labels == 1)
    impostor_below = np.cumsum(sorted_labels == 0)

    frr = np.concatenate([[0.0], genuine_below / float(n_genuine)])
    far = np.concatenate([[1.0], 1.0 - impostor_below / float(n_impostor)])
    thresholds = np.concatenate([[sorted_scores[0] - 1e-6], sorted_scores])
    return DetCurve(thresholds=thresholds, far=far, frr=frr)


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Return ``(eer, threshold)`` where FAR and FRR cross."""
    curve = det_curve(scores, labels)
    difference = curve.far - curve.frr
    crossing = int(np.argmin(np.abs(difference)))
    eer = float((curve.far[crossing] + curve.frr[crossing]) / 2.0)
    return eer, float(curve.thresholds[crossing])


def threshold_at_far(
    scores: Sequence[float], labels: Sequence[int], target_far: float
) -> Tuple[float, float, float]:
    """Lowest threshold whose FAR is at or below ``target_far``.

    This is the operating point to ship, not the EER point. For access control
    CLAUDE.md §8 calls for FAR <= 0.1% and accepting the resulting FRR.
    """
    if not 0.0 < target_far < 1.0:
        raise SpeakerIdError("target FAR must be in (0, 1); got {0}".format(target_far))

    n_impostor = int(np.sum(np.asarray(labels).reshape(-1).astype(int) == 0))
    # You cannot measure a rate of 1-in-N from fewer than N trials. Without
    # this check, a handful of impostor pairs produce an apparent FAR of zero
    # at some threshold and the operator ships a number that was never
    # measured -- precisely the false confidence CLAUDE.md §8 warns about.
    required = int(np.ceil(1.0 / target_far))
    if n_impostor < required:
        raise SpeakerIdError(
            "measuring FAR <= {0} needs at least {1} impostor trials, but the "
            "trial set has too few impostor trials ({2}). Either record more "
            "impostor pairs or choose a less demanding target FAR.".format(
                target_far, required, n_impostor
            )
        )

    curve = det_curve(scores, labels)
    acceptable = np.where(curve.far <= target_far)[0]
    if acceptable.size == 0:
        raise SpeakerIdError(
            "no threshold achieves FAR <= {0}. The trial set cannot support this "
            "operating point: either the models separate these speakers poorly, "
            "or there are too few impostor trials to measure a rate this "
            "low (you have {1}).".format(target_far, int(np.sum(np.asarray(labels) == 0)))
        )
    index = int(acceptable[0])  # curve.far is non-increasing, so this is the lowest threshold
    return float(curve.thresholds[index]), float(curve.far[index]), float(curve.frr[index])


def min_dcf(
    scores: Sequence[float],
    labels: Sequence[int],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_false_alarm: float = 1.0,
) -> Tuple[float, float]:
    """Minimum detection cost, normalized. Returns ``(min_dcf, threshold)``."""
    curve = det_curve(scores, labels)
    cost = c_miss * p_target * curve.frr + c_false_alarm * (1.0 - p_target) * curve.far
    normalizer = min(c_miss * p_target, c_false_alarm * (1.0 - p_target))
    normalized = cost / normalizer
    best = int(np.argmin(normalized))
    return float(normalized[best]), float(curve.thresholds[best])


def far_frr_at_threshold(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> Tuple[float, float]:
    """Measured FAR and FRR at a specific threshold."""
    score_array, label_array = _validate(scores, labels)
    genuine = score_array[label_array == 1]
    impostor = score_array[label_array == 0]
    frr = float(np.mean(genuine < threshold))
    far = float(np.mean(impostor >= threshold))
    return far, frr


# ---------------------------------------------------------------------------
# Open-set identification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenSetReport:
    """What happens when strangers walk up, which is the real question."""

    n_probes: int
    n_enrolled_probes: int
    n_impostor_probes: int
    top1_accuracy: float
    top1_accuracy_enrolled_only: float
    impostor_rejection_rate: float
    false_accept_rate: float
    misidentification_rate: float
    ambiguous_rate: float
    threshold: float
    margin_threshold: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_probes": self.n_probes,
            "n_enrolled_probes": self.n_enrolled_probes,
            "n_impostor_probes": self.n_impostor_probes,
            "top1_accuracy": self.top1_accuracy,
            "top1_accuracy_enrolled_only": self.top1_accuracy_enrolled_only,
            "impostor_rejection_rate": self.impostor_rejection_rate,
            "false_accept_rate": self.false_accept_rate,
            "misidentification_rate": self.misidentification_rate,
            "ambiguous_rate": self.ambiguous_rate,
            "threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
        }

    def summary(self) -> str:
        return (
            "open-set: {0} probes ({1} enrolled, {2} impostor)\n"
            "  top-1 accuracy (all probes)      {3:.4f}\n"
            "  top-1 accuracy (enrolled only)   {4:.4f}\n"
            "  IMPOSTOR REJECTION RATE          {5:.4f}   <- the number that matters\n"
            "  false accept rate (impostors)    {6:.4f}\n"
            "  misidentification (wrong person) {7:.4f}\n"
            "  ambiguous                        {8:.4f}".format(
                self.n_probes,
                self.n_enrolled_probes,
                self.n_impostor_probes,
                self.top1_accuracy,
                self.top1_accuracy_enrolled_only,
                self.impostor_rejection_rate,
                self.false_accept_rate,
                self.misidentification_rate,
                self.ambiguous_rate,
            )
        )


def open_set_report(
    ranked_candidates: Sequence[Sequence[Tuple[str, float]]],
    truth: Sequence[Optional[str]],
    accept_threshold: float,
    margin_threshold: float,
) -> OpenSetReport:
    """Score open-set identification outcomes.

    ``ranked_candidates[i]`` is the per-person scores for probe *i*, best
    first. ``truth[i]`` is the person's id, or ``None`` when the probe is an
    impostor who is not enrolled at all.

    Decisions here mirror ``scoring.decide``: accept only above threshold and
    clear of the runner-up by the margin.
    """
    if len(ranked_candidates) != len(truth):
        raise SpeakerIdError("candidate lists and truth labels have different lengths")
    if not ranked_candidates:
        raise SpeakerIdError("no probes supplied")

    correct = 0
    correct_enrolled = 0
    impostors_rejected = 0
    impostors_accepted = 0
    misidentified = 0
    ambiguous = 0
    n_enrolled = 0
    n_impostor = 0

    for candidates, actual in zip(ranked_candidates, truth):
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        top = ordered[0] if ordered else None
        runner_up = ordered[1] if len(ordered) > 1 else None
        margin = (top[1] - runner_up[1]) if (top and runner_up) else None

        accepted = top is not None and top[1] >= accept_threshold
        is_ambiguous = accepted and margin is not None and margin < margin_threshold
        if is_ambiguous:
            accepted = False
            ambiguous += 1

        predicted = top[0] if accepted and top else None

        if actual is None:
            n_impostor += 1
            if predicted is None:
                impostors_rejected += 1
                correct += 1  # correctly refused to name anybody
            else:
                impostors_accepted += 1
        else:
            n_enrolled += 1
            if predicted == actual:
                correct += 1
                correct_enrolled += 1
            elif predicted is not None:
                misidentified += 1

    total = len(ranked_candidates)
    return OpenSetReport(
        n_probes=total,
        n_enrolled_probes=n_enrolled,
        n_impostor_probes=n_impostor,
        top1_accuracy=correct / float(total),
        top1_accuracy_enrolled_only=(
            correct_enrolled / float(n_enrolled) if n_enrolled else float("nan")
        ),
        impostor_rejection_rate=(
            impostors_rejected / float(n_impostor) if n_impostor else float("nan")
        ),
        false_accept_rate=(impostors_accepted / float(n_impostor) if n_impostor else float("nan")),
        misidentification_rate=(misidentified / float(n_enrolled) if n_enrolled else float("nan")),
        ambiguous_rate=ambiguous / float(total),
        threshold=accept_threshold,
        margin_threshold=margin_threshold,
    )


def format_det_table(curve: DetCurve, rows: int = 10) -> List[str]:
    """A small text DET table, for a report that must print in a terminal."""
    indices = np.linspace(0, curve.thresholds.size - 1, num=min(rows, curve.thresholds.size))
    lines = ["{0:>12} {1:>10} {2:>10}".format("threshold", "FAR", "FRR")]
    for raw_index in indices:
        index = int(round(raw_index))
        lines.append(
            "{0:>12.4f} {1:>10.4f} {2:>10.4f}".format(
                curve.thresholds[index], curve.far[index], curve.frr[index]
            )
        )
    return lines
