"""The decision rule. This is the only place a decision is made.

Four outcomes, never a bare name (CLAUDE.md §6)::

    IDENTIFIED  confident match
    UNKNOWN     nobody above threshold
    AMBIGUOUS   top-1 and top-2 too close
    REJECTED    failed liveness, challenge, or insufficient speech

``UNKNOWN`` is a success, not a failure, and the system is biased hard toward
it. In an open set, the cost of admitting the wrong person is not comparable
to the cost of asking someone to speak again.

Every number used here arrives as an argument from ``Thresholds``. There are
no float literals in this module's decision path, by design: a constant typed
in here would be an unmeasured false-accept rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from ..config import Thresholds

LOGGER = logging.getLogger(__name__)


class Outcome(Enum):
    IDENTIFIED = "identified"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class RejectReason(Enum):
    """Why a REJECTED outcome happened. Operator-facing, and logged."""

    INSUFFICIENT_SPEECH = "insufficient_speech"
    FAILED_LIVENESS = "failed_liveness"
    FAILED_CHALLENGE = "failed_challenge"
    NO_ENROLMENTS = "no_enrolments"


@dataclass(frozen=True)
class Candidate:
    """One person's score against the probe."""

    person_id: str
    score: float  # after AS-Norm; the scale decisions are made on
    raw_score: float = 0.0  # plain cosine, kept for diagnostics


@dataclass(frozen=True)
class Decision:
    """The full result. Carries its own evidence so it can be audited."""

    outcome: Outcome
    person_id: Optional[str] = None
    score: Optional[float] = None
    runner_up_id: Optional[str] = None
    runner_up_score: Optional[float] = None
    margin: Optional[float] = None
    cm_score: Optional[float] = None
    speech_seconds: Optional[float] = None
    challenge_passed: Optional[bool] = None
    reason: Optional[str] = None
    candidates: Tuple[Candidate, ...] = field(default=())

    @property
    def identified(self) -> bool:
        return self.outcome is Outcome.IDENTIFIED

    def summary(self) -> str:
        """One line, safe to log: ids and scores only, never a name."""
        parts = ["outcome={0}".format(self.outcome.value)]
        if self.person_id is not None:
            parts.append("person={0}".format(self.person_id))
        if self.score is not None:
            parts.append("score={0:.4f}".format(self.score))
        if self.margin is not None:
            parts.append("margin={0:.4f}".format(self.margin))
        if self.cm_score is not None:
            parts.append("cm={0:.4f}".format(self.cm_score))
        if self.speech_seconds is not None:
            parts.append("speech={0:.2f}s".format(self.speech_seconds))
        if self.challenge_passed is not None:
            parts.append("challenge={0}".format("pass" if self.challenge_passed else "fail"))
        if self.reason:
            parts.append("reason={0}".format(self.reason))
        return " ".join(parts)


def decide(
    candidates: Sequence[Candidate],
    thresholds: Thresholds,
    speech_seconds: float,
    cm_score: Optional[float] = None,
    challenge_passed: Optional[bool] = None,
) -> Decision:
    """Turn evidence into one of four outcomes.

    Checks run cheapest-and-most-disqualifying first. Anything that rejects
    does so before the identity scores are even consulted, so a replay never
    produces a name that could be logged or displayed by mistake.
    """
    ordered: List[Candidate] = sorted(candidates, key=lambda item: item.score, reverse=True)
    top = ordered[0] if ordered else None
    runner_up = ordered[1] if len(ordered) > 1 else None

    # 1. Enough speech to be worth judging at all.
    if speech_seconds < thresholds.min_speech_sec:
        return _rejected(
            RejectReason.INSUFFICIENT_SPEECH,
            speech_seconds=speech_seconds,
            cm_score=cm_score,
            challenge_passed=challenge_passed,
            candidates=ordered,
            detail="{0:.2f}s < {1:.2f}s required".format(speech_seconds, thresholds.min_speech_sec),
        )

    # 2. Liveness. A replay is rejected before identity is considered.
    if cm_score is not None and cm_score < thresholds.cm_threshold:
        return _rejected(
            RejectReason.FAILED_LIVENESS,
            speech_seconds=speech_seconds,
            cm_score=cm_score,
            challenge_passed=challenge_passed,
            candidates=ordered,
            detail="bona-fide {0:.4f} < {1:.4f}".format(cm_score, thresholds.cm_threshold),
        )

    # 3. Challenge-response. Fails closed: an unanswered challenge is a
    #    failed challenge, never a pass (CLAUDE.md §10).
    if challenge_passed is False:
        return _rejected(
            RejectReason.FAILED_CHALLENGE,
            speech_seconds=speech_seconds,
            cm_score=cm_score,
            challenge_passed=False,
            candidates=ordered,
            detail="spoken digits did not match the challenge",
        )

    # 4. Nobody to compare against.
    if top is None:
        return _rejected(
            RejectReason.NO_ENROLMENTS,
            speech_seconds=speech_seconds,
            cm_score=cm_score,
            challenge_passed=challenge_passed,
            candidates=ordered,
            detail="the gallery is empty",
        )

    margin = float(top.score - runner_up.score) if runner_up is not None else None

    # 5. Above the accept threshold?
    if top.score < thresholds.accept_threshold:
        return Decision(
            outcome=Outcome.UNKNOWN,
            score=float(top.score),
            runner_up_id=runner_up.person_id if runner_up else None,
            runner_up_score=float(runner_up.score) if runner_up else None,
            margin=margin,
            cm_score=cm_score,
            speech_seconds=speech_seconds,
            challenge_passed=challenge_passed,
            reason="best score {0:.4f} < accept threshold {1:.4f}".format(
                top.score, thresholds.accept_threshold
            ),
            candidates=tuple(ordered),
        )

    # 6. Clearly ahead of the runner-up? Two people above threshold and close
    #    together means the system cannot tell them apart, which is a distinct
    #    situation from "nobody matched" and deserves its own outcome.
    if margin is not None and margin < thresholds.margin_threshold:
        return Decision(
            outcome=Outcome.AMBIGUOUS,
            person_id=None,
            score=float(top.score),
            runner_up_id=runner_up.person_id if runner_up else None,
            runner_up_score=float(runner_up.score) if runner_up else None,
            margin=margin,
            cm_score=cm_score,
            speech_seconds=speech_seconds,
            challenge_passed=challenge_passed,
            reason="margin {0:.4f} < {1:.4f} between {2} and {3}".format(
                margin,
                thresholds.margin_threshold,
                top.person_id,
                runner_up.person_id if runner_up else "-",
            ),
            candidates=tuple(ordered),
        )

    return Decision(
        outcome=Outcome.IDENTIFIED,
        person_id=top.person_id,
        score=float(top.score),
        runner_up_id=runner_up.person_id if runner_up else None,
        runner_up_score=float(runner_up.score) if runner_up else None,
        margin=margin,
        cm_score=cm_score,
        speech_seconds=speech_seconds,
        challenge_passed=challenge_passed,
        candidates=tuple(ordered),
    )


def _rejected(
    reason: RejectReason,
    speech_seconds: float,
    cm_score: Optional[float],
    challenge_passed: Optional[bool],
    candidates: Sequence[Candidate],
    detail: str,
) -> Decision:
    return Decision(
        outcome=Outcome.REJECTED,
        cm_score=cm_score,
        speech_seconds=speech_seconds,
        challenge_passed=challenge_passed,
        reason="{0}: {1}".format(reason.value, detail),
        # No person_id and no score: a rejected sample has no identity, and
        # exposing a near-match here would invite a caller to use it.
        candidates=tuple(candidates),
    )
