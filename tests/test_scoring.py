"""Similarity, AS-Norm, and the decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from speakerid.config import Thresholds
from speakerid.errors import SpeakerIdError
from speakerid.scoring.asnorm import AsNorm, load_cohort_file, save_cohort_file
from speakerid.scoring.decide import Candidate, Outcome, decide
from speakerid.scoring.similarity import (
    aggregate_person_scores,
    check_unit_norm,
    cosine,
    cosine_matrix,
    top_k,
)


def unit(seed: int, dim: int = 192) -> np.ndarray:
    vector = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    vector = unit(0)
    assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_cosine_of_opposite_vectors_is_minus_one():
    vector = unit(0)
    assert cosine(vector, -vector) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_rejects_mismatched_shapes():
    with pytest.raises(SpeakerIdError, match="cannot compare"):
        cosine(np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32))


def test_cosine_rejects_a_zero_vector():
    with pytest.raises(SpeakerIdError, match="zero vector"):
        cosine(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_cosine_matrix_matches_pairwise_cosine():
    probe = unit(0)
    matrix = np.stack([unit(1), unit(2), unit(3)], axis=0)
    scores = cosine_matrix(probe, matrix)
    for index in range(3):
        assert scores[index] == pytest.approx(cosine(probe, matrix[index]), abs=1e-6)


def test_cosine_matrix_on_an_empty_gallery():
    assert cosine_matrix(unit(0), np.zeros((0, 0), dtype=np.float32)).size == 0


def test_cosine_matrix_rejects_a_dimension_mismatch():
    with pytest.raises(SpeakerIdError, match="dimension mismatch"):
        cosine_matrix(np.ones(4, dtype=np.float32), np.ones((3, 8), dtype=np.float32))


def test_check_unit_norm_catches_an_unnormalised_export():
    """A model that stopped normalising would inflate every score silently."""
    with pytest.raises(SpeakerIdError, match="not unit-norm"):
        check_unit_norm(np.full((2, 8), 3.0, dtype=np.float32))


def test_top_k_returns_best_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    indices, values = top_k(scores, 2)
    assert list(indices) == [1, 3]
    assert values[0] == pytest.approx(0.9)


def test_top_k_handles_k_larger_than_the_gallery():
    indices, _values = top_k(np.array([0.5, 0.2], dtype=np.float32), 10)
    assert indices.size == 2


def test_aggregate_uses_max_across_a_persons_enrolments():
    """CLAUDE.md §7: a person matches if the probe resembles any enrolment."""
    scores = np.array([0.2, 0.95, 0.4], dtype=np.float32)
    owners = ["alice", "alice", "bob"]
    collapsed = dict(aggregate_person_scores(scores, owners, method="max"))
    assert collapsed["alice"] == pytest.approx(0.95)
    assert collapsed["bob"] == pytest.approx(0.4)


def test_aggregate_returns_people_best_first():
    scores = np.array([0.2, 0.95], dtype=np.float32)
    assert aggregate_person_scores(scores, ["bob", "alice"])[0][0] == "alice"


def test_aggregate_rejects_an_unknown_method():
    with pytest.raises(SpeakerIdError, match="unknown aggregation"):
        aggregate_person_scores(np.array([0.5]), ["a"], method="median")


# ---------------------------------------------------------------------------
# AS-Norm
# ---------------------------------------------------------------------------


def test_asnorm_disabled_passes_scores_through():
    asnorm = AsNorm(np.zeros((0, 0), dtype=np.float32), top_n=300, epsilon=1e-6)
    assert not asnorm.enabled
    raw = np.array([0.5, 0.2], dtype=np.float32)
    means, stds = asnorm.enrollment_stats(np.stack([unit(1), unit(2)]))
    assert np.array_equal(asnorm.normalize(raw, unit(0), means, stds), raw)


def test_asnorm_produces_z_scores():
    cohort = np.stack([unit(100 + i) for i in range(50)], axis=0)
    asnorm = AsNorm(cohort, top_n=20, epsilon=1e-6)
    gallery = np.stack([unit(1), unit(2)], axis=0)
    means, stds = asnorm.enrollment_stats(gallery)

    # A probe identical to enrolment 0 must normalise far above one that is not.
    probe = gallery[0]
    raw = cosine_matrix(probe, gallery)
    normalized = asnorm.normalize(raw, probe, means, stds)
    assert normalized[0] > normalized[1]
    assert normalized[0] > 3.0, "a self-match should be many sigma above the cohort"


def test_asnorm_separates_genuine_from_impostor_better_than_raw():
    """The reason AS-Norm is worth a matrix multiply (CLAUDE.md §9.3)."""
    base = unit(1)

    def jitter(vector, amount, seed):
        noise = np.random.RandomState(seed).randn(vector.size).astype(np.float32)
        combined = vector + amount * noise
        return combined / np.linalg.norm(combined)

    enrolment = jitter(base, 0.3, 10).reshape(1, -1)
    genuine = jitter(base, 0.3, 11)
    impostor = unit(999)
    cohort = np.stack([unit(500 + i) for i in range(60)], axis=0)

    asnorm = AsNorm(cohort, top_n=30, epsilon=1e-6)
    means, stds = asnorm.enrollment_stats(enrolment)

    raw_gap = cosine(enrolment[0], genuine) - cosine(enrolment[0], impostor)
    normalized_genuine = asnorm.normalize(cosine_matrix(genuine, enrolment), genuine, means, stds)[
        0
    ]
    normalized_impostor = asnorm.normalize(
        cosine_matrix(impostor, enrolment), impostor, means, stds
    )[0]
    assert (normalized_genuine - normalized_impostor) > raw_gap


def test_asnorm_rejects_stale_enrolment_statistics():
    """Statistics must be recomputed when the gallery changes."""
    cohort = np.stack([unit(100 + i) for i in range(30)], axis=0)
    asnorm = AsNorm(cohort, top_n=10, epsilon=1e-6)
    means, stds = asnorm.enrollment_stats(np.stack([unit(1)], axis=0))
    with pytest.raises(SpeakerIdError, match="gallery changed"):
        asnorm.normalize(np.array([0.5, 0.6], dtype=np.float32), unit(0), means, stds)


def test_asnorm_rejects_an_unnormalised_cohort():
    with pytest.raises(SpeakerIdError, match="not unit-norm"):
        AsNorm(np.full((5, 8), 2.0, dtype=np.float32), top_n=3, epsilon=1e-6)


def test_cohort_file_round_trip(tmp_path):
    cohort = np.stack([unit(i) for i in range(10)], axis=0)
    path = str(tmp_path / "cohort.npz")
    save_cohort_file(path, cohort, model_id="redimnet@abc", room_id="lobby")
    assert np.allclose(load_cohort_file(path), cohort)


def test_cohort_file_refuses_a_model_mismatch(tmp_path):
    """A cohort is only valid for the model that produced it."""
    path = str(tmp_path / "cohort.npz")
    save_cohort_file(path, np.stack([unit(i) for i in range(5)]), model_id="model-a")
    with pytest.raises(SpeakerIdError, match="Rebuild the cohort"):
        load_cohort_file(path, expected_model_id="model-b")


def test_missing_cohort_file_explains_the_fix(tmp_path):
    with pytest.raises(SpeakerIdError, match="build_cohort"):
        load_cohort_file(str(tmp_path / "absent.npz"))


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------


@pytest.fixture
def limits() -> Thresholds:
    return Thresholds(
        cm_threshold=0.5, accept_threshold=0.6, margin_threshold=0.1, min_speech_sec=2.0
    )


def test_identified_when_clear_and_live(limits):
    decision = decide(
        [Candidate("alice", 0.9), Candidate("bob", 0.4)],
        limits,
        speech_seconds=4.0,
        cm_score=0.95,
    )
    assert decision.outcome is Outcome.IDENTIFIED
    assert decision.person_id == "alice"
    assert decision.margin == pytest.approx(0.5)


def test_unknown_when_nobody_clears_the_threshold(limits):
    """UNKNOWN is a success, not a failure."""
    decision = decide(
        [Candidate("alice", 0.55), Candidate("bob", 0.2)],
        limits,
        speech_seconds=4.0,
        cm_score=0.95,
    )
    assert decision.outcome is Outcome.UNKNOWN
    assert decision.person_id is None


def test_ambiguous_when_the_top_two_are_too_close(limits):
    decision = decide(
        [Candidate("alice", 0.92), Candidate("bob", 0.90)],
        limits,
        speech_seconds=4.0,
        cm_score=0.95,
    )
    assert decision.outcome is Outcome.AMBIGUOUS
    assert decision.person_id is None, "an ambiguous result must not name anybody"
    assert "alice" in decision.reason and "bob" in decision.reason


def test_rejected_on_insufficient_speech_before_anything_else(limits):
    decision = decide([Candidate("alice", 0.99)], limits, speech_seconds=0.5, cm_score=0.99)
    assert decision.outcome is Outcome.REJECTED
    assert "insufficient_speech" in decision.reason


def test_rejected_on_failed_liveness_names_no_one(limits):
    """A replay must never produce an identity that could be logged."""
    decision = decide([Candidate("alice", 0.99)], limits, speech_seconds=4.0, cm_score=0.05)
    assert decision.outcome is Outcome.REJECTED
    assert "failed_liveness" in decision.reason
    assert decision.person_id is None
    assert decision.score is None


def test_liveness_is_checked_before_identity(limits):
    """Order matters: a spoof with a perfect score is still a spoof."""
    decision = decide([Candidate("alice", 1.0)], limits, speech_seconds=4.0, cm_score=0.49)
    assert decision.outcome is Outcome.REJECTED


def test_failed_challenge_rejects(limits):
    decision = decide(
        [Candidate("alice", 0.99)],
        limits,
        speech_seconds=4.0,
        cm_score=0.99,
        challenge_passed=False,
    )
    assert decision.outcome is Outcome.REJECTED
    assert "failed_challenge" in decision.reason


def test_empty_gallery_rejects_rather_than_identifying(limits):
    decision = decide([], limits, speech_seconds=4.0, cm_score=0.99)
    assert decision.outcome is Outcome.REJECTED
    assert "no_enrolments" in decision.reason


def test_single_enrolled_person_has_no_margin_check(limits):
    decision = decide([Candidate("alice", 0.9)], limits, speech_seconds=4.0, cm_score=0.9)
    assert decision.outcome is Outcome.IDENTIFIED
    assert decision.margin is None


def test_candidates_are_sorted_regardless_of_input_order(limits):
    decision = decide(
        [Candidate("bob", 0.3), Candidate("alice", 0.9)],
        limits,
        speech_seconds=4.0,
        cm_score=0.9,
    )
    assert decision.person_id == "alice"
    assert decision.runner_up_id == "bob"


def test_summary_never_contains_a_name(limits):
    decision = decide(
        [Candidate("alice", 0.9), Candidate("bob", 0.2)],
        limits,
        speech_seconds=4.0,
        cm_score=0.9,
    )
    summary = decision.summary()
    assert "alice" in summary  # the person *id*, which is what logs may carry
    assert "score=" in summary and "cm=" in summary


def test_missing_cm_score_does_not_reject(limits):
    """Liveness absent (not yet scored) is distinct from liveness failed."""
    decision = decide([Candidate("alice", 0.9)], limits, speech_seconds=4.0, cm_score=None)
    assert decision.outcome is Outcome.IDENTIFIED
