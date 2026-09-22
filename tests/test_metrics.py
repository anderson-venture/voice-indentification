"""Verification and open-set identification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from speakerid.errors import SpeakerIdError
from speakerid.metrics import (
    det_curve,
    equal_error_rate,
    far_frr_at_threshold,
    format_det_table,
    min_dcf,
    open_set_report,
    threshold_at_far,
)


def separable_trials(gap: float = 1.0, count: int = 200):
    """Genuine scores around +gap/2, impostor scores around -gap/2."""
    rng = np.random.RandomState(0)
    genuine = list(rng.normal(gap / 2.0, 0.2, count))
    impostor = list(rng.normal(-gap / 2.0, 0.2, count))
    return genuine + impostor, [1] * count + [0] * count


# ---------------------------------------------------------------------------
# DET / EER
# ---------------------------------------------------------------------------


def test_perfect_separation_gives_zero_eer():
    scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    labels = [1, 1, 1, 0, 0, 0]
    eer, _threshold = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.0, abs=1e-9)


def test_random_scores_give_eer_near_one_half():
    rng = np.random.RandomState(0)
    scores = list(rng.rand(400))
    labels = [1] * 200 + [0] * 200
    eer, _ = equal_error_rate(scores, labels)
    assert 0.35 < eer < 0.65


def test_far_is_non_increasing_and_frr_non_decreasing():
    scores, labels = separable_trials()
    curve = det_curve(scores, labels)
    assert np.all(np.diff(curve.far) <= 1e-12)
    assert np.all(np.diff(curve.frr) >= -1e-12)


def test_det_curve_spans_the_full_range():
    scores, labels = separable_trials()
    curve = det_curve(scores, labels)
    assert curve.far[0] == pytest.approx(1.0)
    assert curve.frr[0] == pytest.approx(0.0)
    assert curve.far[-1] == pytest.approx(0.0)
    assert curve.frr[-1] == pytest.approx(1.0)


def test_trials_must_contain_both_classes():
    """A threshold cannot be set from genuine trials alone."""
    with pytest.raises(SpeakerIdError, match="both genuine and impostor"):
        equal_error_rate([0.9, 0.8], [1, 1])


def test_labels_must_be_binary():
    with pytest.raises(SpeakerIdError, match="0 .*or 1"):
        equal_error_rate([0.9, 0.8], [1, 2])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(SpeakerIdError, match="different lengths"):
        equal_error_rate([0.9, 0.8], [1])


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------


def test_threshold_at_far_meets_the_target():
    """The shipped operating point is a target FAR, not the EER (§8)."""
    scores, labels = separable_trials(gap=2.0, count=500)
    threshold, far, _frr = threshold_at_far(scores, labels, 0.01)
    assert far <= 0.01
    measured_far, _ = far_frr_at_threshold(scores, labels, threshold)
    assert measured_far <= 0.01 + 1e-9


def test_tighter_far_gives_a_higher_threshold():
    scores, labels = separable_trials(gap=2.0, count=1000)
    loose, _, _ = threshold_at_far(scores, labels, 0.05)
    tight, _, _ = threshold_at_far(scores, labels, 0.001)
    assert tight >= loose


def test_tighter_far_costs_false_rejects():
    scores, labels = separable_trials(gap=1.0, count=500)
    _, _, loose_frr = threshold_at_far(scores, labels, 0.05)
    _, _, tight_frr = threshold_at_far(scores, labels, 0.002)
    assert tight_frr >= loose_frr


def test_far_finer_than_the_trial_set_can_measure_is_refused():
    """A rate of 1-in-N cannot be measured from fewer than N impostor trials.

    Without this the operator gets an apparent FAR of zero from five trials
    and ships a number nobody measured.
    """
    scores = [0.6, 0.5] * 5 + [0.55] * 5
    labels = [1] * 10 + [0] * 5
    with pytest.raises(SpeakerIdError, match="at least 10000 impostor trials"):
        threshold_at_far(scores, labels, 0.0001)


def test_far_within_the_trial_set_resolution_is_allowed():
    scores, labels = separable_trials(gap=2.0, count=1200)
    threshold, far, _frr = threshold_at_far(scores, labels, 0.001)
    assert far <= 0.001


def test_target_far_must_be_a_probability():
    scores, labels = separable_trials()
    with pytest.raises(SpeakerIdError, match="must be in"):
        threshold_at_far(scores, labels, 1.5)


def test_min_dcf_is_non_negative():
    scores, labels = separable_trials()
    value, _threshold = min_dcf(scores, labels, p_target=0.01)
    assert value >= 0.0


def test_min_dcf_is_lower_for_better_separation():
    good, good_labels = separable_trials(gap=3.0, count=300)
    poor, poor_labels = separable_trials(gap=0.2, count=300)
    assert min_dcf(good, good_labels)[0] < min_dcf(poor, poor_labels)[0]


def test_det_table_formats():
    scores, labels = separable_trials()
    lines = format_det_table(det_curve(scores, labels), rows=5)
    assert lines[0].split() == ["threshold", "FAR", "FRR"]
    assert len(lines) == 6


# ---------------------------------------------------------------------------
# Open-set identification -- the part that decides deployability
# ---------------------------------------------------------------------------


def test_impostor_below_threshold_is_rejected():
    ranked = [[("alice", 0.2), ("bob", 0.1)]]
    report = open_set_report(ranked, [None], accept_threshold=0.6, margin_threshold=0.05)
    assert report.impostor_rejection_rate == pytest.approx(1.0)
    assert report.false_accept_rate == pytest.approx(0.0)


def test_impostor_above_threshold_is_a_false_accept():
    ranked = [[("alice", 0.9), ("bob", 0.1)]]
    report = open_set_report(ranked, [None], accept_threshold=0.6, margin_threshold=0.05)
    assert report.false_accept_rate == pytest.approx(1.0)
    assert report.impostor_rejection_rate == pytest.approx(0.0)


def test_enrolled_speaker_is_counted_correct():
    ranked = [[("alice", 0.9), ("bob", 0.1)]]
    report = open_set_report(ranked, ["alice"], accept_threshold=0.6, margin_threshold=0.05)
    assert report.top1_accuracy_enrolled_only == pytest.approx(1.0)
    assert report.misidentification_rate == pytest.approx(0.0)


def test_wrong_person_is_a_misidentification():
    ranked = [[("bob", 0.9), ("alice", 0.1)]]
    report = open_set_report(ranked, ["alice"], accept_threshold=0.6, margin_threshold=0.05)
    assert report.misidentification_rate == pytest.approx(1.0)


def test_close_runner_up_becomes_ambiguous_not_accepted():
    ranked = [[("alice", 0.90), ("bob", 0.89)]]
    report = open_set_report(ranked, ["alice"], accept_threshold=0.6, margin_threshold=0.05)
    assert report.ambiguous_rate == pytest.approx(1.0)
    assert report.top1_accuracy_enrolled_only == pytest.approx(0.0)


def test_top1_accuracy_can_hide_total_impostor_failure():
    """The headline number people quote, and why CLAUDE.md §8 distrusts it.

    Ten enrolled people all identified correctly, ten strangers all admitted.
    Accuracy over enrolled probes is a perfect 1.0; the system is unusable.
    """
    ranked = [[("alice", 0.95), ("bob", 0.1)]] * 10 + [[("alice", 0.95), ("bob", 0.1)]] * 10
    truth = ["alice"] * 10 + [None] * 10
    report = open_set_report(ranked, truth, accept_threshold=0.6, margin_threshold=0.05)

    assert report.top1_accuracy_enrolled_only == pytest.approx(1.0)
    assert report.impostor_rejection_rate == pytest.approx(0.0)
    assert "IMPOSTOR REJECTION RATE" in report.summary()


def test_report_counts_add_up():
    ranked = [[("alice", 0.9)], [("alice", 0.1)], [("alice", 0.95)]]
    truth = ["alice", None, None]
    report = open_set_report(ranked, truth, accept_threshold=0.6, margin_threshold=0.0)
    assert report.n_probes == 3
    assert report.n_enrolled_probes == 1
    assert report.n_impostor_probes == 2


def test_open_set_requires_matching_lengths():
    with pytest.raises(SpeakerIdError, match="different lengths"):
        open_set_report([[("a", 0.5)]], ["a", "b"], 0.5, 0.0)


def test_open_set_requires_probes():
    with pytest.raises(SpeakerIdError, match="no probes"):
        open_set_report([], [], 0.5, 0.0)
