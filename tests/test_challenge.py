"""Challenge-response.

The defence that actually stops replay attacks (CLAUDE.md §10). The most
important behaviours here are the two failure modes: no verifier means no
pass, and a token can only ever be used once.
"""

from __future__ import annotations

import numpy as np
import pytest

from speakerid.challenge import ChallengeManager
from speakerid.errors import ChallengeError


def build(**overrides) -> ChallengeManager:
    settings = dict(
        enabled=True,
        num_digits=5,
        alphabet="0123456789",
        ttl_seconds=30.0,
        require_pass=True,
        max_attempts=3,
        verifier=None,
    )
    settings.update(overrides)
    return ChallengeManager(**settings)


AUDIO = np.zeros(16000, dtype=np.float32)


def accepts_everything(speech, digits):
    return True


def test_issued_digits_have_the_configured_length():
    challenge = build().issue()
    assert len(challenge.digits) == 5
    assert challenge.digits.isdigit()


def test_challenges_are_unpredictable():
    """A predictable sequence would let an attacker pre-record the prompt."""
    manager = build()
    issued = {manager.issue().digits for _ in range(50)}
    assert len(issued) > 25, "digits repeat far too often to be random"


def test_tokens_are_unique():
    manager = build()
    assert len({manager.issue().token for _ in range(20)}) == 20


def test_prompt_is_spaced_for_reading_aloud():
    challenge = build(num_digits=4).issue()
    assert challenge.prompt() == " ".join(challenge.digits)


def test_no_verifier_fails_closed():
    """An unverifiable challenge is a failed challenge, never a pass."""
    manager = build(verifier=None)
    challenge = manager.issue()
    assert manager.verify(challenge.token, AUDIO) is False


def test_verifier_result_is_honoured():
    manager = build(verifier=accepts_everything)
    challenge = manager.issue()
    assert manager.verify(challenge.token, AUDIO) is True


def test_wrong_answer_returns_false_without_raising():
    manager = build(verifier=lambda speech, digits: False)
    challenge = manager.issue()
    assert manager.verify(challenge.token, AUDIO) is False


def test_token_is_single_use():
    """Replaying an entire prior session -- same audio, same token -- fails."""
    manager = build(verifier=accepts_everything)
    challenge = manager.issue()
    assert manager.verify(challenge.token, AUDIO) is True
    with pytest.raises(ChallengeError, match="unknown or already-consumed"):
        manager.verify(challenge.token, AUDIO)


def test_unknown_token_is_refused():
    with pytest.raises(ChallengeError, match="unknown"):
        build().verify("not-a-real-token", AUDIO)


def test_expired_challenge_is_refused():
    manager = build(ttl_seconds=10.0, verifier=accepts_everything)
    challenge = manager.issue(now=1000.0)
    with pytest.raises(ChallengeError, match="expired"):
        manager.verify(challenge.token, AUDIO, now=1011.0)


def test_challenge_valid_just_before_expiry():
    manager = build(ttl_seconds=10.0, verifier=accepts_everything)
    challenge = manager.issue(now=1000.0)
    assert manager.verify(challenge.token, AUDIO, now=1009.0) is True


def test_attempts_are_capped():
    manager = build(max_attempts=2, verifier=lambda speech, digits: False)
    challenge = manager.issue()
    manager.verify(challenge.token, AUDIO)
    manager.verify(challenge.token, AUDIO)
    with pytest.raises(ChallengeError, match="exceeded"):
        manager.verify(challenge.token, AUDIO)


def test_resolve_returns_none_when_disabled():
    """Challenge not in use, so the decision must not consider it."""
    assert build(enabled=False).resolve(None, AUDIO) is None


def test_resolve_fails_closed_when_required_but_absent():
    assert build(enabled=True, require_pass=True).resolve(None, AUDIO) is False


def test_resolve_returns_none_when_optional_and_absent():
    assert build(enabled=True, require_pass=False).resolve(None, AUDIO) is None


def test_expired_challenges_are_purged():
    manager = build(ttl_seconds=5.0)
    manager.issue(now=100.0)
    manager.issue(now=100.0)
    assert len(manager.outstanding) == 2
    assert manager.purge_expired(now=200.0) == 2
    assert manager.outstanding == []


def test_configuration_is_validated():
    with pytest.raises(ChallengeError, match="num_digits"):
        build(num_digits=0)
    with pytest.raises(ChallengeError, match="alphabet"):
        build(alphabet="1")


def test_pipeline_rejects_when_challenge_required_but_missing(enrolled_pipeline, fixture_path):
    """Wired end to end: a required challenge with no token is a rejection."""
    from speakerid.scoring.decide import Outcome

    enrolled_pipeline.challenges = build(enabled=True, require_pass=True, verifier=None)
    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert result.outcome is Outcome.REJECTED
    assert "failed_challenge" in result.decision.reason


def test_pipeline_identifies_when_challenge_passes(enrolled_pipeline, fixture_path):
    from speakerid.scoring.decide import Outcome

    manager = build(enabled=True, require_pass=True, verifier=accepts_everything)
    enrolled_pipeline.challenges = manager
    token = manager.issue().token
    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"), challenge_token=token)
    assert result.outcome is Outcome.IDENTIFIED
    assert result.decision.challenge_passed is True
