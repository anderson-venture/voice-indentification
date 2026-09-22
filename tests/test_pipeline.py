"""End-to-end pipeline behaviour on the fixtures.

This is the test that answers "does the thing work": enrol two synthetic
speakers, then check each of the four outcomes arises from the audio that
should produce it, and that the stage ordering in CLAUDE.md §6 actually holds.
"""

from __future__ import annotations

import pytest

from speakerid.audio.io import load_audio
from speakerid.errors import ConsentError
from speakerid.scoring.decide import Outcome

# ---------------------------------------------------------------------------
# The four outcomes
# ---------------------------------------------------------------------------


def test_enrolled_speaker_is_identified(enrolled_pipeline, fixture_path):
    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert result.outcome is Outcome.IDENTIFIED, result.decision.reason
    assert result.decision.person_id == "alice"


def test_the_other_enrolled_speaker_is_not_confused(enrolled_pipeline, fixture_path):
    enrolled_pipeline.store.add_person(person_id="bob2")
    result = enrolled_pipeline.identify_file(fixture_path("bob_02.wav"))
    assert result.decision.person_id == "bob"


def test_a_stranger_returns_unknown(enrolled_pipeline, fixture_path):
    """Carol is never enrolled. Refusing to name her is the correct answer."""
    result = enrolled_pipeline.identify_file(fixture_path("carol_01.wav"))
    assert result.outcome is Outcome.UNKNOWN, result.decision.reason
    assert result.decision.person_id is None


def test_a_replay_is_rejected(enrolled_pipeline, fixture_path):
    """The replay still *sounds* like Alice -- that is the point.

    Its embedding is close to hers (cosine ~0.97), so identity alone would
    admit it. Only the liveness stage stops it.
    """
    result = enrolled_pipeline.identify_file(fixture_path("alice_01_replay.wav"))
    assert result.outcome is Outcome.REJECTED, result.decision.reason
    assert "failed_liveness" in result.decision.reason
    assert result.decision.person_id is None


def test_too_short_is_rejected(enrolled_pipeline, fixture_path):
    result = enrolled_pipeline.identify_file(fixture_path("too_short.wav"))
    assert result.outcome is Outcome.REJECTED
    assert "insufficient_speech" in result.decision.reason


def test_room_tone_is_rejected(enrolled_pipeline, fixture_path):
    """Non-speech must never reach the embedder (CLAUDE.md §6)."""
    result = enrolled_pipeline.identify_file(fixture_path("room_tone.wav"))
    assert result.outcome is Outcome.REJECTED
    assert "insufficient_speech" in result.decision.reason
    assert result.embedding is None


def test_empty_gallery_rejects(pipeline, fixture_path):
    result = pipeline.identify_file(fixture_path("alice_01.wav"))
    assert result.outcome is Outcome.REJECTED
    assert "no_enrolments" in result.decision.reason


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------


def test_countermeasure_runs_before_the_embedder_and_short_circuits(
    enrolled_pipeline, fixture_path
):
    """CLAUDE.md §6: the CM is 40x cheaper, so a rejection must skip the embedder."""
    embedder_session = enrolled_pipeline.features.embedder.session
    cm_session = enrolled_pipeline.features.countermeasure.session

    embed_before = embedder_session.run_count
    cm_before = cm_session.run_count

    result = enrolled_pipeline.identify_file(fixture_path("alice_01_replay.wav"))

    assert result.outcome is Outcome.REJECTED
    assert cm_session.run_count > cm_before, "the countermeasure must have run"
    assert (
        embedder_session.run_count == embed_before
    ), "the embedder ran on audio the countermeasure had already rejected"


def test_accepted_audio_does_reach_the_embedder(enrolled_pipeline, fixture_path):
    embedder_session = enrolled_pipeline.features.embedder.session
    before = embedder_session.run_count
    enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert embedder_session.run_count > before


def test_insufficient_speech_skips_both_models(enrolled_pipeline, fixture_path):
    embedder_session = enrolled_pipeline.features.embedder.session
    cm_session = enrolled_pipeline.features.countermeasure.session
    embed_before, cm_before = embedder_session.run_count, cm_session.run_count

    enrolled_pipeline.identify_file(fixture_path("room_tone.wav"))

    assert embedder_session.run_count == embed_before
    assert cm_session.run_count == cm_before


# ---------------------------------------------------------------------------
# Enrolment gates (CLAUDE.md §7)
# ---------------------------------------------------------------------------


def test_enrolment_requires_consent(pipeline, fixture_path):
    pipeline.store.add_person(person_id="dave")
    with pytest.raises(ConsentError):
        pipeline.enroll_file("dave", fixture_path("dave_01.wav"))


def test_enrolment_rejects_too_little_speech(pipeline, fixture_path):
    pipeline.store.add_person(person_id="dave")
    pipeline.store.record_consent("dave", policy_version="1", granted_by="tester")
    result = pipeline.enroll_file("dave", fixture_path("too_short.wav"))
    assert not result.accepted
    assert "speech" in result.reason


def test_enrolment_rejects_the_wrong_speaker(enrolled_pipeline, fixture_path):
    """A recording of somebody else must not be added to a person's row."""
    result = enrolled_pipeline.enroll_file("alice", fixture_path("carol_01.wav"))
    assert not result.accepted
    assert "wrong speaker" in result.reason
    assert result.self_cosine is not None


def test_enrolment_caps_the_number_of_utterances(enrolled_pipeline, fixture_path):
    # The fixture already enrolled three; the configured maximum is five.
    assert enrolled_pipeline.enroll_file("alice", fixture_path("alice_04.wav")).accepted
    # A fifth is still allowed, a sixth is not.
    enrolled_pipeline.enroll_file("alice", fixture_path("alice_02.wav"))
    result = enrolled_pipeline.enroll_file("alice", fixture_path("alice_03.wav"))
    assert not result.accepted
    assert "max" in result.reason


def test_enrolment_status_tracks_completeness(pipeline, fixture_path):
    pipeline.store.add_person(person_id="alice")
    pipeline.store.record_consent("alice", policy_version="1", granted_by="tester")
    assert not pipeline.enrollment_status("alice")["complete"]

    for name in ("alice_01.wav", "alice_02.wav", "alice_03.wav"):
        pipeline.enroll_file("alice", fixture_path(name))
    status = pipeline.enrollment_status("alice")
    assert status["complete"]
    assert status["embeddings"] == 3
    assert status["self_consistency"] > 0.9


def test_enrolment_refreshes_the_gallery(pipeline, fixture_path):
    pipeline.store.add_person(person_id="alice")
    pipeline.store.record_consent("alice", policy_version="1", granted_by="tester")
    assert len(pipeline.gallery) == 0
    pipeline.enroll_file("alice", fixture_path("alice_01.wav"))
    assert len(pipeline.gallery) == 1


def test_retained_audio_is_only_registered_on_request(pipeline, fixture_path):
    from sqlalchemy import select

    from speakerid.db.schema import RetainedAudio

    pipeline.store.add_person(person_id="alice")
    pipeline.store.record_consent("alice", policy_version="1", granted_by="tester")
    pipeline.enroll_file("alice", fixture_path("alice_01.wav"))

    with pipeline.store.session() as session:
        assert session.scalars(select(RetainedAudio)).all() == []

    pipeline.enroll_file("alice", fixture_path("alice_02.wav"), retain_audio=True)
    with pipeline.store.session() as session:
        assert len(session.scalars(select(RetainedAudio)).all()) == 1


# ---------------------------------------------------------------------------
# Audit and erasure
# ---------------------------------------------------------------------------


def test_every_decision_writes_an_event(enrolled_pipeline, fixture_path):
    from sqlalchemy import select

    from speakerid.db.schema import RecognitionEvent

    enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    with enrolled_pipeline.store.session() as session:
        events = session.scalars(select(RecognitionEvent)).all()
    assert events, "decisions must be auditable"
    assert events[-1].outcome in {outcome.value for outcome in Outcome}


def test_deleting_a_person_stops_them_being_identified(enrolled_pipeline, fixture_path):
    assert enrolled_pipeline.identify_file(fixture_path("alice_04.wav")).decision.person_id == (
        "alice"
    )

    enrolled_pipeline.store.delete_subject("alice")
    enrolled_pipeline.refresh_gallery()

    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert result.decision.person_id != "alice"


# ---------------------------------------------------------------------------
# Timings and plumbing
# ---------------------------------------------------------------------------


def test_timings_are_populated(enrolled_pipeline, fixture_path):
    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert result.timings.total_ms > 0
    assert result.timings.vad_ms > 0
    assert "total=" in result.timings.summary()


def test_identify_accepts_a_raw_buffer(enrolled_pipeline, fixture_path):
    samples = load_audio(fixture_path("alice_04.wav"), peak_dbfs=-3.0)
    result = enrolled_pipeline.identify(samples)
    assert result.decision.person_id == "alice"


def test_model_id_changes_with_the_model_file(tmp_path):
    from speakerid.pipeline import derive_model_id

    first = tmp_path / "m.onnx"
    first.write_bytes(b"version one")
    original = derive_model_id(str(first))

    first.write_bytes(b"version two")
    assert (
        derive_model_id(str(first)) != original
    ), "a re-export must not be silently scored against the old gallery"


def test_model_id_can_be_pinned_explicitly(tmp_path):
    from speakerid.pipeline import derive_model_id

    path = tmp_path / "m.onnx"
    path.write_bytes(b"whatever")
    assert derive_model_id(str(path), override="pinned-id") == "pinned-id"
