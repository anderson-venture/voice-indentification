"""Enrolment store: consent, retention, and complete erasure (CLAUDE.md §11)."""

from __future__ import annotations

import datetime as _datetime
import os

import numpy as np
import pytest

from speakerid.db.schema import utcnow
from speakerid.errors import ConsentError, StoreError


def unit_vector(seed: int, dim: int = 192) -> np.ndarray:
    vector = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def consented_person(store):
    store.add_person(person_id="alice", display_name="Alice Example")
    store.record_consent("alice", policy_version="2026-09-01", granted_by="operator")
    return "alice"


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------


def test_enrolment_without_consent_is_refused(store):
    """No consent row, no enrolment. The enforcement point."""
    store.add_person(person_id="bob")
    with pytest.raises(ConsentError, match="no active consent"):
        store.add_embedding("bob", unit_vector(1), model_id="m1")


def test_enrolment_succeeds_with_consent(store, consented_person):
    embedding_id = store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    assert embedding_id


def test_revoked_consent_blocks_further_enrolment(store, consented_person):
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    store.revoke_consent(consented_person)
    assert not store.has_active_consent(consented_person)
    with pytest.raises(ConsentError):
        store.add_embedding(consented_person, unit_vector(2), model_id="m1")


def test_revoking_consent_does_not_delete_data(store, consented_person):
    """Revocation and erasure are separate acts; the operator must do both."""
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    store.revoke_consent(consented_person)
    assert store.get_person_embeddings(consented_person, "m1") is not None


def test_consent_for_an_unknown_person_is_refused(store):
    with pytest.raises(StoreError, match="unknown person"):
        store.record_consent("ghost", policy_version="1", granted_by="operator")


def test_duplicate_person_is_refused(store):
    store.add_person(person_id="alice")
    with pytest.raises(StoreError, match="already exists"):
        store.add_person(person_id="alice")


# ---------------------------------------------------------------------------
# Embeddings and the gallery
# ---------------------------------------------------------------------------


def test_all_enrolments_are_kept_not_just_the_mean(store, consented_person):
    """CLAUDE.md §7: store every enrolment, score against max."""
    for seed in range(4):
        store.add_embedding(consented_person, unit_vector(seed), model_id="m1")
    person = store.get_person_embeddings(consented_person, "m1")
    assert person.vectors.shape == (4, 192)


def test_round_trip_through_encryption_preserves_vectors(store, consented_person):
    original = unit_vector(7)
    store.add_embedding(consented_person, original, model_id="m1")
    restored = store.get_person_embeddings(consented_person, "m1").vectors[0]
    assert np.allclose(restored, original, atol=1e-6)


def test_embeddings_are_encrypted_on_disk(store, consented_person, base_config):
    """The raw float bytes must not be readable in the database file."""
    original = unit_vector(11)
    store.add_embedding(consented_person, original, model_id="m1")

    url = base_config.get_str("database.url")
    db_path = base_config.resolve_relative(url[len("sqlite:///") :])
    with open(db_path, "rb") as handle:
        contents = handle.read()
    assert original.tobytes() not in contents


def test_centroid_is_unit_norm(store, consented_person):
    for seed in range(3):
        store.add_embedding(consented_person, unit_vector(seed), model_id="m1")
    centroid = store.get_person_embeddings(consented_person, "m1").centroid
    assert float(np.linalg.norm(centroid)) == pytest.approx(1.0, abs=1e-5)


def test_gallery_only_contains_the_requested_model(store, consented_person):
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    store.add_embedding(consented_person, unit_vector(2), model_id="m2")
    assert store.load_gallery("m1").total_embeddings == 1
    assert store.load_gallery("m2").total_embeddings == 1


def test_gallery_stacked_maps_rows_to_owners(store):
    for person, count in (("alice", 2), ("bob", 3)):
        store.add_person(person_id=person)
        store.record_consent(person, policy_version="1", granted_by="operator")
        for seed in range(count):
            store.add_embedding(person, unit_vector(seed), model_id="m1")

    matrix, owners = store.load_gallery("m1").stacked()
    assert matrix.shape == (5, 192)
    assert owners.count("alice") == 2
    assert owners.count("bob") == 3


def test_empty_gallery_is_not_an_error(store):
    gallery = store.load_gallery("m1")
    assert len(gallery) == 0
    matrix, owners = gallery.stacked()
    assert matrix.size == 0 and owners == []


def test_list_persons_reports_counts_and_consent(store, consented_person):
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    listed = store.list_persons()
    assert len(listed) == 1
    assert listed[0]["embeddings"] == 1
    assert listed[0]["consented"] is True


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_every_row_gets_an_expiry(store, consented_person):
    from sqlalchemy import select

    from speakerid.db.schema import EmbeddingRecord

    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    with store.session() as session:
        record = session.scalars(select(EmbeddingRecord)).first()
    assert record.expires_at is not None


def test_purge_removes_expired_rows_only(store, consented_person):
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    # Nothing has expired yet.
    assert store.purge_expired()["embeddings"] == 0
    assert store.get_person_embeddings(consented_person, "m1") is not None

    future = utcnow() + _datetime.timedelta(days=365 * 10)
    assert store.purge_expired(now=future)["embeddings"] == 1
    assert store.get_person_embeddings(consented_person, "m1") is None


def test_purge_deletes_retained_audio_files(store, consented_person, tmp_path):
    audio = tmp_path / "retained.wav"
    audio.write_bytes(b"RIFF....")
    store.register_retained_audio(str(audio), purpose="evaluation", person_id=consented_person)

    future = utcnow() + _datetime.timedelta(days=365 * 10)
    removed = store.purge_expired(now=future)
    assert removed["retained_audio"] == 1
    assert removed["audio_files"] == 1
    assert not audio.exists(), "the file on disk must go with the row"


def test_purge_is_idempotent(store, consented_person):
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    future = utcnow() + _datetime.timedelta(days=365 * 10)
    store.purge_expired(now=future)
    assert store.purge_expired(now=future)["embeddings"] == 0


# ---------------------------------------------------------------------------
# Erasure
# ---------------------------------------------------------------------------


def test_delete_subject_removes_every_trace(store, consented_person, tmp_path):
    """The erasure path, exercised across every table that mentions a person."""
    store.add_embedding(consented_person, unit_vector(1), model_id="m1")
    store.add_embedding(consented_person, unit_vector(2), model_id="m1")
    store.add_cohort_member(unit_vector(3), model_id="m1", person_id=consented_person)
    store.log_event("identified", person_id=consented_person, score=0.9)

    audio = tmp_path / "enrol.wav"
    audio.write_bytes(b"RIFF....")
    store.register_retained_audio(str(audio), purpose="enrollment", person_id=consented_person)

    assert store.subject_trace_count(consented_person) > 0

    report = store.delete_subject(consented_person)

    assert report.embedding_rows == 2
    assert report.cohort_rows == 1
    assert report.event_rows == 1
    assert report.consent_rows == 1
    assert report.person_rows == 1
    assert report.audio_files_deleted == 1
    assert not audio.exists()
    assert store.subject_trace_count(consented_person) == 0, "erasure left something behind"


def test_delete_subject_leaves_other_people_alone(store):
    for person in ("alice", "bob"):
        store.add_person(person_id=person)
        store.record_consent(person, policy_version="1", granted_by="operator")
        store.add_embedding(person, unit_vector(1), model_id="m1")

    store.delete_subject("alice")

    assert store.subject_trace_count("alice") == 0
    assert store.get_person_embeddings("bob", "m1") is not None


def test_delete_subject_of_an_unknown_person_is_harmless(store):
    report = store.delete_subject("never-existed")
    assert report.total_rows == 0


def test_delete_subject_tolerates_an_already_missing_file(store, consented_person, tmp_path):
    audio = tmp_path / "gone.wav"
    audio.write_bytes(b"x")
    store.register_retained_audio(str(audio), purpose="evaluation", person_id=consented_person)
    os.remove(str(audio))

    report = store.delete_subject(consented_person)
    assert report.audio_files_missing == 1
    assert store.subject_trace_count(consented_person) == 0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_record_ids_and_scores(store):
    from sqlalchemy import select

    from speakerid.db.schema import RecognitionEvent

    store.log_event("rejected", cm_score=0.1, reason="failed_liveness: ...")
    with store.session() as session:
        event = session.scalars(select(RecognitionEvent)).first()
    assert event.outcome == "rejected"
    assert event.person_id is None
    # The schema has no column that could hold a name or audio.
    columns = set(RecognitionEvent.__table__.columns.keys())
    assert not columns & {"display_name", "name", "audio", "waveform"}
