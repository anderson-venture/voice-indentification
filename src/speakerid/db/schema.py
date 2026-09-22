"""SQLAlchemy schema.

Design notes that are policy, not taste (CLAUDE.md §11):

* No enrolment without a consent row. The foreign key exists so the constraint
  is visible in the schema, and ``store`` enforces it before writing.
* Every embedding and every retained audio file carries ``expires_at``, so
  retention is a property of the row rather than of somebody remembering to
  run a cleanup script.
* ``recognition_events`` records person ids and scores. Never names, never
  audio, never transcripts.
"""

from __future__ import annotations

import datetime as _datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Schema version, bumped whenever a migration is required.
SCHEMA_VERSION = 1


def utcnow() -> _datetime.datetime:
    """Timezone-aware UTC now. All stored timestamps are UTC."""
    return _datetime.datetime.now(_datetime.timezone.utc)


class Base(DeclarativeBase):
    pass


class Person(Base):
    """An enrolled subject.

    ``display_name`` is for the operator's console only. It must never reach a
    log line -- logging uses ``id``.
    """

    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    external_ref: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[_datetime.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    consents: Mapped[List[ConsentRecord]] = relationship(
        "ConsentRecord", back_populates="person", cascade="all, delete-orphan"
    )
    embeddings: Mapped[List[EmbeddingRecord]] = relationship(
        "EmbeddingRecord", back_populates="person", cascade="all, delete-orphan"
    )


class ConsentRecord(Base):
    """Recorded, timestamped consent with the policy version shown.

    No consent row, no enrolment. Revocation is a row update, not a delete, so
    that the operator can prove consent was held at the time of enrolment --
    and ``delete_subject`` then removes the lot when erasure is requested.
    """

    __tablename__ = "consent_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("persons.id", ondelete="CASCADE"), index=True, nullable=False
    )
    granted_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    method: Mapped[str] = mapped_column(String(32), default="in_person", nullable=False)
    revoked_at: Mapped[Optional[_datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    person: Mapped[Person] = relationship("Person", back_populates="consents")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class EmbeddingRecord(Base):
    """One enrolment utterance's voiceprint, encrypted.

    All of a person's embeddings are kept, not just the mean: scoring uses
    max-cosine across them, and the centroid is derived on demand for
    diagnostics only (CLAUDE.md §7).
    """

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("persons.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    expires_at: Mapped[Optional[_datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    # Capture-quality provenance, kept so a later accuracy complaint can be
    # traced to a bad enrolment rather than guessed at.
    speech_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snr_db: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    room_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    person: Mapped[Person] = relationship("Person", back_populates="embeddings")


class CohortMember(Base):
    """A non-enrolled embedding used for AS-Norm.

    Tracked in the database rather than only in the .npz so that
    ``delete_subject`` can remove a person's cohort membership too, which is
    required for a complete erasure (CLAUDE.md §11).
    """

    __tablename__ = "cohort_members"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    person_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    room_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    expires_at: Mapped[Optional[_datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class RetainedAudio(Base):
    """An audio file deliberately kept for evaluation.

    Raw audio is deleted immediately after embedding extraction unless an
    explicit ``--retain-audio`` flag is set. When it is set, the file is
    recorded here so retention and erasure can both reach it.
    """

    __tablename__ = "retained_audio"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    person_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    expires_at: Mapped[Optional[_datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class RecognitionEvent(Base):
    """Audit trail of decisions. IDs and scores only -- no names, no audio."""

    __tablename__ = "recognition_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    occurred_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    person_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    runner_up_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    runner_up_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cm_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speech_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    challenge_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    expires_at: Mapped[Optional[_datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class SchemaMeta(Base):
    """Single-row table recording the schema version of this database."""

    __tablename__ = "schema_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[_datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


Index("ix_embeddings_person_model", EmbeddingRecord.person_id, EmbeddingRecord.model_id)
Index("ix_events_outcome_time", RecognitionEvent.outcome, RecognitionEvent.occurred_at)
