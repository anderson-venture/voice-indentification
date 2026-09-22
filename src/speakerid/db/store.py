"""Enrolment store: enroll / delete / list / purge.

Every privacy rule in CLAUDE.md §11 that can be expressed in code is expressed
here, so that compliance is a property of the write path rather than of
operator discipline:

* :meth:`Store.add_embedding` refuses to write without an active consent row.
* Every written row gets an ``expires_at`` from ``database.retention_months``.
* :meth:`Store.purge_expired` runs on startup and on a timer.
* :meth:`Store.delete_subject` removes embeddings, retained audio (files
  included), events and cohort membership, and reports what it removed.
"""

from __future__ import annotations

import datetime as _datetime
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import Config
from ..errors import ConsentError, StoreError
from ..nptypes import FloatArray
from .crypto import EmbeddingCipher, subject_aad
from .schema import (
    SCHEMA_VERSION,
    Base,
    CohortMember,
    ConsentRecord,
    EmbeddingRecord,
    Person,
    RecognitionEvent,
    RetainedAudio,
    SchemaMeta,
    utcnow,
)

LOGGER = logging.getLogger(__name__)

#: Approximate days per month, used to turn a retention policy in months into
#: a concrete expiry timestamp.
_DAYS_PER_MONTH = 30.44


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class PersonEmbeddings:
    """A person's enrolment vectors, decrypted and ready to score against."""

    person_id: str
    vectors: FloatArray  # shape (n_utterances, dim), each row unit-norm
    embedding_ids: Tuple[str, ...] = ()

    @property
    def centroid(self) -> FloatArray:
        """Mean vector, re-normalized. Diagnostics only -- never the score.

        Scoring uses ``max`` over the individual enrolments (CLAUDE.md §7);
        the centroid is kept to spot a person whose enrolments disagree with
        each other, which usually means a bad capture.
        """
        mean = self.vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            raise StoreError("degenerate centroid for person {0}".format(self.person_id))
        return np.ascontiguousarray(mean / norm, dtype=np.float32)

    @property
    def self_consistency(self) -> float:
        """Mean pairwise cosine among a person's own enrolments."""
        if self.vectors.shape[0] < 2:
            return float("nan")
        similarities = self.vectors @ self.vectors.T
        upper = similarities[np.triu_indices(self.vectors.shape[0], k=1)]
        return float(np.mean(upper))


@dataclass
class Gallery:
    """All enrolled people for one model, in memory, ready for scoring."""

    model_id: str
    people: List[PersonEmbeddings] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.people)

    @property
    def person_ids(self) -> List[str]:
        return [person.person_id for person in self.people]

    @property
    def total_embeddings(self) -> int:
        return int(sum(person.vectors.shape[0] for person in self.people))

    def stacked(self) -> Tuple[FloatArray, List[str]]:
        """Return (N, dim) of every enrolment vector plus its owner's id."""
        if not self.people:
            return np.zeros((0, 0), dtype=np.float32), []
        vectors = np.concatenate([person.vectors for person in self.people], axis=0)
        owners: List[str] = []
        for person in self.people:
            owners.extend([person.person_id] * person.vectors.shape[0])
        return np.ascontiguousarray(vectors, dtype=np.float32), owners


@dataclass
class DeletionReport:
    """What ``delete_subject`` actually removed. Returned so it can be tested."""

    person_id: str
    person_rows: int = 0
    consent_rows: int = 0
    embedding_rows: int = 0
    cohort_rows: int = 0
    event_rows: int = 0
    audio_rows: int = 0
    audio_files_deleted: int = 0
    audio_files_missing: int = 0

    @property
    def total_rows(self) -> int:
        return (
            self.person_rows
            + self.consent_rows
            + self.embedding_rows
            + self.cohort_rows
            + self.event_rows
            + self.audio_rows
        )


class Store(object):
    """Encrypted enrolment database."""

    def __init__(
        self,
        engine: Engine,
        cipher: EmbeddingCipher,
        retention_months: int,
        create: bool = True,
    ) -> None:
        self.engine = engine
        self.cipher = cipher
        self.retention_months = retention_months
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        self._purge_timer: Optional[threading.Timer] = None
        if create:
            self.create_all()

    @classmethod
    def from_config(
        cls,
        config: Config,
        cipher: Optional[EmbeddingCipher] = None,
        passphrase: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Store:
        database_url = url or config.get_str("database.url")
        database_url = _resolve_sqlite_path(database_url, config)
        engine = create_engine(database_url, future=True)
        resolved_cipher = cipher or EmbeddingCipher.from_config(config, passphrase)
        return cls(
            engine=engine,
            cipher=resolved_cipher,
            retention_months=config.get_int("database.retention_months"),
        )

    # -- lifecycle -----------------------------------------------------------

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        with self._session_factory() as session:
            existing = session.get(SchemaMeta, 1)
            if existing is None:
                session.add(SchemaMeta(id=1, version=SCHEMA_VERSION))
                session.commit()
            elif existing.version != SCHEMA_VERSION:
                raise StoreError(
                    "database schema version {0} does not match this build's {1}; "
                    "a migration is required".format(existing.version, SCHEMA_VERSION)
                )

    def session(self) -> Session:
        return self._session_factory()

    def close(self) -> None:
        self.stop_purge_timer()
        self.engine.dispose()

    # -- retention -----------------------------------------------------------

    def _expiry(self, now: Optional[_datetime.datetime] = None) -> Optional[_datetime.datetime]:
        if self.retention_months <= 0:
            return None
        moment = now or utcnow()
        return moment + _datetime.timedelta(days=self.retention_months * _DAYS_PER_MONTH)

    def purge_expired(self, now: Optional[_datetime.datetime] = None) -> Dict[str, int]:
        """Delete every row past its retention date. Safe to call repeatedly."""
        moment = now or utcnow()
        removed: Dict[str, int] = {}
        with self._session_factory() as session:
            # Retained audio first, so the files on disk go with the rows.
            expired_audio = session.scalars(
                select(RetainedAudio).where(
                    RetainedAudio.expires_at.is_not(None), RetainedAudio.expires_at <= moment
                )
            ).all()
            files_removed = 0
            for record in expired_audio:
                if _unlink_quietly(record.path):
                    files_removed += 1
                session.delete(record)
            removed["retained_audio"] = len(expired_audio)
            removed["audio_files"] = files_removed

            for model, label in (
                (EmbeddingRecord, "embeddings"),
                (CohortMember, "cohort_members"),
                (RecognitionEvent, "recognition_events"),
            ):
                result = session.execute(
                    delete(model).where(model.expires_at.is_not(None), model.expires_at <= moment)
                )
                removed[label] = int(result.rowcount or 0)
            session.commit()

        if any(count for count in removed.values()):
            LOGGER.info("retention purge removed %s", removed)
        else:
            LOGGER.debug("retention purge: nothing expired")
        return removed

    def start_purge_timer(self, interval_seconds: float) -> None:
        """Run ``purge_expired`` repeatedly in a daemon thread."""
        if interval_seconds <= 0:
            return

        def _tick() -> None:
            try:
                self.purge_expired()
            except Exception:  # never let the timer thread die silently
                LOGGER.exception("scheduled retention purge failed")
            finally:
                self.start_purge_timer(interval_seconds)

        timer = threading.Timer(interval_seconds, _tick)
        timer.daemon = True
        timer.start()
        self._purge_timer = timer

    def stop_purge_timer(self) -> None:
        if self._purge_timer is not None:
            self._purge_timer.cancel()
            self._purge_timer = None

    # -- people and consent --------------------------------------------------

    def add_person(
        self,
        person_id: Optional[str] = None,
        display_name: Optional[str] = None,
        external_ref: Optional[str] = None,
    ) -> str:
        identifier = person_id or new_id()
        with self._session_factory() as session:
            if session.get(Person, identifier) is not None:
                raise StoreError("person {0} already exists".format(identifier))
            session.add(Person(id=identifier, display_name=display_name, external_ref=external_ref))
            session.commit()
        LOGGER.info("created person %s", identifier)  # id only, never the name
        return identifier

    def record_consent(
        self,
        person_id: str,
        policy_version: str,
        granted_by: Optional[str] = None,
        method: str = "in_person",
        notes: Optional[str] = None,
        granted_at: Optional[_datetime.datetime] = None,
    ) -> str:
        with self._session_factory() as session:
            if session.get(Person, person_id) is None:
                raise StoreError("unknown person {0}".format(person_id))
            record = ConsentRecord(
                id=new_id(),
                person_id=person_id,
                policy_version=policy_version,
                granted_by=granted_by,
                method=method,
                notes=notes,
                granted_at=granted_at or utcnow(),
            )
            session.add(record)
            session.commit()
            LOGGER.info("recorded consent for person %s under policy %s", person_id, policy_version)
            return record.id

    def revoke_consent(self, person_id: str, when: Optional[_datetime.datetime] = None) -> int:
        """Mark consent withdrawn. Does not delete -- see ``delete_subject``."""
        moment = when or utcnow()
        with self._session_factory() as session:
            records = session.scalars(
                select(ConsentRecord).where(
                    ConsentRecord.person_id == person_id, ConsentRecord.revoked_at.is_(None)
                )
            ).all()
            for record in records:
                record.revoked_at = moment
            session.commit()
        LOGGER.info("revoked %d consent record(s) for person %s", len(records), person_id)
        return len(records)

    def has_active_consent(self, person_id: str) -> bool:
        with self._session_factory() as session:
            found = session.scalars(
                select(ConsentRecord.id).where(
                    ConsentRecord.person_id == person_id, ConsentRecord.revoked_at.is_(None)
                )
            ).first()
            return found is not None

    def list_persons(self, include_inactive: bool = False) -> List[Dict[str, object]]:
        with self._session_factory() as session:
            statement = select(Person)
            if not include_inactive:
                statement = statement.where(Person.active.is_(True))
            people = session.scalars(statement.order_by(Person.created_at)).all()
            summary: List[Dict[str, object]] = []
            for person in people:
                count = len(
                    session.scalars(
                        select(EmbeddingRecord.id).where(EmbeddingRecord.person_id == person.id)
                    ).all()
                )
                summary.append(
                    {
                        "person_id": person.id,
                        "display_name": person.display_name,
                        "external_ref": person.external_ref,
                        "embeddings": count,
                        "consented": self.has_active_consent(person.id),
                        "created_at": person.created_at,
                    }
                )
            return summary

    # -- embeddings ----------------------------------------------------------

    def add_embedding(
        self,
        person_id: str,
        vector: FloatArray,
        model_id: str,
        speech_seconds: Optional[float] = None,
        snr_db: Optional[float] = None,
        device_id: Optional[str] = None,
        room_id: Optional[str] = None,
        source_sha256: Optional[str] = None,
    ) -> str:
        """Store one enrolment vector, encrypted and bound to its owner.

        Refuses without an active consent row. This is the enforcement point
        for "no consent row, no enrolment".
        """
        if not self.has_active_consent(person_id):
            raise ConsentError(
                "person {0} has no active consent record; enrolment refused "
                "(CLAUDE.md §11)".format(person_id)
            )
        array: FloatArray = np.asarray(vector, dtype=np.float32).reshape(-1)
        blob = self.cipher.encrypt_vector(array, subject_aad(person_id, model_id))
        record = EmbeddingRecord(
            id=new_id(),
            person_id=person_id,
            ciphertext=blob,
            dim=int(array.size),
            model_id=model_id,
            expires_at=self._expiry(),
            speech_seconds=speech_seconds,
            snr_db=snr_db,
            device_id=device_id,
            room_id=room_id,
            source_sha256=source_sha256,
        )
        with self._session_factory() as session:
            if session.get(Person, person_id) is None:
                raise StoreError("unknown person {0}".format(person_id))
            session.add(record)
            session.commit()
        LOGGER.info("stored embedding %s for person %s (model %s)", record.id, person_id, model_id)
        return record.id

    def get_person_embeddings(self, person_id: str, model_id: str) -> Optional[PersonEmbeddings]:
        with self._session_factory() as session:
            records = session.scalars(
                select(EmbeddingRecord)
                .where(
                    EmbeddingRecord.person_id == person_id,
                    EmbeddingRecord.model_id == model_id,
                )
                .order_by(EmbeddingRecord.created_at)
            ).all()
        if not records:
            return None
        return self._decrypt_records(person_id, model_id, records)

    def _decrypt_records(
        self, person_id: str, model_id: str, records: Sequence[EmbeddingRecord]
    ) -> PersonEmbeddings:
        aad = subject_aad(person_id, model_id)
        vectors = np.stack(
            [self.cipher.decrypt_vector(record.ciphertext, aad) for record in records], axis=0
        )
        return PersonEmbeddings(
            person_id=person_id,
            vectors=np.ascontiguousarray(vectors, dtype=np.float32),
            embedding_ids=tuple(record.id for record in records),
        )

    def load_gallery(self, model_id: str) -> Gallery:
        """Decrypt every active person's embeddings for scoring.

        Called once at startup. Decrypting per request would put a KDF and an
        AES pass in the latency budget for no benefit.
        """
        with self._session_factory() as session:
            active_ids = session.scalars(
                select(Person.id).where(Person.active.is_(True)).order_by(Person.created_at)
            ).all()
            gallery = Gallery(model_id=model_id)
            for person_id in active_ids:
                records = session.scalars(
                    select(EmbeddingRecord)
                    .where(
                        EmbeddingRecord.person_id == person_id,
                        EmbeddingRecord.model_id == model_id,
                    )
                    .order_by(EmbeddingRecord.created_at)
                ).all()
                if records:
                    gallery.people.append(self._decrypt_records(person_id, model_id, records))
        LOGGER.info(
            "loaded gallery: %d people, %d embeddings, model %s",
            len(gallery),
            gallery.total_embeddings,
            model_id,
        )
        return gallery

    # -- cohort (AS-Norm) ----------------------------------------------------

    def add_cohort_member(
        self,
        vector: FloatArray,
        model_id: str,
        person_id: Optional[str] = None,
        room_id: Optional[str] = None,
    ) -> str:
        array: FloatArray = np.asarray(vector, dtype=np.float32).reshape(-1)
        blob = self.cipher.encrypt_vector(array, subject_aad(person_id or "", model_id))
        record = CohortMember(
            id=new_id(),
            person_id=person_id,
            ciphertext=blob,
            dim=int(array.size),
            model_id=model_id,
            room_id=room_id,
            expires_at=self._expiry(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
        return record.id

    def load_cohort(self, model_id: str) -> FloatArray:
        """Return the cohort matrix ``(n_members, dim)``, or an empty array."""
        with self._session_factory() as session:
            records = session.scalars(
                select(CohortMember).where(CohortMember.model_id == model_id)
            ).all()
        if not records:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = [
            self.cipher.decrypt_vector(
                record.ciphertext, subject_aad(record.person_id or "", model_id)
            )
            for record in records
        ]
        return np.ascontiguousarray(np.stack(vectors, axis=0), dtype=np.float32)

    # -- retained audio ------------------------------------------------------

    def register_retained_audio(
        self,
        path: str,
        purpose: str,
        person_id: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> str:
        record = RetainedAudio(
            id=new_id(),
            person_id=person_id,
            path=os.path.abspath(path),
            purpose=purpose,
            sha256=sha256,
            expires_at=self._expiry(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
        LOGGER.info("registered retained audio %s for purpose %s", record.id, purpose)
        return record.id

    # -- events --------------------------------------------------------------

    def log_event(
        self,
        outcome: str,
        person_id: Optional[str] = None,
        score: Optional[float] = None,
        runner_up_id: Optional[str] = None,
        runner_up_score: Optional[float] = None,
        margin: Optional[float] = None,
        cm_score: Optional[float] = None,
        speech_seconds: Optional[float] = None,
        reason: Optional[str] = None,
        challenge_passed: Optional[bool] = None,
    ) -> str:
        record = RecognitionEvent(
            id=new_id(),
            outcome=outcome,
            person_id=person_id,
            score=score,
            runner_up_id=runner_up_id,
            runner_up_score=runner_up_score,
            margin=margin,
            cm_score=cm_score,
            speech_seconds=speech_seconds,
            reason=reason,
            challenge_passed=challenge_passed,
            expires_at=self._expiry(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
        return record.id

    # -- erasure -------------------------------------------------------------

    def delete_subject(self, person_id: str) -> DeletionReport:
        """Remove every trace of a person: the erasure path, and it is tested.

        Covers embeddings, cohort membership, retained audio rows *and* the
        files they point at, recognition events, consent records and the
        person row itself.
        """
        report = DeletionReport(person_id=person_id)
        with self._session_factory() as session:
            audio_records = session.scalars(
                select(RetainedAudio).where(RetainedAudio.person_id == person_id)
            ).all()
            for record in audio_records:
                if _unlink_quietly(record.path):
                    report.audio_files_deleted += 1
                else:
                    report.audio_files_missing += 1
                session.delete(record)
            report.audio_rows = len(audio_records)

            for model, attribute in (
                (EmbeddingRecord, "embedding_rows"),
                (CohortMember, "cohort_rows"),
                (RecognitionEvent, "event_rows"),
                (ConsentRecord, "consent_rows"),
            ):
                result = session.execute(delete(model).where(model.person_id == person_id))
                setattr(report, attribute, int(result.rowcount or 0))

            person = session.get(Person, person_id)
            if person is not None:
                session.delete(person)
                report.person_rows = 1
            session.commit()

        LOGGER.info(
            "deleted subject %s: %d rows, %d audio files (%d already absent)",
            person_id,
            report.total_rows,
            report.audio_files_deleted,
            report.audio_files_missing,
        )
        return report

    def subject_trace_count(self, person_id: str) -> int:
        """How many rows still mention this person. Used to verify erasure."""
        with self._session_factory() as session:
            total = 0
            for model in (
                EmbeddingRecord,
                CohortMember,
                RecognitionEvent,
                ConsentRecord,
                RetainedAudio,
            ):
                total += len(
                    session.scalars(select(model.id).where(model.person_id == person_id)).all()
                )
            if session.get(Person, person_id) is not None:
                total += 1
            return total


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unlink_quietly(path: str) -> bool:
    """Delete a file, returning whether it was there. Missing is not an error."""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        LOGGER.error("could not delete retained audio %s: %s", path, exc)
        return False


def _resolve_sqlite_path(url: str, config: Config) -> str:
    """Make a relative sqlite path project-relative and ensure its directory."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix) :]
    if not raw_path or raw_path == ":memory:":
        return url
    absolute = config.resolve_relative(raw_path)
    parent = os.path.dirname(absolute)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return prefix + absolute


def iter_gallery_vectors(gallery: Gallery) -> Iterable[Tuple[str, FloatArray]]:
    for person in gallery.people:
        for row in person.vectors:
            yield person.person_id, row
