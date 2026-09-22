"""Pipeline orchestration: front-end, liveness, identity, decision.

The order below is deliberate and is not an implementation detail
(CLAUDE.md §6)::

    raw audio (any rate)
      -> resample 16 kHz mono float32, peak-normalize
      -> Silero VAD -> speech segments
      -> reject if total speech < min_speech_sec
      -> concatenate speech, take window
      -> [C] AASIST-L   -- if below threshold, stop here
      -> [B] ReDimNet2  -- embedding
      -> AS-Norm against cohort
      -> cosine vs all enrolled -> top-1 and top-2
      -> decide()

The countermeasure runs **before** the embedder because it is far cheaper and
short-circuits the expensive path on rejection. Non-speech never reaches
either model.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio.enhance import Enhancer
from .audio.io import (
    ensure_audio,
    estimate_snr_db,
    load_audio,
)
from .audio.vad import (
    Segment,
    SileroVad,
    concatenate_speech,
    speech_mask,
    total_speech_seconds,
)
from .challenge import ChallengeManager
from .config import Config, Thresholds, load_config, load_thresholds
from .db.store import Gallery, Store
from .errors import SpeakerIdError
from .models.base import SessionFactory, sha256_file
from .models.countermeasure import Countermeasure
from .models.embedder import SpeakerEmbedder
from .nptypes import FloatArray
from .scoring.asnorm import AsNorm
from .scoring.decide import Candidate, Decision, Outcome, decide
from .scoring.similarity import aggregate_person_scores, cosine_matrix

LOGGER = logging.getLogger(__name__)


@dataclass
class Timings:
    """Per-stage wall time in milliseconds. Cheap, and it makes the device
    port measurable rather than guessed at."""

    load_ms: float = 0.0
    vad_ms: float = 0.0
    countermeasure_ms: float = 0.0
    embed_ms: float = 0.0
    score_ms: float = 0.0
    total_ms: float = 0.0

    def summary(self) -> str:
        return (
            "load={0:.1f} vad={1:.1f} cm={2:.1f} embed={3:.1f} score={4:.1f} "
            "total={5:.1f} (ms)".format(
                self.load_ms,
                self.vad_ms,
                self.countermeasure_ms,
                self.embed_ms,
                self.score_ms,
                self.total_ms,
            )
        )


@dataclass
class PipelineResult:
    """A decision plus the evidence and cost behind it."""

    decision: Decision
    segments: Tuple[Segment, ...] = ()
    speech_seconds: float = 0.0
    snr_db: Optional[float] = None
    embedding: Optional[FloatArray] = None
    timings: Timings = field(default_factory=Timings)
    event_id: Optional[str] = None

    @property
    def outcome(self) -> Outcome:
        return self.decision.outcome


@dataclass
class EnrollmentResult:
    """Outcome of one enrolment utterance, accepted or not."""

    accepted: bool
    embedding_id: Optional[str] = None
    speech_seconds: float = 0.0
    snr_db: Optional[float] = None
    self_cosine: Optional[float] = None
    reason: Optional[str] = None
    embedding: Optional[FloatArray] = None


def derive_model_id(model_path: str, override: Optional[str] = None) -> str:
    """Identify an embedding model by filename plus a short content hash.

    Embeddings are only comparable within one model. Folding the file's hash
    into the id means a re-export cannot be silently scored against vectors
    produced by the previous one -- the gallery for the new id is simply empty,
    which is loud and correct, instead of quietly degraded.
    """
    if override:
        return override
    stem = os.path.splitext(os.path.basename(model_path))[0]
    if os.path.isfile(model_path):
        return "{0}@{1}".format(stem, sha256_file(model_path)[:8])
    return stem


class FeatureExtractor(object):
    """The model half of the pipeline: front-end, liveness and embedding.

    Deliberately knows nothing about thresholds, the database or decisions.
    Calibration needs embeddings *before* any threshold exists -- producing
    them is the whole point of calibrating -- so the code that runs the models
    must not require a calibrated file to exist.
    """

    def __init__(
        self,
        config: Config,
        vad: SileroVad,
        countermeasure: Countermeasure,
        embedder: SpeakerEmbedder,
        enhancer: Optional[Enhancer] = None,
    ) -> None:
        self.config = config
        self.vad = vad
        self.countermeasure = countermeasure
        self.embedder = embedder
        self.enhancer = enhancer or Enhancer(enabled=False, backend="none")
        self.sample_rate = config.sample_rate

    @classmethod
    def from_config(
        cls, config: Config, session_factory: Optional[SessionFactory] = None
    ) -> FeatureExtractor:
        return cls(
            config=config,
            vad=SileroVad.from_config(config, session_factory=session_factory),
            countermeasure=Countermeasure.from_config(config, session_factory=session_factory),
            embedder=SpeakerEmbedder.from_config(config, session_factory=session_factory),
            enhancer=Enhancer.from_config(config),
        )

    # -- loading -------------------------------------------------------------

    def load(self, path: str) -> FloatArray:
        return load_audio(
            path,
            target_rate=self.sample_rate,
            peak_dbfs=self.config.get_float("audio.peak_normalize_dbfs"),
            max_seconds=self.config.get_float("audio.max_input_seconds"),
        )

    # -- front end -----------------------------------------------------------

    def prepare(
        self, audio: FloatArray
    ) -> Tuple[FloatArray, List[Segment], float, Optional[float]]:
        """VAD the buffer and return (speech, segments, seconds, snr).

        Everything downstream sees only the speech. Silence, keyboard noise
        and HVAC hum never reach a model (CLAUDE.md §6).
        """
        ensure_audio(audio, self.sample_rate, name="pipeline input")
        audio = self.enhancer(audio)
        segments = self.vad.segments(audio)
        seconds = total_speech_seconds(segments, self.sample_rate)
        snr: Optional[float] = None
        if segments:
            measured = estimate_snr_db(audio, speech_mask(segments, audio.size))
            snr = None if not np.isfinite(measured) else float(measured)
        return concatenate_speech(audio, segments), segments, seconds, snr

    # -- models --------------------------------------------------------------

    def liveness(self, speech: FloatArray) -> float:
        return self.countermeasure.score(speech)

    def embed(self, speech: FloatArray) -> FloatArray:
        return self.embedder.embed(speech)

    def embed_file(self, path: str, min_speech_sec: float) -> Optional[FloatArray]:
        """Embed a file's speech, or return None when it has too little.

        Used by cohort building, calibration and evaluation, which need
        embeddings without a decision. Applies the same VAD gate as the live
        path, so a cohort never contains an embedding of room tone.
        """
        speech, _segments, seconds, _snr = self.prepare(self.load(path))
        if seconds < min_speech_sec or speech.size == 0:
            LOGGER.warning("skipping %s: only %.2fs of speech", os.path.basename(path), seconds)
            return None
        return self.embed(speech)

    def liveness_file(self, path: str, min_speech_sec: float) -> Optional[float]:
        speech, _segments, seconds, _snr = self.prepare(self.load(path))
        if seconds < min_speech_sec or speech.size == 0:
            LOGGER.warning("skipping %s: only %.2fs of speech", os.path.basename(path), seconds)
            return None
        return self.liveness(speech)

    @property
    def model_id(self) -> str:
        return derive_model_id(
            self.embedder.model_path, self.config.get_optional_str("embedder.model_id")
        )


class Pipeline(object):
    """Owns the loaded models and the in-memory gallery for one process."""

    def __init__(
        self,
        config: Config,
        thresholds: Thresholds,
        features: FeatureExtractor,
        store: Store,
        asnorm: AsNorm,
        challenges: Optional[ChallengeManager] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.thresholds = thresholds
        self.features = features
        self.store = store
        self.asnorm = asnorm
        self.challenges = challenges or ChallengeManager.from_config(config)
        self.model_id = model_id or features.model_id
        self.sample_rate = config.sample_rate
        self.aggregation = config.get_str("scoring.aggregation")

        self._gallery: Gallery = Gallery(model_id=self.model_id)
        self._gallery_matrix: FloatArray = np.zeros((0, 0), dtype=np.float32)
        self._gallery_owners: List[str] = []
        self._enrollment_means: FloatArray = np.zeros(0, dtype=np.float32)
        self._enrollment_stds: FloatArray = np.ones(0, dtype=np.float32)
        self.refresh_gallery()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Optional[Config] = None,
        thresholds: Optional[Thresholds] = None,
        store: Optional[Store] = None,
        passphrase: Optional[str] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> Pipeline:
        """Build everything once, at startup.

        Every ONNX session in the process is created here and warmed. Nothing
        downstream may create one per request.
        """
        resolved_config = config or load_config()
        resolved_thresholds = thresholds or load_thresholds()
        resolved_store = store or Store.from_config(resolved_config, passphrase=passphrase)

        if resolved_config.get_bool("database.purge_on_startup"):
            resolved_store.purge_expired()
        interval = resolved_config.get_float("database.purge_interval_seconds")
        resolved_store.start_purge_timer(interval)

        return cls(
            config=resolved_config,
            thresholds=resolved_thresholds,
            features=FeatureExtractor.from_config(resolved_config, session_factory=session_factory),
            store=resolved_store,
            asnorm=AsNorm.from_config(resolved_config),
            challenges=ChallengeManager.from_config(resolved_config),
        )

    # -- gallery -------------------------------------------------------------

    def refresh_gallery(self) -> None:
        """Reload enrolments and recompute AS-Norm enrolment statistics.

        Call after any enrolment or deletion. The statistics depend only on
        the gallery and the cohort, so they belong here and never in the
        request path.
        """
        self._gallery = self.store.load_gallery(self.model_id)
        self._gallery_matrix, self._gallery_owners = self._gallery.stacked()
        self._enrollment_means, self._enrollment_stds = self.asnorm.enrollment_stats(
            self._gallery_matrix
        )
        if len(self._gallery) == 0:
            LOGGER.warning(
                "gallery is empty for model %s; every request will return "
                "REJECTED(no_enrolments)",
                self.model_id,
            )

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    def close(self) -> None:
        self.store.close()

    # -- front end -----------------------------------------------------------

    def _prepare(
        self, audio: FloatArray
    ) -> Tuple[FloatArray, List[Segment], float, Optional[float]]:
        return self.features.prepare(audio)

    def load(self, path: str) -> FloatArray:
        return self.features.load(path)

    # -- identification ------------------------------------------------------

    def identify_file(self, path: str, challenge_token: Optional[str] = None) -> PipelineResult:
        started = time.time()
        audio = self.load(path)
        load_ms = (time.time() - started) * 1000.0
        result = self.identify(audio, challenge_token=challenge_token)
        result.timings.load_ms = load_ms
        result.timings.total_ms += load_ms
        return result

    def identify(self, audio: FloatArray, challenge_token: Optional[str] = None) -> PipelineResult:
        """Run the full pipeline on a buffer and return a decision."""
        started = time.time()
        timings = Timings()

        # -- A. front end ----------------------------------------------------
        stage = time.time()
        speech, segments, speech_seconds, snr = self._prepare(audio)
        timings.vad_ms = (time.time() - stage) * 1000.0

        if speech_seconds < self.thresholds.min_speech_sec or speech.size == 0:
            # Not enough to judge. decide() still issues the verdict so that
            # there remains exactly one decision point in the system.
            decision = decide(
                candidates=(),
                thresholds=self.thresholds,
                speech_seconds=speech_seconds,
                cm_score=None,
                challenge_passed=None,
            )
            timings.total_ms = (time.time() - started) * 1000.0
            return self._finish(decision, segments, speech_seconds, snr, None, timings)

        # -- challenge-response ----------------------------------------------
        challenge_passed = self.challenges.resolve(challenge_token, speech)

        # -- C. liveness, before the expensive path --------------------------
        stage = time.time()
        cm_score = self.features.liveness(speech)
        timings.countermeasure_ms = (time.time() - stage) * 1000.0

        # This comparison skips work; it does not decide anything. decide()
        # below reaches the same conclusion from the same threshold and is the
        # authoritative verdict. Skipping the embedder here is the 40x saving
        # that CLAUDE.md §6 asks for.
        liveness_failed = cm_score < self.thresholds.cm_threshold
        embedding: Optional[FloatArray] = None
        candidates: Sequence[Candidate] = ()

        if not liveness_failed and challenge_passed is not False:
            # -- B. identity -------------------------------------------------
            stage = time.time()
            embedding = self.features.embed(speech)
            timings.embed_ms = (time.time() - stage) * 1000.0

            stage = time.time()
            candidates = self.score(embedding)
            timings.score_ms = (time.time() - stage) * 1000.0
        elif liveness_failed:
            LOGGER.info(
                "liveness below threshold (%.4f < %.4f); skipping the embedder",
                cm_score,
                self.thresholds.cm_threshold,
            )

        decision = decide(
            candidates=candidates,
            thresholds=self.thresholds,
            speech_seconds=speech_seconds,
            cm_score=cm_score,
            challenge_passed=challenge_passed,
        )
        timings.total_ms = (time.time() - started) * 1000.0
        return self._finish(decision, segments, speech_seconds, snr, embedding, timings)

    def score(self, embedding: FloatArray) -> List[Candidate]:
        """Score one embedding against the gallery, AS-Normed, per person."""
        if self._gallery_matrix.size == 0:
            return []
        raw = cosine_matrix(embedding, self._gallery_matrix)
        normalized = self.asnorm.normalize(
            raw, embedding, self._enrollment_means, self._enrollment_stds
        )
        per_person = aggregate_person_scores(
            normalized, self._gallery_owners, method=self.aggregation
        )
        raw_per_person = dict(
            aggregate_person_scores(raw, self._gallery_owners, method=self.aggregation)
        )
        return [
            Candidate(
                person_id=person_id, score=score, raw_score=raw_per_person.get(person_id, 0.0)
            )
            for person_id, score in per_person
        ]

    def _finish(
        self,
        decision: Decision,
        segments: Sequence[Segment],
        speech_seconds: float,
        snr: Optional[float],
        embedding: Optional[FloatArray],
        timings: Timings,
    ) -> PipelineResult:
        event_id = self.store.log_event(
            outcome=decision.outcome.value,
            person_id=decision.person_id,
            score=decision.score,
            runner_up_id=decision.runner_up_id,
            runner_up_score=decision.runner_up_score,
            margin=decision.margin,
            cm_score=decision.cm_score,
            speech_seconds=speech_seconds,
            reason=decision.reason,
            challenge_passed=decision.challenge_passed,
        )
        LOGGER.info("%s | %s", decision.summary(), timings.summary())
        return PipelineResult(
            decision=decision,
            segments=tuple(segments),
            speech_seconds=speech_seconds,
            snr_db=snr,
            embedding=embedding,
            timings=timings,
            event_id=event_id,
        )

    # -- enrolment -----------------------------------------------------------

    def enroll_file(
        self,
        person_id: str,
        path: str,
        device_id: Optional[str] = None,
        room_id: Optional[str] = None,
        retain_audio: bool = False,
    ) -> EnrollmentResult:
        audio = self.load(path)
        digest = sha256_file(path)
        result = self.enroll(
            person_id,
            audio,
            device_id=device_id,
            room_id=room_id,
            source_sha256=digest,
        )
        if result.accepted and retain_audio:
            # Only ever on an explicit flag, and always tracked so that
            # retention and erasure can both reach the file (CLAUDE.md §11).
            self.store.register_retained_audio(
                path=path, purpose="enrollment", person_id=person_id, sha256=digest
            )
        return result

    def enroll(
        self,
        person_id: str,
        audio: FloatArray,
        device_id: Optional[str] = None,
        room_id: Optional[str] = None,
        source_sha256: Optional[str] = None,
    ) -> EnrollmentResult:
        """Quality-gate one utterance and store its embedding.

        The gates are CLAUDE.md §7: enough speech, enough SNR, and consistent
        with the person's existing enrolments. An enrolment accepted in spite
        of them poisons that person's row for as long as it is stored.
        """
        speech, _segments, speech_seconds, snr = self._prepare(audio)

        min_speech = self.config.get_float("enrollment.min_speech_seconds_per_utterance")
        if speech_seconds < min_speech:
            return EnrollmentResult(
                accepted=False,
                speech_seconds=speech_seconds,
                snr_db=snr,
                reason="only {0:.2f}s of speech; {1:.2f}s required".format(
                    speech_seconds, min_speech
                ),
            )

        min_snr = self.config.get_float("enrollment.min_snr_db")
        if snr is not None and snr < min_snr:
            return EnrollmentResult(
                accepted=False,
                speech_seconds=speech_seconds,
                snr_db=snr,
                reason="SNR {0:.1f} dB below the {1:.1f} dB enrolment minimum".format(snr, min_snr),
            )

        embedding = self.features.embed(speech)

        # Consistency with this person's existing enrolments. A low cosine
        # here usually means the wrong person was recorded, or the capture
        # broke -- both worth catching now rather than at the door.
        self_cosine: Optional[float] = None
        existing = self.store.get_person_embeddings(person_id, self.model_id)
        if existing is not None and existing.vectors.size:
            self_cosine = float(np.max(cosine_matrix(embedding, existing.vectors)))
            minimum = self.config.get_float("enrollment.min_self_cosine")
            if self_cosine < minimum:
                return EnrollmentResult(
                    accepted=False,
                    speech_seconds=speech_seconds,
                    snr_db=snr,
                    self_cosine=self_cosine,
                    embedding=embedding,
                    reason=(
                        "cosine {0:.3f} to this person's existing enrolments is below "
                        "{1:.3f}: likely the wrong speaker or a broken recording"
                    ).format(self_cosine, minimum),
                )

        maximum = self.config.get_int("enrollment.max_utterances")
        if existing is not None and existing.vectors.shape[0] >= maximum:
            return EnrollmentResult(
                accepted=False,
                speech_seconds=speech_seconds,
                snr_db=snr,
                self_cosine=self_cosine,
                reason="person already has {0} enrolments (max {1}); delete one "
                "or re-enroll".format(existing.vectors.shape[0], maximum),
            )

        embedding_id = self.store.add_embedding(
            person_id=person_id,
            vector=embedding,
            model_id=self.model_id,
            speech_seconds=speech_seconds,
            snr_db=snr,
            device_id=device_id,
            room_id=room_id,
            source_sha256=source_sha256,
        )
        self.refresh_gallery()
        return EnrollmentResult(
            accepted=True,
            embedding_id=embedding_id,
            speech_seconds=speech_seconds,
            snr_db=snr,
            self_cosine=self_cosine,
            embedding=embedding,
        )

    def enrollment_status(self, person_id: str) -> Dict[str, object]:
        """Is this person enrolled well enough to be recognised reliably?"""
        existing = self.store.get_person_embeddings(person_id, self.model_id)
        count = 0 if existing is None else int(existing.vectors.shape[0])
        minimum = self.config.get_int("enrollment.min_utterances")
        return {
            "person_id": person_id,
            "embeddings": count,
            "min_required": minimum,
            "complete": count >= minimum,
            "self_consistency": None if existing is None else existing.self_consistency,
        }

    # -- embedding utility (cohort building, evaluation) ---------------------

    def embed_file(self, path: str) -> Optional[FloatArray]:
        """Embed a file's speech, or return None when it has too little."""
        return self.features.embed_file(path, self.thresholds.min_speech_sec)


def discard_audio(path: str) -> None:
    """Delete a working copy of raw audio.

    Raw audio is deleted immediately after embedding extraction unless an
    explicit retention flag was set (CLAUDE.md §11).
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOGGER.error("could not delete raw audio: %s", exc)
        raise SpeakerIdError("failed to delete raw audio at the configured path") from exc
