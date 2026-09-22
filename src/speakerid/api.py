"""Optional localhost HTTP API.

Bound to loopback only, and the configuration loader refuses any other host
(see ``config._validate``). This endpoint serves biometric decisions; exposing
it on a network interface would be a different product with a different threat
model.

FastAPI is an optional dependency. Import failures are turned into a clear
message rather than a traceback, because the device build does not install it.

The API is where challenge-response actually works end to end: the challenge
must be issued and verified inside one process, so that the token store is
shared. ``POST /challenge`` then ``POST /identify`` with the returned token.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .challenge import ChallengeManager
from .config import Config, load_config, load_thresholds
from .errors import SpeakerIdError
from .logging_setup import configure_logging
from .nptypes import FloatArray
from .pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


def create_app(
    config: Optional[Config] = None,
    pipeline: Optional[Pipeline] = None,
    passphrase: Optional[str] = None,
) -> Any:
    """Build the FastAPI application.

    The pipeline -- and therefore every ONNX session -- is constructed once,
    here, at startup. Never per request (CLAUDE.md §12).
    """
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SpeakerIdError(
            "the local API needs fastapi and uvicorn, which are optional. "
            "Install them, or use the CLI instead: {0}".format(exc)
        ) from exc

    import io

    import numpy as np
    import soundfile

    resolved_config = config or load_config()
    configure_logging(resolved_config)
    engine = pipeline or Pipeline.from_config(
        config=resolved_config, thresholds=load_thresholds(), passphrase=passphrase
    )
    challenges: ChallengeManager = engine.challenges

    app = FastAPI(
        title="speakerid",
        description="Offline speaker recognition. Localhost only. No audio leaves this host.",
        version="0.1.0",
    )

    def _read_upload(payload: bytes) -> FloatArray:
        from .audio.io import ensure_audio, peak_normalize, resample, to_mono

        try:
            data, rate = soundfile.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=400, detail="could not decode audio: {0}".format(exc)
            ) from exc
        samples = resample(
            np.ascontiguousarray(to_mono(np.asarray(data)), dtype=np.float32),
            int(rate),
            resolved_config.sample_rate,
        )
        samples = peak_normalize(samples, resolved_config.get_float("audio.peak_normalize_dbfs"))
        return ensure_audio(samples, resolved_config.sample_rate, name="uploaded audio")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "enrolled_people": len(engine.gallery),
            "enrolled_embeddings": engine.gallery.total_embeddings,
            "model_id": engine.model_id,
            "asnorm_cohort": engine.asnorm.size,
            "challenge_enabled": challenges.enabled,
        }

    @app.post("/challenge")
    def issue_challenge() -> Dict[str, Any]:
        """Issue digits for the user to read aloud.

        A pre-recorded clip cannot answer a challenge it has never heard,
        which is what defeats casual replay (CLAUDE.md §10).
        """
        if not challenges.enabled:
            raise HTTPException(status_code=409, detail="challenge-response is disabled")
        challenge = challenges.issue()
        return {
            "token": challenge.token,
            "say": challenge.prompt(),
            "expires_in": challenges.ttl_seconds,
        }

    # FastAPI evaluates its dependency markers once and uses them as parameter
    # defaults; that is the framework's documented idiom, so B008 is silenced
    # here rather than worked around.
    _AUDIO_UPLOAD = File(...)

    @app.post("/identify")
    async def identify(  # noqa: B008
        audio: UploadFile = _AUDIO_UPLOAD, challenge_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Identify a speaker. Returns one of four outcomes, never a bare name."""
        samples = _read_upload(await audio.read())
        try:
            result = engine.identify(samples, challenge_token=challenge_token)
        except SpeakerIdError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        decision = result.decision
        # The uploaded audio is never written to disk and goes out of scope
        # here: raw audio is discarded once the embedding is extracted.
        return {
            "outcome": decision.outcome.value,
            "person_id": decision.person_id,
            "score": decision.score,
            "margin": decision.margin,
            "liveness": decision.cm_score,
            "speech_seconds": result.speech_seconds,
            "reason": decision.reason,
            "event_id": result.event_id,
        }

    @app.get("/people")
    def people() -> Dict[str, Any]:
        listed = engine.store.list_persons()
        return {
            "count": len(listed),
            # Ids and counts only; names stay in the operator's own console.
            "people": [
                {
                    "person_id": item["person_id"],
                    "embeddings": item["embeddings"],
                    "consented": item["consented"],
                }
                for item in listed
            ],
        }

    @app.delete("/people/{person_id}")
    def delete_person(person_id: str) -> Dict[str, Any]:
        report = engine.store.delete_subject(person_id)
        engine.refresh_gallery()
        remaining = engine.store.subject_trace_count(person_id)
        if remaining:
            raise HTTPException(
                status_code=500, detail="erasure incomplete: {0} rows remain".format(remaining)
            )
        return {"person_id": person_id, "rows_deleted": report.total_rows}

    return app


def serve(  # pragma: no cover
    config: Optional[Config] = None, passphrase: Optional[str] = None
) -> None:
    """Run the API with uvicorn on the configured loopback address."""
    try:
        import uvicorn
    except ImportError as exc:
        raise SpeakerIdError("uvicorn is not installed: {0}".format(exc)) from exc

    resolved_config = config or load_config()
    host = resolved_config.get_str("api.host")
    port = resolved_config.get_int("api.port")
    LOGGER.info("serving on http://%s:%d (loopback only)", host, port)
    uvicorn.run(create_app(resolved_config, passphrase=passphrase), host=host, port=port)
