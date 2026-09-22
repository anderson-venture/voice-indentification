"""Shared test fixtures.

The suite runs with **no network and no model weights**. Both are deliberate:
CLAUDE.md §12 requires pytest to pass offline, and the ONNX files are shipped
by USB rather than committed, so CI cannot rely on them.

``OnnxModel`` takes a ``session_factory``, and the doubles below plug into
that seam. They are not stubs that return canned answers -- each one computes
a real acoustic quantity, so a test that passes says something true about the
surrounding code:

* the VAD double scores **periodicity**, so it separates the synthetic voices
  from room tone for the same reason a real VAD does, and it is gain-invariant
  (which matters, because loading peak-normalises);
* the embedder double is a **log-spectral shape** vector, so utterances from
  one synthetic speaker really are close together and different speakers
  really are far apart;
* the countermeasure double measures **envelope autocorrelation**, which is
  how reverberation actually gives a replay away: a recording played into a
  room carries both the attacker's reverb tail and yours, and that doubled
  reverberation smooths the energy envelope. Measured over the fixtures it
  separates cleanly and voice-independently -- live speech scores -0.04 to
  0.24, replays 0.70 to 0.85.

Tests that need real weights are *skipped*, never mocked and forgotten.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speakerid.config import Config, Thresholds, load_config  # noqa: E402
from speakerid.db.crypto import EmbeddingCipher  # noqa: E402
from speakerid.db.store import Store  # noqa: E402
from speakerid.models.base import RuntimeOptions  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")

#: Embedding width used by the doubles. Matches config/default.yaml.
EMBEDDING_DIM = 192

#: Separates the synthetic voices from room tone. Measured, not guessed:
#: speech frames score 0.60-0.87, room tone 0.10-0.15.
PERIODICITY_SPEECH = 0.40

#: Envelope autocorrelation lag, in frames of 256 samples (16 ms), so 4
#: frames is ~64 ms -- the scale on which room reverberation smears speech.
ENVELOPE_LAG_FRAMES = 4

#: Decision point for the countermeasure double. Measured across four
#: synthetic voices x three utterances: live 0.235 at worst, replay 0.698 at
#: best, so the boundary sits in the middle of a wide gap.
ENVELOPE_AC_CENTRE = 0.46
ENVELOPE_AC_SLOPE = 0.08


# ---------------------------------------------------------------------------
# Session doubles
# ---------------------------------------------------------------------------


class _Spec(object):
    """Mimics onnxruntime's NodeArg surface: a name and a declared shape."""

    def __init__(self, name: str, shape: Optional[List[Any]] = None) -> None:
        self.name = name
        self.shape = shape or []


class FakeSession(object):
    """Base double. Counts runs so tests can assert on the short-circuit."""

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        input_shapes: Optional[List[List[Any]]] = None,
    ) -> None:
        shapes = input_shapes or [[] for _ in input_names]
        self._inputs = [_Spec(name, shape) for name, shape in zip(input_names, shapes)]
        self._outputs = [_Spec(name) for name in output_names]
        self.run_count = 0

    def get_inputs(self) -> List[_Spec]:
        return self._inputs

    def get_outputs(self) -> List[_Spec]:
        return self._outputs

    def run(self, output_names: Optional[List[str]], feeds: Dict[str, Any]) -> List[np.ndarray]:
        self.run_count += 1
        return self._compute(feeds)

    def _compute(self, feeds: Dict[str, Any]) -> List[np.ndarray]:
        raise NotImplementedError


def _frames(samples: np.ndarray, size: int = 512, hop: int = 256) -> np.ndarray:
    if samples.size < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: samples.size] = samples
        return padded.reshape(1, -1)
    count = 1 + (samples.size - size) // hop
    return np.stack([samples[i * hop : i * hop + size] for i in range(count)], axis=0)


def periodicity(window: np.ndarray) -> float:
    """Normalised autocorrelation peak over the 61-400 Hz pitch range.

    Voiced speech is periodic; room tone is not. Independent of gain, which
    matters because audio is peak-normalised on load.
    """
    centred = window - float(np.mean(window))
    if float(np.max(np.abs(centred))) < 1e-9:
        return 0.0
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    if correlation[0] <= 0:
        return 0.0
    normalised = correlation / correlation[0]
    low, high = 40, min(260, normalised.size)
    if high <= low:
        return 0.0
    return float(np.max(normalised[low:high]))


class FakeVadSession(FakeSession):
    """Silero v5 signature, scoring periodicity instead of running a network.

    Enforces the 512 + 64 context layout the real ONNX graph requires. The
    real model does not complain when given a bare 512-sample window -- it
    just returns ~0.0 for everything, so no speech is ever found. Asserting
    the width here turns that silent failure into a loud test failure.
    """

    def __init__(self, window_samples: int = 512, context_samples: int = 64) -> None:
        super(FakeVadSession, self).__init__(["input", "state", "sr"], ["output", "stateN"])
        self.expected_samples = window_samples + context_samples

    def _compute(self, feeds: Dict[str, Any]) -> List[np.ndarray]:
        window = np.asarray(feeds["input"], dtype=np.float32).reshape(-1)
        if window.size != self.expected_samples:
            raise AssertionError(
                "Silero VAD v5 expects {0} samples (512 window + 64 context); "
                "got {1}. A bare window makes the real model output ~0.0 for "
                "every frame.".format(self.expected_samples, window.size)
            )
        score = periodicity(window)
        # Map the measured range onto a probability the hysteresis can use.
        probability = float(np.clip((score - 0.2) / 0.4, 0.0, 1.0))
        state = np.asarray(feeds.get("state", np.zeros((2, 1, 128), dtype=np.float32)))
        return [np.array([[probability]], dtype=np.float32), state.astype(np.float32)]


def spectral_embedding(samples: np.ndarray, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Mean log-magnitude spectrum of the loudest frames, mean-removed.

    Captures the formant envelope, which is what distinguishes the synthetic
    voices from one another.
    """
    frames = _frames(np.asarray(samples, dtype=np.float32))
    window = np.hanning(frames.shape[1])
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    energy = spectrum.sum(axis=1)
    loudest = energy >= np.percentile(energy, 50)
    if not loudest.any():
        loudest = np.ones_like(energy, dtype=bool)
    mean_spectrum = spectrum[loudest].mean(axis=0) + 1e-10
    log_spectrum = np.log(mean_spectrum)
    source = np.linspace(0.0, 1.0, log_spectrum.size)
    target = np.linspace(0.0, 1.0, dim)
    resampled = np.interp(target, source, log_spectrum)
    centred = resampled - float(np.mean(resampled))
    norm = float(np.linalg.norm(centred)) or 1.0
    return np.ascontiguousarray(centred / norm, dtype=np.float32)


class FakeEmbedderSession(FakeSession):
    """Mirrors the real ReDimNet-B2 export: rank-3 input, unnormalised output."""

    def __init__(self, dim: int = EMBEDDING_DIM, rank: int = 3) -> None:
        shape = ["B", 1, "T"] if rank == 3 else ["B", "T"]
        super(FakeEmbedderSession, self).__init__(["audio"], ["embs"], [shape])
        self.dim = dim

    def _compute(self, feeds: Dict[str, Any]) -> List[np.ndarray]:
        waveform = np.asarray(feeds["audio"], dtype=np.float32).reshape(-1)
        # Deliberately NOT unit-norm, exactly like the real export, so the
        # wrapper's defensive normalisation is actually exercised.
        return [spectral_embedding(waveform, self.dim).reshape(1, -1) * 18.0]


def envelope_autocorrelation(samples: np.ndarray, lag_frames: int = ENVELOPE_LAG_FRAMES) -> float:
    """Self-similarity of the log-energy envelope at a ~64 ms lag.

    Live speech has a sharply modulated envelope: syllables start and stop, so
    frames 64 ms apart are close to uncorrelated. Reverberation fills the gaps
    in and makes neighbouring frames resemble one another, and a replay
    carries two rooms' worth of it -- the attacker's and yours.

    High values therefore mean *more* reverberant, which means *less* likely
    to be a live speaker in front of the microphone.
    """
    frames = _frames(np.asarray(samples, dtype=np.float32))
    if frames.shape[0] <= lag_frames:
        return 0.0
    energy = np.log(np.sqrt((frames**2).mean(axis=1)) + 1e-12)
    centred = energy - float(np.mean(energy))
    power = float(np.dot(centred, centred))
    if power <= 0.0:
        return 0.0
    return float(np.dot(centred[:-lag_frames], centred[lag_frames:]) / power)


class FakeCountermeasureSession(FakeSession):
    """Returns two logits, index 1 = bona fide, as AASIST-L does."""

    def __init__(self) -> None:
        super(FakeCountermeasureSession, self).__init__(["waveform"], ["logits"])

    def _compute(self, feeds: Dict[str, Any]) -> List[np.ndarray]:
        waveform = np.asarray(feeds["waveform"], dtype=np.float32).reshape(-1)
        # Reverberant (high autocorrelation) -> spoof, so the margin is negated.
        reverberance = envelope_autocorrelation(waveform)
        margin = (ENVELOPE_AC_CENTRE - reverberance) / ENVELOPE_AC_SLOPE
        # Logits, not probabilities: the wrapper applies the softmax.
        return [np.array([[-margin, margin]], dtype=np.float32)]


def fake_session_factory(model_path: str, options: RuntimeOptions) -> FakeSession:
    """Dispatch on the configured path, as the real factory does."""
    lowered = model_path.lower()
    if "vad" in lowered:
        return FakeVadSession()
    if "aasist" in lowered or "cm" in os.path.basename(os.path.dirname(lowered)):
        return FakeCountermeasureSession()
    return FakeEmbedderSession()


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir() -> str:
    if not os.path.isdir(FIXTURES_DIR) or not os.listdir(FIXTURES_DIR):
        pytest.skip("fixtures missing; run python -m scripts.make_fixtures")
    return FIXTURES_DIR


@pytest.fixture
def fixture_path(fixtures_dir):
    def _resolve(name: str) -> str:
        path = os.path.join(fixtures_dir, name)
        if not os.path.isfile(path):
            pytest.skip("fixture {0} not generated".format(name))
        return path

    return _resolve


@pytest.fixture
def base_config(tmp_path) -> Config:
    """Default config, redirected so no test touches the real database."""
    return load_config(
        overrides={
            "database": {
                "url": "sqlite:///{0}".format(tmp_path / "test.db"),
                "purge_on_startup": False,
                "purge_interval_seconds": 0,
            },
            "crypto": {"key_source": "passphrase", "salt_path": str(tmp_path / ".salt")},
            "runtime": {"verify_manifest": False, "warmup_iterations": 1},
            "scoring": {"asnorm": {"enabled": False}},
            "challenge": {"enabled": False},
            "logging": {"level": "WARNING"},
        }
    )


@pytest.fixture
def cipher(base_config) -> EmbeddingCipher:
    return EmbeddingCipher.from_config(base_config, passphrase="test-passphrase")


@pytest.fixture
def store(base_config, cipher) -> Store:
    created = Store.from_config(base_config, cipher=cipher)
    yield created
    created.close()


@pytest.fixture
def thresholds() -> Thresholds:
    """Thresholds for tests only.

    Real deployments get these from calibrate.py; hardcoding them here is
    confined to the test suite, where the point is to exercise the decision
    boundaries rather than to be correct for a site.
    """
    return Thresholds(
        cm_threshold=0.5,
        accept_threshold=0.9,
        margin_threshold=0.05,
        min_speech_sec=2.0,
        provenance={"site": "unit-test"},
    )


@pytest.fixture
def features(base_config):
    from speakerid.pipeline import FeatureExtractor

    return FeatureExtractor.from_config(base_config, session_factory=fake_session_factory)


@pytest.fixture
def pipeline(base_config, thresholds, store, features):
    from speakerid.pipeline import Pipeline
    from speakerid.scoring.asnorm import AsNorm

    built = Pipeline(
        config=base_config,
        thresholds=thresholds,
        features=features,
        store=store,
        asnorm=AsNorm.from_config(base_config),
        model_id="fake-embedder",
    )
    yield built


@pytest.fixture
def enrolled_pipeline(pipeline, fixture_path):
    """A pipeline with Alice and Bob enrolled, and consent recorded."""
    for person, files in (
        ("alice", ["alice_01.wav", "alice_02.wav", "alice_03.wav"]),
        ("bob", ["bob_01.wav", "bob_02.wav", "bob_03.wav"]),
    ):
        pipeline.store.add_person(person_id=person, display_name=person.title())
        pipeline.store.record_consent(person, policy_version="test-1", granted_by="tester")
        for name in files:
            result = pipeline.enroll_file(person, fixture_path(name))
            assert result.accepted, "fixture enrolment failed: {0}".format(result.reason)
    pipeline.refresh_gallery()
    return pipeline
