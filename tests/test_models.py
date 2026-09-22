"""ONNX wrappers: manifest verification, session lifecycle, and contracts."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from speakerid.errors import ManifestError, ModelError
from speakerid.models.base import (
    RuntimeOptions,
    sha256_file,
    verify_against_manifest,
)
from speakerid.models.countermeasure import Countermeasure, softmax
from speakerid.models.embedder import SpeakerEmbedder, l2_normalize

from .conftest import (
    EMBEDDING_DIM,
    FakeCountermeasureSession,
    FakeEmbedderSession,
    fake_session_factory,
)


def options(**overrides) -> RuntimeOptions:
    settings = dict(
        intra_op_num_threads=2,
        inter_op_num_threads=1,
        execution_mode="sequential",
        graph_optimization_level="all",
        providers=["CPUExecutionProvider"],
        warmup_iterations=2,
        verify_manifest=False,
    )
    settings.update(overrides)
    return RuntimeOptions(**settings)


def build_embedder(**overrides) -> SpeakerEmbedder:
    settings = dict(
        embedding_dim=EMBEDDING_DIM,
        window_samples=64000,
        max_window_samples=96000,
        multi_window_enabled=True,
        multi_window_trigger_samples=128000,
        multi_window_hop_samples=32000,
        max_windows=8,
    )
    settings.update(overrides)
    return SpeakerEmbedder(
        model_path="models/embed/redimnet2_b2.onnx",
        options=options(),
        session_factory=fake_session_factory,
        **settings,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "model.onnx"
    payload = b"not really a model"
    path.write_bytes(payload)
    assert sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()


def write_manifest(models_dir, relative, digest):
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump({"models": {relative: {"sha256": digest}}}, handle)


def test_manifest_accepts_a_matching_file(tmp_path):
    models_dir = tmp_path / "models"
    model = models_dir / "embed" / "m.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    write_manifest(str(models_dir), "embed/m.onnx", sha256_file(str(model)))
    verify_against_manifest(str(model), str(models_dir))  # must not raise


def test_manifest_rejects_a_changed_file(tmp_path):
    """A silently swapped model is invisible until the accuracy moves."""
    models_dir = tmp_path / "models"
    model = models_dir / "embed" / "m.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    write_manifest(str(models_dir), "embed/m.onnx", sha256_file(str(model)))

    model.write_bytes(b"different weights")
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        verify_against_manifest(str(model), str(models_dir))


def test_manifest_rejects_an_unlisted_model(tmp_path):
    models_dir = tmp_path / "models"
    model = models_dir / "embed" / "rogue.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    write_manifest(str(models_dir), "embed/other.onnx", "0" * 64)
    with pytest.raises(ManifestError, match="not listed"):
        verify_against_manifest(str(model), str(models_dir))


def test_missing_manifest_names_the_fix(tmp_path):
    with pytest.raises(ManifestError, match="fetch_models"):
        verify_against_manifest(str(tmp_path / "x.onnx"), str(tmp_path))


def test_missing_model_file_is_not_downloaded():
    """No model is ever fetched at run time (CLAUDE.md §2)."""
    with pytest.raises(ModelError, match="USB"):
        SpeakerEmbedder(
            model_path="/nonexistent/redimnet2_b2.onnx",
            options=options(verify_manifest=False),
            embedding_dim=EMBEDDING_DIM,
            window_samples=64000,
            max_window_samples=96000,
            multi_window_enabled=False,
            multi_window_trigger_samples=128000,
            multi_window_hop_samples=32000,
            max_windows=8,
        )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_session_is_created_once_and_warmed():
    created = []

    def counting_factory(path, opts):
        session = FakeEmbedderSession()
        created.append(session)
        return session

    embedder = SpeakerEmbedder(
        model_path="models/embed/redimnet2_b2.onnx",
        options=options(warmup_iterations=3),
        embedding_dim=EMBEDDING_DIM,
        window_samples=64000,
        max_window_samples=96000,
        multi_window_enabled=False,
        multi_window_trigger_samples=128000,
        multi_window_hop_samples=32000,
        max_windows=8,
        session_factory=counting_factory,
    )
    assert len(created) == 1, "one session per model, created at startup"
    assert created[0].run_count == 3, "the session must be warmed before first use"

    before = created[0].run_count
    embedder.embed(np.zeros(64000, dtype=np.float32) + 0.1)
    assert len(created) == 1, "inference must not create a second session"
    assert created[0].run_count > before


def test_unknown_optimisation_level_is_rejected():
    from speakerid.models.base import default_session_factory

    pytest.importorskip("onnxruntime")
    with pytest.raises(ModelError, match="graph_optimization_level"):
        default_session_factory("x.onnx", options(graph_optimization_level="turbo"))


# ---------------------------------------------------------------------------
# Embedder contract
# ---------------------------------------------------------------------------


def test_embedding_is_unit_norm_and_correct_width(fixture_path):
    from speakerid.audio.io import load_audio

    embedder = build_embedder()
    embedding = embedder.embed(load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0))
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)


def test_embedder_is_deterministic(fixture_path):
    from speakerid.audio.io import load_audio

    embedder = build_embedder()
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    first = embedder.embed(samples)
    second = embedder.embed(samples)
    assert np.array_equal(first, second)


def test_embedder_rejects_a_dimension_mismatch():
    embedder = SpeakerEmbedder(
        model_path="models/embed/redimnet2_b2.onnx",
        options=options(),
        embedding_dim=512,  # the config says 512, the model returns 192
        window_samples=64000,
        max_window_samples=96000,
        multi_window_enabled=False,
        multi_window_trigger_samples=128000,
        multi_window_hop_samples=32000,
        max_windows=8,
        session_factory=lambda path, opts: FakeEmbedderSession(EMBEDDING_DIM),
    )
    with pytest.raises(ModelError, match="does not match"):
        embedder.embed(np.zeros(64000, dtype=np.float32) + 0.1)


def test_embedder_refuses_an_empty_buffer():
    with pytest.raises(ModelError, match="empty"):
        build_embedder().embed(np.zeros(0, dtype=np.float32))


def test_multi_window_pooling_engages_on_long_speech(fixture_path):
    """Above the trigger, several windows are pooled rather than one cropped."""
    from speakerid.audio.io import load_audio

    embedder = build_embedder(multi_window_trigger_samples=16000)
    samples = load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0)
    before = embedder.session.run_count
    embedder.embed(samples)
    assert embedder.session.run_count - before > 1


def test_pooled_embedding_is_renormalised():
    left = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    left[0] = 1.0
    right = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    right[1] = 1.0
    pooled = SpeakerEmbedder.pool([left, right])
    assert float(np.linalg.norm(pooled)) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalize_rejects_a_dead_embedding():
    with pytest.raises(ModelError, match="near-zero norm"):
        l2_normalize(np.zeros(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# Countermeasure contract
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    result = softmax(np.array([2.0, -1.0], dtype=np.float32))
    assert float(np.sum(result)) == pytest.approx(1.0)


def test_softmax_is_stable_on_large_logits():
    result = softmax(np.array([1000.0, 999.0], dtype=np.float64))
    assert np.all(np.isfinite(result))


def build_cm(**overrides) -> Countermeasure:
    settings = dict(input_samples=64600, bona_fide_index=1)
    settings.update(overrides)
    return Countermeasure(
        model_path="models/cm/aasist_l.onnx",
        options=options(),
        session_factory=fake_session_factory,
        **settings,
    )


def test_countermeasure_returns_a_probability(fixture_path):
    from speakerid.audio.io import load_audio

    score = build_cm().score(load_audio(fixture_path("alice_01.wav"), peak_dbfs=-3.0))
    assert 0.0 <= score <= 1.0


def test_countermeasure_rejects_a_bad_index():
    cm = build_cm(bona_fide_index=5)
    with pytest.raises(ModelError, match="bona_fide_index"):
        cm.score(np.zeros(64600, dtype=np.float32) + 0.1)


def test_countermeasure_pads_short_input_by_repetition():
    """A 1 s clip must still produce a score, filled by repeating speech."""
    cm = build_cm()
    short = np.sin(np.linspace(0, 200, 16000)).astype(np.float32)
    assert 0.0 <= cm.score(short) <= 1.0


def test_countermeasure_rejects_the_wrong_logit_count():
    class ThreeLogitSession(FakeCountermeasureSession):
        def _compute(self, feeds):
            return [np.array([[0.1, 0.2, 0.7]], dtype=np.float32)]

    cm = Countermeasure(
        model_path="models/cm/aasist_l.onnx",
        options=options(),
        input_samples=64600,
        bona_fide_index=1,
        session_factory=lambda path, opts: ThreeLogitSession(),
    )
    with pytest.raises(ModelError, match="expected 2"):
        cm.score(np.zeros(64600, dtype=np.float32) + 0.1)
