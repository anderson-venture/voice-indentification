"""Export parity: the ONNX graph must reproduce the PyTorch checkpoint.

CLAUDE.md §5: an embedding export is accepted only if its output has cosine
similarity >= 0.9999 against the reference checkpoint on a fixed waveform, and
remains unit-norm.

This needs the real model files and a torch install, neither of which lives in
the repository. When they are absent the tests **skip** -- they are not mocked
into passing, because a mocked parity check is worse than no parity check: it
reports success for something nobody verified.

To run them for real, on a development machine:

    python -m scripts.fetch_models            # writes models/MANIFEST.json
    SPEAKERID_REFERENCE_CHECKPOINT=/path/to/redimnet_b2.pt \\
        pytest tests/test_export_parity.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scripts.export_onnx import PARITY_COSINE_MIN, parity_waveform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDER_ONNX = os.path.join(PROJECT_ROOT, "models", "embed", "redimnet2_b2.onnx")
CHECKPOINT_ENV = "SPEAKERID_REFERENCE_CHECKPOINT"


def test_parity_waveform_is_deterministic():
    """Runs anywhere: the reference input must be identical on every machine."""
    assert np.array_equal(parity_waveform(), parity_waveform())
    assert parity_waveform().dtype == np.float32
    assert float(np.max(np.abs(parity_waveform()))) == pytest.approx(0.7, abs=1e-5)


def test_parity_waveform_has_signal_in_it():
    waveform = parity_waveform()
    assert float(np.std(waveform)) > 0.05, "a silent reference would prove nothing"


def _shape_for(session, waveform):
    """Lay a waveform out for whichever rank this export declares.

    The shipped ReDimNet-B2 export takes ``[batch, 1, samples]``; CLAUDE.md §5
    describes ``[1, N]``. Both are legitimate exports of the same architecture,
    so the production wrapper reads the rank from the graph and these tests do
    the same rather than hardcoding one.
    """
    shape = list(session.get_inputs()[0].shape or [])
    return waveform.reshape(1, 1, -1) if len(shape) == 3 else waveform.reshape(1, -1)


@pytest.mark.skipif(
    not os.path.isfile(EMBEDDER_ONNX),
    reason="models/embed/redimnet2_b2.onnx absent; models ship by USB, not in git",
)
def test_onnx_export_is_loadable_and_honours_the_contract():
    """Interface contract from CLAUDE.md §5: one waveform in, [1, 192] out."""
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(EMBEDDER_ONNX, providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1

    waveform = parity_waveform(64000)
    output = np.asarray(
        session.run(None, {session.get_inputs()[0].name: _shape_for(session, waveform)})[0]
    )
    assert output.shape == (1, 192)
    assert np.all(np.isfinite(output))


@pytest.mark.skipif(
    not os.path.isfile(EMBEDDER_ONNX),
    reason="models/embed/redimnet2_b2.onnx absent",
)
def test_wrapper_normalises_an_export_that_does_not():
    """The shipped export returns vectors of norm ~18, not unit-norm.

    CLAUDE.md §5 says the embedder output is L2-normalized. This particular
    export is not, which is exactly why ``SpeakerEmbedder`` normalises
    defensively -- trusting the stated contract would have inflated every
    cosine score by a factor of ~18 and silently moved the operating point.
    """
    ort = pytest.importorskip("onnxruntime")
    from speakerid.models.base import RuntimeOptions
    from speakerid.models.embedder import SpeakerEmbedder

    session = ort.InferenceSession(EMBEDDER_ONNX, providers=["CPUExecutionProvider"])
    waveform = parity_waveform(64000)
    raw = np.asarray(
        session.run(None, {session.get_inputs()[0].name: _shape_for(session, waveform)})[0]
    ).reshape(-1)
    assert float(np.linalg.norm(raw)) > 2.0, "this export is unexpectedly unit-norm already"

    embedder = SpeakerEmbedder(
        model_path=EMBEDDER_ONNX,
        options=RuntimeOptions(2, 1, "sequential", "all", ["CPUExecutionProvider"], 1, False),
        embedding_dim=192,
        window_samples=64000,
        max_window_samples=96000,
        multi_window_enabled=False,
        multi_window_trigger_samples=128000,
        multi_window_hop_samples=32000,
        max_windows=8,
    )
    embedding = embedder.embed_window(waveform)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)
    # Same direction, unit length: normalisation must not change the identity.
    assert float(np.dot(embedding, raw / np.linalg.norm(raw))) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(
    not os.path.isfile(EMBEDDER_ONNX),
    reason="models/embed/redimnet2_b2.onnx absent",
)
def test_onnx_export_accepts_the_full_window_range():
    """The embedder must take 4 s to 6 s (64000 to 96000 samples)."""
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(EMBEDDER_ONNX, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    for num_samples in (64000, 80000, 96000):
        feed = _shape_for(session, parity_waveform(num_samples))
        output = session.run(None, {name: feed})[0]
        assert np.asarray(output).shape == (1, 192)


@pytest.mark.skipif(
    not os.path.isfile(EMBEDDER_ONNX) or not os.environ.get(CHECKPOINT_ENV),
    reason="needs both the ONNX export and {0} pointing at the checkpoint".format(CHECKPOINT_ENV),
)
def test_onnx_matches_the_pytorch_reference():
    """The parity bar itself. Skipped unless both halves are present."""
    pytest.importorskip("torch")
    import torch
    from scripts.export_onnx import verify_parity

    checkpoint = os.environ[CHECKPOINT_ENV]
    model = torch.load(checkpoint, map_location="cpu")
    model.eval()

    waveform = parity_waveform(64000).reshape(1, -1)
    with torch.no_grad():
        reference = model(torch.from_numpy(waveform)).cpu().numpy().reshape(-1)

    cosine, norm = verify_parity(EMBEDDER_ONNX, reference, waveform)
    assert cosine >= PARITY_COSINE_MIN, "export does not reproduce the checkpoint"
    assert norm == pytest.approx(1.0, abs=1e-3), "export stopped normalising"


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(PROJECT_ROOT, "models", "MANIFEST.json")),
    reason="models/MANIFEST.json absent until scripts/fetch_models.py has run",
)
def test_every_manifest_entry_is_present_and_unchanged():
    from speakerid.models.base import load_manifest, verify_against_manifest

    models_dir = os.path.join(PROJECT_ROOT, "models")
    manifest = load_manifest(models_dir)
    for relative in manifest["models"]:
        path = os.path.join(models_dir, relative)
        assert os.path.isfile(path), "{0} is in the manifest but missing".format(relative)
        verify_against_manifest(path, models_dir)
