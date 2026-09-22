"""Export PyTorch checkpoints to ONNX.

RUNS ON A CONNECTED DEVELOPMENT MACHINE ONLY. PyTorch is a development
dependency and is never imported at runtime (CLAUDE.md §2, §4).

An export is only accepted if it reproduces the reference PyTorch output:
cosine similarity >= 0.9999 on a fixed test waveform, and still unit-norm.
``tests/test_export_parity.py`` re-checks this whenever both the checkpoint
and the ONNX file are available.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from speakerid.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

#: Parity bar an export must clear before it may be trusted (CLAUDE.md §5).
PARITY_COSINE_MIN = 0.9999

#: Fixed waveform used for parity, so the number is comparable across runs.
PARITY_SEED = 20260920
PARITY_SAMPLES = 64000


def parity_waveform(num_samples: int = PARITY_SAMPLES) -> np.ndarray:
    """A deterministic pseudo-speech waveform.

    Not real speech, but fixed: parity is about the export matching the
    checkpoint, not about either being accurate.
    """
    rng = np.random.RandomState(PARITY_SEED)
    noise = rng.randn(num_samples).astype(np.float32)
    time = np.arange(num_samples, dtype=np.float32) / 16000.0
    tone = 0.3 * np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 2.5 * time)).astype(np.float32)
    waveform = (0.1 * noise + tone) * envelope
    peak = float(np.max(np.abs(waveform)))
    return np.ascontiguousarray(waveform / peak * 0.7, dtype=np.float32)


def export_embedder(
    checkpoint_path: str,
    output_path: str,
    num_samples: int = PARITY_SAMPLES,
    opset: int = 14,
    simplify: bool = True,
) -> str:
    """Export a speaker-embedding checkpoint and verify parity."""
    import torch  # dev-only dependency

    LOGGER.info("loading checkpoint %s", checkpoint_path)
    model = torch.load(checkpoint_path, map_location="cpu")
    if not hasattr(model, "forward"):
        raise SystemExit(
            "{0} does not contain a callable module. Load the architecture and "
            "its state_dict in a wrapper script, then call export_embedder with "
            "the assembled module.".format(checkpoint_path)
        )
    model.eval()

    dummy = torch.from_numpy(parity_waveform(num_samples)).reshape(1, -1)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with torch.no_grad():
        reference = model(dummy).cpu().numpy().reshape(-1)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["waveform"],
        output_names=["embedding"],
        # The embedder must accept 4-6 s, so the sample axis is dynamic.
        dynamic_axes={"waveform": {1: "num_samples"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    LOGGER.info("exported to %s", output_path)

    if simplify:
        _simplify(output_path)

    cosine, norm = verify_parity(output_path, reference, dummy.numpy())
    if cosine < PARITY_COSINE_MIN:
        raise SystemExit(
            "export parity FAILED: cosine {0:.6f} < {1}. The ONNX graph does not "
            "reproduce the checkpoint; do not ship it.".format(cosine, PARITY_COSINE_MIN)
        )
    LOGGER.info("parity ok: cosine=%.6f, output norm=%.6f", cosine, norm)
    return output_path


def _simplify(path: str) -> None:
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
    except ImportError:
        LOGGER.warning("onnxsim not available; shipping the unsimplified graph")
        return
    model = onnx.load(path)
    simplified, ok = onnx_simplify(model)
    if ok:
        onnx.save(simplified, path)
        LOGGER.info("simplified graph")
    else:
        LOGGER.warning("onnxsim could not validate the simplified graph; keeping the original")


def verify_parity(
    onnx_path: str, reference: np.ndarray, waveform: np.ndarray
) -> Tuple[float, float]:
    """Compare the ONNX output against the PyTorch reference."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    produced = np.asarray(session.run(None, {name: waveform.astype(np.float32)})[0]).reshape(-1)

    reference_norm = reference / (np.linalg.norm(reference) or 1.0)
    produced_norm = produced / (np.linalg.norm(produced) or 1.0)
    cosine = float(np.dot(reference_norm, produced_norm))
    return cosine, float(np.linalg.norm(produced))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=PARITY_SAMPLES)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level="INFO")
    export_embedder(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.samples,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
    LOGGER.info("remember to re-run scripts/fetch_models.py to refresh MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
