"""Export AASIST-L to ONNX from the official clovaai checkpoint.

RUNS ON A CONNECTED DEVELOPMENT MACHINE ONLY (CLAUDE.md §2, §5).

The AASIST authors publish PyTorch weights, not ONNX, so the countermeasure
slot has to be exported here rather than downloaded ready-made. The upstream
``Model.forward`` returns ``(last_hidden, output)``; only the second is the
2-class logit vector this system consumes, so the export wraps the module to
return that alone and to match the declared contract::

    input:  float32 [1, 64600]   raw waveform, 16 kHz, ~4.04 s
    output: float32 [1, 2]       logits; softmax, index 1 = bona fide

Parity is verified against the PyTorch module on a fixed waveform before the
file is accepted.

Usage (needs the upstream repo's ``models/AASIST.py`` and the ``.pth``)::

    python -m scripts.export_aasist \\
        --source-dir /path/to/aasist \\
        --output models/cm/aasist_l.onnx
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from typing import Any, Optional, Sequence

import numpy as np

from speakerid.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

#: Fixed by the AASIST-L architecture; also the value in config/default.yaml.
NB_SAMP = 64600

#: An export is accepted only if it reproduces the checkpoint this closely.
PARITY_ATOL = 1e-4


def _load_model_module(source_dir: str) -> Any:
    """Import the upstream ``models/AASIST.py`` without installing the repo."""
    module_path = os.path.join(source_dir, "models", "AASIST.py")
    if not os.path.isfile(module_path):
        raise SystemExit(
            "could not find {0}. Clone https://github.com/clovaai/aasist and "
            "pass its path with --source-dir.".format(module_path)
        )
    spec = importlib.util.spec_from_file_location("aasist_model", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit("could not import {0}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["aasist_model"] = module
    spec.loader.exec_module(module)
    return module


def build_model(source_dir: str, weights_path: str, config_path: str) -> Any:
    """Instantiate AASIST-L and load the published weights."""
    import torch

    module = _load_model_module(source_dir)
    with open(config_path, encoding="utf-8") as handle:
        model_config = json.load(handle)["model_config"]

    model = module.Model(model_config)
    # weights_only=True: a .pth is a pickle, and this one is downloaded from
    # the internet. The published checkpoint is a plain tensor state dict, so
    # restricting unpickling costs nothing and closes an arbitrary-code path
    # in a script whose whole job is building a security component.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # The published checkpoint is a DataParallel state dict.
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("loaded AASIST-L: %d parameters", total)
    return model


class LogitsOnly(object):
    """Placeholder so the wrapper class is built only once torch is imported."""


def _wrap(model: Any) -> Any:
    """Return a module whose forward is just the 2-class logits."""
    import torch.nn as nn

    class _LogitsOnly(nn.Module):
        def __init__(self, inner: Any) -> None:
            super(_LogitsOnly, self).__init__()
            self.inner = inner

        def forward(self, waveform):  # type: ignore[no-untyped-def]
            # Upstream returns (last_hidden, output); only the logits are part
            # of this system's contract.
            _hidden, logits = self.inner(waveform)
            return logits

    wrapped = _LogitsOnly(model)
    wrapped.eval()
    return wrapped


def parity_waveform(num_samples: int = NB_SAMP, seed: int = 20260920) -> np.ndarray:
    """Deterministic pseudo-speech, identical on every machine."""
    rng = np.random.RandomState(seed)
    time = np.arange(num_samples, dtype=np.float32) / 16000.0
    tone = 0.3 * np.sin(2.0 * np.pi * 180.0 * time).astype(np.float32)
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * time)).astype(np.float32)
    wave = (0.1 * rng.randn(num_samples).astype(np.float32) + tone) * envelope
    return np.ascontiguousarray(wave / (np.max(np.abs(wave)) or 1.0) * 0.7, dtype=np.float32)


def export(
    source_dir: str,
    weights_path: str,
    config_path: str,
    output_path: str,
    opset: int = 14,
) -> str:
    import torch

    model = _wrap(build_model(source_dir, weights_path, config_path))
    waveform = parity_waveform()
    dummy = torch.from_numpy(waveform).reshape(1, -1)

    with torch.no_grad():
        reference = model(dummy).cpu().numpy().reshape(-1)
    LOGGER.info("reference logits: %s", np.round(reference, 5).tolist())

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    LOGGER.info("exported to %s (%.1f KB)", output_path, os.path.getsize(output_path) / 1024.0)

    produced = _run_onnx(output_path, waveform)
    delta = float(np.max(np.abs(produced - reference)))
    if delta > PARITY_ATOL:
        raise SystemExit(
            "export parity FAILED: max |onnx - torch| = {0:.3e} > {1:.0e}. "
            "Do not ship this file.".format(delta, PARITY_ATOL)
        )
    LOGGER.info("parity ok: max abs difference %.3e", delta)
    return output_path


def _run_onnx(path: str, waveform: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: waveform.reshape(1, -1)})[0]).reshape(-1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", required=True, help="Clone of clovaai/aasist")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="models/cm/aasist_l.onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args(argv)

    configure_logging(level="INFO")
    weights = args.weights or os.path.join(args.source_dir, "models", "weights", "AASIST-L.pth")
    config = args.config or os.path.join(args.source_dir, "config", "AASIST-L.conf")
    export(args.source_dir, weights, config, args.output, args.opset)
    LOGGER.info("now re-run scripts/fetch_models.py to refresh MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
