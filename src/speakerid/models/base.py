"""ONNX Runtime session management.

Rules this module exists to enforce (CLAUDE.md §12):

* One session per model, created once at startup, never per request.
  Creating a session per request is a 100x latency bug.
* Threads pinned from config, so the device does not oversubscribe its cores.
* Every session warmed with a dummy input before first real use, because the
  first inference pays the memory-arena and kernel-selection cost.
* Every model file's SHA-256 verified against ``models/MANIFEST.json`` on load.

Inference is ONNX Runtime only. PyTorch is a development-time dependency for
export and optional fine-tuning; it is never imported at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..errors import ManifestError, ModelError
from ..nptypes import FloatArray

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "MANIFEST.json"

#: A session factory takes a model path plus runtime options and returns an
#: object with ONNX Runtime's ``get_inputs``/``get_outputs``/``run`` surface.
#: Tests inject a fake through this seam, which is why the test suite needs no
#: model weights and no network.
SessionFactory = Callable[[str, "RuntimeOptions"], Any]


class RuntimeOptions(object):
    """Resolved ONNX Runtime settings, read once from config."""

    def __init__(
        self,
        intra_op_num_threads: int,
        inter_op_num_threads: int,
        execution_mode: str,
        graph_optimization_level: str,
        providers: Sequence[str],
        warmup_iterations: int,
        verify_manifest: bool,
    ) -> None:
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.execution_mode = execution_mode
        self.graph_optimization_level = graph_optimization_level
        self.providers = list(providers)
        self.warmup_iterations = warmup_iterations
        self.verify_manifest = verify_manifest

    @classmethod
    def from_config(cls, config: Config) -> RuntimeOptions:
        return cls(
            intra_op_num_threads=config.get_int("runtime.intra_op_num_threads"),
            inter_op_num_threads=config.get_int("runtime.inter_op_num_threads"),
            execution_mode=config.get_str("runtime.execution_mode"),
            graph_optimization_level=config.get_str("runtime.graph_optimization_level"),
            providers=config.get_str_list("runtime.providers"),
            warmup_iterations=config.get_int("runtime.warmup_iterations"),
            verify_manifest=config.get_bool("runtime.verify_manifest"),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(models_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(models_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise ManifestError(
            "models/{0} is missing. Models are immutable binary assets and must "
            "be checksummed before use; run scripts/fetch_models.py on a "
            "connected machine to produce it.".format(MANIFEST_FILENAME)
        )
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        raise ManifestError(
            "models/{0} is not valid JSON: {1}".format(MANIFEST_FILENAME, exc)
        ) from exc
    if not isinstance(manifest, dict):
        raise ManifestError("models/{0} must contain a JSON object".format(MANIFEST_FILENAME))
    entries = manifest.get("models")
    if not isinstance(entries, dict):
        raise ManifestError("models/{0} has no 'models' mapping".format(MANIFEST_FILENAME))
    loaded: Dict[str, Any] = manifest
    return loaded


def verify_against_manifest(model_path: str, models_dir: str) -> None:
    """Fail closed if a model file is unknown to, or differs from, the manifest.

    A silently swapped model file is indistinguishable from a working system
    until the accuracy numbers move, so this is checked on every load.
    """
    manifest = load_manifest(models_dir)
    key = os.path.relpath(os.path.abspath(model_path), os.path.abspath(models_dir))
    key = key.replace(os.sep, "/")
    entry = manifest["models"].get(key)
    if entry is None:
        raise ManifestError(
            "{0} is not listed in models/{1}; refusing to load an "
            "unrecorded model".format(key, MANIFEST_FILENAME)
        )
    expected = str(entry.get("sha256", "")).lower()
    if not expected:
        raise ManifestError("manifest entry for {0} has no sha256".format(key))
    actual = sha256_file(model_path)
    if actual != expected:
        raise ManifestError(
            "SHA-256 mismatch for {0}:\n  expected {1}\n  actual   {2}\n"
            "The model file has changed since it was recorded.".format(key, expected, actual)
        )
    LOGGER.debug("manifest verified for %s", key)


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def default_session_factory(model_path: str, options: RuntimeOptions) -> Any:
    """Build a real ``onnxruntime.InferenceSession`` with pinned threads."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ModelError(
            "onnxruntime is required for inference but is not installed: {0}".format(exc)
        ) from exc

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = options.intra_op_num_threads
    sess_options.inter_op_num_threads = options.inter_op_num_threads
    sess_options.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL
        if options.execution_mode == "parallel"
        else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if options.graph_optimization_level not in levels:
        raise ModelError(
            "unknown runtime.graph_optimization_level: {0}".format(options.graph_optimization_level)
        )
    sess_options.graph_optimization_level = levels[options.graph_optimization_level]
    # Telemetry is off by default in onnxruntime, but say so explicitly: this
    # system is air-gapped and the intent must be visible in the code.
    sess_options.log_severity_level = 3

    return ort.InferenceSession(model_path, sess_options, providers=options.providers)


class OnnxModel(object):
    """Base class for a single loaded ONNX model.

    Subclasses implement :meth:`_warmup_feeds` to describe a dummy input, and
    call :meth:`_run` to infer. They must not create sessions themselves.
    """

    #: Human-readable slot name, used in log lines and error messages.
    slot = "model"

    def __init__(
        self,
        model_path: str,
        options: RuntimeOptions,
        models_dir: Optional[str] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        self.model_path = model_path
        self.options = options
        self.models_dir = models_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(model_path))
        )
        factory = session_factory or default_session_factory

        if session_factory is None:
            # Only a real on-disk model is checksummed; an injected session has
            # no file to verify and is a test or an in-memory export.
            if not os.path.isfile(model_path):
                raise ModelError(
                    "{0} model not found at {1}. Models ship by USB, never by "
                    "download at first run (CLAUDE.md §2).".format(self.slot, model_path)
                )
            if options.verify_manifest:
                verify_against_manifest(model_path, self.models_dir)

        LOGGER.info("loading %s session from %s", self.slot, model_path)
        self.session = factory(model_path, options)
        self.input_names = [spec.name for spec in self.session.get_inputs()]
        self.output_names = [spec.name for spec in self.session.get_outputs()]
        self.input_shapes = [
            list(getattr(spec, "shape", []) or []) for spec in self.session.get_inputs()
        ]
        self._validate_signature()
        self._warmup()

    # -- hooks ---------------------------------------------------------------

    def _validate_signature(self) -> None:
        """Check the model's inputs match what this wrapper will feed it."""

    def input_rank(self, index: int = 0, default: int = 2) -> int:
        """Declared rank of an input, or ``default`` when the model is silent.

        Exports of the same architecture disagree about waveform layout: some
        take ``[batch, samples]``, others ``[batch, 1, samples]``. Reading the
        rank from the graph and shaping the feed to match means a re-export
        with a different convention is handled rather than rejected at runtime.
        """
        if index >= len(self.input_shapes):
            return default
        shape = self.input_shapes[index]
        return len(shape) if shape else default

    def _warmup_feeds(self) -> Dict[str, FloatArray]:
        raise NotImplementedError

    # -- inference -----------------------------------------------------------

    def _warmup(self) -> None:
        feeds = self._warmup_feeds()
        for _ in range(max(0, self.options.warmup_iterations)):
            self._run(feeds)
        LOGGER.debug(
            "%s warmed with %d iteration(s)", self.slot, max(0, self.options.warmup_iterations)
        )

    def _run(self, feeds: Dict[str, FloatArray]) -> List[FloatArray]:
        try:
            outputs = self.session.run(None, feeds)
        except Exception as exc:  # onnxruntime raises a variety of C++ bindings
            raise ModelError("{0} inference failed: {1}".format(self.slot, exc)) from exc
        return [np.asarray(item) for item in outputs]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<{0} slot={1} path={2}>".format(
            type(self).__name__, self.slot, os.path.basename(self.model_path)
        )
