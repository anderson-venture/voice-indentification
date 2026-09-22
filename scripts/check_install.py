"""Preflight: does this copy of the tree actually work on this machine?

Run it first thing after copying the folder to the offline machine. It answers
the only question that matters at that moment -- will this run here -- and it
answers it by doing the work, not by inspecting version strings.

    venv\\Scripts\\python -m scripts.check_install

Exits 0 if every check passes, 1 otherwise. It makes no network calls; one
check actively proves that the pipeline does not need any.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import List, Tuple

RESULTS: List[Tuple[str, bool, str]] = []

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    sys.stdout.write(
        "  [{0}] {1}{2}\n".format(
            "PASS" if ok else "FAIL", name, "  -- " + detail if detail else ""
        )
    )
    sys.stdout.flush()
    return bool(ok)


def guarded(name, fn):
    try:
        ok, detail = fn()
        return check(name, ok, detail)
    except Exception as exc:  # noqa: BLE001 -- a preflight reports, never raises
        return check(name, False, "{0}: {1}".format(type(exc).__name__, exc))


# -- 1. the interpreter is the one in this folder ---------------------------


def _interpreter():
    inside = os.path.abspath(sys.prefix).lower().startswith(PROJECT.lower())
    return inside, "{0}.{1}.{2} at {3}".format(
        sys.version_info[0], sys.version_info[1], sys.version_info[2], sys.prefix
    )


def _stdlib_is_local():
    """The venv holds no stdlib; it loads it from python38/ via pyvenv.cfg."""
    base = os.path.abspath(sys.base_prefix)
    return base.lower().startswith(PROJECT.lower()), "base_prefix {0}".format(base)


def _no_foreign_paths():
    foreign = [
        p for p in sys.path if p and not os.path.abspath(p).lower().startswith(PROJECT.lower())
    ]
    return not foreign, "{0} entries outside the project".format(len(foreign)) + (
        ": " + "; ".join(foreign[:3]) if foreign else ""
    )


# -- 2. the binaries load on this CPU ---------------------------------------


def _imports():
    missing = []
    for module in (
        "numpy",
        "scipy",
        "onnxruntime",
        "soundfile",
        "librosa",
        "sqlalchemy",
        "cryptography",
        "pydantic",
        "click",
        "yaml",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            missing.append("{0} ({1})".format(module, type(exc).__name__))
    return not missing, "runtime deps: " + (", ".join(missing) if missing else "all import")


def _optional_imports():
    present, absent = [], []
    for module in ("torch", "torchaudio", "onnx", "onnxsim", "fastapi", "uvicorn", "keyring"):
        try:
            importlib.import_module(module)
            present.append(module)
        except Exception:  # noqa: BLE001
            absent.append(module)
    # Optional by definition, so absence is reported rather than failed.
    return True, "{0} present{1}".format(
        len(present), "; missing " + ", ".join(absent) if absent else ""
    )


def _numeric_kernels():
    """Catch an illegal-instruction class of failure on an older CPU."""
    import numpy as np

    a = np.random.RandomState(0).randn(256, 256).astype(np.float32)
    value = float(np.linalg.svd(a, compute_uv=False)[0])
    ok = value > 0 and np.isfinite(value)
    detail = "numpy BLAS ok"
    try:
        import torch

        t = torch.rand(64, 64)
        ok = ok and bool(torch.isfinite((t @ t.T).sum()))
        detail += ", torch matmul ok"
    except ImportError:
        detail += ", torch absent"
    return ok, detail


# -- 3. the models load and run ---------------------------------------------


def _models():
    from speakerid.config import load_config
    from speakerid.models.base import verify_against_manifest

    config = load_config()
    models_dir = os.path.join(PROJECT, "models")
    names = []
    for key in ("vad_model.model_path", "embedder.model_path", "countermeasure.model_path"):
        path = config.resolve_path(key)
        if not os.path.isfile(path):
            return False, "missing {0}".format(path)
        verify_against_manifest(path, models_dir)
        names.append(os.path.basename(path))
    return True, "checksums match: " + ", ".join(names)


def _inference():
    """One real pass through all three models on a synthetic buffer."""
    import numpy as np
    import onnxruntime as ort

    from speakerid.config import load_config

    config = load_config()
    rng = np.random.RandomState(0)

    def session(key):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.get_int("runtime.intra_op_num_threads")
        return ort.InferenceSession(
            config.resolve_path(key), opts, providers=["CPUExecutionProvider"]
        )

    vad = session("vad_model.model_path")
    out, _state = vad.run(
        None,
        {
            "input": rng.randn(1, 576).astype(np.float32) * 0.05,
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "sr": np.array(16000, dtype=np.int64),
        },
    )
    vad_ok = out.shape == (1, 1)

    cm = session("countermeasure.model_path")
    spec = cm.get_inputs()[0]
    logits = cm.run(None, {spec.name: rng.randn(1, 64600).astype(np.float32) * 0.05})[0]
    cm_ok = logits.shape == (1, 2) and np.all(np.isfinite(logits))

    emb = session("embedder.model_path")
    spec = emb.get_inputs()[0]
    wav = (rng.randn(64000) * 0.05).astype(np.float32)
    wav = wav.reshape(1, 1, -1) if len(spec.shape) == 3 else wav.reshape(1, -1)
    vector = emb.run(None, {spec.name: wav})[0]
    emb_ok = vector.shape == (1, 192) and np.all(np.isfinite(vector))

    return (vad_ok and cm_ok and emb_ok), "vad {0}, cm {1}, embedder {2}".format(
        out.shape, logits.shape, vector.shape
    )


def _pipeline_offline():
    """Build the whole pipeline with the network stubbed out.

    If anything in the import or construction path reaches for a socket, this
    is where it shows up -- on the offline machine that failure would
    otherwise look like an unexplained hang.
    """
    import shutil
    import socket
    import tempfile

    from speakerid.config import Thresholds, load_config
    from speakerid.pipeline import Pipeline

    # Stand-in thresholds and a scratch database: a fresh copy is uncalibrated
    # and has no cohort by design, and both refusals are checked separately.
    # The question here is only whether construction reaches for a socket.
    stand_in = Thresholds(
        cm_threshold=0.5, accept_threshold=0.5, margin_threshold=0.1, min_speech_sec=2.0
    )
    scratch = tempfile.mkdtemp(prefix="speakerid-preflight-")
    config = load_config(
        overrides={
            "database": {"url": "sqlite:///" + scratch.replace("\\", "/") + "/probe.db"},
            "crypto": {"salt_path": os.path.join(scratch, ".salt")},
            "scoring": {"asnorm": {"enabled": False}},
        }
    )

    real = socket.socket

    class Blocked(Exception):
        pass

    def refuse(*args, **kwargs):
        raise Blocked("the pipeline attempted a network connection")

    socket.socket = refuse  # type: ignore[assignment]
    try:
        pipeline = Pipeline.from_config(config=config, thresholds=stand_in, passphrase="preflight")
        return True, "built {0} with sockets disabled".format(type(pipeline).__name__)
    finally:
        socket.socket = real  # type: ignore[assignment]
        shutil.rmtree(scratch, ignore_errors=True)


def _cohort():
    """Absent on a fresh copy, and that is a deployment step, not a fault."""
    from speakerid.config import load_config

    config = load_config()
    if not config.get_bool("scoring.asnorm.enabled"):
        return True, "AS-Norm disabled in config"
    path = config.resolve_path("scoring.asnorm.cohort_path")
    if os.path.isfile(path):
        return True, "cohort present at {0}".format(path)
    return True, (
        "no cohort yet -- expected; run `speakerid build-cohort` on 200-500 "
        "in-room recordings of non-enrolled people (SETUP 9)"
    )


# -- 4. writability and thresholds ------------------------------------------


def _writable():
    from speakerid.config import load_config

    config = load_config()
    target = os.path.dirname(config.resolve_path("crypto.salt_path")) or PROJECT
    created = not os.path.isdir(target)
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".preflight_write_probe")
        with open(probe, "wb") as handle:
            handle.write(b"ok")
        os.remove(probe)
        return True, "can write to {0}".format(target)
    except OSError as exc:
        return False, "cannot write to {0}: {1}".format(target, exc)
    finally:
        # A preflight reports; it does not leave the data directory behind.
        if created and os.path.isdir(target) and not os.listdir(target):
            os.rmdir(target)


def _thresholds():
    """Not calibrated is the expected, correct state on a fresh copy."""
    from speakerid.config import load_thresholds
    from speakerid.errors import ConfigError

    try:
        thresholds = load_thresholds()
    except (ConfigError, Exception) as exc:  # noqa: BLE001
        if "not calibrated" in str(exc):
            return True, "not calibrated yet -- expected; run `speakerid calibrate` (SETUP 9)"
        raise
    return True, "calibrated: {0}".format(thresholds.describe()[:70])


def main():
    sys.stdout.write("\nPreflight for {0}\n\n".format(PROJECT))

    sys.stdout.write(" Interpreter\n")
    guarded("runs the interpreter inside this folder", _interpreter)
    guarded("standard library loads from python38/", _stdlib_is_local)
    guarded("no sys.path entry outside the folder", _no_foreign_paths)

    sys.stdout.write("\n Binaries on this CPU\n")
    guarded("runtime dependencies import", _imports)
    guarded("optional dependencies", _optional_imports)
    guarded("numeric kernels execute", _numeric_kernels)

    sys.stdout.write("\n Models\n")
    guarded("model files match MANIFEST.json", _models)
    guarded("all three models run a real inference", _inference)

    sys.stdout.write("\n Runtime\n")
    guarded("pipeline builds with sockets disabled", _pipeline_offline)
    guarded("data directory is writable", _writable)

    sys.stdout.write("\n Deployment steps still outstanding\n")
    guarded("thresholds state", _thresholds)
    guarded("AS-Norm cohort", _cohort)

    failed = [r for r in RESULTS if not r[1]]
    sys.stdout.write("\n" + "-" * 66 + "\n")
    if failed:
        sys.stdout.write("{0} of {1} checks FAILED:\n".format(len(failed), len(RESULTS)))
        for name, _ok, detail in failed:
            sys.stdout.write("  - {0}  {1}\n".format(name, detail))
        sys.stdout.write("\nSee docs/SETUP.md 4 and 5.\n")
        return 1
    sys.stdout.write(
        "All {0} checks passed. This copy works on this machine.\n".format(len(RESULTS))
    )
    sys.stdout.write("Next: calibrate before it will identify anyone (SETUP 9).\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
