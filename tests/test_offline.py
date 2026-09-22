"""The system must never reach the network.

CLAUDE.md §2: fully offline at runtime and at test time. No network calls
ever, no downloads at import, no telemetry. The machine is assumed to have no
route to the internet, so anything that tries is a bug that would only show up
on the air-gapped box -- the worst possible place to find it.

These tests make the guarantee executable rather than aspirational.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys

import pytest

RUNTIME_MODULES = [
    "speakerid",
    "speakerid.config",
    "speakerid.errors",
    "speakerid.metrics",
    "speakerid.challenge",
    "speakerid.logging_setup",
    "speakerid.pipeline",
    "speakerid.audio.io",
    "speakerid.audio.vad",
    "speakerid.audio.enhance",
    "speakerid.models.base",
    "speakerid.models.embedder",
    "speakerid.models.countermeasure",
    "speakerid.db.crypto",
    "speakerid.db.schema",
    "speakerid.db.store",
    "speakerid.scoring.similarity",
    "speakerid.scoring.asnorm",
    "speakerid.scoring.decide",
    "speakerid.cli",
]

#: Development-only dependencies. Importing one at runtime would drag PyTorch
#: onto a device that is supposed to run ONNX Runtime alone (CLAUDE.md §4).
DEV_ONLY_MODULES = ("torch", "torchaudio", "onnx", "onnxsim", "sklearn", "matplotlib")


@pytest.fixture
def preserve_modules():
    """Restore sys.modules after a test that re-imports the package.

    Removing a module and importing it again produces a *new* class object,
    while anything that imported the old one still points at it. That makes
    later monkeypatching silently ineffective, so whatever these tests evict
    has to be put back.
    """
    watched = tuple(RUNTIME_MODULES) + DEV_ONLY_MODULES
    saved = {name: sys.modules[name] for name in watched if name in sys.modules}
    try:
        yield
    finally:
        for name in watched:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture
def no_network(monkeypatch):
    """Make every outbound connection raise."""

    def refuse(*args, **kwargs):
        raise AssertionError("the system attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    if hasattr(socket, "getaddrinfo"):
        monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return True


def test_runtime_modules_import_without_network(no_network, preserve_modules):
    for name in RUNTIME_MODULES:
        if name in sys.modules:
            del sys.modules[name]
    for name in RUNTIME_MODULES:
        importlib.import_module(name)


def test_runtime_does_not_import_pytorch(preserve_modules):
    """PyTorch is an export-time tool and must not be on the device."""
    for name in RUNTIME_MODULES:
        sys.modules.pop(name, None)
    for dev_module in DEV_ONLY_MODULES:
        sys.modules.pop(dev_module, None)

    for name in RUNTIME_MODULES:
        importlib.import_module(name)

    still_absent = [name for name in DEV_ONLY_MODULES if name not in sys.modules]
    assert "torch" in still_absent, "importing the runtime package pulled in PyTorch"


def test_full_pipeline_runs_without_network(no_network, enrolled_pipeline, fixture_path):
    """The decision path itself, start to finish, with sockets disabled."""
    result = enrolled_pipeline.identify_file(fixture_path("alice_04.wav"))
    assert result.decision.person_id == "alice"


def test_enrolment_runs_without_network(no_network, pipeline, fixture_path):
    pipeline.store.add_person(person_id="dave")
    pipeline.store.record_consent("dave", policy_version="1", granted_by="tester")
    assert pipeline.enroll_file("dave", fixture_path("dave_01.wav")).accepted


#: Scripts that run only on a connected development machine. Importing one
#: from the package would put model downloading or PyTorch on the device.
CONNECTED_MACHINE_ONLY = ("scripts.fetch_models", "scripts.export_onnx")


def _imported_modules(path):
    """Module names a file actually imports, via AST rather than text search.

    A string search would flag ``models/base.py``, which names the fetch
    script in an error message so the operator knows what to run -- exactly
    the kind of helpful reference that must stay allowed.
    """
    import ast

    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update("{0}.{1}".format(node.module, a.name) for a in node.names)
    return found


def test_no_runtime_module_imports_a_connected_machine_script():
    """scripts/fetch_models.py and export_onnx.py never reach the device."""
    import speakerid

    source_dir = os.path.dirname(os.path.abspath(speakerid.__file__))
    offenders = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            for imported in _imported_modules(path):
                if imported.startswith(CONNECTED_MACHINE_ONLY):
                    offenders.append("{0} imports {1}".format(path, imported))
    assert offenders == [], offenders


def test_no_runtime_module_imports_a_development_dependency():
    """PyTorch, onnx and scikit-learn are export-time tools only."""
    import speakerid

    source_dir = os.path.dirname(os.path.abspath(speakerid.__file__))
    offenders = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            for imported in _imported_modules(path):
                if imported.split(".")[0] in DEV_ONLY_MODULES:
                    offenders.append("{0} imports {1}".format(path, imported))
    assert offenders == [], offenders


def test_no_runtime_module_contains_a_url_it_would_fetch():
    """Documentation URLs are fine; a call that would retrieve one is not."""
    import speakerid

    source_dir = os.path.dirname(os.path.abspath(speakerid.__file__))
    forbidden = (
        "urllib.request",
        "requests.get",
        "requests.post",
        "httpx.",
        "urlopen",
        "hf_hub_download",
        "snapshot_download",
    )
    offenders = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
            for needle in forbidden:
                if needle in content:
                    offenders.append("{0}: {1}".format(path, needle))
    assert offenders == [], "runtime code can fetch from the network: {0}".format(offenders)


def test_api_binds_loopback_only():
    """A biometric endpoint on 0.0.0.0 is a different product."""
    from speakerid.config import load_config

    assert load_config().get_str("api.host") in ("127.0.0.1", "localhost", "::1")
