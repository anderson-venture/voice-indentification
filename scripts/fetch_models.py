"""Acquire models and write models/MANIFEST.json.

RUNS ON A CONNECTED DEVELOPMENT MACHINE ONLY.

Nothing in ``src/speakerid`` may import this module. The runtime system is
air-gapped: models arrive by USB as immutable binary assets, and a package
that downloads a model "just for convenience" on first run is exactly the
failure CLAUDE.md §2 forbids.

Workflow
--------
1. On a connected machine, obtain the ONNX exports. The quickest path is the
   ``speakeronnx`` PyPI package, which registers ready-made exports including
   ``redimnet-b2``, ``campplus``, ``wespeaker-ecapa512`` and ``titanet-small``.
   Install it, pull the weights, copy the ``.onnx`` files into ``models/``,
   then **drop the dependency**.
2. Silero VAD v5 comes from ``snakers4/silero-vad`` (MIT); AASIST-L from
   ``clovaai/aasist`` (MIT) -- see ``scripts/export_onnx.py`` for the export.
3. Run this script to checksum everything into ``models/MANIFEST.json``.
4. Commit the refreshed ``.onnx`` files together with ``MANIFEST.json``.

This script does not download anything by itself. Fetching is left to the
tools that own each model, so that the URL, licence and version are recorded
by a human in the manifest rather than buried in a helper function.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import logging
import os
from typing import Dict, Optional, Sequence

from speakerid.logging_setup import configure_logging
from speakerid.models.base import MANIFEST_FILENAME, sha256_file

LOGGER = logging.getLogger(__name__)

#: Slot -> (relative path, description, source, licence). Mirrors the model
#: inventory table in CLAUDE.md §5.
EXPECTED_MODELS: Dict[str, Dict[str, str]] = {
    "vad": {
        "path": "vad/silero_vad.onnx",
        "model": "Silero VAD v5",
        "source": "https://github.com/snakers4/silero-vad",
        "licence": "MIT",
        "notes": "~1.8 MB ONNX; 512-sample windows at 16 kHz.",
    },
    "embedder": {
        "path": "embed/redimnet2_b2.onnx",
        "model": "ReDimNet2-B2",
        "source": "https://github.com/IDRnD/redimnet",
        "licence": "Apache-2.0",
        "notes": "3.6M params / 0.95 GMACs; input [1, N] 64000..96000, output [1, 192].",
    },
    "embedder_fallback": {
        "path": "embed/campplus.onnx",
        "model": "CAM++",
        "source": "https://github.com/modelscope/3D-Speaker",
        "licence": "Apache-2.0",
        "notes": "7.2M params / 1.15 GMACs; fallback and A/B comparison.",
        "optional": "true",
    },
    "countermeasure": {
        "path": "cm/aasist_l.onnx",
        "model": "AASIST-L",
        "source": "https://github.com/clovaai/aasist",
        "licence": "MIT",
        "notes": "85,306 params; input [1, 64600], output [1, 2] logits.",
    },
}


def build_manifest(models_dir: str, allow_missing: bool = False) -> Dict[str, object]:
    """Checksum every model present and record its provenance."""
    entries: Dict[str, Dict[str, str]] = {}
    missing = []
    for slot, spec in sorted(EXPECTED_MODELS.items()):
        relative = spec["path"]
        absolute = os.path.join(models_dir, relative)
        if not os.path.isfile(absolute):
            if spec.get("optional") == "true" or allow_missing:
                LOGGER.warning("%s not present at %s; skipping", slot, relative)
                missing.append(relative)
                continue
            raise SystemExit(
                "required model {0} is missing at {1}.\n"
                "Obtain it from {2} ({3}) on a connected machine, export it to "
                "ONNX, and place it there.".format(slot, absolute, spec["source"], spec["licence"])
            )
        digest = sha256_file(absolute)
        entries[relative] = {
            "slot": slot,
            "sha256": digest,
            "bytes": str(os.path.getsize(absolute)),
            "model": spec["model"],
            "source": spec["source"],
            "licence": spec["licence"],
            "notes": spec["notes"],
        }
        LOGGER.info("%-20s %s  %s", slot, digest[:16], relative)

    return {
        "schema": 1,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "note": (
            "Models are immutable binary assets. Every SHA-256 here is verified "
            "on load; a mismatch refuses to start rather than running a model "
            "nobody recorded."
        ),
        "missing": missing,
        "models": entries,
    }


def write_manifest(models_dir: str, manifest: Dict[str, object]) -> str:
    path = os.path.join(models_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record only the models present. For partial transfers.",
    )
    args = parser.parse_args(argv)

    configure_logging(level="INFO")
    models_dir = args.models_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    manifest = build_manifest(models_dir, allow_missing=args.allow_missing)
    path = write_manifest(models_dir, manifest)
    LOGGER.info("wrote %s with %d model(s)", path, len(manifest["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
