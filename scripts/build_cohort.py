"""Build AS-Norm cohort embeddings.

The cohort is 200-500 embeddings of people who are **not** enrolled, recorded
in the deployment environment. AS-Norm uses them to ask "how unusual is this
score?" instead of comparing a raw cosine to a fixed number. It typically cuts
error substantially and costs one matrix multiply at inference (CLAUDE.md §9.3).

A cohort of clean studio speech normalises against the wrong distribution and
buys nothing. Record it in the room.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from speakerid.config import Config, load_config
from speakerid.errors import SpeakerIdError
from speakerid.logging_setup import configure_logging
from speakerid.pipeline import FeatureExtractor
from speakerid.scoring.asnorm import MIN_USEFUL_COHORT, save_cohort_file

from ._common import find_audio, progress

LOGGER = logging.getLogger(__name__)

#: The size CLAUDE.md §9.3 asks for. Below it, warn loudly.
RECOMMENDED_MIN = 200


def build_cohort(
    config: Config,
    audio_dir: str,
    output_path: Optional[str] = None,
    room_id: Optional[str] = None,
    passphrase: Optional[str] = None,
) -> Dict[str, object]:
    """Embed every recording under ``audio_dir`` into a cohort matrix."""
    destination = output_path or config.resolve_path("scoring.asnorm.cohort_path")
    min_speech_sec = config.get_float("calibration.min_speech_sec")
    features = FeatureExtractor.from_config(config)

    paths = find_audio(audio_dir)
    if not paths:
        raise SpeakerIdError("no audio files found under {0}".format(audio_dir))

    embeddings: List[np.ndarray] = []
    skipped: List[str] = []
    for index, path in enumerate(paths, start=1):
        progress("embedding cohort", index, len(paths), every=25)
        embedding = features.embed_file(path, min_speech_sec)
        if embedding is None:
            skipped.append(os.path.basename(path))
            continue
        embeddings.append(embedding)

    if not embeddings:
        raise SpeakerIdError(
            "no cohort embeddings produced: every file had less than {0:.1f}s of "
            "detected speech. Check the VAD threshold against this audio before "
            "anything else.".format(min_speech_sec)
        )

    cohort = np.stack(embeddings, axis=0).astype(np.float32)
    save_cohort_file(destination, cohort, model_id=features.model_id, room_id=room_id or "")

    if cohort.shape[0] < MIN_USEFUL_COHORT:
        LOGGER.error("cohort of %d is too small to give stable statistics", cohort.shape[0])
    elif cohort.shape[0] < RECOMMENDED_MIN:
        LOGGER.warning(
            "cohort of %d is below the %d recommended in CLAUDE.md §9.3; AS-Norm "
            "will help less than it could",
            cohort.shape[0],
            RECOMMENDED_MIN,
        )

    return {
        "cohort_path": destination,
        "size": int(cohort.shape[0]),
        "dim": int(cohort.shape[1]),
        "model_id": features.model_id,
        "room_id": room_id,
        "n_files_scanned": len(paths),
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:10],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--room-id", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(override_path=args.config)
    configure_logging(config)
    summary = build_cohort(config, args.audio_dir, args.output, args.room_id)
    LOGGER.info("cohort: %d embeddings -> %s", summary["size"], summary["cohort_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
