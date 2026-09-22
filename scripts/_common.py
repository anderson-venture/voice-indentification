"""Shared helpers for the operator-facing scripts."""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from speakerid.config import Config
from speakerid.errors import SpeakerIdError
from speakerid.pipeline import FeatureExtractor

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


def find_audio(directory: str) -> List[str]:
    """All readable audio under a directory, sorted for reproducibility."""
    found: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if name.lower().endswith(AUDIO_EXTENSIONS):
                found.append(os.path.join(root, name))
    return sorted(found)


def read_csv_rows(path: str, required: Sequence[str]) -> List[Dict[str, str]]:
    """Read a CSV and check its header, with an error the operator can act on."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SpeakerIdError("{0} is empty".format(path))
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise SpeakerIdError(
                "{0} is missing column(s) {1}. Expected header: {2}".format(
                    path, ", ".join(missing), ",".join(required)
                )
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise SpeakerIdError("{0} contains no rows".format(path))
    return rows


def parse_bool(value: str, field: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "t"):
        return True
    if lowered in ("0", "false", "no", "n", "f"):
        return False
    raise SpeakerIdError("could not read {0}={1!r} as a boolean".format(field, value))


class EmbeddingCache(object):
    """Embed each distinct file once.

    Trial lists reuse the same recordings many times over; without this, a
    calibration run spends most of its time re-embedding the same audio.
    """

    def __init__(self, features: FeatureExtractor, min_speech_sec: float) -> None:
        self.features = features
        self.min_speech_sec = min_speech_sec
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self.skipped: List[str] = []

    def get(self, path: str) -> Optional[np.ndarray]:
        resolved = os.path.abspath(path)
        if resolved not in self._cache:
            if not os.path.isfile(resolved):
                raise SpeakerIdError(
                    "audio file listed in the trial set is missing: {0}".format(path)
                )
            embedding = self.features.embed_file(resolved, self.min_speech_sec)
            if embedding is None:
                self.skipped.append(path)
            self._cache[resolved] = embedding
        return self._cache[resolved]

    def __len__(self) -> int:
        return len([item for item in self._cache.values() if item is not None])


def resolve_against(config: Config, path: str) -> str:
    return config.resolve_relative(path)


def progress(label: str, index: int, total: int, every: int = 25) -> None:
    if total and (index % every == 0 or index == total):
        LOGGER.info("%s: %d/%d", label, index, total)


def summarize_scores(scores: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(array)), float(np.std(array))
