"""Configuration loading.

Two files, deliberately separate:

* ``config/default.yaml``   -- engineering tunables, versioned, safe defaults.
* ``config/thresholds.yaml`` -- the four decision numbers. Site-specific,
  produced by calibration, and **shipped empty**. Reading them when they are
  absent raises rather than defaulting (CLAUDE.md §8).

No module outside this one may read a YAML file, and no module anywhere may
hardcode a number that reaches a decision.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .errors import ConfigError, ThresholdsNotCalibratedError

LOGGER = logging.getLogger(__name__)

#: Repository root, resolved from this file's location (src/speakerid/config.py).
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(PACKAGE_ROOT))

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default.yaml")
DEFAULT_THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "config", "thresholds.yaml")

#: Every key calibrate.py must write. Missing any one of them is fatal.
#: Sanity bound on calibrated score thresholds. Raw cosine is in [-1, 1];
#: AS-Norm z-scores are unbounded but never legitimately this large.
_SCORE_SCALE_LIMIT = 20.0

REQUIRED_THRESHOLD_KEYS = (
    "cm_threshold",
    "accept_threshold",
    "margin_threshold",
    "min_speech_sec",
)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML mapping, tolerating an empty file (which yields ``{}``)."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError("configuration file not found: {0}".format(path)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError("malformed YAML in {0}: {1}".format(path, exc)) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("{0} must contain a mapping at the top level".format(path))
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Mappings merge key-by-key; every other type (including lists) is replaced
    wholesale, because a half-overridden list is never what the operator meant.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class _Missing:
    """Sentinel distinguishing "no default supplied" from ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<MISSING>"


_MISSING = _Missing()


@dataclass(frozen=True)
class Config:
    """Immutable view over the merged configuration tree.

    Values are reached by dotted path so that call sites read as a statement of
    which knob they use::

        cfg.get_float("vad.threshold")
    """

    data: Dict[str, Any] = field(default_factory=dict)
    source_paths: Tuple[str, ...] = ()
    project_root: str = PROJECT_ROOT

    # -- lookup --------------------------------------------------------------

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise ConfigError("missing configuration key: {0}".format(dotted))
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> Dict[str, Any]:
        node = self.get(dotted)
        if not isinstance(node, Mapping):
            raise ConfigError("configuration key {0} is not a section".format(dotted))
        return dict(node)

    def get_int(self, dotted: str, default: Any = _MISSING) -> int:
        value = self.get(dotted, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError("configuration key {0} must be an int".format(dotted))
        return int(value)

    def get_float(self, dotted: str, default: Any = _MISSING) -> float:
        value = self.get(dotted, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("configuration key {0} must be a number".format(dotted))
        return float(value)

    def get_bool(self, dotted: str, default: Any = _MISSING) -> bool:
        value = self.get(dotted, default)
        if not isinstance(value, bool):
            raise ConfigError("configuration key {0} must be a bool".format(dotted))
        return value

    def get_str(self, dotted: str, default: Any = _MISSING) -> str:
        value = self.get(dotted, default)
        if not isinstance(value, str):
            raise ConfigError("configuration key {0} must be a string".format(dotted))
        return value

    def get_optional_str(self, dotted: str) -> Optional[str]:
        value = self.get(dotted, None)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError("configuration key {0} must be a string or null".format(dotted))
        return value

    def get_str_list(self, dotted: str) -> List[str]:
        value = self.get(dotted)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigError("configuration key {0} must be a list of strings".format(dotted))
        return list(value)

    # -- paths ---------------------------------------------------------------

    def resolve_path(self, dotted: str) -> str:
        """Resolve a configured path relative to the project root."""
        return self.resolve_relative(self.get_str(dotted))

    def resolve_relative(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.project_root, path)

    # -- convenience ---------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        rate: int = self.get_int("audio.sample_rate")
        return rate


def load_config(
    override_path: Optional[str] = None,
    default_path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load ``default.yaml``, then a site file, then in-process overrides."""
    base_path = default_path or DEFAULT_CONFIG_PATH
    data = _read_yaml(base_path)
    sources = [base_path]

    if override_path:
        data = deep_merge(data, _read_yaml(override_path))
        sources.append(override_path)
    if overrides:
        data = deep_merge(data, overrides)

    config = Config(data=data, source_paths=tuple(sources))
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Catch the configuration mistakes that would otherwise fail silently."""
    sample_rate = config.get_int("audio.sample_rate")
    if sample_rate != 16000:
        # Every model in the inventory is 16 kHz. A different rate here would
        # not resample the models, it would just feed them wrong-speed audio.
        raise ConfigError("audio.sample_rate must be 16000; got {0}".format(sample_rate))
    if config.get_int("audio.channels") != 1:
        raise ConfigError("audio.channels must be 1 (mono)")
    if config.get_int("vad.window_samples") != 512:
        raise ConfigError("vad.window_samples must be 512 for Silero VAD v5 at 16 kHz")
    if config.get_int("embedder.embedding_dim") <= 0:
        raise ConfigError("embedder.embedding_dim must be positive")
    if config.get_float("embedder.window_seconds") > config.get_float(
        "embedder.max_window_seconds"
    ):
        raise ConfigError("embedder.window_seconds exceeds embedder.max_window_seconds")

    retention = config.get_int("database.retention_months")
    if retention < 0 or retention > 24:
        # CLAUDE.md §11: no default retention longer than 24 months.
        raise ConfigError(
            "database.retention_months must be between 0 and 24; got {0}".format(retention)
        )

    api_host = config.get_str("api.host")
    if api_host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError(
            "api.host must be a loopback address; {0!r} would expose biometric "
            "endpoints on the network".format(api_host)
        )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """The four numbers that decide an outcome, plus calibration provenance."""

    cm_threshold: float
    accept_threshold: float
    margin_threshold: float
    min_speech_sec: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def describe(self) -> str:
        calibrated_at = self.provenance.get("calibrated_at", "unknown")
        site = self.provenance.get("site", "unspecified site")
        target_far = self.provenance.get("target_far", "unspecified")
        return (
            "thresholds(cm={0:.4f}, accept={1:.4f}, margin={2:.4f}, min_speech={3:.2f}s) "
            "calibrated {4} for {5} at target FAR {6}".format(
                self.cm_threshold,
                self.accept_threshold,
                self.margin_threshold,
                self.min_speech_sec,
                calibrated_at,
                site,
                target_far,
            )
        )


_UNCALIBRATED_MESSAGE = (
    "config/thresholds.yaml is not calibrated for this site: {reason}.\n"
    "The system refuses to make identity decisions with guessed thresholds.\n"
    "Run:  python -m speakerid.cli calibrate --trials <trials.csv> --target-far 0.001\n"
    "on recordings captured on the deployment hardware in the deployment room.\n"
    "Do not copy published benchmark numbers into this file (CLAUDE.md §8)."
)


def load_thresholds(path: Optional[str] = None) -> Thresholds:
    """Load site thresholds, or raise a loud, actionable error.

    There is no default and no fallback. Every way of not having a calibrated
    value -- missing file, empty file, absent key, non-numeric, out of range --
    ends here.
    """
    thresholds_path = path or DEFAULT_THRESHOLDS_PATH
    try:
        data = _read_yaml(thresholds_path)
    except ConfigError as exc:
        raise ThresholdsNotCalibratedError(_UNCALIBRATED_MESSAGE.format(reason=str(exc))) from exc

    if not data:
        raise ThresholdsNotCalibratedError(
            _UNCALIBRATED_MESSAGE.format(reason="the file is empty (as shipped)")
        )

    missing = [key for key in REQUIRED_THRESHOLD_KEYS if key not in data]
    if missing:
        raise ThresholdsNotCalibratedError(
            _UNCALIBRATED_MESSAGE.format(reason="missing keys {0}".format(", ".join(missing)))
        )

    values: Dict[str, float] = {}
    for key in REQUIRED_THRESHOLD_KEYS:
        raw = data[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ThresholdsNotCalibratedError(
                _UNCALIBRATED_MESSAGE.format(reason="{0} is not a number".format(key))
            )
        values[key] = float(raw)

    # cm_threshold is a probability and must be one. accept_threshold and
    # margin_threshold live on whatever scale scoring produces: raw cosine in
    # [-1, 1] when AS-Norm is off, and an unbounded z-score when it is on.
    # calibrate.py derives them on exactly the scale decide() sees, so the
    # check here is for obvious nonsense, not for a fixed range.
    if not 0.0 <= values["cm_threshold"] <= 1.0:
        raise ThresholdsNotCalibratedError(
            _UNCALIBRATED_MESSAGE.format(
                reason="cm_threshold={0} is not a probability in [0, 1]".format(
                    values["cm_threshold"]
                )
            )
        )
    if values["margin_threshold"] < 0.0:
        raise ThresholdsNotCalibratedError(
            _UNCALIBRATED_MESSAGE.format(reason="margin_threshold must not be negative")
        )
    for key in ("accept_threshold", "margin_threshold"):
        if not -_SCORE_SCALE_LIMIT <= values[key] <= _SCORE_SCALE_LIMIT:
            raise ThresholdsNotCalibratedError(
                _UNCALIBRATED_MESSAGE.format(
                    reason="{0}={1} is implausible on any score scale this "
                    "system produces".format(key, values[key])
                )
            )
    if values["min_speech_sec"] <= 0.0:
        raise ThresholdsNotCalibratedError(
            _UNCALIBRATED_MESSAGE.format(reason="min_speech_sec must be positive")
        )

    provenance = {key: value for key, value in data.items() if key not in REQUIRED_THRESHOLD_KEYS}
    thresholds = Thresholds(
        cm_threshold=values["cm_threshold"],
        accept_threshold=values["accept_threshold"],
        margin_threshold=values["margin_threshold"],
        min_speech_sec=values["min_speech_sec"],
        provenance=provenance,
        source_path=thresholds_path,
    )

    if "calibrated_at" not in provenance:
        LOGGER.warning(
            "thresholds at %s carry no calibration provenance; they may have been "
            "hand-edited rather than produced by calibrate.py",
            thresholds_path,
        )
    LOGGER.info("loaded %s", thresholds.describe())
    return thresholds
