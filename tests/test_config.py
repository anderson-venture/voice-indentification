"""Configuration and threshold loading.

The most important behaviour here is a refusal: an uncalibrated site must not
be able to make a decision (CLAUDE.md §8).
"""

from __future__ import annotations

import pytest

from speakerid.config import deep_merge, load_config, load_thresholds
from speakerid.errors import ConfigError, ThresholdsNotCalibratedError


def test_shipped_thresholds_file_refuses_to_load():
    """config/thresholds.yaml ships empty and must stay that way."""
    with pytest.raises(ThresholdsNotCalibratedError) as excinfo:
        load_thresholds()
    message = str(excinfo.value)
    assert "not calibrated" in message
    assert "calibrate" in message  # the error tells the operator what to run


def test_thresholds_reject_missing_keys(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text("accept_threshold: 0.5\n", encoding="utf-8")
    with pytest.raises(ThresholdsNotCalibratedError) as excinfo:
        load_thresholds(str(path))
    assert "cm_threshold" in str(excinfo.value)


def test_thresholds_reject_non_probability_cm(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        "cm_threshold: 1.7\naccept_threshold: 0.5\nmargin_threshold: 0.1\nmin_speech_sec: 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ThresholdsNotCalibratedError):
        load_thresholds(str(path))


def test_thresholds_load_with_provenance(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        "cm_threshold: 0.5\naccept_threshold: 0.62\nmargin_threshold: 0.08\n"
        "min_speech_sec: 2.0\ncalibrated_at: '2026-09-20T10:00:00Z'\nsite: 'north gate'\n",
        encoding="utf-8",
    )
    loaded = load_thresholds(str(path))
    assert loaded.accept_threshold == pytest.approx(0.62)
    assert loaded.provenance["site"] == "north gate"
    assert "north gate" in loaded.describe()


def test_asnorm_zscore_thresholds_are_allowed(tmp_path):
    """AS-Norm produces z-scores, so accept_threshold is not bounded by 1.0."""
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        "cm_threshold: 0.5\naccept_threshold: 3.4\nmargin_threshold: 0.9\n" "min_speech_sec: 2.0\n",
        encoding="utf-8",
    )
    assert load_thresholds(str(path)).accept_threshold == pytest.approx(3.4)


def test_deep_merge_replaces_lists_but_merges_mappings():
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2, 3]}
    override = {"a": {"c": 9}, "list": [7]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 9}, "list": [7]}


def test_sample_rate_is_locked_to_16k():
    with pytest.raises(ConfigError, match="16000"):
        load_config(overrides={"audio": {"sample_rate": 8000}})


def test_retention_cannot_exceed_24_months():
    """CLAUDE.md §11: no default retention longer than 24 months."""
    with pytest.raises(ConfigError, match="retention_months"):
        load_config(overrides={"database": {"retention_months": 60}})


def test_api_must_bind_loopback():
    with pytest.raises(ConfigError, match="loopback"):
        load_config(overrides={"api": {"host": "0.0.0.0"}})


def test_missing_key_raises_rather_than_defaulting():
    config = load_config()
    with pytest.raises(ConfigError, match="missing configuration key"):
        config.get_float("vad.nonexistent_knob")


def test_typed_getters_reject_wrong_types():
    config = load_config(overrides={"vad": {"threshold": "loud"}})
    with pytest.raises(ConfigError, match="must be a number"):
        config.get_float("vad.threshold")
