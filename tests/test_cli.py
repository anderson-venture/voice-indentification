"""CLI behaviour.

The CLI is a thin wrapper, so these tests check the wiring and the operator
experience: that an uncalibrated system says so loudly, that erasure reports
what it removed, and that destructive commands confirm first.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from speakerid.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def site_config(tmp_path, base_config):
    """A config file on disk pointing at a temporary database."""
    import yaml

    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "database": {
                    "url": "sqlite:///{0}".format(tmp_path / "cli.db"),
                    "purge_on_startup": False,
                    "purge_interval_seconds": 0,
                },
                "crypto": {"salt_path": str(tmp_path / ".salt")},
                "runtime": {"verify_manifest": False},
                "logging": {"level": "ERROR"},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def invoke(runner, site_config, *args):
    return runner.invoke(
        cli,
        ["--config", site_config, "--passphrase", "test-pass"] + list(args),
        catch_exceptions=False,
    )


def test_help_lists_the_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("enroll", "identify", "delete", "status", "calibrate", "purge"):
        assert command in result.output


def test_status_reports_uncalibrated_thresholds(runner, site_config):
    """The operator must be told, in red, that nothing is calibrated yet."""
    result = invoke(runner, site_config, "status")
    assert result.exit_code == 0
    assert "NOT CALIBRATED" in result.output


def test_status_reports_every_model_slot(runner, site_config):
    """Status must account for all three stages, present or not.

    Deliberately does not assume weights exist: they ship by USB, so a clean
    checkout has none and a provisioned machine has all three.
    """
    result = invoke(runner, site_config, "status")
    for slot in ("vad", "embedder", "countermeasure"):
        assert slot in result.output
    assert "ok" in result.output or "MISSING" in result.output


def test_status_json_is_machine_readable(runner, site_config):
    result = runner.invoke(
        cli,
        ["--config", site_config, "--passphrase", "p", "--json", "status"],
        catch_exceptions=False,
    )
    payload = json.loads(result.output)
    assert payload["thresholds"] is None
    assert payload["thresholds_error"]
    assert payload["sample_rate"] == 16000


def test_add_person_then_consent_then_list(runner, site_config):
    created = invoke(runner, site_config, "add-person", "--person-id", "alice", "--name", "Alice")
    assert created.exit_code == 0
    assert "consent" in created.output  # tells the operator the next step

    listed = invoke(runner, site_config, "list")
    assert "alice" in listed.output
    assert "NO" in listed.output, "consent must show as absent before it is recorded"

    invoke(runner, site_config, "consent", "--person-id", "alice", "--granted-by", "operator")
    listed = invoke(runner, site_config, "list")
    assert "yes" in listed.output


def test_list_is_empty_initially(runner, site_config):
    assert "no people enrolled" in invoke(runner, site_config, "list").output


def test_delete_requires_confirmation(runner, site_config):
    invoke(runner, site_config, "add-person", "--person-id", "alice")
    result = runner.invoke(
        cli,
        ["--config", site_config, "--passphrase", "p", "delete", "--person-id", "alice"],
        input="n\n",
    )
    assert result.exit_code != 0  # aborted
    assert "alice" in invoke(runner, site_config, "list").output


def test_delete_with_yes_erases(runner, site_config):
    invoke(runner, site_config, "add-person", "--person-id", "alice")
    result = invoke(runner, site_config, "delete", "--person-id", "alice", "--yes")
    assert result.exit_code == 0
    assert "0 trace(s) remaining" in result.output
    assert "no people enrolled" in invoke(runner, site_config, "list").output


def test_purge_reports_what_it_removed(runner, site_config):
    result = invoke(runner, site_config, "purge")
    assert result.exit_code == 0
    assert "purged" in result.output


def test_challenge_prints_digits_and_a_token(runner, site_config):
    result = invoke(runner, site_config, "challenge")
    assert result.exit_code == 0
    assert "say:" in result.output
    assert "token:" in result.output


def test_identify_without_thresholds_fails_loudly(runner, site_config, fixture_path):
    """No calibration, no decision -- even from the CLI."""
    from speakerid.errors import ThresholdsNotCalibratedError

    with pytest.raises(ThresholdsNotCalibratedError):
        runner.invoke(
            cli,
            [
                "--config",
                site_config,
                "--passphrase",
                "p",
                "identify",
                fixture_path("alice_01.wav"),
            ],
            catch_exceptions=False,
        )


def test_verify_models_without_a_manifest(tmp_path):
    """A models directory with no manifest must refuse, and say what to run."""
    from speakerid.errors import ManifestError
    from speakerid.models.base import verify_against_manifest

    empty = tmp_path / "models"
    empty.mkdir()
    with pytest.raises(ManifestError, match="fetch_models"):
        verify_against_manifest(str(empty / "embed" / "x.onnx"), str(empty))
