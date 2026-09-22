"""Command-line interface.

Operator-facing entry points. Everything here is a thin wrapper: the rules
live in the modules, so the CLI cannot accidentally become a second place
where decisions are made.

    python -m speakerid.cli --help
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import click

from .challenge import ChallengeManager
from .config import Config, load_config, load_thresholds
from .db.store import Store
from .errors import SpeakerIdError
from .logging_setup import configure_logging
from .models.base import load_manifest, verify_against_manifest
from .pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

_PASSPHRASE_HELP = (
    "Encryption passphrase. Omit it to read the SPEAKERID_KEY environment "
    "variable instead; never pass it in a shared shell history."
)


class Context(object):
    """Shared state assembled once per invocation."""

    def __init__(self, config: Config, passphrase: Optional[str], json_output: bool) -> None:
        self.config = config
        self.passphrase = passphrase
        self.json_output = json_output
        self._store: Optional[Store] = None
        self._pipeline: Optional[Pipeline] = None

    def store(self) -> Store:
        if self._store is None:
            self._store = Store.from_config(self.config, passphrase=self.passphrase)
        return self._store

    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline.from_config(
                config=self.config,
                thresholds=load_thresholds(),
                store=self.store(),
                passphrase=self.passphrase,
            )
        return self._pipeline

    def emit(self, payload: Dict[str, Any], human: str) -> None:
        if self.json_output:
            click.echo(json.dumps(payload, default=str, sort_keys=True))
        else:
            click.echo(human)


pass_context = click.make_pass_decorator(Context)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Site configuration overlaid on config/default.yaml.",
)
@click.option("--passphrase", envvar="SPEAKERID_KEY", default=None, help=_PASSPHRASE_HELP)
@click.option("--json", "json_output", is_flag=True, help="Machine-readable output.")
@click.option("--log-level", default=None, help="Override logging.level.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Optional[str],
    passphrase: Optional[str],
    json_output: bool,
    log_level: Optional[str],
) -> None:
    """Offline speaker recognition and anti-spoofing.

    This system never sends audio anywhere and makes no network calls.
    """
    config = load_config(override_path=config_path)
    configure_logging(config, level=log_level)
    ctx.obj = Context(config, passphrase, json_output)


# ---------------------------------------------------------------------------
# Status and self-checks
# ---------------------------------------------------------------------------


@cli.command()
@pass_context
def status(context: Context) -> None:
    """Report configuration, models, thresholds and enrolment counts."""
    config = context.config
    models: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {
        "config_sources": list(config.source_paths),
        "sample_rate": config.sample_rate,
        "asnorm_enabled": config.get_bool("scoring.asnorm.enabled"),
        "enhance_enabled": config.get_bool("enhance.enabled"),
        "challenge_enabled": config.get_bool("challenge.enabled"),
        "retention_months": config.get_int("database.retention_months"),
        "models": models,
        "thresholds": None,
        "thresholds_error": None,
    }
    for slot, key in (
        ("vad", "vad_model.model_path"),
        ("embedder", "embedder.model_path"),
        ("countermeasure", "countermeasure.model_path"),
    ):
        path = config.resolve_path(key)
        models[slot] = {
            "path": os.path.relpath(path, config.project_root),
            "present": os.path.isfile(path),
        }
    try:
        thresholds = load_thresholds()
        report["thresholds"] = {
            "cm_threshold": thresholds.cm_threshold,
            "accept_threshold": thresholds.accept_threshold,
            "margin_threshold": thresholds.margin_threshold,
            "min_speech_sec": thresholds.min_speech_sec,
            "provenance": thresholds.provenance,
        }
    except SpeakerIdError as exc:
        report["thresholds_error"] = str(exc)

    if context.json_output:
        click.echo(json.dumps(report, default=str, indent=2, sort_keys=True))
        return

    click.echo("config:     {0}".format(", ".join(report["config_sources"])))
    click.echo("models:")
    for slot, info in report["models"].items():
        mark = "ok     " if info["present"] else "MISSING"
        click.echo("  {0:<15} {1}  {2}".format(slot, mark, info["path"]))
    if report["thresholds"]:
        click.echo("thresholds: calibrated")
        for key in ("cm_threshold", "accept_threshold", "margin_threshold", "min_speech_sec"):
            click.echo("  {0:<18} {1}".format(key, report["thresholds"][key]))
    else:
        click.secho("thresholds: NOT CALIBRATED", fg="red", bold=True)
        click.echo(report["thresholds_error"])


@cli.command("verify-models")
@pass_context
def verify_models(context: Context) -> None:
    """Check every model file against models/MANIFEST.json."""
    config = context.config
    models_dir = config.resolve_relative("models")
    manifest = load_manifest(models_dir)
    failures = 0
    for relative, entry in sorted(manifest["models"].items()):
        path = os.path.join(models_dir, relative)
        if not os.path.isfile(path):
            click.secho("MISSING  {0}".format(relative), fg="red")
            failures += 1
            continue
        try:
            verify_against_manifest(path, models_dir)
        except SpeakerIdError as exc:
            click.secho("MISMATCH {0}: {1}".format(relative, exc), fg="red")
            failures += 1
        else:
            click.secho(
                "ok       {0}  {1}".format(relative, entry.get("sha256", "")[:16]), fg="green"
            )
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# People, consent, enrolment
# ---------------------------------------------------------------------------


@cli.command("add-person")
@click.option("--person-id", default=None, help="Explicit id; generated when omitted.")
@click.option(
    "--name", "display_name", default=None, help="Operator-facing label. Never written to a log."
)
@click.option("--external-ref", default=None, help="Badge or HR reference.")
@pass_context
def add_person(
    context: Context,
    person_id: Optional[str],
    display_name: Optional[str],
    external_ref: Optional[str],
) -> None:
    """Create a person record. Consent is still required before enrolment."""
    identifier = context.store().add_person(person_id, display_name, external_ref)
    context.emit(
        {"person_id": identifier},
        "created person {0}\nRecord consent before enrolling:\n"
        "  speakerid consent --person-id {0} --granted-by <operator>".format(identifier),
    )


@cli.command()
@click.option("--person-id", required=True)
@click.option("--granted-by", required=True, help="Who witnessed and recorded consent.")
@click.option("--method", default="in_person", help="in_person | written | electronic")
@click.option(
    "--policy-version", default=None, help="Defaults to enrollment.policy_version from config."
)
@click.option("--notes", default=None)
@pass_context
def consent(
    context: Context,
    person_id: str,
    granted_by: str,
    method: str,
    policy_version: Optional[str],
    notes: Optional[str],
) -> None:
    """Record consent. Without this row, enrolment is refused."""
    version = policy_version or context.config.get_str("enrollment.policy_version")
    record_id = context.store().record_consent(
        person_id=person_id,
        policy_version=version,
        granted_by=granted_by,
        method=method,
        notes=notes,
    )
    context.emit(
        {"consent_id": record_id, "person_id": person_id, "policy_version": version},
        "recorded consent {0} for {1} under policy {2}".format(record_id, person_id, version),
    )


@cli.command("revoke-consent")
@click.option("--person-id", required=True)
@pass_context
def revoke_consent(context: Context, person_id: str) -> None:
    """Withdraw consent. Existing data is kept -- use `delete` to erase it."""
    count = context.store().revoke_consent(person_id)
    context.emit(
        {"person_id": person_id, "revoked": count},
        "revoked {0} consent record(s) for {1}\n"
        "Data is retained until you run: speakerid delete --person-id {1}".format(count, person_id),
    )


@cli.command()
@click.option("--person-id", required=True)
@click.argument(
    "audio_files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--device-id", default=None, help="Which microphone captured this.")
@click.option("--room-id", default=None, help="Which room. Enrol where you deploy.")
@click.option(
    "--retain-audio",
    is_flag=True,
    help="Keep the source audio under the retention schedule. Off by default.",
)
@pass_context
def enroll(
    context: Context,
    person_id: str,
    audio_files: Tuple[str, ...],
    device_id: Optional[str],
    room_id: Optional[str],
    retain_audio: bool,
) -> None:
    """Enrol utterances for a person.

    Record 3-5 utterances of at least 5 s each, on the deployment hardware in
    the deployment room. Enrolling on a laptop in a quiet office and deploying
    on a factory floor will not work (CLAUDE.md §7).
    """
    pipeline = context.pipeline()
    results = []
    for path in audio_files:
        outcome = pipeline.enroll_file(
            person_id, path, device_id=device_id, room_id=room_id, retain_audio=retain_audio
        )
        results.append(
            {
                "file": os.path.basename(path),
                "accepted": outcome.accepted,
                "speech_seconds": outcome.speech_seconds,
                "snr_db": outcome.snr_db,
                "self_cosine": outcome.self_cosine,
                "reason": outcome.reason,
            }
        )
        if not context.json_output:
            if outcome.accepted:
                click.secho(
                    "accepted {0}  ({1:.1f}s speech)".format(
                        os.path.basename(path), outcome.speech_seconds
                    ),
                    fg="green",
                )
            else:
                click.secho(
                    "rejected {0}: {1}".format(os.path.basename(path), outcome.reason), fg="yellow"
                )

    state = pipeline.enrollment_status(person_id)
    if context.json_output:
        click.echo(json.dumps({"results": results, "status": state}, default=str))
    elif not state["complete"]:
        click.secho(
            "{0} has {1} of {2} required enrolments".format(
                person_id, state["embeddings"], state["min_required"]
            ),
            fg="yellow",
        )
    else:
        click.secho(
            "{0} is enrolled ({1} utterances)".format(person_id, state["embeddings"]), fg="green"
        )


@cli.command("list")
@click.option("--include-inactive", is_flag=True)
@pass_context
def list_people(context: Context, include_inactive: bool) -> None:
    """List enrolled people. Shows counts, never voiceprints."""
    people = context.store().list_persons(include_inactive=include_inactive)
    if context.json_output:
        click.echo(json.dumps(people, default=str))
        return
    if not people:
        click.echo("no people enrolled")
        return
    click.echo("{0:<34} {1:>6} {2:>10}  {3}".format("person_id", "embs", "consent", "name"))
    for person in people:
        click.echo(
            "{0:<34} {1:>6} {2:>10}  {3}".format(
                person["person_id"],
                person["embeddings"],
                "yes" if person["consented"] else "NO",
                person["display_name"] or "-",
            )
        )


@cli.command()
@click.option("--person-id", required=True)
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@pass_context
def delete(context: Context, person_id: str, yes: bool) -> None:
    """Erase a person: embeddings, audio, events, consent and cohort rows."""
    if not yes:
        click.confirm(
            "Permanently erase every record for {0}? This cannot be undone.".format(person_id),
            abort=True,
        )
    store = context.store()
    report = store.delete_subject(person_id)
    remaining = store.subject_trace_count(person_id)
    payload = {
        "person_id": person_id,
        "rows_deleted": report.total_rows,
        "audio_files_deleted": report.audio_files_deleted,
        "remaining_traces": remaining,
    }
    context.emit(
        payload,
        "erased {0}: {1} rows, {2} audio file(s); {3} trace(s) remaining".format(
            person_id, report.total_rows, report.audio_files_deleted, remaining
        ),
    )
    if remaining:
        raise SystemExit(1)


@cli.command()
@pass_context
def purge(context: Context) -> None:
    """Delete everything past its retention date."""
    removed = context.store().purge_expired()
    context.emit(removed, "purged: {0}".format(removed))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--challenge-token",
    default=None,
    help="Token from `speakerid challenge`, when challenge-response is in use.",
)
@pass_context
def identify(context: Context, audio_file: str, challenge_token: Optional[str]) -> None:
    """Identify the speaker in an audio file.

    Prints one of four outcomes. UNKNOWN is a success, not a failure.
    """
    result = context.pipeline().identify_file(audio_file, challenge_token=challenge_token)
    decision = result.decision
    payload = {
        "outcome": decision.outcome.value,
        "person_id": decision.person_id,
        "score": decision.score,
        "runner_up_id": decision.runner_up_id,
        "runner_up_score": decision.runner_up_score,
        "margin": decision.margin,
        "cm_score": decision.cm_score,
        "speech_seconds": result.speech_seconds,
        "reason": decision.reason,
        "timings_ms": result.timings.summary(),
    }
    if context.json_output:
        click.echo(json.dumps(payload, default=str))
        return

    colours = {"identified": "green", "unknown": "yellow", "ambiguous": "yellow", "rejected": "red"}
    click.secho(decision.outcome.value.upper(), fg=colours.get(decision.outcome.value), bold=True)
    if decision.person_id:
        click.echo("  person   {0}".format(decision.person_id))
    if decision.score is not None:
        click.echo("  score    {0:.4f}".format(decision.score))
    if decision.margin is not None:
        click.echo("  margin   {0:.4f}".format(decision.margin))
    if decision.cm_score is not None:
        click.echo("  liveness {0:.4f}".format(decision.cm_score))
    click.echo("  speech   {0:.2f}s".format(result.speech_seconds))
    if decision.reason:
        click.echo("  reason   {0}".format(decision.reason))
    click.echo("  timing   {0}".format(result.timings.summary()))


@cli.command()
@pass_context
def challenge(context: Context) -> None:
    """Issue a challenge. Have the user read the digits aloud, then identify."""
    manager = ChallengeManager.from_config(context.config)
    issued = manager.issue()
    context.emit(
        {"token": issued.token, "digits": issued.digits, "expires_at": issued.expires_at},
        "say: {0}\ntoken: {1}\n(expires in {2:.0f}s)\n\n"
        "Note: challenge tokens live in the issuing process. Use the API for a "
        "real challenge-response flow; this command is for testing the prompt.".format(
            issued.prompt(), issued.token, manager.ttl_seconds
        ),
    )


# ---------------------------------------------------------------------------
# Calibration and evaluation -- delegated to scripts/
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--trials",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV of trials: enroll_path,test_path,same_speaker",
)
@click.option(
    "--target-far",
    default=0.001,
    show_default=True,
    help="Operating point. For access control, 0.001 or lower.",
)
@click.option(
    "--cm-trials",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV of liveness trials: path,bona_fide. Your own replay recordings.",
)
@click.option(
    "--cm-threshold",
    default=None,
    type=float,
    help="Set the liveness threshold by hand instead. Recorded as uncalibrated.",
)
@click.option(
    "--cm-target-far",
    default=None,
    type=float,
    help="Target spoof-accept rate for the liveness threshold. Defaults to "
    "--target-far; set it separately when the replay corpus is smaller than "
    "the speaker trial set.",
)
@click.option(
    "--identification-trials",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV: path,person_id,role (enroll|probe). Sweeps the margin.",
)
@click.option("--site", default="unspecified", help="Recorded in the threshold provenance.")
@click.option("--output", default=None, help="Defaults to config/thresholds.yaml.")
@pass_context
def calibrate(
    context: Context,
    trials: str,
    target_far: float,
    cm_trials: Optional[str],
    cm_threshold: Optional[float],
    cm_target_far: Optional[float],
    identification_trials: Optional[str],
    site: str,
    output: Optional[str],
) -> None:
    """Derive thresholds from your own recordings. Run before deploying."""
    from scripts.calibrate import run_calibration

    destination = output or os.path.join(context.config.project_root, "config", "thresholds.yaml")
    summary = run_calibration(
        config=context.config,
        trials_csv=trials,
        target_far=target_far,
        site=site,
        output_path=destination,
        cm_trials_csv=cm_trials,
        cm_threshold_override=cm_threshold,
        cm_target_far=cm_target_far,
        identification_trials_csv=identification_trials,
        passphrase=context.passphrase,
    )
    context.emit(summary, json.dumps(summary, indent=2, default=str))


@cli.command()
@click.option("--trials", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--report", default=None, help="Write a JSON report here.")
@pass_context
def evaluate(context: Context, trials: str, report: Optional[str]) -> None:
    """Report EER, minDCF, FAR/FRR, top-1 accuracy and impostor rejection."""
    from scripts.evaluate import run_evaluation

    summary = run_evaluation(
        config=context.config,
        trials_csv=trials,
        report_path=report,
        passphrase=context.passphrase,
    )
    context.emit(summary, json.dumps(summary, indent=2, default=str))


@cli.command("build-cohort")
@click.argument("audio_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", default=None, help="Defaults to scoring.asnorm.cohort_path.")
@click.option("--room-id", default=None)
@pass_context
def build_cohort(
    context: Context, audio_dir: str, output: Optional[str], room_id: Optional[str]
) -> None:
    """Build AS-Norm cohort embeddings from non-enrolled recordings."""
    from scripts.build_cohort import build_cohort as run_build

    summary = run_build(
        config=context.config,
        audio_dir=audio_dir,
        output_path=output,
        room_id=room_id,
        passphrase=context.passphrase,
    )
    context.emit(summary, json.dumps(summary, indent=2, default=str))


def main() -> None:
    try:
        cli()  # pylint: disable=no-value-for-parameter
    except SpeakerIdError as exc:
        click.secho("error: {0}".format(exc), fg="red", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
