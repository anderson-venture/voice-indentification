"""Derive decision thresholds from the operator's own recordings.

Run this before the system is allowed to make a decision. ``thresholds.yaml``
ships empty and the loader refuses to default it, so this script is not
optional (CLAUDE.md §8).

Why published numbers cannot be reused
--------------------------------------
ReDimNet2-B2's 0.57% EER on VoxCeleb1-O is clean-speech, 1-to-1 verification,
English, short segments. A deployment is noisy, reverberant, possibly
multilingual, and **open-set 1-to-N identification**, where false accepts
scale roughly with the number of enrolled people. A model at 1% verification
EER can produce 10-20% identification errors against 500 enrolled people in
noise.

So the operating point is set from the operator's own DET curve at a **target
FAR**, not at EER.

Inputs
------
``--trials`` CSV, header ``enroll_path,test_path,same_speaker``::

    enroll_path,test_path,same_speaker
    audio/alice_01.wav,audio/alice_02.wav,1
    audio/alice_01.wav,audio/bob_01.wav,0

``--cm-trials`` CSV, header ``path,bona_fide`` -- live captures and replays of
them through real phones and speakers (CLAUDE.md §10).

``--identification-trials`` CSV (optional), header ``path,person_id,role``
where ``role`` is ``enroll`` or ``probe`` and ``person_id`` is empty for a
probe from someone who is not enrolled. This is the same format
``evaluate.py`` takes. Supplying it lets the margin threshold be swept against
real 1-to-N behaviour instead of estimated from the impostor score spread.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import getpass
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from speakerid.config import Config, load_config
from speakerid.errors import SpeakerIdError
from speakerid.logging_setup import configure_logging
from speakerid.metrics import (
    det_curve,
    equal_error_rate,
    format_det_table,
    min_dcf,
    open_set_report,
    threshold_at_far,
)
from speakerid.pipeline import FeatureExtractor
from speakerid.scoring.asnorm import AsNorm
from speakerid.scoring.similarity import cosine, cosine_matrix

from ._common import EmbeddingCache, parse_bool, progress, read_csv_rows

LOGGER = logging.getLogger(__name__)


def _score_pairs(
    rows: Sequence[Dict[str, str]], cache: EmbeddingCache, asnorm: AsNorm
) -> Tuple[List[float], List[int], int]:
    """Score every verification trial, AS-Normed exactly as at runtime."""
    scores: List[float] = []
    labels: List[int] = []
    skipped = 0
    for index, row in enumerate(rows, start=1):
        progress("scoring trials", index, len(rows), every=100)
        enroll = cache.get(row["enroll_path"])
        test = cache.get(row["test_path"])
        if enroll is None or test is None:
            skipped += 1
            continue
        raw = cosine(enroll, test)
        if asnorm.enabled:
            means, stds = asnorm.enrollment_stats(enroll.reshape(1, -1))
            score = float(asnorm.normalize(np.array([raw], dtype=np.float32), test, means, stds)[0])
        else:
            score = raw
        scores.append(score)
        labels.append(1 if parse_bool(row["same_speaker"], "same_speaker") else 0)
    return scores, labels, skipped


def _calibrate_cm(
    rows: Sequence[Dict[str, str]],
    features: FeatureExtractor,
    min_speech_sec: float,
    target_far: float,
) -> Dict[str, object]:
    """Choose the bona-fide probability floor from labelled replay data."""
    scores: List[float] = []
    labels: List[int] = []
    for index, row in enumerate(rows, start=1):
        progress("scoring liveness", index, len(rows), every=50)
        score = features.liveness_file(row["path"], min_speech_sec)
        if score is None:
            continue
        scores.append(score)
        labels.append(1 if parse_bool(row["bona_fide"], "bona_fide") else 0)

    if not scores:
        raise SpeakerIdError("no usable countermeasure trials: every file was too short")

    eer, eer_threshold = equal_error_rate(scores, labels)
    threshold, measured_far, measured_frr = threshold_at_far(scores, labels, target_far)
    return {
        "cm_threshold": threshold,
        "cm_eer": eer,
        "cm_eer_threshold": eer_threshold,
        "cm_far_at_threshold": measured_far,
        "cm_frr_at_threshold": measured_frr,
        "cm_n_trials": len(scores),
        "cm_n_bona_fide": int(sum(labels)),
        "cm_n_spoof": int(len(labels) - sum(labels)),
    }


def _sweep_margin(
    rows: Sequence[Dict[str, str]],
    cache: EmbeddingCache,
    asnorm: AsNorm,
    accept_threshold: float,
) -> Dict[str, object]:
    """Pick the margin that best separates enrolled speakers from strangers.

    Swept against real 1-to-N trials, which is the only setting where the
    top-1/top-2 gap means anything. The trial file uses the same
    ``path,person_id,role`` schema as ``evaluate.py``: one format for both
    scripts, and the gallery is built from the rows the operator labelled
    ``enroll`` rather than inferred.
    """
    gallery: Dict[str, List[np.ndarray]] = {}
    probes: List[Tuple[str, Optional[str]]] = []
    for row in rows:
        role = row["role"].strip().lower()
        person = (row.get("person_id") or "").strip() or None
        if role == "enroll":
            if person is None:
                raise SpeakerIdError("an enrol row must name a person_id: {0}".format(row["path"]))
            embedding = cache.get(row["path"])
            if embedding is not None:
                gallery.setdefault(person, []).append(embedding)
        elif role == "probe":
            probes.append((row["path"], person))
        else:
            raise SpeakerIdError("role must be 'enroll' or 'probe'; got {0!r}".format(role))

    if not gallery:
        raise SpeakerIdError("the identification trial file has no usable enrol rows")
    if not probes:
        raise SpeakerIdError("the identification trial file has no probe rows")

    person_ids = sorted(gallery)
    matrix = np.concatenate([np.stack(gallery[pid], axis=0) for pid in person_ids], axis=0)
    owners: List[str] = []
    for pid in person_ids:
        owners.extend([pid] * len(gallery[pid]))
    means, stds = asnorm.enrollment_stats(matrix)

    ranked: List[List[Tuple[str, float]]] = []
    truth: List[Optional[str]] = []
    for index, (probe_path, person) in enumerate(probes, start=1):
        progress("scoring identification trials", index, len(probes), every=50)
        probe = cache.get(probe_path)
        if probe is None:
            continue
        raw = cosine_matrix(probe, matrix)
        scored = asnorm.normalize(raw, probe, means, stds) if asnorm.enabled else raw
        best: Dict[str, float] = {}
        for value, owner in zip(scored, owners):
            best[owner] = max(best.get(owner, float("-inf")), float(value))
        ranked.append(sorted(best.items(), key=lambda item: item[1], reverse=True))
        truth.append(person)

    if not ranked:
        raise SpeakerIdError("no usable identification trials")

    best_margin = 0.0
    best_report = None
    best_score = float("-inf")
    for margin in np.linspace(0.0, 2.0, num=41):
        report = open_set_report(ranked, truth, accept_threshold, float(margin))
        rejection = report.impostor_rejection_rate
        accuracy = report.top1_accuracy_enrolled_only
        rejection = 0.0 if np.isnan(rejection) else rejection
        accuracy = 0.0 if np.isnan(accuracy) else accuracy
        # Weight impostor rejection above hit rate -- in an open set, admitting
        # a stranger costs more than asking an employee to repeat themselves --
        # but never so much that rejecting everyone wins. A margin so large
        # that no one is ever identified scores accuracy 0 and must lose to a
        # margin that keeps both.
        objective = rejection + 0.5 * accuracy
        # `>=` breaks ties toward the LARGER margin. When two operating points
        # are equally good on this data, the more demanding one is the safer
        # bet: it returns AMBIGUOUS where the other would have guessed, and
        # CLAUDE.md §6 says to bias hard that way.
        if objective >= best_score:
            best_score = objective
            best_margin = float(margin)
            best_report = report

    assert best_report is not None
    if best_report.top1_accuracy_enrolled_only == 0.0:
        LOGGER.error(
            "the margin sweep could not find an operating point that identifies "
            "anyone. Every margin either admits strangers or rejects genuine "
            "speakers, which means the embeddings do not separate these people "
            "on this data. Do not deploy; revisit VAD tuning and enrolment."
        )
    return {
        "margin_threshold": best_margin,
        "margin_source": "swept against identification trials",
        "identification": best_report.to_dict(),
    }


def run_calibration(
    config: Config,
    trials_csv: str,
    target_far: float,
    site: str,
    output_path: str,
    cm_trials_csv: Optional[str] = None,
    cm_threshold_override: Optional[float] = None,
    identification_trials_csv: Optional[str] = None,
    cm_target_far: Optional[float] = None,
    passphrase: Optional[str] = None,
) -> Dict[str, object]:
    """Produce ``thresholds.yaml`` from labelled trials."""
    if cm_trials_csv is None and cm_threshold_override is None:
        raise SpeakerIdError(
            "no countermeasure trials supplied. AASIST-L's published EER is on "
            "synthetic speech, not physical replay, so its threshold must be set "
            "on your own replay recordings (CLAUDE.md §10): record enrolled "
            "employees, play those recordings back through 8-10 phones, laptops "
            "and Bluetooth speakers at 0.3 m / 1 m / 3 m in each deployment room, "
            "and label them in a CSV.\\n"
            "Pass --cm-trials <file>, or --cm-threshold <value> to set it by hand "
            "and have that recorded as uncalibrated in the provenance."
        )

    min_speech_sec = config.get_float("calibration.min_speech_sec")
    features = FeatureExtractor.from_config(config)
    asnorm = AsNorm.from_config(config)
    cache = EmbeddingCache(features, min_speech_sec)

    rows = read_csv_rows(trials_csv, ("enroll_path", "test_path", "same_speaker"))
    LOGGER.info("scoring %d verification trials", len(rows))
    scores, labels, skipped = _score_pairs(rows, cache, asnorm)
    if not scores:
        raise SpeakerIdError("every trial was skipped; check the paths in the trial CSV")

    eer, eer_threshold = equal_error_rate(scores, labels)
    accept_threshold, measured_far, measured_frr = threshold_at_far(scores, labels, target_far)
    dcf, dcf_threshold = min_dcf(
        scores,
        labels,
        p_target=config.get_float("calibration.dcf_p_target"),
        c_miss=config.get_float("calibration.dcf_c_miss"),
        c_false_alarm=config.get_float("calibration.dcf_c_false_alarm"),
    )

    summary: Dict[str, object] = {
        "accept_threshold": accept_threshold,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "min_dcf": dcf,
        "min_dcf_threshold": dcf_threshold,
        "target_far": target_far,
        "measured_far": measured_far,
        "measured_frr": measured_frr,
        "n_trials": len(scores),
        "n_genuine": int(sum(labels)),
        "n_impostor": int(len(labels) - sum(labels)),
        "n_skipped_trials": skipped,
        "n_files_embedded": len(cache),
        "asnorm_enabled": asnorm.enabled,
        "asnorm_cohort_size": asnorm.size,
    }

    # -- countermeasure ------------------------------------------------------
    if cm_trials_csv is not None:
        cm_rows = read_csv_rows(cm_trials_csv, ("path", "bona_fide"))
        # The liveness operating point is a separate question from the
        # identification one, and is measured on a separate (usually much
        # smaller) trial set, so it gets its own target rather than inheriting
        # one the replay corpus cannot resolve.
        resolved_cm_far = target_far if cm_target_far is None else cm_target_far
        summary.update(_calibrate_cm(cm_rows, features, min_speech_sec, resolved_cm_far))
        summary["cm_target_far"] = resolved_cm_far
        summary["cm_calibrated"] = True
    else:
        summary["cm_threshold"] = float(cm_threshold_override)
        summary["cm_calibrated"] = False
        LOGGER.warning(
            "cm_threshold was set by hand to %.4f and is NOT calibrated on replay "
            "data. The liveness score is unvalidated for this site.",
            cm_threshold_override,
        )

    # -- margin --------------------------------------------------------------
    if identification_trials_csv is not None:
        ident_rows = read_csv_rows(identification_trials_csv, ("path", "person_id", "role"))
        summary.update(_sweep_margin(ident_rows, cache, asnorm, accept_threshold))
    else:
        summary.update(_estimate_margin(config, scores, labels))

    # -- write ---------------------------------------------------------------
    summary["min_speech_sec"] = min_speech_sec
    summary["site"] = site
    summary["calibrated_at"] = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
    summary["calibrated_by"] = _safe_username()
    summary["model_id"] = features.model_id

    _write_thresholds(output_path, summary)
    LOGGER.info("wrote calibrated thresholds to %s", output_path)

    for line in format_det_table(det_curve(scores, labels)):
        LOGGER.info("  %s", line)
    return summary


def _estimate_margin(
    config: Config, scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, object]:
    """Fallback margin when no identification trials were supplied.

    Uses the spread of the impostor score distribution as the scale on which
    two candidates are indistinguishable. This is weaker than sweeping real
    1-to-N trials and the provenance says so.
    """
    impostors = np.asarray(
        [score for score, label in zip(scores, labels) if label == 0], dtype=np.float64
    )
    margin_z = config.get_float("calibration.margin_z")
    margin = float(margin_z * np.std(impostors)) if impostors.size else 0.0
    LOGGER.warning(
        "margin_threshold estimated as %.4f from the impostor score spread. "
        "Supply --identification-trials to sweep it against real 1-to-N data.",
        margin,
    )
    return {
        "margin_threshold": margin,
        "margin_source": "estimated from impostor score spread (no identification trials)",
    }


def _safe_username() -> str:
    try:
        return getpass.getuser()
    except Exception:  # getuser raises on hosts with no passwd entry
        return "unknown"


def _write_thresholds(path: str, summary: Dict[str, object]) -> None:
    """Write thresholds.yaml with full provenance, by hand.

    Written as text rather than via yaml.dump so the file keeps its warnings
    and stays readable to the operator who has to defend these numbers.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines = [
        "# Calibrated decision thresholds -- generated by scripts/calibrate.py.",
        "# Site-specific. Do not copy this file between deployments, and re-run",
        "# calibration whenever the enrolled population grows by ~50% or the",
        "# environment changes (CLAUDE.md §8).",
        "",
        "cm_threshold: {0!r}".format(round(float(summary["cm_threshold"]), 6)),
        "accept_threshold: {0!r}".format(round(float(summary["accept_threshold"]), 6)),
        "margin_threshold: {0!r}".format(round(float(summary["margin_threshold"]), 6)),
        "min_speech_sec: {0!r}".format(round(float(summary["min_speech_sec"]), 6)),
        "",
        "# --- provenance (read by the loader, logged at startup) ---",
        "calibrated_at: {0!r}".format(str(summary["calibrated_at"])),
        "calibrated_by: {0!r}".format(str(summary["calibrated_by"])),
        "site: {0!r}".format(str(summary["site"])),
        "model_id: {0!r}".format(str(summary["model_id"])),
        "target_far: {0!r}".format(float(summary["target_far"])),
        "measured_far: {0!r}".format(round(float(summary["measured_far"]), 6)),
        "measured_frr: {0!r}".format(round(float(summary["measured_frr"]), 6)),
        "eer: {0!r}".format(round(float(summary["eer"]), 6)),
        "min_dcf: {0!r}".format(round(float(summary["min_dcf"]), 6)),
        "n_trials: {0!r}".format(int(summary["n_trials"])),
        "n_genuine: {0!r}".format(int(summary["n_genuine"])),
        "n_impostor: {0!r}".format(int(summary["n_impostor"])),
        "asnorm_enabled: {0!r}".format(bool(summary["asnorm_enabled"])),
        "asnorm_cohort_size: {0!r}".format(int(summary["asnorm_cohort_size"])),
        "cm_calibrated: {0!r}".format(bool(summary["cm_calibrated"])),
        "margin_source: {0!r}".format(str(summary["margin_source"])),
    ]
    if "cm_eer" in summary:
        lines.append("cm_eer: {0!r}".format(round(float(summary["cm_eer"]), 6)))
        lines.append("cm_n_trials: {0!r}".format(int(summary["cm_n_trials"])))
        lines.append("cm_target_far: {0!r}".format(float(summary["cm_target_far"])))
        lines.append(
            "cm_spoof_accept_rate: {0!r}".format(round(float(summary["cm_far_at_threshold"]), 6))
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", required=True)
    parser.add_argument("--cm-trials", default=None)
    parser.add_argument("--cm-threshold", type=float, default=None)
    parser.add_argument(
        "--cm-target-far",
        type=float,
        default=None,
        help="Target spoof-accept rate for the liveness threshold. Defaults to "
        "--target-far; set it separately when the replay corpus is smaller "
        "than the speaker trial set.",
    )
    parser.add_argument("--identification-trials", default=None)
    parser.add_argument("--target-far", type=float, default=None)
    parser.add_argument("--site", default="unspecified")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(override_path=args.config)
    configure_logging(config)
    target_far = args.target_far
    if target_far is None:
        target_far = config.get_float("calibration.target_far")
    output = args.output or os.path.join(config.project_root, "config", "thresholds.yaml")

    summary = run_calibration(
        config=config,
        trials_csv=args.trials,
        target_far=target_far,
        site=args.site,
        output_path=output,
        cm_trials_csv=args.cm_trials,
        cm_threshold_override=args.cm_threshold,
        identification_trials_csv=args.identification_trials,
        cm_target_far=args.cm_target_far,
    )
    LOGGER.info(
        "accept_threshold=%.4f at FAR=%.5f (FRR=%.4f), EER=%.4f",
        summary["accept_threshold"],
        summary["measured_far"],
        summary["measured_frr"],
        summary["eer"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
