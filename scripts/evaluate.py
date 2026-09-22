"""Measure the system on held-out data from the deployment site.

Reports EER, minDCF, FAR/FRR at the shipped threshold, top-1 identification
accuracy, and **open-set rejection rate for impostors** -- the last being the
number that actually matters and the one people forget to measure
(CLAUDE.md §8).

Milestone 7 is where you find out whether the system works at all. Expect it
to send you back to VAD tuning and re-enrolment more than once.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from speakerid.config import Config, load_config, load_thresholds
from speakerid.errors import SpeakerIdError
from speakerid.logging_setup import configure_logging
from speakerid.metrics import (
    det_curve,
    equal_error_rate,
    far_frr_at_threshold,
    format_det_table,
    min_dcf,
    open_set_report,
)
from speakerid.pipeline import FeatureExtractor
from speakerid.scoring.asnorm import AsNorm
from speakerid.scoring.similarity import cosine, cosine_matrix

from ._common import EmbeddingCache, parse_bool, progress, read_csv_rows

LOGGER = logging.getLogger(__name__)


def run_evaluation(
    config: Config,
    trials_csv: str,
    report_path: Optional[str] = None,
    identification_trials_csv: Optional[str] = None,
    cm_trials_csv: Optional[str] = None,
    compare_enhance: bool = False,
    passphrase: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate at the thresholds that are actually deployed."""
    thresholds = load_thresholds()
    min_speech_sec = thresholds.min_speech_sec
    features = FeatureExtractor.from_config(config)
    asnorm = AsNorm.from_config(config)
    cache = EmbeddingCache(features, min_speech_sec)

    rows = read_csv_rows(trials_csv, ("enroll_path", "test_path", "same_speaker"))
    scores: List[float] = []
    labels: List[int] = []
    for index, row in enumerate(rows, start=1):
        progress("scoring trials", index, len(rows), every=100)
        enroll = cache.get(row["enroll_path"])
        test = cache.get(row["test_path"])
        if enroll is None or test is None:
            continue
        raw = cosine(enroll, test)
        if asnorm.enabled:
            means, stds = asnorm.enrollment_stats(enroll.reshape(1, -1))
            raw = float(asnorm.normalize(np.array([raw], dtype=np.float32), test, means, stds)[0])
        scores.append(raw)
        labels.append(1 if parse_bool(row["same_speaker"], "same_speaker") else 0)

    if not scores:
        raise SpeakerIdError("every trial was skipped; check the paths in the trial CSV")

    eer, eer_threshold = equal_error_rate(scores, labels)
    dcf, dcf_threshold = min_dcf(
        scores,
        labels,
        p_target=config.get_float("calibration.dcf_p_target"),
        c_miss=config.get_float("calibration.dcf_c_miss"),
        c_false_alarm=config.get_float("calibration.dcf_c_false_alarm"),
    )
    far, frr = far_frr_at_threshold(scores, labels, thresholds.accept_threshold)

    report: Dict[str, object] = {
        "thresholds": {
            "cm_threshold": thresholds.cm_threshold,
            "accept_threshold": thresholds.accept_threshold,
            "margin_threshold": thresholds.margin_threshold,
            "min_speech_sec": thresholds.min_speech_sec,
            "provenance": thresholds.provenance,
        },
        "verification": {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "min_dcf": dcf,
            "min_dcf_threshold": dcf_threshold,
            "far_at_deployed_threshold": far,
            "frr_at_deployed_threshold": frr,
            "n_trials": len(scores),
            "n_genuine": int(sum(labels)),
            "n_impostor": int(len(labels) - sum(labels)),
            "n_files_skipped": len(cache.skipped),
        },
        "det_table": format_det_table(det_curve(scores, labels)),
        "asnorm": {"enabled": asnorm.enabled, "cohort_size": asnorm.size},
        "open_set": None,
        "countermeasure": None,
        "enhance_comparison": None,
    }

    # -- open-set identification: the part that decides success -------------
    if identification_trials_csv:
        report["open_set"] = _evaluate_open_set(
            identification_trials_csv,
            cache,
            asnorm,
            thresholds.accept_threshold,
            thresholds.margin_threshold,
        )
    else:
        LOGGER.warning(
            "no --identification-trials supplied, so open-set impostor rejection "
            "was NOT measured. Verification EER alone does not tell you whether "
            "this system is safe to deploy (CLAUDE.md §8)."
        )

    # -- countermeasure on physical replay ----------------------------------
    if cm_trials_csv:
        report["countermeasure"] = _evaluate_cm(
            cm_trials_csv, features, min_speech_sec, thresholds.cm_threshold
        )

    if compare_enhance:
        report["enhance_comparison"] = _compare_enhance(config, rows, labels, min_speech_sec)

    if report_path:
        parent = os.path.dirname(os.path.abspath(report_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        LOGGER.info("wrote report to %s", report_path)

    _log_report(report)
    return report


def _evaluate_open_set(
    path: str,
    cache: EmbeddingCache,
    asnorm: AsNorm,
    accept_threshold: float,
    margin_threshold: float,
) -> Dict[str, object]:
    """Score 1-to-N probes against a gallery built from enrolment rows.

    CSV header: ``probe_path,person_id`` for probes, plus ``enroll_path``
    rows in a second file. Here the gallery comes from rows whose
    ``role`` column is ``enroll``.
    """
    rows = read_csv_rows(path, ("path", "person_id", "role"))
    gallery: Dict[str, List[np.ndarray]] = {}
    probes: List[Tuple[str, Optional[str]]] = []
    for row in rows:
        role = row["role"].strip().lower()
        person = (row["person_id"] or "").strip() or None
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
        raise SpeakerIdError("the identification trial file contains no usable enrol rows")
    if not probes:
        raise SpeakerIdError("the identification trial file contains no probe rows")

    person_ids = sorted(gallery)
    matrix = np.concatenate([np.stack(gallery[pid], axis=0) for pid in person_ids], axis=0)
    owners: List[str] = []
    for pid in person_ids:
        owners.extend([pid] * len(gallery[pid]))
    means, stds = asnorm.enrollment_stats(matrix)

    ranked: List[List[Tuple[str, float]]] = []
    truth: List[Optional[str]] = []
    for index, (probe_path, person) in enumerate(probes, start=1):
        progress("scoring probes", index, len(probes), every=50)
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

    result = open_set_report(ranked, truth, accept_threshold, margin_threshold)
    payload = result.to_dict()
    payload["n_enrolled_people"] = len(person_ids)
    payload["summary"] = result.summary()
    return payload


def _evaluate_cm(
    path: str, features: FeatureExtractor, min_speech_sec: float, cm_threshold: float
) -> Dict[str, object]:
    rows = read_csv_rows(path, ("path", "bona_fide"))
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
        raise SpeakerIdError("no usable countermeasure trials")

    eer, _ = equal_error_rate(scores, labels)
    far, frr = far_frr_at_threshold(scores, labels, cm_threshold)
    return {
        "eer": eer,
        "spoof_accept_rate": far,  # replays that got through
        "bona_fide_reject_rate": frr,  # live people turned away
        "n_trials": len(scores),
        "n_bona_fide": int(sum(labels)),
        "n_spoof": int(len(labels) - sum(labels)),
        "note": (
            "If these replays were not recorded through real phones and speakers "
            "in the deployment rooms, this number does not describe the threat "
            "(CLAUDE.md §10)."
        ),
    }


def _compare_enhance(
    config: Config, rows: Sequence[Dict[str, str]], labels: Sequence[int], min_speech_sec: float
) -> Dict[str, object]:
    """Run the trial set with and without denoising and report both EERs.

    This is the A/B that CLAUDE.md §9.5 requires before enhancement may be
    switched on. Raw audio into a well-augmented model often wins.
    """
    from speakerid.config import load_config as _load

    results: Dict[str, object] = {}
    for enabled in (False, True):
        variant = _load(
            overrides={
                "enhance": {"enabled": enabled, "backend": "spectral_gate" if enabled else "none"}
            }
        )
        features = FeatureExtractor.from_config(variant)
        cache = EmbeddingCache(features, min_speech_sec)
        scores: List[float] = []
        kept: List[int] = []
        for row, label in zip(rows, labels):
            enroll = cache.get(row["enroll_path"])
            test = cache.get(row["test_path"])
            if enroll is None or test is None:
                continue
            scores.append(cosine(enroll, test))
            kept.append(label)
        eer, _ = equal_error_rate(scores, kept)
        results["enhanced" if enabled else "raw"] = {"eer": eer, "n_trials": len(scores)}

    raw_eer = float(results["raw"]["eer"])  # type: ignore[index]
    enhanced_eer = float(results["enhanced"]["eer"])  # type: ignore[index]
    results["verdict"] = (
        "enhancement helps: {0:.4f} -> {1:.4f} EER".format(raw_eer, enhanced_eer)
        if enhanced_eer < raw_eer
        else "leave enhancement OFF: {0:.4f} -> {1:.4f} EER".format(raw_eer, enhanced_eer)
    )
    return results


def _log_report(report: Dict[str, object]) -> None:
    verification = report["verification"]
    LOGGER.info("verification: EER=%.4f minDCF=%.4f", verification["eer"], verification["min_dcf"])
    LOGGER.info(
        "at the deployed threshold: FAR=%.5f FRR=%.4f",
        verification["far_at_deployed_threshold"],
        verification["frr_at_deployed_threshold"],
    )
    open_set = report.get("open_set")
    if open_set:
        for line in str(open_set["summary"]).splitlines():
            LOGGER.info("%s", line)
    else:
        LOGGER.warning("open-set impostor rejection: NOT MEASURED")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", required=True)
    parser.add_argument(
        "--identification-trials",
        default=None,
        help="CSV: path,person_id,role where role is enroll|probe",
    )
    parser.add_argument("--cm-trials", default=None, help="CSV: path,bona_fide")
    parser.add_argument(
        "--compare-enhance",
        action="store_true",
        help="A/B denoising on this trial set (CLAUDE.md §9.5)",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(override_path=args.config)
    configure_logging(config)
    run_evaluation(
        config=config,
        trials_csv=args.trials,
        report_path=args.report,
        identification_trials_csv=args.identification_trials,
        cm_trials_csv=args.cm_trials,
        compare_enhance=args.compare_enhance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
