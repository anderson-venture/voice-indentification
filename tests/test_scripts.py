"""The operator-facing scripts: cohort building, calibration, evaluation.

These run the real code paths on synthetic audio, so the round trip
"record -> calibrate -> deploy -> evaluate" is exercised end to end without
any real voices or model weights.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from scripts.build_cohort import build_cohort
from scripts.calibrate import run_calibration
from scripts.evaluate import run_evaluation
from scripts.make_fixtures import cohort_audio

from speakerid.audio.io import write_audio
from speakerid.config import load_config, load_thresholds
from speakerid.errors import SpeakerIdError

from .conftest import fake_session_factory


@pytest.fixture
def patched_features(monkeypatch):
    """Make the scripts build FeatureExtractors backed by the doubles."""
    from speakerid.pipeline import FeatureExtractor

    original = FeatureExtractor.from_config

    @classmethod
    def with_doubles(cls, config, session_factory=None):
        return original.__func__(cls, config, session_factory=fake_session_factory)

    monkeypatch.setattr(FeatureExtractor, "from_config", with_doubles)
    return True


@pytest.fixture
def audio_dir(tmp_path):
    """A directory of cohort recordings, written from the synthesiser."""
    directory = tmp_path / "cohort_audio"
    directory.mkdir()
    for index, clip in enumerate(cohort_audio(count=12, seconds=6.0)):
        write_audio(str(directory / "cohort_{0:02d}.wav".format(index)), clip)
    return str(directory)


@pytest.fixture
def script_config(tmp_path):
    return load_config(
        overrides={
            "database": {"url": "sqlite:///{0}".format(tmp_path / "s.db")},
            "crypto": {"salt_path": str(tmp_path / ".salt")},
            "runtime": {"verify_manifest": False, "warmup_iterations": 1},
            "scoring": {"asnorm": {"enabled": False}},
            "logging": {"level": "ERROR"},
        }
    )


# ---------------------------------------------------------------------------
# build_cohort
# ---------------------------------------------------------------------------


def test_build_cohort_writes_a_usable_matrix(patched_features, script_config, audio_dir, tmp_path):
    output = str(tmp_path / "cohort.npz")
    summary = build_cohort(script_config, audio_dir, output_path=output, room_id="lobby")

    assert summary["size"] == 12
    assert summary["dim"] == 192
    assert os.path.isfile(output)

    from speakerid.scoring.asnorm import load_cohort_file

    cohort = load_cohort_file(output)
    assert cohort.shape == (12, 192)
    norms = np.linalg.norm(cohort, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_cohort_refuses_an_empty_directory(patched_features, script_config, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SpeakerIdError, match="no audio files"):
        build_cohort(script_config, str(empty))


def test_build_cohort_skips_files_with_no_speech(
    patched_features, script_config, audio_dir, tmp_path
):
    """Room tone must never become a cohort member."""
    rng = np.random.RandomState(0)
    write_audio(
        os.path.join(audio_dir, "zz_room_tone.wav"),
        (0.004 * rng.randn(5 * 16000)).astype(np.float32),
    )
    summary = build_cohort(script_config, audio_dir, output_path=str(tmp_path / "c.npz"))
    assert summary["n_skipped"] == 1
    assert summary["size"] == 12


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def trial_files(tmp_path):
    """Many utterances per voice, so the trial list can be large enough."""
    from scripts.make_fixtures import VOICES, synth_utterance

    directory = tmp_path / "trials"
    directory.mkdir()
    paths = {}
    for voice in VOICES:
        paths[voice] = []
        for index in range(6):
            path = directory / "{0}_{1}.wav".format(voice, index)
            write_audio(str(path), synth_utterance(voice, 9.0, 4242 + index * 31))
            paths[voice].append(str(path))
    return paths


@pytest.fixture
def verification_trials(tmp_path, trial_files):
    rows = []
    voices = sorted(trial_files)
    for voice in voices:
        files = trial_files[voice]
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                rows.append([files[i], files[j], 1])
    for a_index, voice_a in enumerate(voices):
        for voice_b in voices[a_index + 1 :]:
            for left in trial_files[voice_a]:
                for right in trial_files[voice_b]:
                    rows.append([left, right, 0])
    return write_csv(tmp_path / "trials.csv", ["enroll_path", "test_path", "same_speaker"], rows)


@pytest.fixture
def cm_trials(tmp_path, trial_files):
    from scripts.make_fixtures import simulate_replay

    from speakerid.audio.io import load_audio

    directory = tmp_path / "cm"
    directory.mkdir()
    rows = []
    for voice, files in sorted(trial_files.items()):
        for index, path in enumerate(files):
            rows.append([path, 1])
            replay_path = directory / "{0}_{1}_replay.wav".format(voice, index)
            write_audio(
                str(replay_path), simulate_replay(load_audio(path, peak_dbfs=-3.0), 7000 + index)
            )
            rows.append([str(replay_path), 0])
    return write_csv(tmp_path / "cm_trials.csv", ["path", "bona_fide"], rows)


def test_calibration_requires_countermeasure_data(
    patched_features, script_config, verification_trials, tmp_path
):
    """AASIST-L's published EER is synthetic speech, not physical replay."""
    with pytest.raises(SpeakerIdError, match="replay recordings"):
        run_calibration(
            config=script_config,
            trials_csv=verification_trials,
            target_far=0.05,
            site="test",
            output_path=str(tmp_path / "th.yaml"),
        )


def test_calibration_writes_loadable_thresholds(
    patched_features, script_config, verification_trials, cm_trials, tmp_path
):
    output = str(tmp_path / "thresholds.yaml")
    summary = run_calibration(
        config=script_config,
        trials_csv=verification_trials,
        cm_trials_csv=cm_trials,
        target_far=0.05,
        site="north gate",
        output_path=output,
    )

    assert summary["measured_far"] <= 0.05
    assert summary["cm_calibrated"] is True

    loaded = load_thresholds(output)
    assert loaded.provenance["site"] == "north gate"
    assert loaded.provenance["calibrated_at"]
    assert loaded.provenance["target_far"] == pytest.approx(0.05)
    assert 0.0 <= loaded.cm_threshold <= 1.0


def test_calibration_records_an_uncalibrated_cm_threshold_honestly(
    patched_features, script_config, verification_trials, tmp_path
):
    output = str(tmp_path / "thresholds.yaml")
    summary = run_calibration(
        config=script_config,
        trials_csv=verification_trials,
        cm_threshold_override=0.5,
        target_far=0.05,
        site="test",
        output_path=output,
    )
    assert summary["cm_calibrated"] is False
    assert load_thresholds(output).provenance["cm_calibrated"] is False


def test_calibration_separates_the_synthetic_speakers(
    patched_features, script_config, verification_trials, cm_trials, tmp_path
):
    """A sanity check on the whole chain: these voices are separable."""
    summary = run_calibration(
        config=script_config,
        trials_csv=verification_trials,
        cm_trials_csv=cm_trials,
        target_far=0.05,
        site="test",
        output_path=str(tmp_path / "th.yaml"),
    )
    assert summary["eer"] < 0.2, "the pipeline cannot tell the fixtures apart"
    assert summary["cm_eer"] < 0.2, "the countermeasure cannot spot the replays"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluation_reports_open_set_rejection(
    patched_features,
    script_config,
    verification_trials,
    cm_trials,
    trial_files,
    tmp_path,
    monkeypatch,
):
    """The number CLAUDE.md §8 says actually matters."""
    output = str(tmp_path / "thresholds.yaml")
    run_calibration(
        config=script_config,
        trials_csv=verification_trials,
        cm_trials_csv=cm_trials,
        target_far=0.05,
        site="test",
        output_path=output,
    )
    monkeypatch.setenv("SPEAKERID_THRESHOLDS", output)
    monkeypatch.setattr("speakerid.config.DEFAULT_THRESHOLDS_PATH", output)
    monkeypatch.setattr("scripts.evaluate.load_thresholds", lambda: load_thresholds(output))

    voices = sorted(trial_files)
    enrolled, impostors = voices[:2], voices[2:]
    rows = []
    for voice in enrolled:
        for path in trial_files[voice][:3]:
            rows.append([path, voice, "enroll"])
        for path in trial_files[voice][3:]:
            rows.append([path, voice, "probe"])
    for voice in impostors:
        for path in trial_files[voice]:
            rows.append([path, "", "probe"])  # a stranger: no id
    identification = write_csv(tmp_path / "ident.csv", ["path", "person_id", "role"], rows)

    report = run_evaluation(
        config=script_config,
        trials_csv=verification_trials,
        identification_trials_csv=identification,
        cm_trials_csv=cm_trials,
        report_path=str(tmp_path / "report.json"),
    )

    open_set = report["open_set"]
    assert open_set["n_enrolled_people"] == 2
    assert open_set["n_impostor_probes"] == len(impostors) * 6
    assert 0.0 <= open_set["impostor_rejection_rate"] <= 1.0
    assert "IMPOSTOR REJECTION RATE" in open_set["summary"]
    assert os.path.isfile(str(tmp_path / "report.json"))

    assert report["countermeasure"]["spoof_accept_rate"] <= 0.25


def test_evaluation_refuses_to_run_without_calibration(patched_features, script_config, tmp_path):
    """Evaluating against guessed thresholds would report a meaningless FAR."""
    from speakerid.errors import ThresholdsNotCalibratedError

    bad = write_csv(tmp_path / "bad.csv", ["a", "b"], [[1, 2]])
    with pytest.raises(ThresholdsNotCalibratedError):
        run_evaluation(config=script_config, trials_csv=bad)


def test_evaluation_refuses_a_malformed_trial_file(
    patched_features, script_config, tmp_path, monkeypatch
):
    thresholds_path = tmp_path / "th.yaml"
    thresholds_path.write_text(
        "cm_threshold: 0.5\naccept_threshold: 0.6\nmargin_threshold: 0.05\n"
        "min_speech_sec: 2.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evaluate.load_thresholds", lambda: load_thresholds(str(thresholds_path))
    )
    bad = write_csv(tmp_path / "bad.csv", ["a", "b"], [[1, 2]])
    with pytest.raises(SpeakerIdError, match="missing column"):
        run_evaluation(config=script_config, trials_csv=bad)
