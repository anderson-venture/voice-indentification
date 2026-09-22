"""Generate the deterministic synthetic audio fixtures in tests/fixtures/.

Real speech cannot be committed to this repository: it is biometric data, and
a test corpus of real voices is exactly the thing CLAUDE.md §11 is about. So
the fixtures are synthesised.

They are built to exercise the parts of the system that matter:

* **Speakers** are harmonic stacks with distinct f0 and formant envelopes, so
  two "people" are genuinely separable by spectral shape, and two utterances
  from one person are genuinely similar. The fake embedder in the test suite
  measures spectral shape, so identification tests pass or fail for a real
  reason rather than by construction.
* **Pauses and room tone** give the VAD something to segment. Silence is never
  digital zero, because real rooms are not.
* **Replays** are simulated the way a real one differs from a live voice:
  convolved with a short room impulse response (the attacker's room, then
  yours -- doubled reverberation), high-pass filtered because loudspeakers are
  weak below ~150 Hz, and mildly clipped. Those are the cues a countermeasure
  is supposed to find.

Everything is seeded. Re-running this script reproduces the files byte for
byte, which is what makes the golden-file embedding test meaningful.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from speakerid.audio.io import TARGET_SAMPLE_RATE, peak_normalize, write_audio
from speakerid.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

#: Master seed. Changing it changes every fixture and breaks the golden test,
#: which is the point: the golden test exists to notice exactly that.
MASTER_SEED = 20260920

#: A small phone or Bluetooth loudspeaker passes roughly 150 Hz - 6 kHz. A
#: replay is band-limited at BOTH ends, and the top-end roll-off is the part
#: that survives VAD selection, because it lives inside the speech band
#: (CLAUDE.md §10).
SPEAKER_HIGHPASS_HZ = 150.0
SPEAKER_LOWPASS_HZ = 5500.0

#: Synthetic speaker definitions: (f0 in Hz, formant centres in Hz).
VOICES: Dict[str, Tuple[float, Tuple[float, ...]]] = {
    "alice": (196.0, (700.0, 1220.0, 2600.0)),
    "bob": (112.0, (530.0, 1840.0, 2480.0)),
    "carol": (233.0, (850.0, 1610.0, 2900.0)),
    "dave": (128.0, (620.0, 1400.0, 2300.0)),
}


def _formant_envelope(frequencies: np.ndarray, formants: Sequence[float]) -> np.ndarray:
    """Resonant gain curve: the part that makes a voice sound like a person."""
    envelope = np.zeros_like(frequencies)
    for index, centre in enumerate(formants):
        bandwidth = 90.0 + 40.0 * index
        envelope += (1.0 / (1.0 + index)) * np.exp(-0.5 * ((frequencies - centre) / bandwidth) ** 2)
    # A gentle spectral tilt, as in real voiced speech.
    envelope += 0.05 * np.exp(-frequencies / 2500.0)
    return envelope


def synth_voiced(
    f0: float,
    formants: Sequence[float],
    num_samples: int,
    rng: np.random.RandomState,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """One voiced stretch: a harmonic stack shaped by the formant envelope."""
    time = np.arange(num_samples, dtype=np.float64) / sample_rate
    # Small random pitch drift, so repeated utterances are not identical.
    jitter = 1.0 + 0.012 * np.sin(2.0 * np.pi * 1.7 * time + rng.uniform(0, 6.28))
    phase = 2.0 * np.pi * f0 * np.cumsum(jitter) / sample_rate

    signal = np.zeros(num_samples, dtype=np.float64)
    harmonics = int((sample_rate / 2.0) // f0)
    centres = np.array([h * f0 for h in range(1, harmonics + 1)], dtype=np.float64)
    gains = _formant_envelope(centres, formants)
    for index, gain in enumerate(gains, start=1):
        signal += gain * np.sin(index * phase + rng.uniform(0, 6.28))

    # Breath noise, shaped so it does not flatten the spectral signature.
    signal += 0.06 * np.sum(gains) * rng.randn(num_samples) / max(harmonics, 1)

    # Syllable-rate amplitude modulation.
    syllables = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.6 * time + rng.uniform(0, 6.28))
    return (signal * syllables).astype(np.float32)


def synth_utterance(
    voice: str,
    seconds: float,
    seed: int,
    noise_level: float = 0.004,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """A full utterance: speech bursts separated by pauses, over room tone."""
    if voice not in VOICES:
        raise KeyError("unknown synthetic voice: {0}".format(voice))
    f0, formants = VOICES[voice]
    rng = np.random.RandomState(seed)

    total = int(seconds * sample_rate)
    # Room tone: never digital silence, because a real room never is.
    audio = (noise_level * rng.randn(total)).astype(np.float32)

    position = int(0.25 * sample_rate)
    while position < total - int(0.4 * sample_rate):
        burst = int(rng.uniform(0.7, 1.6) * sample_rate)
        burst = min(burst, total - position - int(0.2 * sample_rate))
        if burst <= 0:
            break
        speech = synth_voiced(f0, formants, burst, rng, sample_rate)
        speech = speech / (np.max(np.abs(speech)) or 1.0)
        # Fade the edges so bursts do not click.
        ramp = int(0.02 * sample_rate)
        window = np.ones(burst, dtype=np.float32)
        window[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        window[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        audio[position : position + burst] += 0.7 * speech * window
        # Generous pauses: real enrolment recordings have them, and they give
        # the SNR estimator clean non-speech to measure the noise floor from.
        position += burst + int(rng.uniform(0.45, 0.80) * sample_rate)

    return peak_normalize(np.ascontiguousarray(audio, dtype=np.float32), -3.0)


# ---------------------------------------------------------------------------
# Replay simulation
# ---------------------------------------------------------------------------


def _room_impulse(
    seconds: float, decay: float, seed: int, sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Crude exponentially-decaying noise impulse response."""
    rng = np.random.RandomState(seed)
    length = int(seconds * sample_rate)
    time = np.arange(length, dtype=np.float64) / sample_rate
    impulse = rng.randn(length) * np.exp(-time / decay)
    impulse[0] += 1.0
    return (impulse / np.max(np.abs(impulse))).astype(np.float32)


def _bandlimit(
    samples: np.ndarray,
    low_hz: float = SPEAKER_HIGHPASS_HZ,
    high_hz: float = SPEAKER_LOWPASS_HZ,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Restrict to a small loudspeaker's passband.

    The low cut is what a phone speaker cannot reproduce; the high cut is what
    its driver and codec throw away. The high cut matters most for detection,
    because it sits inside the speech band and therefore survives the VAD.
    """
    from scipy.signal import butter, sosfilt

    nyquist = sample_rate / 2.0
    sos = butter(
        4, [low_hz / nyquist, min(high_hz / nyquist, 0.99)], btype="bandpass", output="sos"
    )
    return np.ascontiguousarray(sosfilt(sos, samples.astype(np.float64)), dtype=np.float32)


def simulate_replay(
    samples: np.ndarray,
    seed: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    clip: float = 0.86,
) -> np.ndarray:
    """Turn a live recording into what a replay of it sounds like.

    The chain an attacker actually creates: the voice was recorded in *their*
    room, played through a small loudspeaker that cannot reproduce the low
    end, and re-recorded in *yours*. Hence doubled reverberation, a missing
    bottom octave, and a little amplifier clipping.
    """
    # Realistic small-room reverberation. An attacker records in their room,
    # then plays it back into yours, so the speech carries BOTH tails -- the
    # doubled reverberation CLAUDE.md §10 describes. RT60 here is a few
    # hundred milliseconds, as in an ordinary office.
    attacker_room = _room_impulse(0.50, 0.085, seed, sample_rate)
    your_room = _room_impulse(0.40, 0.070, seed + 1, sample_rate)

    stage = np.convolve(samples.astype(np.float64), attacker_room, mode="same")
    stage = _bandlimit(stage.astype(np.float32), sample_rate=sample_rate)
    stage = np.convolve(stage.astype(np.float64), your_room, mode="same")
    stage = np.clip(stage / (np.max(np.abs(stage)) or 1.0), -clip, clip)

    # Re-recording adds the second room's noise floor.
    rng = np.random.RandomState(seed + 2)
    stage = stage + 0.002 * rng.randn(stage.size)
    return peak_normalize(np.ascontiguousarray(stage, dtype=np.float32), -3.0)


# ---------------------------------------------------------------------------
# Fixture set
# ---------------------------------------------------------------------------


def fixture_plan() -> List[Tuple[str, str, float, int]]:
    """(filename, voice, seconds, seed) for every committed fixture."""
    plan: List[Tuple[str, str, float, int]] = []
    for index in range(1, 5):
        plan.append(("alice_{0:02d}.wav".format(index), "alice", 9.0, MASTER_SEED + index))
    for index in range(1, 4):
        plan.append(("bob_{0:02d}.wav".format(index), "bob", 9.0, MASTER_SEED + 100 + index))
    # Carol is never enrolled: she is the impostor in the open-set tests.
    plan.append(("carol_01.wav", "carol", 9.0, MASTER_SEED + 200))
    plan.append(("dave_01.wav", "dave", 9.0, MASTER_SEED + 300))
    return plan


def generate(output_dir: str) -> List[str]:
    """Write every fixture. Deterministic: same inputs, same bytes."""
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    for filename, voice, seconds, seed in fixture_plan():
        audio = synth_utterance(voice, seconds, seed)
        path = os.path.join(output_dir, filename)
        write_audio(path, audio)
        written.append(path)
        LOGGER.info("wrote %s (%.1fs)", filename, seconds)

    # A replay of alice_01: same voice, different provenance.
    alice = synth_utterance("alice", 9.0, MASTER_SEED + 1)
    replay_path = os.path.join(output_dir, "alice_01_replay.wav")
    write_audio(replay_path, simulate_replay(alice, MASTER_SEED + 900))
    written.append(replay_path)

    # Room tone only: the VAD must find no speech here, and the embedder must
    # never see it.
    rng = np.random.RandomState(MASTER_SEED + 1000)
    silence = (0.004 * rng.randn(int(5.0 * TARGET_SAMPLE_RATE))).astype(np.float32)
    silence_path = os.path.join(output_dir, "room_tone.wav")
    write_audio(silence_path, silence)
    written.append(silence_path)

    # Too short to judge: must produce REJECTED(insufficient_speech).
    short = synth_utterance("alice", 1.1, MASTER_SEED + 1100)
    short_path = os.path.join(output_dir, "too_short.wav")
    write_audio(short_path, short)
    written.append(short_path)

    LOGGER.info("wrote %d fixtures to %s", len(written), output_dir)
    return written


def cohort_audio(count: int = 24, seconds: float = 3.0) -> List[np.ndarray]:
    """Cohort recordings, generated in memory rather than committed.

    A cohort is 200-500 clips in production. The test suite needs only enough
    to exercise the AS-Norm code path, and generating them keeps the
    repository small.
    """
    voices = list(VOICES)
    clips: List[np.ndarray] = []
    for index in range(count):
        voice = voices[index % len(voices)]
        clips.append(synth_utterance(voice, seconds, MASTER_SEED + 5000 + index * 7))
    return clips


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    configure_logging(level="INFO")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures"
    )
    generate(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
