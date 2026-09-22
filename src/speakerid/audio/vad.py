"""Silero VAD wrapper (CLAUDE.md stage A).

VAD quality is the single highest-leverage component in this system
(CLAUDE.md §9.1). Architecture choice is perhaps 20% of real-world accuracy;
whether the embedder sees speech or room tone is most of the rest.

Tune ``vad.threshold`` on the operator's own audio before touching anything
else. The defaults in config/default.yaml are a starting point, not a setting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..errors import ModelError
from ..models.base import OnnxModel, RuntimeOptions, SessionFactory
from ..nptypes import BoolArray, FloatArray
from .io import TARGET_SAMPLE_RATE, ensure_audio

LOGGER = logging.getLogger(__name__)

#: Silero VAD v5 advances 512 samples per window at 16 kHz.
SILERO_WINDOW_SAMPLES = 512

#: ...but the v5 ONNX graph expects 64 samples of the PREVIOUS window
#: prepended to each one, so the tensor it actually receives is 576 wide.
#: This is not optional and not cosmetic: feeding a bare 512-sample window
#: makes the network output ~0.0 for every frame, including clear speech, so
#: the system detects no speech at all and rejects every request. The JIT
#: model hides this by carrying the context internally; the ONNX export does
#: not, and the caller has to do it.
SILERO_CONTEXT_SAMPLES = 64

#: Hidden state shape for the v5 export: (2, batch, 128).
SILERO_STATE_DIM = 128


@dataclass(frozen=True)
class Segment:
    """A half-open speech interval ``[start, end)`` in samples."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid segment: [{0}, {1})".format(self.start, self.end))

    @property
    def duration_samples(self) -> int:
        return self.end - self.start

    def duration_seconds(self, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        return self.duration_samples / float(sample_rate)

    def start_seconds(self, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        return self.start / float(sample_rate)

    def end_seconds(self, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        return self.end / float(sample_rate)


def total_speech_samples(segments: Sequence[Segment]) -> int:
    return int(sum(segment.duration_samples for segment in segments))


def total_speech_seconds(
    segments: Sequence[Segment], sample_rate: int = TARGET_SAMPLE_RATE
) -> float:
    return total_speech_samples(segments) / float(sample_rate)


def speech_mask(segments: Sequence[Segment], length: int) -> BoolArray:
    """Boolean mask over the original buffer, True where VAD found speech."""
    mask: BoolArray = np.zeros(length, dtype=bool)
    for segment in segments:
        mask[min(segment.start, length) : min(segment.end, length)] = True
    return mask


def concatenate_speech(samples: FloatArray, segments: Sequence[Segment]) -> FloatArray:
    """Splice the speech regions together, dropping everything else.

    This is what gets fed to the countermeasure and the embedder. Non-speech
    never reaches either of them.
    """
    ensure_audio(samples, name="concatenate_speech input")
    if not segments:
        return np.zeros(0, dtype=np.float32)
    pieces = [samples[segment.start : segment.end] for segment in segments]
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


class SileroVad(OnnxModel):
    """Streaming Silero VAD, run window-by-window over a whole buffer."""

    slot = "vad"

    def __init__(
        self,
        model_path: str,
        options: RuntimeOptions,
        threshold: float,
        neg_threshold: float,
        window_samples: int,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
        max_speech_seconds: float,
        sample_rate: int = TARGET_SAMPLE_RATE,
        models_dir: Optional[str] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        if window_samples != SILERO_WINDOW_SAMPLES:
            raise ModelError(
                "Silero VAD v5 requires {0}-sample windows at 16 kHz; got {1}".format(
                    SILERO_WINDOW_SAMPLES, window_samples
                )
            )
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.window_samples = window_samples
        self.context_samples = SILERO_CONTEXT_SAMPLES
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.max_speech_samples = int(max_speech_seconds * sample_rate)
        self._state_input: Optional[str] = None
        self._legacy_state_inputs: Tuple[str, ...] = ()
        super(SileroVad, self).__init__(
            model_path, options, models_dir=models_dir, session_factory=session_factory
        )

    @classmethod
    def from_config(
        cls, config: Config, session_factory: Optional[SessionFactory] = None
    ) -> SileroVad:
        return cls(
            model_path=config.resolve_path("vad_model.model_path"),
            options=RuntimeOptions.from_config(config),
            threshold=config.get_float("vad.threshold"),
            neg_threshold=config.get_float("vad.neg_threshold"),
            window_samples=config.get_int("vad.window_samples"),
            min_speech_ms=config.get_int("vad.min_speech_ms"),
            min_silence_ms=config.get_int("vad.min_silence_ms"),
            speech_pad_ms=config.get_int("vad.speech_pad_ms"),
            max_speech_seconds=config.get_float("vad.max_speech_seconds"),
            sample_rate=config.sample_rate,
            models_dir=config.resolve_relative("models"),
            session_factory=session_factory,
        )

    # -- OnnxModel hooks -----------------------------------------------------

    def _validate_signature(self) -> None:
        names = set(self.input_names)
        if "state" in names:
            self._state_input = "state"  # v5 export
        elif {"h", "c"}.issubset(names):
            self._legacy_state_inputs = ("h", "c")  # v4 export
        elif len(self.input_names) == 1:
            self._state_input = None  # stateless test double
        else:
            raise ModelError(
                "unrecognised Silero VAD signature; inputs were {0}".format(self.input_names)
            )

    def _warmup_feeds(self) -> Dict[str, FloatArray]:
        return self._build_feeds(
            np.zeros((1, self.window_samples + self.context_samples), dtype=np.float32),
            self._initial_state(),
        )

    # -- state plumbing ------------------------------------------------------

    def _initial_state(self) -> Dict[str, FloatArray]:
        if self._state_input:
            return {self._state_input: np.zeros((2, 1, SILERO_STATE_DIM), dtype=np.float32)}
        if self._legacy_state_inputs:
            return {
                name: np.zeros((2, 1, SILERO_STATE_DIM), dtype=np.float32)
                for name in self._legacy_state_inputs
            }
        return {}

    def _build_feeds(
        self, window: FloatArray, state: Dict[str, FloatArray]
    ) -> Dict[str, FloatArray]:
        feeds: Dict[str, FloatArray] = {self.input_names[0]: window}
        feeds.update(state)
        if "sr" in self.input_names:
            feeds["sr"] = np.array(self.sample_rate, dtype=np.int64)
        return feeds

    # -- inference -----------------------------------------------------------

    def speech_probabilities(self, samples: FloatArray) -> FloatArray:
        """Per-window P(speech), one value per ``window_samples`` of input."""
        ensure_audio(samples, self.sample_rate, name="vad input")

        num_windows = int(np.ceil(samples.size / float(self.window_samples)))
        if num_windows == 0:
            return np.zeros(0, dtype=np.float32)

        # Zero-pad only the final partial window; the VAD is a detector, not a
        # generative model, so a short tail costs nothing.
        padded_length = num_windows * self.window_samples
        if samples.size < padded_length:
            padded: FloatArray = np.zeros(padded_length, dtype=np.float32)
            padded[: samples.size] = samples
        else:
            padded = samples

        state = self._initial_state()
        # The context carried into the first window is silence, matching the
        # reference implementation's cold start.
        context: FloatArray = np.zeros((1, self.context_samples), dtype=np.float32)
        probabilities: FloatArray = np.zeros(num_windows, dtype=np.float32)
        for index in range(num_windows):
            offset = index * self.window_samples
            window = padded[offset : offset + self.window_samples].reshape(1, -1)
            feed = np.concatenate([context, window], axis=1).astype(np.float32)
            outputs = self._run(self._build_feeds(feed, state))
            probabilities[index] = float(np.asarray(outputs[0]).reshape(-1)[0])
            state = self._next_state(outputs)
            context = window[:, -self.context_samples :]
        return probabilities

    def _next_state(self, outputs: Sequence[FloatArray]) -> Dict[str, FloatArray]:
        if self._state_input and len(outputs) > 1:
            return {self._state_input: np.asarray(outputs[1], dtype=np.float32)}
        if self._legacy_state_inputs and len(outputs) > 2:
            return {
                self._legacy_state_inputs[0]: np.asarray(outputs[1], dtype=np.float32),
                self._legacy_state_inputs[1]: np.asarray(outputs[2], dtype=np.float32),
            }
        return self._initial_state() if not self._state_input else {}

    def segments(self, samples: FloatArray) -> List[Segment]:
        """Return speech segments in sample coordinates."""
        probabilities = self.speech_probabilities(samples)
        return self.segments_from_probabilities(probabilities, samples.size)

    # -- probability -> segments --------------------------------------------

    def segments_from_probabilities(
        self, probabilities: FloatArray, total_samples: int
    ) -> List[Segment]:
        """Hysteresis thresholding, then duration filtering and padding.

        Split out from inference so it can be tested exhaustively against
        hand-built probability sequences, with no model involved.
        """
        raw: List[List[int]] = []
        triggered = False
        start = 0
        silence_start: Optional[int] = None

        for index, probability in enumerate(probabilities):
            position = index * self.window_samples
            if not triggered:
                if probability >= self.threshold:
                    triggered = True
                    start = position
                    silence_start = None
                continue

            # Triggered: stay in speech until we drop below the lower
            # (negative) threshold for long enough. The gap between the two
            # thresholds stops a single quiet frame from splitting a word.
            if probability >= self.neg_threshold:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = position
                elif position - silence_start >= self.min_silence_samples:
                    raw.append([start, silence_start])
                    triggered = False
                    silence_start = None
                    continue

            if position - start >= self.max_speech_samples:
                # Force a cut so one continuous talker cannot produce an
                # unbounded buffer.
                raw.append([start, position])
                triggered = False
                silence_start = None

        if triggered:
            raw.append([start, total_samples])

        return self._postprocess(raw, total_samples)

    def _postprocess(self, raw: List[List[int]], total_samples: int) -> List[Segment]:
        segments: List[Segment] = []
        for start, end in raw:
            end = min(end, total_samples)
            if end - start < self.min_speech_samples:
                continue  # too short to be a word; almost always a noise burst
            padded_start = max(0, start - self.speech_pad_samples)
            padded_end = min(total_samples, end + self.speech_pad_samples)
            segments.append(Segment(padded_start, padded_end))
        return _merge_overlapping(segments)


def _merge_overlapping(segments: Sequence[Segment]) -> List[Segment]:
    """Merge segments that padding pushed into each other."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item.start)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start <= last.end:
            merged[-1] = Segment(last.start, max(last.end, segment.end))
        else:
            merged.append(segment)
    return merged
