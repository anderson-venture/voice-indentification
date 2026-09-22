# Deviations from CLAUDE.md

Every departure from the standing brief, why it was made, and what would undo
it. Anything not listed here was implemented as specified.

---

## 1. Python 3.8 provisioning (RESOLVED)

**Brief:** §2 — "Python 3.8.x."

**How to get 3.8 per platform:**

* **Windows** — `python-3.8.10-amd64.exe` from python.org. 3.8.10 is the last
  release for which a Windows installer was published, so the patch version
  is forced.
* **Linux** — distro package where one exists. Ubuntu 24.04 ships no 3.8, has
  no `python3-venv` and no system `pip`; this development machine therefore
  uses a prebuilt portable CPython **3.8.20** from `python-build-standalone`
  at `~/.local/share/cpython-3.8`. A genuine CPython, no root required. That
  is a local development convenience only — nothing in the repository depends
  on it.

**The pins are verified, not assumed.** Every one installs on real 3.8.20:

    numpy 1.24.4 · scipy 1.10.1 · onnxruntime 1.19.2 · soundfile 0.12.1
    librosa 0.10.2.post1 · sqlalchemy 2.0.36 · cryptography 43.0.3
    pydantic 1.10.18 · click 8.1.7 · pyyaml 6.0.2 · onnx 1.16.2
    pandas 2.0.3 · matplotlib 3.7.5 · pytest 8.3.4 · black 24.8.0
    ruff 0.6.9 · mypy 1.14.1 · torch 2.4.1+cpu

The full suite, `ruff`, `black --check` and `mypy --strict` all pass under
3.8.20. This closes the pin-validity half of milestone 1.

### Distribution: plain git clone

The project is distributed as a git repository, not as an offline bundle. The
three ONNX models are committed (25 MB total, largest file 22 MB — under
GitHub's limits), so `git clone` yields a working tree and `pip install -r
requirements.txt` completes the setup. CLAUDE.md §4 permits exactly this
("committed via git-lfs or shipped by USB").

An earlier iteration built platform-specific offline bundles carrying a
portable CPython and a full wheelhouse. That was removed as unnecessary
complexity once it was clear the target machine can reach a network at setup
time. §2's "fully offline" constraint is about the **running system** — which
still makes no network calls, enforced by `tests/test_offline.py` — not about
`pip install`. If a genuinely air-gapped install is needed later, rebuild the
wheelhouse with:

    pip download -r requirements.txt -d wheels --only-binary=:all: \
        --python-version 38 --platform <target-platform>

### Two things the pinned toolchain caught

* `ruff` 0.6.9 rejects the selector `UP045`, which had been added to the
  ignore list for a newer ruff. Removed.
* `black` 24.8.0 wanted 31 files reformatted. Applied; §12 asks for it.

### Local-only mypy note

`disallow_untyped_decorators` is off for `speakerid.cli` and `speakerid.api`.
Click's command decorators and FastAPI's route decorators are untyped from
mypy's point of view, which would mark every handler untyped by inference
rather than by authorship; the bodies are still fully checked. `scripts/` is
excluded because §12 requires strict typing of `src/`, and the scripts are
development tooling covered by ruff and the tests.

### Requirements files renamed (§3)

§3 shows a single `requirements.lock.txt` holding runtime and dev pins
together. That is split into three files with plainer names, because the
distinction matters operationally and a single list hid it:

| File | Contents |
|---|---|
| `requirements.txt` | the §3 "Runtime (required, offline)" block |
| `requirements-dev.txt` | pytest, pytest-cov, black, ruff, mypy |
| `requirements-export.txt` | torch, torchaudio, onnx, onnxsim — §4's "dev/export only" |

Versions are unchanged from §3. Keeping PyTorch in a file of its own makes it
hard to install ~2 GB of export-time tooling onto a device by accident.

---

## 2. Resampling uses `scipy.signal.resample_poly`, not librosa

**Brief:** §3 lists `librosa` as a runtime dependency.

**Reason:** `resample_poly` is deterministic, anti-aliased, has no hidden
state and no lazy-loaded numba kernels. The golden-file test requires that the
same wav produce the same samples on every machine, and librosa's resampler
varies with the installed backend (`soxr`, `samplerate` or its own fallback).
`resample_poly` also removes a large transitive dependency tree from the
device image.

`librosa` remains in `requirements.txt` as specified, but no runtime
module imports it. Dropping it from the shipped set would be reasonable.

---

## 3. `accept_threshold` and `margin_threshold` are not bounded to [0, 1]

**Brief:** §8 shows starting points of `0.55` and `0.08`, which read as cosine
values.

**Reason:** with AS-Norm enabled — the default, and the recommendation in §9.3
— the score reaching `decide()` is a **z-score against the cohort**, not a raw
cosine. It is unbounded and routinely exceeds 1.0. Validating those two keys
against [0, 1] would have rejected every correctly calibrated file.

**What was done:** `cm_threshold` is still validated as a probability in
[0, 1], because it is one. The two score thresholds are checked only against
an implausibility bound (`±20`). `calibrate.py` derives them on exactly the
scale `decide()` sees, and records `asnorm_enabled` in the threshold
provenance so the scale is never ambiguous.

---

## 4. Model acquisition (RESOLVED, with substitutions)

**Brief:** §5 — models are acquired once on a connected machine, converted to
ONNX, checksummed into `models/MANIFEST.json`, and verified on load.

All three required slots are now filled and verified. `models/MANIFEST.json`
exists and `speakerid verify-models` passes.

| Slot | File | Source | Params | Note |
|---|---|---|---|---|
| VAD | `vad/silero_vad.onnx` | `snakers4/silero-vad` (MIT) | — | v5 ONNX, as specified |
| Embedder | `embed/redimnet2_b2.onnx` | `OpenVoiceOS/redimnet-b2-vox2-onnx` (HF) | ~5.1M | **substitution, see below** |
| Countermeasure | `cm/aasist_l.onnx` | exported from `clovaai/aasist` (MIT) | **85,306** | matches §5 exactly |
| Embedder fallback | `embed/campplus.onnx` | — | — | not fetched; optional |

### 4a. Embedder is ReDimNet-**B2** (VoxCeleb2), not ReDimNet**2**-B2

§5 specifies ReDimNet2-B2 at 3.6M params / 0.95 GMACs. The export actually
obtained carries ~5.08M initializer parameters and is labelled
`redimnet-b2-vox2`. It satisfies the functional contract — 192-d output, raw
waveform input, Apache-2.0 — and separates speakers well (see
`docs/FIRST_RUN.md`), but it is **not** the exact checkpoint the brief names.
Anyone reproducing the published 0.57% VoxCeleb1-O figure should source the
true ReDimNet2-B2 and re-run `fetch_models.py`; `model_id` carries the file
hash, so swapping it invalidates the existing gallery loudly rather than
silently.

### 4b. `speakeronnx` exists but was not used as a dependency

§5's "fastest acquisition path" is real: the package is on PyPI and registers
`redimnet-b2` among others. It requires Python ≥ 3.9, and §5 itself says to
drop the dependency after pulling weights, so the ONNX file was fetched
directly from the HuggingFace repo the package points at. Nothing in the
shipped tree references it.

### 4c. The embedder's real interface differs from the stated contract

Two mismatches, both found by running the file rather than trusting the table:

1. **Input rank.** §5 says `float32 [1, N]`. The export declares
   `[B, 1, T_samples]` and *rejects* rank-2 input outright. `SpeakerEmbedder`
   now reads the rank from the graph and shapes the feed accordingly, so
   either convention works.
2. **Output is not L2-normalized.** §5 says it is. The export returns vectors
   of norm ≈ 18. The wrapper already re-normalized defensively, which is the
   only reason this did not silently inflate every cosine score by ~18× and
   move the operating point. `tests/test_export_parity.py` now asserts both
   the un-normalized raw output and the unit-norm wrapper output.

### 4d. AASIST-L had to be exported, not downloaded

No ready-made AASIST-L ONNX matching the 85K-parameter architecture was
found, so `scripts/export_aasist.py` builds it from the official PyTorch
checkpoint. The exported model has **85,306 parameters**, exactly the figure
in §5, and ONNX-vs-PyTorch parity is 9.5e-07 max absolute difference. The
script refuses to write a file that fails parity, and loads the checkpoint
with `weights_only=True` because a `.pth` is a pickle fetched over the
network.

---

## 5. Countermeasure calibration is mandatory, with an explicit escape hatch

**Brief:** §10 — fine-tune AASIST-L on physical replay data.

**What was done:** `calibrate.py` **refuses to run** without `--cm-trials`,
and the error explains what to record (8–10 devices, 0.3 m / 1 m / 3 m, each
deployment room). An operator who insists may pass `--cm-threshold` by hand;
the resulting file then records `cm_calibrated: false`, which the loader logs
at startup and `evaluate.py` reports.

Fine-tuning AASIST-L on EchoFake plus in-house replays is **not implemented**.
It needs a GPU and the corpus, neither of which is available here. The
architecture supports it: retrain, re-export, re-run `fetch_models.py`, and
the new checksum flows through automatically.

---

## 6. `min_speech_sec` used during calibration is a config value

**Brief:** §8 lists `min_speech_sec` among the calibrated thresholds.

**Tension:** calibration needs to extract embeddings *before* any threshold
file exists, but the VAD gate it applies while doing so determines the
conditions under which the operating point was measured.

**Resolution:** `calibration.min_speech_sec` in `config/default.yaml` gates
the calibration run, and `calibrate.py` writes that same value into
`thresholds.yaml` as the deployed `min_speech_sec`. The number the system
enforces is therefore always the number the measurement was made under.

---

## 6b. Bugs found only by running the real models

Three defects that the synthetic test doubles could not have caught, each
fixed and each now covered by a regression test.

### The Silero VAD context window — the serious one

Silero VAD v5's ONNX graph expects **64 samples of the previous window
prepended** to each 512-sample window, so the tensor it receives is 576 wide.
The JIT model carries that context internally; the ONNX export does not, and
the caller must do it. The original implementation fed a bare 512-sample
window.

The failure mode is what makes this dangerous: the model does **not** raise.
It returns ~0.0 for every frame, including unambiguous speech. Measured on a
real 10.4 s speech clip:

| Feed | max P(speech) | frames > 0.5 |
|---|---|---|
| 512 samples, no context | 0.078 | 0.0% |
| 512 + 64 context | **1.000** | **89.3%** |

With the bug, the VAD finds no speech anywhere, `min_speech_sec` rejects
everything, and every request returns `REJECTED(insufficient_speech)`. It
looks exactly like a threshold-tuning problem and is not one.

Fixed in `SileroVad.speech_probabilities`, which now carries the rolling
context. The VAD test double **asserts** the 576-sample width, so a
regression fails loudly instead of silently detecting no speech, and
`tests/test_vad.py` checks that the context really is the previous window's
tail.

### The embedder input rank and normalization

See §4c. Both were assumptions taken from the brief's table that the actual
file contradicts.

### The margin sweep could never identify anyone

`calibrate.py`'s `--identification-trials` sweep built its gallery by
union-find over the verification trial pairs, keying people by file path,
then compared those paths against the operator's person labels. They could
never match, so top-1 accuracy was always 0 and the optimiser happily chose
the margin that rejects every probe (`margin_threshold: 1.0`, impostor
rejection 1.0, hit rate 0.0) — a degenerate operating point that would have
been shipped as calibrated.

Three changes:

* the sweep now uses the same `path,person_id,role` trial schema as
  `evaluate.py` and builds the gallery from rows the operator labelled
  `enroll`. One format for both scripts, no inference;
* ties break toward the **larger** margin, because between two equally
  scoring operating points the more demanding one returns `AMBIGUOUS` where
  the other guesses (§6 says to bias that way). On the demo corpus this moved
  the chosen margin from 0.0 to 0.55 with no loss of accuracy;
* if the best achievable operating point still identifies nobody, calibration
  now logs an explicit "do not deploy" error instead of writing the number.

### Liveness and identification now have separate target FARs

`--target-far` set the operating point for both the speaker threshold and the
liveness threshold, but they are different questions measured on different
trial sets — here 536 impostor pairs versus 179 replay clips. Requesting
FAR ≤ 1% correctly failed because the replay corpus cannot resolve a rate
that fine. Added `--cm-target-far`, defaulting to `--target-far`.

---

## 7. Not implemented

Honest scope statement. None of these are stubbed as if they worked.

| Item | Brief | Status |
|---|---|---|
| Digit-keyword verifier for challenge-response | §10.1 | **Not implemented.** `ChallengeManager` owns issuing, TTL and single-use consumption. Verification takes a `DigitVerifier` callable; with none attached it **fails closed**. A small keyword model must be wired in before this gates a door. |
| AASIST-L replay fine-tuning | §10 | Not implemented — needs GPU + EchoFake. Now demonstrably needed: see `docs/FIRST_RUN.md`. |
| Large-margin fine-tuning with MUSAN/RIR | §9.6 | Not implemented — needs GPU. |
| Microphone-array direction-of-arrival | §10.2 | Not implemented — single-channel only. Requires hardware. |
| INT8 quantisation / device port | §13 milestone 9 | Not started, as instructed: milestone 7 comes first. |
| CAM++ fallback embedder | §5 | Supported by configuration (`embedder.model_path`); not fetched, no A/B run. |
| Offline wheelhouse | §3, milestone 1 | Removed. The project installs from PyPI after `git clone`; §3's `pip download` recipe still applies if a genuinely air-gapped install is needed. |
| AS-Norm cohort | §9.3 | No cohort built: the available corpus has two speakers, and a cohort drawn from the enrolled speakers would normalise against the wrong distribution. AS-Norm was disabled for the demo run. |

---

## 8. Test fixtures are synthetic

Real speech cannot be committed: it is biometric data, and a corpus of real
voices in a git repository is the exact failure §11 is about.

`scripts/make_fixtures.py` synthesises four "speakers" as harmonic stacks with
distinct f0 and formant envelopes, and simulates replay through a physically
motivated chain — the attacker's room impulse response, a small loudspeaker's
150 Hz–6 kHz passband, amplifier clipping, then your room's impulse response.
That doubled reverberation is the cue the countermeasure double detects, and
it separates live from replayed audio across all four voices with a wide
margin.

This validates **plumbing and logic**, never accuracy. Accuracy numbers come
only from milestone 7 on the operator's own recordings.
