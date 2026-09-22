# Deviations from CLAUDE.md

Every departure from the standing brief, why it was made, and what would undo
it. Anything not listed here was implemented as specified.

---

## 1. Python 3.8 provisioning and distribution (RESOLVED)

**Brief:** §2 — "Python 3.8.x." §3 — a wheelhouse built with `pip download`,
installed with `pip install --no-index`.

**What is done instead:** the interpreter and the environment are *copied*,
not installed. The project folder carries:

* `python38\` — a verbatim copy of a CPython **3.8.10** install (62 MB).
  3.8.10 is the last release python.org published a Windows installer for, so
  the patch version is forced.
* `venv\` — the environment itself (1.7 GB), with `venv\pyvenv.cfg` pointing
  `home` at `python38\` rather than at any system install.

Installing on the offline machine is one `robocopy` to the identical absolute
path. `docs/SETUP.md` is the procedure.

**Why not the wheelhouse in §3.** A wheelhouse still needs a Python 3.8 on the
target, at a path the venv can find, of a matching patch release — and then a
`pip install` step that can fail. Copying the interpreter removes all of it.
It also removes the wheelhouse itself: there is no `wheels\` folder, and
`pip install` on the target is expected to fail. The cost is that nothing new
can be installed there, which is why the optional and export dependencies are
installed up front (see the table below).

**The pins are verified, not assumed.** Every package in
`requirements.lock.txt` is installed and importable on Windows under CPython
3.8.10, including `torch 2.4.1+cpu`, `torchaudio 2.4.1+cpu`, `onnxruntime
1.19.2`, `numba 0.58.1` and `llvmlite 0.41.1`.

**Self-containment was tested, not assumed.** With the machine's own Python
3.8 renamed out of the way, the full suite still passed (255 passed,
2 skipped), every heavy dependency still imported, and `verify-models` still
matched all three checksums. Nothing under `E:\Project\voice-indentification`
reads from the system Python install.

### Deviations from §3's package list

* **`pandas` and `matplotlib` are not installed.** §3 lists both under dev
  tooling, but no module in `src/`, `scripts/` or `tests/` imports either.
  `scripts/evaluate.py` and `scripts/calibrate.py` compute DET points and
  metrics with numpy and write JSON; plotting is left to whatever the
  operator already has.
* **`keyring` is installed but not in §3.** `crypto.key_source: keyring` is a
  documented option in `config/default.yaml`, and on an offline machine an
  uninstallable option is a dead end. The default remains `passphrase`.
* **`fastapi` / `uvicorn` are installed**, per §3's "Optional local API". The
  CLI does not need them.

### Two things the pinned toolchain caught

* `ruff` 0.6.9 rejects the selector `UP045`, which had been added to the
  ignore list for a newer ruff. Removed.
* `black` 24.8.0 wanted files reformatted. Applied; §12 asks for it.

`pyproject.toml` excludes `python38\` and `venv\` from both `ruff` and
`black`. Without that, a bare `ruff check .` walks into the vendored stdlib
and reports 13,611 errors, including Python 2 syntax in the bundled Tcl
support files.

### Local-only mypy note

`disallow_untyped_decorators` is off for `speakerid.cli` and `speakerid.api`.
Click's command decorators and FastAPI's route decorators are untyped from
mypy's point of view, which would mark every handler untyped by inference
rather than by authorship; the bodies are still fully checked. `scripts/` is
excluded because §12 requires strict typing of `src/`, and the scripts are
development tooling covered by ruff and the tests.

### `import speakerid` without an install

The package is not `pip install`ed. `venv\Lib\site-packages\_speakerid_src.pth`
holds the relative path `..\..\..\src`, which `site` resolves against
site-packages at startup. This replaces the `PYTHONPATH=src` that earlier
instructions used, works in every process without shell setup, and survives
relocation because it is relative. `venv\Scripts\speakerid.bat` provides the
`speakerid` command in place of a pip-generated `.exe`, for the same reason:
it calls `%~dp0python.exe`, relative to itself.

### Requirements files (§3)

§3 shows a single `requirements.lock.txt` holding runtime and dev pins
together. There are five files, because the distinction matters operationally
and a single list hid it:

| File | Contents |
|---|---|
| `requirements.txt` | the §3 "Runtime (required, offline)" block |
| `requirements-dev.txt` | pytest, pytest-cov, black, ruff, mypy |
| `requirements-optional.txt` | fastapi, uvicorn, keyring |
| `requirements-export.txt` | torch, torchaudio, onnx, onnxsim — §4's "dev/export only" |
| `requirements.lock.txt` | `pip freeze --all` of `venv\`: a record, not an install step |

Versions are unchanged from §3. Keeping PyTorch in a file of its own makes it
hard to install 1.1 GB of export-time tooling onto a device by accident —
though on this deployment it is already in `venv\`, because §10's replay
fine-tuning is impossible without it and the target cannot install it later.

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

## 6c. Bug found only by running on Windows

### The PBKDF2 salt was corrupted by Windows text mode

`_load_or_create_salt()` created the salt with

    os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

On Windows a descriptor opened without `os.O_BINARY` is in **text mode**, and
`os.write` expands every `0x0A` byte to `0x0D 0x0A`. A random 16-byte salt
contains at least one `0x0A` about 6% of the time, so roughly one salt in
sixteen was written 17 bytes or longer.

The length check on the read path turned that into a `CryptoError` on the next
run rather than a silently wrong key, which is the right failure — but it is
still a database that cannot be opened, and the 6% rate made it look like
flakiness rather than a bug. It first appeared as an intermittent
`test_delete_requires_confirmation` failure: *salt file ... is 17 bytes,
expected 16*.

Fixed by adding `getattr(os, "O_BINARY", 0)` to the flags — the constant does
not exist on POSIX. `tests/test_crypto.py::test_salt_survives_newline_bytes`
pins `os.urandom` to a salt containing `0x0A`, `0x0D` and `0x1A` and asserts a
byte-for-byte round trip; it fails without the flag.

This is why §12's "pytest must pass" is not the same statement as "pytest
passed once". The suite had passed on Linux, where the flag is meaningless.

---

## 6d. The countermeasure is not cheaper than the embedder

§6 orders the pipeline countermeasure-first and justifies it as "40× cheaper".
Timed on the shipped exports — one 4 s window, ONNX Runtime CPU, median of 60
runs after warmup — the opposite holds at every thread count:

| intra-op threads | ReDimNet-B2 | AASIST-L |
|---|---|---|
| 1 | 283 ms | 565 ms |
| 2 | 344 ms | 535 ms |
| 4 | 432 ms | 498 ms |

AASIST-L has 85K parameters against ReDimNet's 5.1M, so the arithmetic really
is ~60× lighter. The export is not. It is 2,670 graph nodes — 973 `Constant`,
228 `Shape`, 216 `Unsqueeze`, 64 `Range` — because the sinc filterbank is
rebuilt at inference time rather than baked into a constant. ReDimNet spends
836 nodes and 108 convolutions, which ONNX Runtime is good at.

**The ordering is unchanged.** Rejecting a replay before extracting a
voiceprint from it is a §11 property and does not depend on which model is
faster. What changes is the latency budget: roughly 0.9 s per decision at two
threads, not the near-free liveness check §6 implies. Re-exporting AASIST-L
with a constant filterbank is the obvious milestone-9 target, ahead of INT8.

`docs/MODELS.md` carries the full measurement alongside the architecture of
all three models.

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
| Offline wheelhouse | §3, milestone 1 | Removed, and not needed: `venv\` and `python38\` are copied to the target whole, so there is nothing to install there. See §1. |
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
