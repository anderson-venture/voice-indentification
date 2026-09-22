# CLAUDE.md — Offline Speaker Recognition & Anti-Spoofing System

This file is the standing brief for any AI assistant working in this repository.
Read it fully before writing code. Follow it over your own defaults.

---

## 1. What this project is

An **on-premises, fully offline** system that identifies which enrolled employee is
speaking, and rejects audio that is a recording played back from a phone or speaker.

It is **not** a speech-to-text system. It never sends audio anywhere. It runs on a
local machine during development and is intended to be ported to an embedded
device (Cortex-A class SoC or better) later.

### The three jobs — never merge them

| Stage | Job | Model |
|---|---|---|
| A. Front-end | Find speech, clean it | Silero VAD (+ optional denoiser) |
| B. Identification | *Who* is this voice | ReDimNet2-B2 speaker embedder |
| C. Liveness | Is this a live human or a replay | AASIST-L countermeasure |

**Critical design rule:** identification is *not* a classifier over employees.
The embedding model is frozen. Enrollment means storing a vector in a database.
Adding an employee = inserting a row. **Never fine-tune or retrain the embedding
model to add a person.** If you find yourself writing a softmax layer sized to the
employee count, stop — that is the wrong architecture.

---

## 2. Hard constraints

1. **Python 3.8.x.** Do not use walrus-only-in-3.9 syntax, `dict |` merge (3.9+),
   `list[str]` / `dict[str, int]` builtin generics in annotations (3.9+),
   `str.removeprefix` (3.9+), `match` statements (3.10+), or `X | None` unions (3.10+).
   Use `typing.List`, `typing.Dict`, `typing.Optional`. Add
   `from __future__ import annotations` at the top of every module.
2. **Fully offline at runtime and at test time.** No network calls, ever. No
   `huggingface_hub` downloads at import time. No telemetry. Assume the machine
   has no route to the internet.
3. **CPU-only by default.** GPU is optional and must never be required.
4. **Inference runs through ONNX Runtime**, not PyTorch. PyTorch is a
   *development-only* dependency used for export and optional fine-tuning.
5. **Biometric data.** Raw audio is never persisted except during an explicit,
   consented enrollment or evaluation session. Embeddings are encrypted at rest.
   See §11.

---

## 3. Environment

### Pinned versions (all verified installable on Python 3.8)

```
# Runtime (required, offline)
numpy==1.24.4              # last 3.8-compatible release
scipy==1.10.1              # last 3.8-compatible release
onnxruntime==1.19.2        # last 3.8-compatible release
soundfile==0.12.1
librosa==0.10.2.post1
sqlalchemy==2.0.36
cryptography==43.0.3
pydantic==1.10.18          # v1 line; v2 works but v1 is lighter for 3.8
click==8.1.7
pyyaml==6.0.2

# Optional local API
fastapi==0.110.3
uvicorn==0.33.0

# Dev / export only (NOT shipped to the device)
torch==2.4.1               # last version supporting Python 3.8
torchaudio==2.4.1
onnx==1.16.2
onnxsim==0.4.36
scikit-learn==1.3.2        # last 3.8-compatible release
pandas==2.0.3              # last 3.8-compatible release
matplotlib==3.7.5
pytest==8.3.4
pytest-cov==5.0.0
black==24.8.0
ruff==0.6.9
mypy==1.14.1
```

> If any pin fails to resolve, fix it in `requirements.lock.txt` and note the
> substitution in `docs/DEVIATIONS.md`. Do not silently upgrade to a version
> that drops 3.8 — pip will happily install a 3.9-only wheel's metadata failure
> at the worst moment.

### Setup

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links=vendor/wheels -r requirements.lock.txt
```

### Building the offline wheelhouse (run ONCE on a connected machine)

```bash
# On the connected machine, same OS/arch/python as the target:
pip download -r requirements.lock.txt -d vendor/wheels --python-version 38 \
    --only-binary=:all: --platform manylinux2014_x86_64
```

Then transfer `vendor/wheels/` and `models/` to the offline machine by USB.
Everything after that must work with `--no-index`.

---

## 4. Repository layout

```
.
├── CLAUDE.md                   # this file
├── requirements.lock.txt
├── config/
│   ├── default.yaml            # all tunables; NO hardcoded constants in code
│   └── thresholds.yaml         # calibrated per deployment site — see §8
├── models/                     # ONNX files, committed via git-lfs or shipped by USB
│   ├── vad/silero_vad.onnx
│   ├── embed/redimnet2_b2.onnx
│   ├── embed/campplus.onnx     # fallback / A-B comparison
│   └── cm/aasist_l.onnx
├── src/speakerid/
│   ├── __init__.py
│   ├── audio/
│   │   ├── io.py               # load/resample to 16 kHz mono float32
│   │   ├── vad.py              # Silero wrapper -> List[Segment]
│   │   └── enhance.py          # optional denoise; OFF by default
│   ├── models/
│   │   ├── base.py             # OnnxModel: session mgmt, warmup, thread pinning
│   │   ├── embedder.py         # -> np.ndarray (192,), L2-normalized
│   │   └── countermeasure.py   # -> float in [0,1], 1.0 = bona fide
│   ├── db/
│   │   ├── schema.py           # SQLAlchemy models
│   │   ├── store.py            # enroll / delete / list / purge
│   │   └── crypto.py           # AES-GCM at rest
│   ├── scoring/
│   │   ├── similarity.py       # cosine
│   │   ├── asnorm.py           # adaptive score normalization vs cohort
│   │   └── decide.py           # the ONLY place a decision is made
│   ├── pipeline.py             # orchestrates A -> C -> B -> decision
│   ├── cli.py                  # click entrypoints
│   └── api.py                  # optional localhost FastAPI
├── scripts/
│   ├── export_onnx.py          # torch -> onnx (dev machine only)
│   ├── calibrate.py            # derive thresholds from YOUR data
│   ├── evaluate.py             # EER / DET / open-set identification rates
│   └── build_cohort.py         # cohort embeddings for AS-Norm
├── tests/
│   ├── fixtures/               # tiny synthetic wavs, committed
│   └── ...
└── docs/
    ├── DEVIATIONS.md
    ├── THREAT_MODEL.md
    └── DPIA.md                 # data protection impact assessment
```

---

## 5. Model inventory and acquisition

All models are acquired **once**, on a connected machine, converted to ONNX, and
then treated as immutable binary assets. Record SHA-256 of every file in
`models/MANIFEST.json` and verify on load.

| Slot | Model | Params | Source | Licence |
|---|---|---|---|---|
| VAD | Silero VAD v5 | ~1.8 MB ONNX | `snakers4/silero-vad` (GitHub) | MIT |
| Embedder (primary) | ReDimNet2-B2 | 3.6M / 0.95 GMACs | `IDRnD/redimnet` (GitHub) | Apache-2.0 |
| Embedder (fallback) | CAM++ | 7.2M / 1.15 GMACs | `modelscope/3D-Speaker` or WeSpeaker zoo | Apache-2.0 / CC-BY-4.0 |
| Countermeasure | AASIST-L | 85,306 params | `clovaai/aasist` (GitHub) | MIT |

**Fastest acquisition path:** the `speakeronnx` PyPI package registers ready-made
ONNX exports including `redimnet-b2`, `campplus`, `wespeaker-ecapa512` and
`titanet-small`. Use it on the connected machine to pull weights, then copy the
`.onnx` files into `models/` and **drop the dependency** — do not ship a package
that downloads on first use.

**Verify before trusting any export:** an embedding export is only accepted if its
output has cosine similarity ≥ 0.9999 against the reference PyTorch checkpoint on a
fixed test waveform, and remains unit-norm. Put this in `tests/test_export_parity.py`.

### Interface contract (do not deviate)

```python
# Embedder
input:  float32 [1, N]      # raw waveform, 16 kHz mono, N = 4s..6s (64000..96000)
output: float32 [1, 192]    # L2-normalized

# Countermeasure (AASIST-L)
input:  float32 [1, 64600]  # raw waveform, 16 kHz, ~4.04 s, pad/crop as needed
output: float32 [1, 2]      # logits; apply softmax, index 1 = bona fide
```

Short clips are handled by **repeating** clean speech to fill the window, not by
zero-padding. Long clips are **centre-cropped**, or windowed with mean-pooled
embeddings (preferred when >8 s of speech is available).

---

## 6. Pipeline order — this order is deliberate

```
raw audio (any rate)
  -> resample 16 kHz mono float32, peak-normalize to -3 dBFS
  -> Silero VAD -> speech segments
  -> reject if total speech < min_speech_sec (default 2.0)
  -> concatenate speech, take window
  -> [C] AASIST-L  --> if spoof_score < cm_threshold: REJECT, stop here
  -> [B] ReDimNet2 --> embedding
  -> AS-Norm against cohort
  -> cosine vs all enrolled -> top-1 and top-2
  -> decide()
```

**Run the countermeasure before the embedder**, not after. It is 40× cheaper and
short-circuits the expensive path on rejection.

**Never feed non-speech to the embedder.** Silence, keyboard noise and HVAC hum
produce confident-looking garbage embeddings. This is the single most common
source of false accepts in systems like this.

### `decide()` returns one of four outcomes — never just a name

```python
class Outcome(Enum):
    IDENTIFIED = "identified"      # confident match
    UNKNOWN    = "unknown"         # nobody above threshold
    AMBIGUOUS  = "ambiguous"       # top-1 and top-2 too close (margin check)
    REJECTED   = "rejected"        # failed liveness or insufficient speech
```

Returning `UNKNOWN` is a success, not a failure. Bias hard toward it.

---

## 7. Enrollment rules

- **3–5 utterances minimum** per person, ≥ 5 s of speech each.
- Recorded **on the deployment hardware, in the deployment room**. Enrollment on a
  laptop in a quiet office and deployment on a factory floor will not work.
- Store **all** enrollment embeddings per person, not just the mean. Score against
  `max(cosine)` across a person's embeddings; also keep the centroid for diagnostics.
- Reject an enrollment utterance at capture time if: VAD speech < 4 s, SNR < 10 dB,
  or its cosine to the person's existing embeddings is < 0.5 (likely wrong person
  or a broken recording).
- Re-enrollment path must exist. Voices change with illness, age, and headsets.

---

## 8. Thresholds — calibrate, never hardcode

`config/thresholds.yaml` is **site-specific** and produced by `scripts/calibrate.py`
from the operator's own recordings. Ship it empty with a loud failure if unset.

Starting points only, to be overwritten:

```yaml
cm_threshold: 0.50          # bona-fide probability floor
accept_threshold: 0.55      # cosine, after AS-Norm
margin_threshold: 0.08      # top1 - top2 must exceed this
min_speech_sec: 2.0
```

### Why you cannot reuse published numbers

Published EERs (ReDimNet2-B2 at 0.57% on VoxCeleb1-O, CAM++ at 0.71%) are
**clean-speech, 1-to-1 verification, English, 2-second segments**. Your deployment
is noisy, reverberant, possibly multilingual, and **open-set 1-to-N identification**.

Open-set false accepts scale roughly with N. A model at 1% verification EER can
produce 10–20% identification errors against 500 enrolled people in noise. So:

- Set the operating point from your own DET curve at a **target FAR**, not at EER.
- For access control, target FAR ≤ 0.1% and accept the resulting FRR.
- Re-run calibration whenever N grows by ~50% or the environment changes.

`scripts/evaluate.py` must report, on the operator's own held-out data:
EER, minDCF, FAR/FRR at the chosen threshold, **top-1 identification accuracy**,
and **open-set rejection rate for impostors**. The last one is the number that
actually matters and the one people forget to measure.

---

## 9. Noise robustness (the requirement that actually decides success)

Architecture choice is maybe 20% of real-world accuracy. In priority order:

1. **VAD quality.** Tune Silero's threshold on your own audio. Most wins are here.
2. **Enrollment in-environment.** See §7.
3. **AS-Norm** against a cohort of ~200–500 non-enrolled embeddings recorded in the
   same environment. Build with `scripts/build_cohort.py`. This alone typically cuts
   error substantially and costs nothing at inference beyond a matrix multiply.
4. **Longer windows.** 2 s is marginal. 4–6 s is where accuracy stabilizes.
   Accumulate across a speaking turn where the UX allows.
5. **Denoising — off by default.** `enhance.py` exists but is disabled. Aggressive
   denoising strips speaker-identifying detail. If you enable GTCRN or DeepFilterNet,
   you must show a measured improvement on the operator's own eval set. Benchmark
   both ways; raw audio into a well-augmented model often wins.
6. **Fine-tuning with augmentation** (optional, dev machine, needs GPU): MUSAN noise
   + RIR reverberation + the operator's own room recordings, followed by
   large-margin fine-tuning. This is where the remaining gains live — robustness-
   tuned ReDimNet variants show several-fold error reduction under channel and codec
   mismatch versus the stock checkpoint.

---

## 10. Anti-spoofing — read this before trusting the CM model

AASIST-L reports 0.99% EER on ASVspoof 2019 LA with 85K parameters. **That benchmark
is synthetic speech (TTS / voice conversion), not physical replay from a speaker in
a room**, which is the actual threat here. Countermeasures that look excellent on
lab-generated synthetic audio frequently fail on physical replay.

Therefore:

- **Fine-tune AASIST-L on physical replay data.** Use the EchoFake corpus (physical
  replay across varied devices and environments) plus **your own** replay recordings:
  record enrolled employees, then play those recordings back through 8–10 different
  phones, laptops and Bluetooth speakers, at 0.3 m / 1 m / 3 m, in each deployment
  room. A few hundred such clips beat any architecture change.
- **Do not present the CM score as a security guarantee.** Treat it as one signal.

### System-level defences — implement these, they matter more

1. **Challenge-response.** Display or speak a random 4–6 digit number; the user
   repeats it. A pre-recorded clip cannot answer. This defeats essentially all
   casual replay attacks. Requires only a small digit-keyword model.
   **Build this into the access-control flow from day one.**
2. **Microphone array.** Two or more mics give direction-of-arrival and a crude
   source-distance estimate. Loudspeakers are point sources with band-limited
   response (weak < 150 Hz) and carry doubled reverberation — the attacker's room
   plus yours.
3. **Multi-factor.** Voice + badge, or voice + face. Voice alone is a weak
   credential in 2026 and weakening: a 10-second clip from a public video is enough
   for zero-shot cloning. Document this in `docs/THREAT_MODEL.md` and make sure the
   operator has signed off on the residual risk.

---

## 11. Privacy, security, legal

Voiceprints are biometric data under Illinois BIPA, Texas CUBI, Washington's MY
Health My Data, GDPR Art. 9, and a growing list of others. BIPA carries per-violation
statutory damages and has produced nine-figure settlements.

Non-negotiable in code:

- Enrollment requires a recorded consent flag in the DB with a timestamp and the
  policy version shown. No consent row, no enrollment.
- **Retention and destruction schedule enforced in code** — `store.purge_expired()`
  runs on startup and on a timer. Default retention: configurable, no default longer
  than 24 months.
- Embeddings encrypted at rest (AES-GCM, key from OS keyring or an operator-supplied
  passphrase, never committed).
- Raw audio deleted immediately after embedding extraction unless an explicit
  `--retain-audio` evaluation flag is set.
- A working `delete_subject(person_id)` that removes embeddings, audio, logs and
  cohort membership. Test it.
- Logs contain person IDs and scores, never raw audio and never names.

> This is a flag, not legal advice. The operator must consult counsel before the
> first enrollment. Retrofitting compliance here is far harder than building it in.

---

## 12. Coding conventions

- `black` line length 100, `ruff` for lint, `mypy --strict` on `src/`.
- `from __future__ import annotations` in every module (3.8 compatibility).
- No bare `except`. No `print` — use `logging`.
- All tunables come from `config/`. If you type a float literal into a decision
  path, you have made a mistake.
- Every ONNX session is created once, at startup, with
  `sess_options.intra_op_num_threads` from config, and warmed with a dummy input.
  Creating sessions per request is a 100× latency bug.
- Functions that touch audio take and return `np.ndarray` float32 at 16 kHz mono.
  Enforce with an assert at every boundary. Sample-rate bugs are silent and lethal.

### Testing

- `pytest` must pass **with no network**. Mark anything that would reach out and
  ensure it is skipped, not mocked-and-forgotten.
- Required tests: export parity (§5), VAD segment boundaries, embedder
  determinism, end-to-end pipeline on fixtures, DB encrypt/decrypt round-trip,
  `delete_subject` completeness, and a **replay-rejection test** on fixture audio.
- Golden-file test: a fixed wav must produce a fixed embedding hash. This catches
  silent preprocessing regressions, which are otherwise invisible.

---

## 13. Build order

| # | Milestone | Done when |
|---|---|---|
| 1 | Env + wheelhouse + model manifest | `pip install --no-index` works on an air-gapped box |
| 2 | Audio IO + VAD | Segments on a noisy fixture match hand-labels ±100 ms |
| 3 | Embedder + parity test | Cosine ≥ 0.9999 vs PyTorch reference |
| 4 | DB + enroll/delete CLI | Round-trip encrypted, `delete_subject` verified |
| 5 | Scoring + AS-Norm + decide | Four outcomes returned, no hardcoded thresholds |
| 6 | Countermeasure wired in | Runs before the embedder, short-circuits |
| 7 | `calibrate.py` + `evaluate.py` | DET curve and open-set rejection rate on real data |
| 8 | Challenge-response flow | Random digits, replay of a prior session fails |
| 9 | Local API / device port | INT8 quantized, latency measured on target hardware |

Do not skip to 9. Milestone 7 is where you find out whether the system works at
all, and it is the one most likely to send you back to 2 and 7 repeatedly.

---

## 14. Things that will be tempting and are wrong

- Training a classifier head over employees. (§1)
- Feeding whole audio files to the embedder without VAD. (§6)
- Copying published EER numbers into `thresholds.yaml`. (§8)
- Enabling denoising because it sounds better to a human ear. (§9)
- Trusting the CM score against replay without in-domain fine-tuning. (§10)
- Downloading a model at first run "just for convenience". (§2)
- Storing the mean embedding only and discarding the individual enrollments. (§7)
- Reporting top-1 accuracy as the headline metric while ignoring impostor
  rejection. (§8)