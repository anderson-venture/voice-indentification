# Offline Speaker Recognition & Anti-Spoofing

On-premises, fully offline. Identifies which enrolled employee is speaking and
rejects audio replayed from a phone or speaker. It never sends audio anywhere
and makes no network calls.

Not a speech-to-text system. The standing brief is [Claude.md](Claude.md);
read it before changing anything here.

## Three jobs, never merged

| Stage | Job | Model |
|---|---|---|
| A. Front-end | Find speech, clean it | Silero VAD v5 |
| C. Liveness | Live human or a replay | AASIST-L |
| B. Identity | *Who* is this voice | ReDimNet2-B2 |

Identification is **not** a classifier over employees. The embedding model is
frozen; enrolling someone inserts a database row. Adding a person never
touches a model.

## Pipeline

```
raw audio (any rate)
  -> 16 kHz mono float32, peak-normalized to -3 dBFS
  -> Silero VAD -> speech segments
  -> reject if total speech < min_speech_sec
  -> concatenate speech, take window
  -> [C] AASIST-L   -- below threshold: REJECT, skip the embedder entirely
  -> [B] ReDimNet2  -- embedding
  -> AS-Norm against the in-room cohort
  -> cosine vs all enrolled -> top-1, top-2
  -> decide()
```

The countermeasure runs **before** the embedder: it is ~40× cheaper and
short-circuits the expensive path. Non-speech never reaches either model.

`decide()` returns one of four outcomes, never a bare name:

| Outcome | Meaning |
|---|---|
| `IDENTIFIED` | Confident match |
| `UNKNOWN` | Nobody above threshold — **a success, not a failure** |
| `AMBIGUOUS` | Top-1 and top-2 too close |
| `REJECTED` | Failed liveness or challenge, or too little speech |

## Quick start

```powershell
py -3.8 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src"
.venv\Scripts\python -m pytest
```

Full setup, including copying the installed tree to an offline machine:
[docs/SETUP.md](docs/SETUP.md).

The suite passes with **no network and no model weights**: tests that need
real ONNX files skip rather than pretend (`pytest -rs` lists them).

### Deploying

```bash
# 1. Models, on a connected machine, then transferred by USB
python -m scripts.fetch_models      # checksums into models/MANIFEST.json
speakerid verify-models

# 2. Cohort for AS-Norm: 200-500 clips of NON-enrolled people, in the room
speakerid build-cohort ./cohort_audio --room-id lobby

# 3. Calibrate. Nothing decides anything until this has been run.
speakerid calibrate \
    --trials trials.csv \
    --cm-trials replay_trials.csv \
    --identification-trials ident.csv \
    --target-far 0.001 --site "north gate"

# 4. Enrol: 3-5 utterances, >= 5 s each, on the deployment hardware, in the room
speakerid add-person --person-id alice --name "Alice Example"
speakerid consent --person-id alice --granted-by "J. Operator"
speakerid enroll --person-id alice --room-id lobby a1.wav a2.wav a3.wav

# 5. Measure. This is the milestone that tells you whether it works.
speakerid evaluate --trials heldout.csv --report reports/eval.json

# 6. Use
speakerid identify sample.wav
speakerid delete --person-id alice      # complete, verified erasure
```

`speakerid status` reports models, thresholds and calibration state at any
time.

## Two things that will bite you

**Thresholds ship empty, and the system refuses to run without them.** That is
deliberate. Published EERs (ReDimNet2-B2 0.57% on VoxCeleb1-O) are
clean-speech 1-to-1 verification; this is open-set 1-to-N in noise, where
false accepts scale with the number of enrolled people. Calibrate at a target
FAR on your own recordings — not at EER, and never by copying a benchmark.

**The countermeasure is not a security guarantee.** AASIST-L's benchmark is
synthetic speech, not a phone held up to the microphone. Until it is
fine-tuned on your own replay recordings and the challenge-response digit
verifier is implemented, replay resistance is low. See
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) §4 — the operator must sign it.

## Privacy

Voiceprints are biometric data under BIPA, CUBI, My Health My Data and GDPR
Art. 9. Enforced in code: no consent row, no enrolment; every record carries
an expiry and is purged automatically; embeddings are AES-256-GCM encrypted
and bound to their subject; raw audio is discarded after extraction unless
explicitly retained; `delete_subject()` is complete and tested; logs carry
person IDs and scores, never names or audio.

The operator must consult counsel and complete
[docs/DPIA.md](docs/DPIA.md) **before the first enrolment**.

## Layout

```
config/       default.yaml (tunables) + thresholds.yaml (site, ships EMPTY)
models/       ONNX assets + MANIFEST.json -- by USB, never in git
src/speakerid/
  audio/      io.py (the 16 kHz contract), vad.py, enhance.py (off by default)
  models/     base.py (sessions, warmup, checksums), embedder.py, countermeasure.py
  db/         schema.py, store.py (consent, retention, erasure), crypto.py
  scoring/    similarity.py, asnorm.py, decide.py (the ONLY decision point)
  pipeline.py challenge.py metrics.py cli.py api.py
scripts/      fetch_models, export_onnx, build_cohort, calibrate, evaluate, make_fixtures
docs/         DEVIATIONS.md, THREAT_MODEL.md, DPIA.md
```

## Status

Milestones 2–8 are implemented and tested on **CPython 3.8.20 with real ONNX
weights**: 255 tests pass, and `ruff`, `black` and `mypy --strict` are clean
under the pinned toolchain. Not yet run on Windows.

The system has been run end to end on real voices — see
[docs/FIRST_RUN.md](docs/FIRST_RUN.md) for the measured numbers and, more
importantly, for what they do not show. The short version: identification
works well, **AASIST-L does not work as shipped** and must be fine-tuned on
physical replay before it protects anything.

Outstanding: milestone 7 must be redone on the operator's own recordings, and
milestone 9 has deliberately not been started.
[docs/DEVIATIONS.md](docs/DEVIATIONS.md) lists every departure from the brief.
