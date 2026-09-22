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

The countermeasure runs **before** the embedder so that a replay is rejected
before a voiceprint is ever extracted from it. Not for speed — measured, it is
the slower of the two ([docs/MODELS.md](docs/MODELS.md)). Non-speech never
reaches either model.

`decide()` returns one of four outcomes, never a bare name:

| Outcome | Meaning |
|---|---|
| `IDENTIFIED` | Confident match |
| `UNKNOWN` | Nobody above threshold — **a success, not a failure** |
| `AMBIGUOUS` | Top-1 and top-2 too close |
| `REJECTED` | Failed liveness or challenge, or too little speech |

## Quick start

This repository is the source half. The other half is the environment —
`venv\` and `python38\`, 1.8 GB of binaries that GitHub will not take — which
is distributed as a zip on Google Drive. [docs/SETUP.md](docs/SETUP.md) §1 has
the link, and is the only setup document you need.

Once both halves are in place at `E:\Project\voice-indentification`, there is
no install step:

```powershell
cd E:\Project\voice-indentification
venv\Scripts\Activate.ps1
pytest                    # 255 passed, 2 skipped
speakerid status
```

The folder must live at that exact path — `venv\` is not relocatable, and
SETUP §4 says why.

The suite makes **no network calls** (`tests/test_offline.py` enforces it) and
skips rather than pretends when something is genuinely absent — `pytest -rs`
names the two skips.

### Deploying

```bash
# 1. Models are already in models/ and checksummed; confirm they survived the copy
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
#    Use scripts.evaluate, not `speakerid evaluate`: only it takes
#    --identification-trials, and open-set impostor rejection is the number
#    that matters (CLAUDE.md §8). `speakerid evaluate` reports verification
#    metrics only.
python -m scripts.evaluate --trials heldout.csv \
    --identification-trials ident.csv --cm-trials replay_trials.csv \
    --report reports/eval.json

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
verifier is implemented, **replay resistance is low** — do not put this in
front of anything valuable, and treat voice as one factor rather than the only
one. [docs/FIRST_RUN.md](docs/FIRST_RUN.md) has this measured rather than
predicted.

## Layout

```
python38/     vendored CPython 3.8.10 -- the venv's interpreter and stdlib
venv/         the environment. Not in git: see docs/SETUP.md §1b
config/       default.yaml (tunables) + thresholds.yaml (site, ships EMPTY)
models/       ONNX assets + MANIFEST.json, verified on every load
src/speakerid/
  audio/      io.py (the 16 kHz contract), vad.py, enhance.py (off by default)
  models/     base.py (sessions, warmup, checksums), embedder.py, countermeasure.py
  db/         schema.py, store.py (consent, retention, erasure), crypto.py
  scoring/    similarity.py, asnorm.py, decide.py (the ONLY decision point)
  pipeline.py challenge.py metrics.py cli.py api.py
scripts/      fetch_models, export_onnx, build_cohort, calibrate, evaluate,
              make_fixtures, make_env_archive (cuts the SETUP §1b zip)
docs/         SETUP.md, MODELS.md, FIRST_RUN.md, DEVIATIONS.md, DPIA.md
```

## Status

Milestones 2–8 are implemented. Verified on **Windows, CPython 3.8.10, with
real ONNX weights**: 255 tests pass and 2 skip, and `ruff`, `black` and
`mypy --strict` are clean. The three models load, warm up and run, and
`verify-models` matches every checksum.

The system has been run end to end on real voices — see
[docs/FIRST_RUN.md](docs/FIRST_RUN.md) for the measured numbers and, more
importantly, for what they do not show. The short version: identification
works well, **AASIST-L does not work as shipped** and must be fine-tuned on
physical replay before it protects anything.

Outstanding: milestone 7 must be redone on the operator's own recordings, and
milestone 9 has deliberately not been started.
[docs/DEVIATIONS.md](docs/DEVIATIONS.md) lists every departure from the brief.
