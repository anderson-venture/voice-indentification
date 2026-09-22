# First real run — bring-up record

First end-to-end execution with **real weights and real voices**, on
CPython 3.8.20. This is a bring-up record, not a site calibration: the numbers
below describe a public-sample corpus on a workstation, and say nothing about
any deployment. Milestone 7 still has to be redone on the operator's own
recordings, in the deployment room, on the deployment hardware.

## What was run

| Stage | Model | Provenance |
|---|---|---|
| A. VAD | Silero VAD v5 | `snakers4/silero-vad`, ONNX as published |
| B. Embedder | ReDimNet-B2 (VoxCeleb2), 192-d | HF `OpenVoiceOS/redimnet-b2-vox2-onnx` |
| C. Countermeasure | AASIST-L, 85,306 params | exported here from `clovaai/aasist`, parity 9.5e-07 |

All three verified against `models/MANIFEST.json` on load.

## Corpus

No real voice is committed to this repository. The corpus was assembled in a
scratch directory from public samples and deleted from the repo tree:

* **alice** — LibriSpeech speaker 1272 (73 utterances merged into 28 clips)
* **bob** — a LibriVox audiobook narrator (12 slices from across the recording)
* **strangers** — 5 unrelated clips, never enrolled
* **replays** — 4 simulated playback chains per clip (room IR → 150 Hz–6 kHz
  loudspeaker passband → clipping → second room IR)

## Results

### Verification — 980 trials (444 genuine, 536 impostor)

| Metric | Value |
|---|---|
| EER | **0.0000** |
| minDCF | 0.0000 |
| FAR at deployed threshold | 0.0093 (target 0.01) |
| FRR at deployed threshold | 0.0000 |

Raw cosine separation: same-speaker 0.82–0.90, different-speaker max 0.296.

### Open-set identification — 17 probes, 2 enrolled people

| Metric | Value |
|---|---|
| top-1 accuracy (enrolled) | 1.0000 |
| **impostor rejection rate** | **1.0000** |
| false accept rate | 0.0000 |
| misidentification | 0.0000 |

**Read this with suspicion.** Two enrolled people is not an open set in any
meaningful sense, and §8 is explicit that false accepts scale with N. A
perfect score here is evidence the plumbing works, not evidence the system is
accurate.

### Countermeasure — 224 trials (45 bona fide, 179 simulated replay)

| Metric | Value |
|---|---|
| EER | 0.0669 |
| spoof accept rate | 0.0503 |
| bona fide reject rate | 0.0667 |

### Calibrated thresholds

```yaml
cm_threshold: 0.003074
accept_threshold: 0.286379
margin_threshold: 0.55
min_speech_sec: 2.0
```

## The four outcomes, observed

| Input | Outcome | Evidence |
|---|---|---|
| alice, held-out | `IDENTIFIED` alice | score 0.840, margin 0.671 |
| bob, held-out | `IDENTIFIED` bob | score 0.901, margin 0.702 |
| stranger (`2.flac`) | `UNKNOWN` | 0.244 < 0.286 accept threshold |
| stranger (`mlk`) | `REJECTED` | liveness 0.001 |
| replay of alice | `REJECTED` | liveness 0.000 |
| replay of bob | `AMBIGUOUS` | passed liveness at 0.004, caught by margin |
| 0.7 s clip | `REJECTED` | 0.76 s < 2.0 s |

## What this run actually proved, and what it did not

**Proved:** the pipeline runs end to end on 3.8 with real ONNX weights; VAD
segments real speech; the embedder separates real speakers with a wide margin;
consent is enforced (enrolment was refused until a consent row existed);
calibration produces a threshold file with provenance; all four outcomes arise
from the inputs that should produce them; erasure is complete.

**Did not prove — and these matter more:**

1. **AASIST-L is not usable as shipped.** The calibrated `cm_threshold` is
   **0.003** — three thousandths. AASIST-L assigns near-zero bona-fide
   probability to most of this corpus, genuine and replayed alike, because
   compressed audiobook and LibriSpeech audio is far outside ASVspoof 2019 LA.
   Two genuine speakers scored 0.034 and 0.228; one replay scored 0.004 and
   **passed** the liveness stage, caught only by the margin check. A threshold
   that low is not a working detector, it is a formality. This is precisely
   what §10 warns about, now measured rather than predicted. It must be
   fine-tuned on real physical replay before it protects anything.
2. **The replays were simulated, not physical.** A synthesised room-IR and
   band-limiting chain is not a phone held up to a microphone. §10 asks for
   8–10 real devices at 0.3 m / 1 m / 3 m in each deployment room.
3. **Two speakers is not an open set.** Re-measure against the real enrolled
   population.
4. **AS-Norm was off.** A cohort needs 200–500 non-enrolled speakers recorded
   in the room; this corpus has two speakers total.
5. **The FRR of 0.0000 is not credible** for a deployment. Clean read speech
   in quiet conditions is the easiest possible case.

## Reproducing

The corpus builder and trial-set scripts were kept out of the repository
deliberately: they download from the public internet, and nothing in the
shipped tree may do that. On a connected machine:

```bash
python -m scripts.fetch_models                 # checksums models/MANIFEST.json
python -m speakerid.cli verify-models
python -m speakerid.cli calibrate \
    --trials trials.csv --cm-trials cm_trials.csv \
    --identification-trials ident.csv \
    --target-far 0.001 --cm-target-far 0.05 --site "<site>"
python -m scripts.evaluate --trials heldout.csv \
    --identification-trials ident.csv --report reports/eval.json
```

Trial file formats are documented in `scripts/calibrate.py` and
`scripts/evaluate.py`; both identification files use `path,person_id,role`.
