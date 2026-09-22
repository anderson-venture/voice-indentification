# End-to-end verification record

The system has been run end to end on real voices, twice. This is the record
of what happened and, more usefully, what it did not prove.

Neither run is a site calibration. Both used public corpora on a workstation
and say nothing about any deployment. Milestone 7 still has to be redone on
the operator's own recordings, in the deployment room, on the deployment
hardware.

---

## The run that counts: 2026-09-22, Windows, full CLI

Every step below went through the shipped `speakerid` CLI as a separate
process — no importing of internals — against the tree that is about to be
copied to the offline machine. CPython 3.8.10, ONNX Runtime CPU, real weights.

**Corpus.** LibriSpeech dev-clean, 40 speakers, split by role: 6 enrolled
(4 enrolment clips + 4 held-out probes each), 8 impostors never enrolled
(6 clips each), 20 cohort speakers for AS-Norm (3 clips each). Clips are 10 s
of concatenated utterances. Replays were simulated — attacker's room impulse
response, 150 Hz–6 kHz loudspeaker passband, soft clipping, capture room
impulse response, sensor noise.

**22 of 24 checks passed. Both failures are the countermeasure.**

### What passed

| Check | Result |
|---|---|
| Three models match `MANIFEST.json` | ok |
| `identify` before calibration | refused, exit 2 |
| Enrolment without a consent row | refused: *"person 1462 has no active consent record"* |
| AS-Norm cohort from 20 non-enrolled speakers | 60 embeddings built |
| Calibration writes thresholds with provenance | ok |
| Six people enrolled | 23 of 24 utterances accepted |
| **No genuine probe given the wrong name** | **0 misidentified of 24** |
| **Open-set impostor rejection** | **48/48 rejected** |
| **No replay ever accepted** | **0 of 24 admitted** |
| 0.7 s clip | `REJECTED`, insufficient_speech |
| Room tone | `REJECTED` at the VAD, never embedded |
| Challenge-response with no verifier | `REJECTED`, fails closed |
| Embeddings at rest | `SPKR1` ciphertext, no `RIFF` payload anywhere in the DB |
| Retained audio rows | 0 |
| Recognition events | 99 logged, by person id only |
| `delete_subject` | person gone, 0 embeddings left, no longer identified |

One enrolment utterance out of 24 was refused at capture time by the §7
quality gate, and speaker 1993 enrolled with 3 clips rather than 4. The gate
working is the point; three is still above the minimum.

### Calibrated operating point

```yaml
accept_threshold: 2.5550      # AS-Norm z-score scale, not cosine
margin_threshold: 2.0         # swept against identification trials
cm_threshold:     0.0196
min_speech_sec:   2.0
```

| Verification (1,248 trials: 96 genuine, 1,152 impostor) | |
|---|---|
| EER | 0.0000 |
| minDCF | 0.0000 |
| FAR at deployed threshold | 0.00087 (target 0.001) |
| FRR at deployed threshold | 0.0000 |

| Open set (72 probes: 24 enrolled, 48 impostor) | |
|---|---|
| top-1 accuracy | 1.0000 |
| **impostor rejection rate** | **1.0000** |
| false accept rate | 0.0000 |
| misidentification rate | 0.0000 |

AS-Norm was **on**, against a 60-clip cohort from 20 speakers who are not
enrolled — the first run had it off.

### The two failures, which are one problem

```
[FAIL] genuine users are accepted end to end
       17/24 accepted; 7 rejected by liveness
[FAIL] the countermeasure alone stops every replay
       23/24 stopped at liveness; the rest fell through to later checks
```

**AASIST-L rejects 29% of legitimate users.** Calibration predicted it in its
own report — `cm_frr_at_threshold: 0.2917` — and the live run delivered
exactly that: 7 of 24 genuine probes refused. Every one came back `rejected`,
not misnamed. The embedder never made a mistake; the liveness stage locked out
nearly a third of the people it is meant to admit.

**And it still let one replay through.** 23 of 24 replays were stopped at
liveness. The 24th passed the countermeasure and was caught downstream by the
margin check, landing as `AMBIGUOUS`. No replay was admitted, so the system
held — but it held because of a check that exists to resolve close calls
between enrolled people, not because the countermeasure did its job.

Measured against the trial set, AASIST-L scored **12.5% EER** on simulated
replay. Its published figure is 0.99% on ASVspoof 2019 LA. That benchmark is
synthetic speech — TTS and voice conversion — not a recording played into a
room, and the gap between the two numbers is the whole of CLAUDE.md §10
restated as a measurement.

At the calibrated point the model is simultaneously too strict for real users
and too permissive for replays. That is not a threshold that needs tuning; it
is a model that has not seen this kind of audio. **It must be fine-tuned on
physical replay recordings before it protects anything**, and until it is,
the honest configuration is to treat the CM as advisory and let
challenge-response carry the weight — which needs the digit verifier that is
also not implemented.

---

## The earlier run: Linux, CPython 3.8.20

The first bring-up used a smaller corpus — LibriSpeech speaker 1272 and a
LibriVox narrator, two enrolled people, AS-Norm off — on a machine with a
network, before this tree moved to Windows. Its identification numbers were
perfect and meaningless at that scale: two enrolled people is not an open set.

Its one durable finding was the countermeasure, and the September run
reproduced it independently on a different corpus, on a different OS, with
AS-Norm enabled and six enrolled speakers. The first run calibrated
`cm_threshold` to 0.003 and saw one replay pass liveness. The second
calibrated to 0.0196 and saw one replay pass liveness. Same failure, twice,
from different directions.

---

## Redoing this on your own recordings

The corpus builder and trial-set scripts are deliberately not in this
repository: they download from the public internet, and nothing in the shipped
tree may do that. The steps below work offline, against recordings you capture
on the deployment hardware — which is the run that actually matters:

```bash
speakerid verify-models
speakerid build-cohort ./cohort_audio --room-id <room>
speakerid calibrate \
    --trials trials.csv --cm-trials cm_trials.csv \
    --identification-trials ident.csv \
    --target-far 0.001 --cm-target-far 0.05 --site "<site>"
python -m scripts.evaluate --trials heldout.csv \
    --identification-trials ident.csv --report reports/eval.json
```

Trial file formats are documented in `scripts/calibrate.py` and
`scripts/evaluate.py`; both identification files use `path,person_id,role`.

Two things the tooling will tell you that are worth listening to. Calibration
**refuses** a target FAR it cannot measure — asking for FAR ≤ 0.001 with 576
impostor trials fails with *"needs at least 1000 impostor trials"*, because
you cannot measure a 1-in-1000 rate from 576 samples. And it records
`cm_frr_at_threshold` in the provenance, which is where this run's 29%
lockout was visible before anyone tried to log in.

---

## What neither run proved

1. **The replays were simulated, not physical.** A synthesised room-IR and
   band-limiting chain is not a phone held up to a microphone. §10 asks for
   8–10 real devices at 0.3 m / 1 m / 3 m in each deployment room, and the
   real thing is likely to be harder than what was tested here.
2. **Six enrolled speakers is a small open set.** False accepts scale with N.
   48/48 impostor rejection against six enrolled people says the plumbing is
   right, not that the system is accurate at 500.
3. **LibriSpeech is clean read speech in quiet conditions** — the easiest
   input this system will ever see. An FRR of 0.0000 on the verification
   trials is not a number any deployment will reproduce.
4. **The digit verifier is still not implemented**, so the defence §10 calls
   strongest was tested only for failing closed, which it does.
5. **Nothing here was measured on the target hardware.** Latency on this
   workstation is roughly 0.9 s per decision at two threads; see
   `docs/MODELS.md` for the per-model breakdown and for why the countermeasure
   is the slow one.
