# Threat model

Scope: an on-premises, fully offline system that identifies which enrolled
employee is speaking and rejects audio replayed from a phone or speaker.

The operator must read §4 and sign off on the residual risk before the first
enrolment.

---

## 1. Assets

| Asset | Why it matters |
|---|---|
| Enrolment embeddings | Biometric identifiers. Irrevocable: an employee cannot be issued a new voice. |
| The encryption key | Compromise exposes every voiceprint at once. |
| Recognition events | Reveal who was where, and when. |
| Model files | A swapped embedder silently changes every decision. |
| `thresholds.yaml` | An edited threshold is an undetectable change to the false-accept rate. |

---

## 2. Adversaries

* **Opportunist.** Holds a phone up to the mic playing a recording taken in
  the corridor. No special skill or equipment.
* **Prepared attacker.** Collects target audio from a public video or a
  meeting recording, uses zero-shot voice cloning, plays it through good
  hardware. A 10-second sample is enough in 2026.
* **Insider.** Has filesystem or database access on the host.
* **Curious bystander.** Reads logs or console output they are not meant to.

---

## 3. Attacks and what actually stops them

### 3.1 Physical replay — the primary threat

Playing a recording of an enrolled employee at the microphone.

| Defence | Strength | Status |
|---|---|---|
| **Challenge-response** | **Strong.** A recording cannot say a number chosen one second ago. Defeats essentially all casual replay. | Issuing, TTL and single-use consumption implemented. **The digit verifier is not implemented and the flow fails closed.** |
| AASIST-L countermeasure | **Weak until fine-tuned.** Its 0.99% EER is ASVspoof 2019 LA — *synthetic* speech, not a speaker in a room. Countermeasures that excel on lab-generated audio routinely fail on physical replay. | Wired in, runs before the embedder. **Not fine-tuned on replay data.** |
| Microphone array (DoA, source distance) | Moderate. Loudspeakers are point sources, weak below 150 Hz, carrying doubled reverberation. | **Not implemented.** Single channel. |
| Second factor (badge, face) | **Strong.** | Out of scope for this component. |

**Assessment:** with the CM alone and no verifier attached, replay resistance
is **low**. The CM score is a signal, never a guarantee.

### 3.2 Synthetic speech / voice cloning

A 10-second public clip is enough for zero-shot cloning. AASIST-L is
genuinely designed for this and is the right tool, but the shipped threshold
is uncalibrated until the operator runs `calibrate.py --cm-trials`.
Challenge-response also helps: a cloned voice must still answer live, which
requires real-time synthesis and a working capture path.

### 3.3 Impostor admitted by similarity alone

Open-set false accepts scale roughly with the number of enrolled people. A
model at 1% verification EER can produce 10–20% identification errors against
500 enrolled people in noise.

Mitigations, all implemented:

* thresholds set from the site's own DET curve at a target FAR, never at EER;
* a **margin** check — top-1 must beat top-2 — yielding `AMBIGUOUS` rather
  than a guess;
* `UNKNOWN` as a first-class outcome, biased toward;
* AS-Norm against an in-room cohort;
* `evaluate.py` reports **impostor rejection rate**, and warns loudly when it
  was not measured.

### 3.4 Non-speech producing a confident embedding

Silence, HVAC hum and keyboard noise produce confident-looking garbage
embeddings — the most common source of false accepts in systems like this.

Mitigated: VAD gates everything; the embedder is never called on a buffer that
did not pass it; `min_speech_sec` rejects too-short input; the test suite
asserts room tone yields `REJECTED`.

### 3.5 Tampering with models or thresholds

* Every model file's SHA-256 is verified against `models/MANIFEST.json` on
  load; a mismatch refuses to start.
* The embedding model's content hash is folded into `model_id`, so a
  re-exported model cannot be scored against vectors from the previous one —
  its gallery is simply empty, which is loud rather than quietly degraded.
* `thresholds.yaml` carries provenance (`calibrated_at`, `target_far`,
  `measured_far`, `n_trials`), logged at startup. A hand-edited file missing
  provenance produces a warning.

**Gap:** the manifest is not signed. An attacker who can write to `models/`
can rewrite both the model and its checksum. Sign it, or hold it on read-only
media, where the deployment allows.

### 3.6 Database theft

Embeddings are AES-256-GCM encrypted at rest, each record bound via GCM AAD to
its `person_id` **and** `model_id`, so rows cannot be swapped between people.
The key is never stored in the database or the repository; it comes from a
passphrase, the environment, or the OS keyring.

**Gap:** a running process holds the key and the decrypted gallery in memory.
Memory disclosure on the host exposes them. Out of scope for a single-host
design.

### 3.7 Leakage through logs

Logs carry person IDs and scores only. `PrivacyFilter` strips identity- and
audio-bearing extras, `RecognitionEvent` has no column that could hold a name
or audio, and challenge digits are deliberately never logged.

---

## 4. Residual risk — operator sign-off required

1. **Voice alone is a weak credential, and weakening.** Zero-shot cloning from
   a short public sample is routine. Voice should be one factor, not the only
   one.
2. **Replay resistance is low as shipped.** The challenge verifier is not
   implemented, and the CM is not fine-tuned on physical replay. Until both are
   addressed, do not place this in front of anything valuable.
3. **Accuracy is unknown until milestone 7.** No published number describes
   this deployment.
4. **Biometric data is irrevocable.** A compromised voiceprint cannot be
   reissued.

Signed: ______________________  Date: ____________

---

## 5. Out of scope

Physical security of the host; compromise of the OS or root; coercion of an
enrolled employee; denial of service by jamming the microphone.
