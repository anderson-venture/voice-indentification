# Data Protection Impact Assessment

**This is a template and an engineering record, not legal advice.** The
operator must have counsel complete and approve it **before the first
enrolment**. Retrofitting compliance here is far harder than building it in.

Voiceprints are biometric data under Illinois BIPA, Texas CUBI, Washington's
My Health My Data, GDPR Art. 9 and a growing list of others. BIPA carries
per-violation statutory damages and has produced nine-figure settlements.

---

## 1. Processing

| Question | Answer |
|---|---|
| What is processed | Speech audio, transiently; derived 192-d embeddings, persistently. |
| Purpose | Identifying which enrolled employee is speaking, for access control. |
| Legal basis | **Operator to complete.** Consent is recorded in code, but consent as a lawful basis in an employment relationship is contested in several jurisdictions. Take advice. |
| Special category (GDPR Art. 9) | Yes — biometric data for unique identification. |
| Data subjects | Enrolled employees; plus anyone who speaks near the microphone. |
| Recipients | None. The system is offline and makes no network calls. |
| Transfers | None. |
| Retention | Configurable, hard-capped at 24 months, enforced in code. |
| Automated decision-making | Yes. See §6. |

---

## 2. Necessity and proportionality

To be completed by the operator, and the hardest section to pass:

* Why is voice necessary, rather than a badge or PIN?
* What is the less intrusive alternative, and why was it rejected?
* How is the **bystander** problem handled — people who speak near the
  microphone but are not enrolled? Their audio is processed even though they
  never consented. The design minimises this (audio is discarded immediately,
  no embedding is stored for an unknown speaker), but processing occurs.

---

## 3. Technical measures implemented

| Requirement | Implementation | Verified by |
|---|---|---|
| No enrolment without recorded consent | `Store.add_embedding` raises `ConsentError` without an active row | `test_store.py::test_enrolment_without_consent_is_refused` |
| Consent records policy version, timestamp, witness | `ConsentRecord` | schema |
| Retention enforced in code | `expires_at` on every row; `purge_expired()` on startup and on a timer | `test_store.py::test_purge_removes_expired_rows_only` |
| Retention capped at 24 months | Config validation rejects more | `test_config.py::test_retention_cannot_exceed_24_months` |
| Embeddings encrypted at rest | AES-256-GCM, key from keyring/passphrase/env, never committed | `test_crypto.py`, `test_store.py::test_embeddings_are_encrypted_on_disk` |
| Records bound to their subject | GCM AAD over `person_id` + `model_id` | `test_crypto.py::test_record_is_bound_to_its_subject` |
| Raw audio deleted after extraction | Not persisted unless `--retain-audio`; retained files tracked and expired | `test_pipeline.py::test_retained_audio_is_only_registered_on_request` |
| Right to erasure | `delete_subject()` removes embeddings, audio **files**, events, consent and cohort rows | `test_store.py::test_delete_subject_removes_every_trace` |
| Erasure is verifiable | `subject_trace_count()` returns 0; the CLI exits non-zero if not | `test_cli.py::test_delete_with_yes_erases` |
| Logs carry no names or audio | `PrivacyFilter`; event schema has no such column | `test_store.py::test_events_record_ids_and_scores` |
| No data leaves the host | No networking code; enforced by test | `test_offline.py` |

---

## 4. Data subject rights

| Right | How it is served |
|---|---|
| Access | `speakerid list`; per-person detail via the store API. The embedding itself is not meaningfully intelligible, and disclosing it creates a new copy of a biometric identifier — take advice before releasing it. |
| Rectification | Re-enrolment. `speakerid delete` then enrol again. |
| **Erasure** | `speakerid delete --person-id <id>`. Verified complete, exits non-zero otherwise. |
| Restriction | `speakerid revoke-consent` stops further enrolment and flags the record; data is retained until erasure is requested. |
| Objection | Operator process. |
| Portability | Not applicable — no interoperable voiceprint format. |

---

## 5. Risks

| Risk | Likelihood | Impact | Mitigation | Residual |
|---|---|---|---|---|
| Voiceprint database stolen | Low | **Severe** — irrevocable identifiers | AES-GCM at rest, key held outside the DB, offline host | Key compromise or memory disclosure while running |
| Wrong person admitted | **Medium** | High | Target-FAR calibration, margin check, AS-Norm, `UNKNOWN` bias | Unknown until milestone 7 |
| Replay attack succeeds | **Medium–High** | High | CM before embedder; challenge-response scaffolding | **High** until the digit verifier is implemented and the CM is fine-tuned on physical replay. AASIST-L's published EER is synthetic speech, not a speaker played into a room; `FIRST_RUN.md` measures it failing on this corpus. Voice must be one factor, not the only one. |
| Bystander audio processed | High | Low–Medium | Audio discarded immediately; nothing stored for unknown speakers | Transient processing still occurs |
| Retention overrun | Low | Medium | Enforced in code, not by procedure | Requires the process to actually run |
| Function creep (attendance, productivity) | **Medium** | High | Purpose stated here; events hold ids and scores only | **Operator governance.** The likeliest way this system becomes unlawful. |

---

## 6. Automated decision-making

The system outputs one of four outcomes — `IDENTIFIED`, `UNKNOWN`,
`AMBIGUOUS`, `REJECTED` — and is biased toward refusing to name anyone.

The operator must ensure:

* a **human-reviewable fallback** exists for `UNKNOWN`, `AMBIGUOUS` and
  `REJECTED`, so nobody is locked out by a false reject with no recourse;
* employees are told the system exists, what it decides, and how to contest a
  decision;
* `RecognitionEvent` rows are retained long enough to investigate a complaint,
  and no longer.

---

## 7. Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Data Protection Officer | | | |
| System owner | | | |
| Legal counsel | | | |

**Do not enrol anybody until this section is complete.**
