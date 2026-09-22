# The three models

The rest of the docs explain how this system is wired together. This one
explains what the three models actually are: the idea each one is built on,
the structure that implements it, and what its published numbers do and do not
promise.

Two kinds of statement appear below, and they are kept apart on purpose:

* **From the paper** — what the authors claim, cited at the end.
* **Measured here** — read out of the `.onnx` files in `models/` on this
  machine. Where the two disagree, the file wins, and the disagreement is
  called out. The scripts that produced these numbers were throwaway; the
  facts are reproducible with `onnx.load()` and a loop over `initializer`.

A summary first, then one section each.

| Slot | Model | Params (measured) | Input | Output |
|---|---|---|---|---|
| A. VAD | Silero VAD v5 | 545,280 | `[B, 576]` + state + `sr` | speech probability per 512-sample hop |
| B. Embedder | ReDimNet-B2 | 5,080,359 | `[B, 1, T]` raw audio | `[B, 192]` speaker vector |
| C. Countermeasure | AASIST-L | 84,892 | `[B, 64600]` raw audio | `[B, 2]` logits |

---

## A. Silero VAD v5 — finding the speech

### What it is for

Deciding, 31 times a second, whether the last 32 ms of audio was speech. It
does not know who is speaking or what was said. Everything downstream depends
on it: CLAUDE.md §9 puts VAD quality at the top of the robustness list, above
the choice of embedder, and §6 forbids letting non-speech reach the embedder
at all, because silence and HVAC hum produce confident-looking garbage
vectors.

### The idea

Silero VAD is not a published-paper model — it is an engineering artefact
released by its authors under MIT, with claims about training data that cannot
be checked. What it is, structurally, is a **small stateful frame
classifier**: a fixed spectral front end, a few convolutions, and a recurrent
layer that carries context across chunks so a 32 ms decision is informed by
far more than 32 ms of audio.

That last part is the design's point. Purely feed-forward VADs have to choose
between a long window (latency) and a short one (no context). A carried LSTM
state gives long context at short-window cost.

### Structure, measured

545,280 float32 weights across 28 tensors, 689 graph nodes. Its weights live
in `Constant` nodes rather than as graph initializers, so a naive parameter
count over `graph.initializer` reports **zero** — a real trap if you are
trying to audit the file.

The pipeline, read off the graph:

```
576 samples (512 new + 64 carried context)
  -> STFT as a fixed Conv1d basis   stft.forward_basis_buffer  (258, 1, 256)
  -> 129 magnitude bins
  -> encoder.0  Conv1d (128, 129, 3) + ReLU
  -> encoder.1  Conv1d ( 64, 128, 3) + ReLU
  -> encoder.2  Conv1d              + ReLU
  -> encoder.3  Conv1d (128,  64, 3) + ReLU
  -> decoder.rnn  LSTM, hidden 128   weight_ih/hh (512, 128) = 4 gates x 128
  -> Conv1d -> Sigmoid -> P(speech)
```

**The file contains two complete models.** The top-level graph is an `If` node
switching on the `sr` input: a 16 kHz branch with a 256-point STFT and 129
bins, and an 8 kHz branch with a 128-point STFT and 65 bins
(`stft.forward_basis_buffer` is `(258, 1, 256)` in one and `(130, 1, 128)` in
the other). Roughly half the 2.27 MB file is a code path this deployment never
executes, since `audio/io.py` guarantees 16 kHz.

### What bit us

The ONNX export does **not** carry the 64-sample inter-window context
internally the way the JIT model does — the caller must prepend it. Feeding a
bare 512-sample window returns ~0.0 for every frame, including unambiguous
speech, and raises nothing. `docs/DEVIATIONS.md` §6b has the measurement; it is
the single most dangerous defect found in this codebase so far.

---

## B. ReDimNet-B2 — who is speaking

### What it is for

Turning a few seconds of speech into a 192-dimensional vector such that two
recordings of the same person land close together and two different people
land far apart. It is **frozen**. Enrolling someone stores a vector; it never
trains anything (CLAUDE.md §1).

### The idea

Speaker-recognition architectures had split into two camps. 2D models treat
the spectrogram as an image and keep frequency and time as separate axes,
which preserves fine spectral detail but is expensive. 1D models (x-vector,
ECAPA-TDNN) collapse frequency into channels immediately and process time
only, which is cheap but throws away frequency structure early.

ReDimNet's move is to **stop choosing**. The paper's observation is that a 2D
feature map of shape `(C, F, T)` and a 1D feature map of shape `(C·F, T)`
contain the same numbers in a different arrangement, so you can reshape freely
between them *provided the volume `C·F` is held constant*. Every stage is
sized so that as channels grow, frequency bins shrink in proportion.

The payoff is skip connections that would otherwise be impossible: a 2D map
from an early stage can be added to a 1D map from a late one, because after
reshaping they are the same shape. The paper describes aggregating "1D and 2D
feature maps together with 2D feature maps from other model stages."

### Structure, measured

5,080,359 parameters, 836 nodes, 108 convolutions, exported by PyTorch
2.10.0+cu128.

**The mel front end is inside the graph.** The model takes raw audio and
computes its own features — `spec.torchfbank.2.mel_scale.fb` is a `(257, 72)`
filterbank, i.e. a 512-point FFT reduced to **72 mel bins**, matching the
paper's "72-dimensional mean-normalized Mel filter bank log-energies" over
25 ms frames with a 15 ms step, 20–7600 Hz. There is a `(1, 1, 2)`
pre-emphasis filter ahead of it. Nothing in `src/` computes a spectrogram,
and nothing should.

Six backbone stages, with capacity concentrated at the end:

| Stage | Parameters |
|---|---|
| stage0 | 274,160 |
| stage1 | 277,696 |
| stage2 | 279,936 |
| stage3 | 496,080 |
| stage4 | 496,080 |
| stage5 | 1,249,056 |
| stem, pooling, head | 2,007,351 |

The blocks are ConvNeXt-shaped: grouped depthwise convolutions
(`conv_block.dwconvs.0` at `(16, 4, 3, 3)`) followed by pointwise
convolutions, and the `red_dim_conv` / `exp_dim_conv` pairs in each stage are
the reduce/expand around the reshape.

The head is Attentive Statistics Pooling with global context, exactly as the
paper says:

```
1152 channels
  -> pool.linear1 (128, 3456, 1)     3456 = 1152 x 3  (frame, mean, std context)
  -> pool.linear2 (1152, 128, 1)     attention weights per channel
  -> weighted mean and std           -> 2304 = 1152 x 2
  -> linear (192, 2304)              -> 192-d embedding
  -> batch norm
```

### Published numbers, and what they describe

| Variant | Params | GMACs | Vox1-O | Vox1-E | Vox1-H |
|---|---|---|---|---|---|
| B0 | 1.0M | 0.43 | 1.16 | 1.25 | 2.20 |
| B1 | 2.2M | 0.54 | 0.85 | 0.97 | 1.73 |
| **B2** | **4.7M** | **0.90** | **0.57** | **0.76** | **1.32** |
| B3 | 3.0M | 3.00 | 0.50 | 0.73 | 1.33 |
| B4 | 6.3M | 4.80 | 0.51 | 0.68 | 1.26 |
| B5 | 9.2M | 9.87 | 0.43 | 0.61 | 1.08 |
| B6 | 15.0M | 20.27 | 0.40 | 0.55 | 1.05 |

Trained on VoxCeleb2 with MUSAN noise and RIR reverberation, speed
perturbation at 0.9/1.1x, additive angular margin ramped 0.0 → 0.2 over 40
epochs on 2 s crops, then fine-tuned on 6 s utterances at margin 0.5.

That augmentation recipe is why CLAUDE.md §9.5 says raw audio into a
well-augmented model often beats aggressive denoising: the noise robustness is
already trained in, and a denoiser strips speaker detail the model was taught
to see through.

### Naming, and a discrepancy

CLAUDE.md §5 calls this slot "ReDimNet2-B2, 3.6M / 0.95 GMACs". Three names
are in play and only one is real:

* There is no "ReDimNet**2**". The paper is *Reshape Dimensions Network*; the
  variants are B0–B6. `docs/DEVIATIONS.md` §4a already records that the file
  is ReDimNet-**B2** trained on VoxCeleb2. The filename
  `redimnet2_b2.onnx` preserves the brief's mistake and is not worth a
  rename — `MANIFEST.json` pins the content by hash regardless.
* The brief's 3.6M does not match anything. The paper's B2 is 4.7M; this file
  measures **5,080,359**. The gap is plausibly the mel front end and pooling
  head being inside the exported graph, which a backbone-only parameter count
  would exclude. It has not been run down, and nothing depends on it.

Also worth repeating from `models/README.md`: the graph's raw output is
**not** unit-norm — a real window comes back with a norm around 18.
`embed_window()` normalizes before returning.

---

## C. AASIST-L — live human or a loudspeaker

### What it is for

Scoring whether the audio came from a person in the room or from a playback
device. It runs *before* the embedder (CLAUDE.md §6) and its verdict can end
the request.

### The idea

Spoofing artefacts are not evenly spread. Some live in particular frequency
bands — a phone speaker rolling off below 150 Hz — and some live in
particular moments, like the transition where a splice was made. Systems
before AASIST handled this by ensembling: one subsystem tuned to spectral
artefacts, another to temporal, scores fused at the end. Expensive, and
awkward to tune.

AASIST's move is to model both in **one graph**, then let the two compete.
Concretely:

1. A **sinc-convolution encoder** in the style of RawNet2 reads the raw
   waveform — no spectrogram — followed by residual blocks. Working from raw
   audio matters here: a mel filterbank is a lossy, perceptually motivated
   projection, and replay artefacts are not guaranteed to survive it.
2. The resulting map is collapsed **two ways** to build two sets of graph
   nodes: `max` over time gives spectral nodes, `max` over frequency gives
   temporal nodes.
3. Graph attention runs within each domain, then a **heterogeneous stacking
   graph attention layer (HS-GAL)** runs *across* them, with separate
   projection vectors for spectral–spectral, spectral–temporal and
   temporal–temporal edges — because the similarity that matters differs by
   edge type.
4. A **stack node** collects information from every other node, giving the
   network somewhere to accumulate evidence that is neither purely spectral
   nor purely temporal.
5. The **max graph operation** runs two parallel branches of two HS-GALs each
   and takes an element-wise maximum. This is the ensemble idea folded inside
   the model: branches specialise, and the more confident one wins per element.
6. A **readout** concatenates the max and mean of both node sets plus the
   stack node, and a linear layer produces two logits.

### Structure, measured

84,892 parameters — the paper's "85K", confirmed to four digits. 2,670 nodes,
15 convolutions.

Every piece of the description above is visible in the tensor names:

| Measured | What it is |
|---|---|
| `inner.encoder.0` … `inner.encoder.5` | the six residual blocks |
| `inner.pos_S` `(1, 23, 24)` | 23 spectral nodes, 24 dims each |
| `inner.GAT_layer_S`, `GAT_layer_T` | the within-domain attention, dim 24 |
| `inner.HtrgGAT_layer_ST11/12/21/22` | two branches x two HS-GALs = the max graph operation |
| `att_weight11`, `att_weight22`, `att_weight12`, `att_weightM` | the per-edge-type projections, dim 32 |
| `inner.master1`, `inner.master2` `(1, 1, 24)` | one stack node per branch |
| `inner.out_layer.weight` `(2, 160)` | readout: 160 = 5 x 32, then 2 logits |

The encoder is narrower than full AASIST — 32 and 24 filters where the large
model uses 32 and 64. That is the whole of the L.

### Published numbers, and the warning attached

On ASVspoof 2019 LA:

| System | Params | EER | min t-DCF |
|---|---|---|---|
| AASIST | 297K | 0.83% | 0.0275 |
| AASIST-L | 85K | 0.99% | 0.0309 |

**ASVspoof 2019 LA is synthetic speech** — text-to-speech and voice
conversion. It is not a phone held up to a microphone. CLAUDE.md §10 flags the
gap in advance; `docs/FIRST_RUN.md` then measured it, and the result was worse
than the warning: on a real corpus the calibrated `cm_threshold` came out at
**0.003**, because the model assigns near-zero bona-fide probability to
genuine and replayed audio alike once the recording conditions leave its
training distribution. A threshold that low is not a detector, it is a
formality.

Nothing about the architecture is at fault. It is a domain mismatch, and the
fix is the one §10 already specifies: fine-tune on physical replay recordings
made in the deployment rooms.

---

## Measured cost, and one claim that does not survive it

CLAUDE.md §6 justifies running the countermeasure first on the grounds that it
is "40× cheaper" than the embedder and short-circuits the expensive path.
**That is not true of these exported graphs.** Measured on this machine, one
4-second window, ONNX Runtime CPU, median of 60 runs after warmup:

| intra-op threads | ReDimNet-B2 | AASIST-L | ratio |
|---|---|---|---|
| 1 | 283 ms | 565 ms | CM is 2.00x **slower** |
| 2 | 344 ms | 535 ms | CM is 1.55x **slower** |
| 4 | 432 ms | 498 ms | CM is 1.15x **slower** |

Silero VAD costs 53 ms to cover the same 4 seconds, across 125 hops.

The cause is visible in the graphs. AASIST-L has 85K parameters but **2,670
nodes**, including 973 `Constant`, 228 `Shape`, 216 `Unsqueeze` and 64 `Range`
ops — the export rebuilds the sinc filterbank at inference time instead of
baking it into a constant. ReDimNet-B2 has 60x the parameters in 836 nodes and
spends its time in 108 convolutions, which ONNX Runtime executes well. The
countermeasure is cheap in arithmetic and expensive in graph overhead.

What follows from this:

* **The ordering is still right**, for a different reason than the brief
  gives. Running liveness first means a replay is rejected before a voiceprint
  is ever extracted from it, which is a privacy and correctness property
  (§11), not a speed one. Keep it.
* **The "40× cheaper" sentence in §6 is wrong** and should not be repeated
  into a latency budget. On this hardware the pipeline costs roughly
  VAD + CM + embedder ≈ 0.9 s per 4-second decision at 2 threads.
* **Milestone 9 has an obvious first target.** Re-exporting AASIST-L with the
  filterbank folded to a constant, before reaching for INT8 quantization,
  looks like the larger win. Nobody has tried it.

---

## What none of these benchmarks tell you

Every headline number above — 0.57% EER, 0.99% EER — was produced under
conditions this deployment does not meet. They are clean-speech, 1-to-1
verification, on curated English corpora, against attacks generated in a lab.
This system is open-set 1-to-N identification, in a room, against a phone.

CLAUDE.md §8 has the arithmetic: open-set false accepts scale with the number
of enrolled people, and a model at 1% verification EER can produce 10–20%
identification errors against 500 enrolled people in noise. That is why
`config/thresholds.yaml` ships empty and the loader refuses to invent values.

The models are good. The published numbers are real. Neither fact tells you
what this system will do at your door, and only `scripts/calibrate.py` and
`scripts/evaluate.py` on your own recordings will.

---

## Sources

* Yakovlev et al., *Reshape Dimensions Network for Speaker Recognition*,
  Interspeech 2024 — [arXiv:2407.18223](https://arxiv.org/abs/2407.18223),
  [ISCA archive](https://www.isca-archive.org/interspeech_2024/yakovlev24_interspeech.html)
* [IDRnD/redimnet](https://github.com/IDRnD/redimnet) — implementation,
  pretrained variants, [EVALUATION.md](https://github.com/IDRnD/redimnet/blob/master/EVALUATION.md)
* Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
  Graph Attention Networks*, ICASSP 2022 —
  [arXiv:2110.01200](https://arxiv.org/abs/2110.01200),
  [IEEE Xplore](https://ieeexplore.ieee.org/iel7/9745891/9746004/09747766.pdf)
* [clovaai/aasist](https://github.com/clovaai/aasist) — implementation and
  AASIST-L configuration
* [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — model, MIT
  licence, 8 kHz / 16 kHz support
* Measurements in this document: the files in `models/`, read with `onnx` and
  timed with `onnxruntime` 1.19.2 on CPython 3.8.10, Windows, CPU only.
