# Models

The `.onnx` files **are** committed (25 MB total). CLAUDE.md §5 treats them as
immutable binary assets acquired once on a connected machine; §4 allows them
in git. Committing keeps `git clone` a single step, and every file's SHA-256
is verified against `MANIFEST.json` on load, so a corrupt copy is caught.

Expected layout:

```
models/
├── MANIFEST.json          # SHA-256 of every file; generated, verified on load
├── vad/silero_vad.onnx    # Silero VAD v5        MIT
├── embed/redimnet2_b2.onnx# ReDimNet2-B2         Apache-2.0   (primary)
├── embed/campplus.onnx    # CAM++                Apache-2.0   (fallback, optional)
└── cm/aasist_l.onnx       # AASIST-L             MIT
```

All three files are present and verified. `speakerid verify-models` is the
only model command you need on the offline machine.

## Re-acquiring or re-exporting — connected machine only

1. Fetch the exports. The quickest path is the `speakeronnx` PyPI package,
   which registers ready-made ONNX exports including `redimnet-b2`,
   `campplus`, `wespeaker-ecapa512` and `titanet-small`. Pull the weights,
   copy the `.onnx` files here, then **drop the dependency** — nothing that
   downloads at first run may ship.
2. Silero VAD v5: `snakers4/silero-vad`. AASIST-L: `clovaai/aasist`, exported
   with `scripts/export_onnx.py`.
3. Verify parity before trusting any embedder export: cosine ≥ 0.9999 against
   the reference PyTorch checkpoint on a fixed waveform, still unit-norm.

   ```bash
   SPEAKERID_REFERENCE_CHECKPOINT=/path/to/checkpoint.pt \
       pytest tests/test_export_parity.py -v
   ```
4. Write the manifest:

   ```bash
   python -m scripts.fetch_models
   ```
5. Commit the refreshed `.onnx` files and `MANIFEST.json`.

## Interface contracts

These are the signatures of the files actually in this folder, read back from
the ONNX graphs. They differ from the contract stated in CLAUDE.md §5 in two
places, both noted below and both handled in code.

```python
# Embedder -- models/embed/redimnet2_b2.onnx
input:  "audio"     float32 [B, 1, T_samples]   # 16 kHz mono, T = 64000..96000
output: "embs"      float32 [B, 192]

# Countermeasure -- models/cm/aasist_l.onnx
input:  "waveform"  float32 [batch, 64600]      # 16 kHz, ~4.04 s
output: "logits"    float32 [batch, 2]          # softmax; index 1 = bona fide

# VAD -- models/vad/silero_vad.onnx
input:  "input" [B, 512], "state" [2, B, 128], "sr" int64 scalar
output: "output" [B, 1], "stateN" [2, B, 128]
```

Two differences from CLAUDE.md §5:

1. **The embedder input is rank 3**, `[B, 1, T]`, not `[1, N]`.
   `embedder._shape_waveform()` reads the declared rank and reshapes to match,
   so both rank-2 and rank-3 exports work.
2. **The embedder output is not L2-normalized.** A real window comes back with
   a norm around 18. `embed_window()` normalizes before returning, so the
   `(192,)` vector the rest of the system sees is unit-norm as documented —
   but do not assume the raw graph output is.

Input names are read from the session, never hardcoded, so a re-export that
renames them still works.

`speakerid verify-models` checks every file against the manifest. A mismatch
refuses to start: a silently swapped model is invisible until the accuracy
moves.
