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

## Acquisition, on a connected machine

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
   make models          # python -m scripts.fetch_models
   ```
5. Commit the refreshed `.onnx` files and `MANIFEST.json`.

## Interface contracts

```python
# Embedder
input:  float32 [1, N]      # 16 kHz mono, N = 64000..96000
output: float32 [1, 192]    # L2-normalized

# Countermeasure (AASIST-L)
input:  float32 [1, 64600]  # 16 kHz, ~4.04 s
output: float32 [1, 2]      # logits; softmax, index 1 = bona fide

# VAD (Silero v5)
input:  input [1, 512], state [2, 1, 128], sr int64
output: output [1, 1], stateN [2, 1, 128]
```

`speakerid verify-models` checks every file against the manifest. A mismatch
refuses to start: a silently swapped model is invisible until the accuracy
moves.
