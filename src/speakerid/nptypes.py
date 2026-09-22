"""NumPy array type aliases.

``np.ndarray`` is generic, so ``mypy --strict`` rejects the bare form. Naming
the dtypes also documents intent: ``FloatArray`` is the 16 kHz mono float32
audio contract, and a function returning ``BoolArray`` is returning a mask,
not samples.

``numpy.typing.NDArray`` exists from NumPy 1.21, so this is safe on the
pinned 1.24.4 / Python 3.8 target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: Audio samples and embeddings. The dtype the whole system speaks.
FloatArray = npt.NDArray[np.float32]

#: Accumulators and metrics, where float32 would lose precision.
Float64Array = npt.NDArray[np.float64]

#: Masks, e.g. "which samples did the VAD call speech".
BoolArray = npt.NDArray[np.bool_]

#: Indices returned by argsort/argpartition.
IntArray = npt.NDArray[np.int64]

#: Arrays whose dtype genuinely varies, such as raw ONNX outputs.
AnyArray = npt.NDArray[Any]

__all__ = ["FloatArray", "Float64Array", "BoolArray", "IntArray", "AnyArray"]
