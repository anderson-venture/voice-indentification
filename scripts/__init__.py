"""Development and operations scripts.

These are not shipped to the device. ``export_onnx.py`` and ``fetch_models.py``
run on a connected development machine; ``build_cohort.py``, ``calibrate.py``
and ``evaluate.py`` run offline on the operator's own recordings.
"""

from __future__ import annotations
