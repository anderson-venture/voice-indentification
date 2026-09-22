"""Offline speaker recognition and anti-spoofing.

Three separate jobs that are never merged (CLAUDE.md §1):

    A. front-end  -- find speech, clean it            (Silero VAD)
    B. identity   -- *who* is this voice              (ReDimNet2-B2 embedder)
    C. liveness   -- live human or a replay           (AASIST-L countermeasure)

Identification is not a classifier over employees. The embedding model is
frozen; enrolment inserts a row in a database. Adding a person never touches
the model.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
