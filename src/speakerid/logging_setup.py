"""Logging configuration.

One rule, enforced by the filter below: log lines carry person IDs and scores,
never names and never audio (CLAUDE.md §11). A log file that accumulates
"recognised Jane Doe" is a biometric record with none of the protections the
database has.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

from .config import Config

#: Extra attributes a caller might attach that would leak identity or audio
#: into a log sink. Deliberately excludes LogRecord's own fields -- `name` is
#: the logger's name and stripping it breaks every format string.
_FORBIDDEN_EXTRA_FIELDS = (
    "display_name",
    "person_name",
    "subject_name",
    "speaker_name",
    "audio",
    "waveform",
    "samples",
    "transcript",
    "passphrase",
)

#: Built-in LogRecord attributes, which must never be removed.
_RESERVED_RECORD_FIELDS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))


class PrivacyFilter(logging.Filter):
    """Drop record attributes that must never reach a log sink.

    Only ever removes extras a caller attached, never a field the logging
    machinery owns.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for field in _FORBIDDEN_EXTRA_FIELDS:
            if field in _RESERVED_RECORD_FIELDS:
                continue
            if hasattr(record, field):
                delattr(record, field)
        return True


def configure_logging(
    config: Optional[Config] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Install handlers. Safe to call more than once."""
    resolved_level = level or (config.get_str("logging.level") if config else "INFO")
    resolved_format = (
        config.get_str("logging.format")
        if config
        else "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    resolved_file = log_file or (config.get_optional_str("logging.file") if config else None)

    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(resolved_format)
    privacy = PrivacyFilter()

    # stderr, so that CLI output on stdout stays machine-readable.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(privacy)
    root.addHandler(stream_handler)

    if resolved_file:
        path = config.resolve_relative(resolved_file) if config else resolved_file
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(privacy)
        root.addHandler(file_handler)
