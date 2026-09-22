"""Cut the environment archive that `docs/SETUP.md` §1b distributes.

`venv/` and `python38/` are 1.8 GB of binaries and cannot go in git: two files
inside `venv/` are over GitHub's 100 MB hard limit. They travel as one zip on
Google Drive instead.

PowerShell 5.1's ``Compress-Archive`` is not used here. It buffers and falls
over near 2 GB, which is uncomfortably close to what this archive holds.
``zipfile`` with ``allowZip64=True`` streams and does not care.

Run on the connected machine, from the project root::

    venv\\Scripts\\python -m scripts.make_env_archive

It prints the SHA-256 to paste into `docs/SETUP.md` §1b beside the share link.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import zipfile
from typing import List, Tuple

#: Directories included in the archive, relative to the project root. Extracting
#: the zip at the project root must recreate exactly these.
PAYLOAD = ("venv", "python38")

DEFAULT_NAME = "voice-indentification-env-py38-win64.zip"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _collect(root: str) -> Tuple[List[Tuple[str, str]], int]:
    """Every file to archive, as (absolute path, path inside the zip)."""
    entries = []  # type: List[Tuple[str, str]]
    total = 0
    for name in PAYLOAD:
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            raise SystemExit("{0} not found; run this from the project root".format(base))
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                absolute = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(absolute)
                except OSError:
                    # A file that vanished between walk and stat is not worth
                    # failing a 1.8 GB archive over, but say so.
                    sys.stderr.write("skipped (unreadable): {0}\n".format(absolute))
                    continue
                entries.append((absolute, os.path.relpath(absolute, root)))
                total += size
    return entries, total


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        help="Archive path. Defaults to {0} beside the project folder, so the "
        "archive is never written inside what it is archiving.".format(DEFAULT_NAME),
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="No compression. Much faster, roughly 1.8 GB out; use when the "
        "upload is cheaper than the wait.",
    )
    args = parser.parse_args()

    root = _project_root()
    output = args.output or os.path.join(os.path.dirname(root), DEFAULT_NAME)
    output = os.path.abspath(output)
    if output.lower().startswith(root.lower() + os.sep):
        raise SystemExit("refusing to write the archive inside {0}".format(root))

    entries, total = _collect(root)
    sys.stderr.write(
        "archiving {0} files, {1:.1f} GB, from {2}\n".format(
            len(entries), total / float(1 << 30), root
        )
    )

    mode = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    started = time.time()
    done = 0
    with zipfile.ZipFile(output, "w", mode, allowZip64=True) as archive:
        for index, (absolute, arcname) in enumerate(entries, 1):
            try:
                archive.write(absolute, arcname)
            except OSError as exc:
                raise SystemExit("failed on {0}: {1}".format(absolute, exc)) from exc
            done += 1
            if index % 2000 == 0:
                sys.stderr.write(
                    "  {0}/{1} files, {2:.0f}s elapsed\n".format(
                        index, len(entries), time.time() - started
                    )
                )

    size = os.path.getsize(output)
    sys.stderr.write("hashing...\n")
    print("archive: {0}".format(output))
    print("files:   {0}".format(done))
    print("size:    {0:.2f} GB".format(size / float(1 << 30)))
    print("sha256:  {0}".format(_sha256(output)))
    print("")
    print("Upload it, then put the share link and this SHA-256 into docs/SETUP.md 1b.")


if __name__ == "__main__":
    main()
