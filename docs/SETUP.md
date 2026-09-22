# Setup

Clone the repository and install. The ONNX models are committed, so a clone
gives you a working tree with nothing else to fetch.

Python **3.8** is required, and it is a hard constraint — the pins in
`requirements.txt` are the last releases that support it (CLAUDE.md §2).

---

## Windows

Python 3.8.10 is the **last version python.org shipped a Windows installer
for**, so that is the one to install:

<https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe>

Tick **"Add Python 3.8 to PATH"** in the installer.

```powershell
git clone <your-repo-url> E:\project\voice-identification
cd E:\project\voice-identification

py -3.8 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt

$env:PYTHONPATH = "src"
.venv\Scripts\python -m speakerid.cli verify-models
.venv\Scripts\python -m pytest
```

Set `PYTHONPATH` once per shell, or permanently with
`setx PYTHONPATH src` (reopen the terminal afterwards).

## Linux / macOS

```bash
git clone <your-repo-url> && cd voice-identification
python3.8 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

export PYTHONPATH=src
.venv/bin/python -m speakerid.cli verify-models
.venv/bin/python -m pytest
```

On Ubuntu, `python3.8 -m venv` needs `apt install python3.8-venv`.

---

## The three requirements files

| File | When |
|---|---|
| `requirements.txt` | always — running the system |
| `requirements-dev.txt` | to run the tests, lint, format, type-check |
| `requirements-export.txt` | **only** to re-export or fine-tune a model (~2 GB, PyTorch) |

`requirements-export.txt` is deliberately separate. Inference runs through
ONNX Runtime; PyTorch is a development-only tool and never goes on a deployed
device (CLAUDE.md §4). The three models are already exported and committed, so
you need it only if you are changing a model — most notably fine-tuning
AASIST-L on replay data, which `docs/FIRST_RUN.md` explains is necessary
before the countermeasure protects anything.

```bash
pip install -r requirements-export.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The optional localhost API additionally needs `fastapi` and `uvicorn`; the CLI
does not.

---

## Then: it still will not identify anyone

`config/thresholds.yaml` ships empty and the loader refuses to default it.
That is deliberate (CLAUDE.md §8) and it is the last setup step:

```
python -m speakerid.cli status        # shows: thresholds NOT CALIBRATED
python -m speakerid.cli calibrate --trials trials.csv --cm-trials cm_trials.csv \
    --identification-trials ident.csv --target-far 0.001 --cm-target-far 0.05 \
    --site "<this room>"
```

Calibrate on recordings made on that hardware, in that room. Never copy a
calibrated `thresholds.yaml` between sites — the numbers describe one
microphone, one acoustic environment, one enrolled population.

---

## Notes for Windows

**Nothing is pre-built.** A `.venv` cannot be copied between machines, and
especially not between operating systems: a Linux venv holds `.so` binaries
and a `bin/` directory, Windows needs `.pyd` and `Scripts/`. Always create it
on the machine you are going to use.

**The PBKDF2 salt cannot be permission-restricted.** Windows ignores POSIX
mode bits, so `data\.salt` inherits the directory ACL instead of being created
`0600`. The code logs a warning when this happens, and the POSIX-only test is
skipped on Windows while a separate test still checks the salt is created
correctly. The salt is not itself a secret — the passphrase carries the
strength — but restrict `data\` with an ACL and prefer full-disk encryption.

**Verified from Linux, not on Windows.** Every pin has a genuine
`cp38`/`win_amd64` wheel, both requirements files resolve cleanly for that
platform, and the path handling is separator-safe (`models/MANIFEST.json` keys
are normalised, and SQLAlchemy accepts `sqlite:///C:\...` URLs). But the test
suite has never actually been executed on Windows — `pytest` after install
will tell you in about 40 seconds.
